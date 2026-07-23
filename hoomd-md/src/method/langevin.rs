// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `Langevin`.

use hoomd_simulation::macrostate::Temperature;
use rand::{Rng, distr::Distribution};

use hoomd_microstate::{
    Body,
    Microstate,
    SiteKey,
    Tagged,
    Transform,
    boundary::{GenerateGhosts, Wrap},
    property::{
        AngularMomentum,
        CustomBodyCartesian2,
        CustomBodyCartesian3,
        Mass,
        MomentOfInertia,
        Momentum,
        NetForce,
        NetTorque,
        NetVirial,
        Orientation,
        Position,
        RotationalMotionTypes
    }
};
use hoomd_spatial::PointUpdate;
use hoomd_vector::{Angle, Cartesian, Outer, Versor, Wedge};
use rand_distr::Uniform;
use crate::{
    RotationalKineticEnergy,
    RotationalMotion,
    TranslationalKineticEnergy,
    TranslationalMotion,
    method::{Gamma, GammaR},
    thermostat::NoThermostat
};

/// Integrate bodies' degrees of freedom in the microstate according to
/// Langevin equations of motion, modelling the NVE or NVT ensemble.
/// 
/// The `Langevin` implementation follows the same symplectic integration scheme
/// used in [`ConstantVolume`](crate::ConstantVolume), but with drag and random
/// forces and torques applied to each body *i*:
/// 
/// ```math
/// \begin{align*}
/// \vec{F}_i &= \vec{F}_C - \gamma \cdot \vec{v}_i + \vec{F}_R \\
/// \vec{\tau}_i &= \vec{\tau}_C - \vec{\gamma}_R \cdot \vec{\omega}_i + \vec{\tau}_R \\
/// \end{align*}
/// ```
/// 
/// where $` \vec{F}_C `$ and $` \vec{\tau}_C `$ are the force and torque on the
/// body from all other bodies and external interactions, $` \gamma `$ and
/// $` \vec{\gamma}_R `$ are the translational and rotational drag coefficients, and
/// $` \vec{F}_R `$ and $` \vec{\tau}_R `$ are random forces and torques. These
/// random forces and torques are uniform
/// 
/// ```math
/// \begin{align*}
/// \left< \vec{F}_R \right> &= 0 \\
/// \left< \vec{\tau}_R \right> &= 0 \\
/// \end{align*}
/// ```
/// 
/// and their magnitudes are chosen via the [fluctuation-dissipation theorem]
/// to be consistent with the specified drag and temperature
/// 
/// ```math
/// \begin{align*}
/// \left< \left| \vec{F}_R \right|^2 \right> &= \frac{2 d k T \gamma}{\Delta t} \\
/// \left< \left| \vec{\tau}_R \right|^2 \right> &= \frac{2 d_R k T \gamma_R}{\Delta t} \\
/// \end{align*}
/// ```
/// 
/// where $` d `$ and $` d_R `$ are the number of translational and rotational
/// degrees of freedom. Note that $` d_R `$ is determined by the number of
/// non-zero components of the body's moment of inertia.
/// 
/// [fluctuation-dissipation theorem]: https://en.wikipedia.org/wiki/Fluctuation%E2%80%93dissipation_theorem
/// 
/// To create a `Langevin`, use [`Langevin::builder`].
/// 
/// TODO: example
pub struct Langevin {
    /// The time step size.
    pub delta_t: f64,
}

/// Langevin forces and virials in N-dimensional cartesian space.
impl Langevin {
    /// Apply drag and random forces and virials to selected bodies in the microstate.
    /// 
    /// Drag forces are parameterized by [`Langevin.gamma`] and oppose the
    /// direction of motion. Random forces are uniform and have magnitudes that
    /// are consistent with the drag and system temperature in accordance with
    /// the fluctuation-dissipation theorem. Drag and random virials are
    /// calculated directly from these forces. For details, see
    /// [above](crate::method::langevin).
    /// 
    /// TODO: check whether virials should be updated
    /// TODO: communicate that this is defined only for CARTESIAN and is not
    /// generic across VECTOR
    #[inline]
    fn apply_drag_and_random_forces_and_virials<const N: usize, B, S, X, C, M, R, F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &self,
        rng: &mut R,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    )
    where
        B: Position<Position = Cartesian<N>>
            + Momentum<Momentum = Cartesian<N>>
            + NetForce<NetForce = Cartesian<N>>
            + NetVirial<NetVirial = <Cartesian<N> as Outer>::Tensor>
            + Mass
            + Gamma
            + Transform<S>
            + Clone,
        S: Position<Position = Cartesian<N>> + Default,
        X: PointUpdate<Cartesian<N>, SiteKey>,
        C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
        M: Temperature,
        R: Rng + ?Sized,
    {
        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue;
            }
            let mut body_properties = body.item.properties.clone();

            // Calculate the drag force
            let g = body_properties.gamma();
            let v = *body_properties.momentum() / body_properties.mass();
            let f_drag = v * g * -1.0;
            
            // Pick a random force
            let magnitude = (6.0 * macrostate.temperature() * g / self.delta_t).sqrt();
            let uniform = Uniform::new_inclusive(-1.0, 1.0).unwrap();
            let f_rand = Cartesian::<N>::from(
                core::array::from_fn(|_| magnitude * uniform.sample(rng))
            );

            // Apply drag and random forces and update virial accordingly
            *body_properties.net_force_mut() += f_drag + f_rand;
            let position = *body_properties.position();
            *body_properties.net_virial_mut() += (f_drag + f_rand).outer(&position);
        }
    }
}

/// Langevin torques in 3-dimensional cartesian space.
impl Langevin
{
    /// Apply drag and random torques to selected bodies in the microstate.
    /// 
    /// Drag torques are parameterized by [`Langevin.gamma_r`]. Random torques
    /// are uniform and have magnitudes that are consistent with the drag and
    /// system temperature in accordance with the fluctuation-dissipation
    /// theorem. For details, see [above](crate::method::langevin).
    #[inline]
    fn apply_drag_and_random_torques_3d<B, S, X, C, M, R, F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &self,
        rng: &mut R,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    )
    where
        B: Transform<S>
            + Clone
            + AngularMomentum<AngularMomentum = Cartesian<3>>
            + MomentOfInertia<MomentOfInertia = [f64; 3]>
            + NetTorque<NetTorque = Cartesian<3>>
            + GammaR<GammaR = [f64; 3]>,
        S: Position<Position = Cartesian<3>> + Default,
        X: PointUpdate<Cartesian<3>, SiteKey>,
        C: Wrap<B>
            + Wrap<S>
            + GenerateGhosts<S>,
        M: Temperature,
        R: Rng + ?Sized,
    {       
        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue;
            }
            let mut body_properties = body.item.properties.clone();

            // Calculate the drag torque
            let g_r = body_properties.gamma_r();
            let moi = body_properties.moment_of_inertia();
            let angular_momentum = body_properties.angular_momentum();
            let w = Cartesian::<3>::from(
                core::array::from_fn(|i| angular_momentum[i] / moi[i])
            );
            let t_drag = Cartesian::<3>::from(
                core::array::from_fn(|i| w[i] * g_r[i] * -1.0)
            );

            // Pick a random torque
            let uniform = Uniform::new_inclusive(-1.0, 1.0).unwrap();
            let t_rand = Cartesian::<3>::from(
                core::array::from_fn(|i| {
                    let magnitude = 
                        if moi[i] > 0.0 {
                            (6.0 * macrostate.temperature() * g_r[i] / self.delta_t).sqrt()
                        } else {
                            0.0
                        };
                    magnitude * uniform.sample(rng)
                })
            );

            // Apply drag and random torques
            *body_properties.net_torque_mut() += t_drag + t_rand;
        }
    }
}

/// Langevin torques in 2-dimensional cartesian space.
/// 
/// TODO: discuss how we link the return type of GammaR with the system's vector-space.
impl Langevin {
    /// Apply drag and random torques to selected bodies in the microstate.
    /// 
    /// Drag torques are parameterized by [`Langevin.gamma_r`]. Random torques
    /// are uniform and have magnitudes that are consistent with the drag and
    /// system temperature in accordance with the fluctuation-dissipation
    /// theorem. For details, see [above](crate::method::langevin).
    #[inline]
    fn apply_drag_and_random_torques_2d<B, S, X, C, M, R, F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &self,
        rng: &mut R,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    )
    where
        B: Transform<S>
            + Clone
            + AngularMomentum<AngularMomentum = f64>
            + MomentOfInertia<MomentOfInertia = f64>
            + NetTorque<NetTorque = f64>
            + GammaR<GammaR = f64>,
        S: Position<Position = Cartesian<2>> + Default,
        X: PointUpdate<Cartesian<2>, SiteKey>,
        C: Wrap<B>
            + Wrap<S>
            + GenerateGhosts<S>,
        M: Temperature,
        R: Rng + ?Sized,
    {       
        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue;
            }
            let mut body_properties = body.item.properties.clone();

            // Calculate the drag torque
            let g_r = body_properties.gamma_r();
            let moi = body_properties.moment_of_inertia();
            let angular_momentum = body_properties.angular_momentum();
            let w = angular_momentum / moi;
            let t_drag = w * g_r * -1.0;

            // Pick a random torque
            let uniform = Uniform::new_inclusive(-1.0, 1.0).unwrap();
            let magnitude = 
                if *moi > 0.0 {
                    (6.0 * macrostate.temperature() * g_r / self.delta_t).sqrt()
                } else {
                    0.0
                };
            let t_rand = magnitude * uniform.sample(rng);

            // Apply drag and random torques
            *body_properties.net_torque_mut() += t_drag + t_rand;
        }
    }
}

impl<const N: usize, B, S, X, C, M> TranslationalMotion<B, S, X, C, M> for Langevin
where
    B: Position<Position = Cartesian<N>>
        + Momentum<Momentum = Cartesian<N>>
        + NetForce<NetForce = Cartesian<N>>
        + NetVirial<NetVirial = <Cartesian<N> as Outer>::Tensor>
        + NetTorque
        + AngularMomentum
        + Gamma
        + Mass
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<N>> + Default,
    X: PointUpdate<Cartesian<N>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    M: Temperature,
{
    /// Integrate selected body positions forward a full step and their momenta forward a half step.
    ///
    /// This method is identical to [`ConstantVolume::integrate_translation_half_step_one_with_filter`].
    fn integrate_translation_half_step_one_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let (kinetic_energy, degrees_of_freedom) =
            microstate.translational_kinetic_energy_with_filter(&should_integrate_body);

        crate::method::constant_volume::integrate_translation_half_step_one_with_filter(
            self.delta_t,
            microstate,
            &mut NoThermostat,
            kinetic_energy,
            degrees_of_freedom,
            macrostate,
            should_integrate_body
        );
    }

    /// Apply drag and random forces and virials to bodies, then integrate
    /// selected body momenta forward a half step.
    ///
    /// Aside from the application of drag and random forces and virials, this
    /// method is identical to
    /// [`ConstantVolume::integrate_translation_half_step_two_with_filter`].
    fn integrate_translation_half_step_two_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();
        self.apply_drag_and_random_forces_and_virials(
            &mut rng,
            microstate,
            macrostate,
            &should_integrate_body,
        );

        crate::method::constant_volume::integrate_translation_half_step_two_with_filter(
            self.delta_t,
            microstate,
            &mut NoThermostat,
            macrostate,
            should_integrate_body
        );
    }
}

// TODO: check that we want to make users use the CustomBodyCartesian3 and 2 in
// order to use Langevin (because currently GammaR is not implemented on
// DynamicOrientedPoint).

/// Rotational motion in 3-dimensional cartesian space.
impl<T, S, X, C, M> RotationalMotion<CustomBodyCartesian3<T>, S, X, C, M>
    for Langevin
where
    T: Transform<S>
        + Position<Position = Cartesian<3>>
        + Orientation<Rotation = Versor>
        + AngularMomentum<AngularMomentum = <Versor as RotationalMotionTypes>::AngularMomentum>
        + MomentOfInertia<MomentOfInertia = <Versor as RotationalMotionTypes>::MomentOfInertia>
        + NetTorque<NetTorque = <Cartesian<3> as Wedge>::Bivector>,
    S: Position<Position = Cartesian<3>> + Default,
    X: PointUpdate<Cartesian<3>, SiteKey>,
    C: Wrap<CustomBodyCartesian3<T>>
        + Wrap<S>
        + GenerateGhosts<S>,
    M: Temperature,
    CustomBodyCartesian3<T>: Copy
        + Transform<S>
        + GammaR<GammaR = [f64; 3]>,
    Microstate<CustomBodyCartesian3<T>, S, X, C>: RotationalKineticEnergy<CustomBodyCartesian3<T>, S>
{
    /// Integrate selected body orientations forward a full step and their angular momenta forward a half step.
    ///
    /// This method is identical to [`ConstantVolume::integrate_rotation_half_step_one_with_filter`].
    fn integrate_rotation_half_step_one_with_filter<
        F: Fn(&Tagged<Body<CustomBodyCartesian3<T>, S>>) -> bool
    >(
        &mut self,
        microstate: &mut Microstate<CustomBodyCartesian3<T>, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        crate::method::constant_volume::integrate_rotation_half_step_one_with_filter_cartesian3(
            self.delta_t,
            microstate,
            &mut NoThermostat,
            macrostate,
            should_integrate_body
        );
    }

    /// Apply drag and random torques to bodies, then integrate selected body
    /// angular momenta forward a half step.
    ///
    /// Aside from the application of drag and random torques, this method is
    /// identical to [`ConstantVolume::integrate_rotation_half_step_two_with_filter`].
    fn integrate_rotation_half_step_two_with_filter<
        F: Fn(&Tagged<Body<CustomBodyCartesian3<T>, S>>) -> bool
    >(
        &mut self,
        microstate: &mut Microstate<CustomBodyCartesian3<T>, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();
        self.apply_drag_and_random_torques_3d(
            &mut rng,
            microstate,
            macrostate,
            &should_integrate_body,
        );
        crate::method::constant_volume::integrate_rotation_half_step_two_with_filter_cartesian3(
            self.delta_t,
            microstate,
            &mut NoThermostat,
            macrostate,
            should_integrate_body
        );
    }
}

/// Rotational motion in 2-dimensional cartesian space.
impl<T, S, X, C, M> RotationalMotion<CustomBodyCartesian2<T>, S, X, C, M>
    for Langevin
where
    T: Transform<S>
        + Position<Position = Cartesian<2>>
        + Orientation<Rotation = Angle>
        + AngularMomentum<AngularMomentum = <Angle as RotationalMotionTypes>::AngularMomentum>
        + MomentOfInertia<MomentOfInertia = <Angle as RotationalMotionTypes>::MomentOfInertia>
        + NetTorque<NetTorque = <Cartesian<2> as Wedge>::Bivector>,
    S: Position<Position = Cartesian<2>> + Default,
    X: PointUpdate<Cartesian<2>, SiteKey>,
    C: Wrap<CustomBodyCartesian2<T>>
        + Wrap<S>
        + GenerateGhosts<S>,
    M: Temperature,
    CustomBodyCartesian2<T>: Copy
        + Transform<S>
        + GammaR<GammaR = f64>,
    Microstate<CustomBodyCartesian2<T>, S, X, C>: RotationalKineticEnergy<CustomBodyCartesian2<T>, S>,
{
    /// Integrate selected body orientations forward a full step and their angular momenta forward a half step.
    ///
    /// This method is identical to [`ConstantVolume::integrate_rotation_half_step_one_with_filter`].
    fn integrate_rotation_half_step_one_with_filter<
        F: Fn(&Tagged<Body<CustomBodyCartesian2<T>, S>>) -> bool
    >(
        &mut self,
        microstate: &mut Microstate<CustomBodyCartesian2<T>, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        crate::method::constant_volume::integrate_rotation_half_step_one_with_filter_cartesian2(
            self.delta_t,
            microstate,
            &mut NoThermostat,
            macrostate,
            should_integrate_body
        );
    }

    /// Apply drag and random torques to bodies, then integrate selected body
    /// angular momenta forward a half step.
    ///
    /// Aside from the application of drag and random torques, this method is
    /// identical to [`ConstantVolume::integrate_rotation_half_step_two_with_filter`].
    fn integrate_rotation_half_step_two_with_filter<
        F: Fn(&Tagged<Body<CustomBodyCartesian2<T>, S>>) -> bool
    >(
        &mut self,
        microstate: &mut Microstate<CustomBodyCartesian2<T>, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();
        self.apply_drag_and_random_torques_2d(
            &mut rng,
            microstate,
            macrostate,
            &should_integrate_body,
        );

        crate::method::constant_volume::integrate_rotation_half_step_two_with_filter_cartesian2(
            self.delta_t,
            microstate,
            &mut NoThermostat,
            macrostate,
            should_integrate_body
        );
    }
}

#[cfg(test)]
mod tests {
    use hoomd_microstate::property::Point;
    use super::*;
    use strum::VariantNames;
    use strum_macros::VariantNames;

    #[test]
    fn test_langevin() {
        type B2 = DynamicOrientedPoint<Cartesian<2>, Angle>;
        type B3 = DynamicOrientedPoint<Cartesian<3>, Versor>;

        let dt = 2.0;
        
        let lan = Langevin::<2, B2, _, _, _>::builder(dt).build();
        assert_eq!(lan.delta_t, dt);

        let lan = Langevin::<3, B3, _, _, _>::builder(dt).build();
        assert_eq!(lan.delta_t, dt);
    }

    fn test_custom_gamma() {
        // Ensure that creating a type-dependent gamma works in Langevin for 2D.
        // type PositionVector = Cartesian<2>;
        // type BodyProperties = DynamicOrientedPoint<PositionVector, Angle>;

        // #[derive(Clone, Copy, Default, PartialEq, VariantNames)]
        // enum BodyType {
        //     #[default]
        //     A,
        //     B
        // }
        
        // #[derive(Position, Orientation, Mass, PartialEq, VariantNames)]
        // struct BodyProperties {
        //     position: PositionVector,
        //     site_type: SiteType,
        // }



        // impl Transform<SiteProperties> for BodyProperties {
        //     fn transform(&self, site_properties: &SiteProperties) -> SiteProperties {
        //         SiteProperties {
        //             position: self.position + site_properties.position,
        //             ..*site_properties
        //         }
        //     }
        // }

        // impl Gamma<BodyProperties> for SiteProperties {
        //     fn value(&self, body_properties: &BodyProperties) -> f64 {
        //         todo!()
        //     }
        
        //     fn value_mut(&mut self, body_properties: &BodyProperties) -> &mut f64 {
        //         todo!()
        //     }
        // }

    }

    fn test_custom_gamma_r() {}
}
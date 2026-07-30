// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `Langevin`.

use std::ops::{Add, AddAssign, MulAssign};

use hoomd_simulation::macrostate::Temperature;
use rand::{Rng, distr::Distribution};

use hoomd_microstate::{
    Body, Microstate, SiteKey, Tagged, Transform, boundary::{GenerateGhosts, Wrap}, property::{
        AngularMomentum, Drag, DynamicOrientedPoint, Mass, MomentOfInertia, Momentum, NetForce, NetTorque, NetVirial, Orientation, Position, RotationalDrag, RotationalMotionTypes
    }
};
use hoomd_spatial::PointUpdate;
use hoomd_vector::{Angle, Cartesian, Outer, Versor, Wedge};
use rand_distr::Uniform;
use crate::{
    RotationalKineticEnergy, RotationalMotion, Thermostat, TranslationalKineticEnergy, TranslationalMotion, method::SymplecticIntegrateRotation, thermostat::NoThermostat
};

/// Integrate bodies' degrees of freedom in the microstate according to
/// Langevin equations of motion, modelling the NVE or NVT ensemble.
/// 
/// The `Langevin` implementation follows the same symplectic integration scheme
/// used in [`ConstantVolume`], but with drag and random forces and torques
/// applied to each body *i*:
/// 
/// ```math
/// \begin{align*}
/// \vec{F}_i &= \vec{F}_C - \gamma \cdot \vec{v}_i + \vec{F}_R \\
/// \vec{\tau}_i &= \vec{\tau}_C - \gamma_R \cdot \vec{\omega}_i + \vec{\tau}_R \\
/// \end{align*}
/// ```
/// 
/// where $` \vec{F}_C `$ and $` \vec{\tau}_C `$ are the force and torque on the
/// body from all other bodies and external interactions, $` \gamma `$ and
/// $` \gamma_R `$ are the translational and rotational drag coefficients, and
/// $` \vec{F}_R `$ and $` \vec{\tau}_R `$ are random forces and torques. These
/// random forces and torques are zero-centered
/// 
/// ```math
/// \begin{align*}
/// \lang \vec{F}_R \rang &= \vec{0} \\
/// \lang \vec{\tau}_R \rang &= \vec{0} \\
/// \end{align*}
/// ```
/// 
/// and their magnitudes are uniformly distributed, with variances chosen via
/// the [fluctuation-dissipation theorem] to be consistent with the specified
/// drag and temperature
/// 
/// ```math
/// \begin{align*}
/// \lang F_{R,j} \cdot F_{R,j} \rang &= 2 k T \gamma / \Delta t \\
/// \lang \tau_{R,j} \cdot \tau_{R,j} \rang &= 2 k T \gamma_{R,j} / \Delta t \\
/// \end{align*}
/// ```
/// 
/// for each component $` j `$  of the force vector and torque bivector.
/// 
/// [fluctuation-dissipation theorem]: https://en.wikipedia.org/wiki/Fluctuation%E2%80%93dissipation_theorem
/// 
/// Because `Langevin` rescales momentum and angular momentum according to the
/// system's temperature, it can be considered as a kind of thermostat. For this
/// reason, contrary to [`ConstantVolume`], it does not store thermostats in its
/// fields. To create a `Langevin`, provide a value for `delta_t`.
/// 
/// ```
/// use hoomd_md::method::Langevin;
/// 
/// let delta_t = 0.001;
/// let mut langevin = Langevin{ delta_t };
/// ```
/// 
/// To use `Langevin`, assign translational and rotational drag coefficients
/// directly to bodies.
/// 
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # use hoomd_md::method::Langevin;
/// # let delta_t = 0.001;
/// # let mut langevin = Langevin{ delta_t };
/// use hoomd_microstate::{
///     Microstate,
///     Body,
///     property::{DynamicOrientedPoint, Point},
/// };
/// use hoomd_vector::Versor;
/// use hoomd_simulation::macrostate::Isothermal;
/// use hoomd_interaction::{
///     PairwiseCutoff,
///     Rigid,
///     pairwise::Isotropic,
///     univariate::LennardJones
/// };
/// use hoomd_md::RotationalMotion;
/// 
/// // Microstate
/// let mut microstate = Microstate::default();
/// microstate.add_body(Body::single_site(
///     DynamicOrientedPoint::<_, Versor> {
///         drag: 2.0,
///         rotational_drag: [1.0, 2.0, 3.0],
///         ..Default::default()
///     },
///     Point::default(),
/// ));
/// 
/// // Macrostate
/// let macrostate = Isothermal { temperature: 1.0 };
/// 
/// // Interaction Model
/// let lj = Rigid(PairwiseCutoff(Isotropic {
///     interaction: LennardJones::<12, 6> {
///         epsilon: 1.0,
///         sigma: 1.0
///     },
///     r_cut: 4.0,
/// }));
/// 
/// // Integrate
/// langevin.integrate_translation_and_rotation(
///     &mut microstate,
///     &macrostate,
///     &lj
/// );
/// # Ok(())
/// # }
/// ```
/// 
/// [`ConstantVolume`]: (crate::method::ConstantVolume)
pub struct Langevin {
    /// The time step size.
    pub delta_t: f64,
}

/// Langevin forces and virials in N-dimensional cartesian space.
impl Langevin {
    /// Apply drag and random forces and virials to selected bodies in the microstate.
    /// 
    /// Drag forces are parameterized by `gamma` and oppose the direction of
    /// motion. Random forces are uniform and have magnitudes that scale with
    /// drag and temperature. Drag and random virials are calculated directly
    /// from these forces. For details, see [above](Langevin).
    #[inline]
    pub fn apply_drag_and_random_forces_and_virials<const N: usize, B, S, X, C, M, R, F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &self,
        rng: &mut R,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_update_body: F,
    )
    where
        B: Position<Position = Cartesian<N>>
            + Momentum<Momentum = Cartesian<N>>
            + NetForce<NetForce = Cartesian<N>>
            + NetVirial<NetVirial = <Cartesian<N> as Outer>::Tensor>
            + Mass
            + Drag
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
            if !should_update_body(body) {
                continue;
            }
            let mut body_properties = body.item.properties.clone();

            // Calculate the drag force
            let g = body_properties.drag();
            let v = *body_properties.momentum() / body_properties.mass();
            let f_drag = v * *g * -1.0;
            
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

/// Calculate drag and random torque.
/// 
/// This trait binds drag and random torque calculation to the types that
/// represent orientation and its associated quantities. Implement this trait on
/// an orientation type to enable its use with [`Langevin`].
pub trait DragAndRandomTorque
where
    <Self as DragAndRandomTorque>::Rotation: RotationalMotionTypes,
{
    /// Type that represents a body's orientation.
    type Rotation;

    /// Type that represents a body's net torque.
    type NetTorque;

    /// Calculate drag and random torque.
    /// 
    /// Drag torques are parameterized by `rotational_drag` and oppose the
    /// direction of motion. Random torques are uniform and have magnitudes that
    /// scale with drag and temperature. For details, see [`Langevin`].
    fn drag_and_random_torque<R: Rng + ?Sized>(
        rotational_drag: &<Self::Rotation as RotationalMotionTypes>::RotationalDrag,
        moment_of_inertia: &<Self::Rotation as RotationalMotionTypes>::MomentOfInertia,
        angular_momentum: &<Self::Rotation as RotationalMotionTypes>::AngularMomentum,
        temperature: f64,
        delta_t: f64,
        rng: &mut R,
    ) -> (Self::NetTorque, Self::NetTorque);
}

impl DragAndRandomTorque for Angle {
    type Rotation = Angle;
    type NetTorque = <Cartesian<2> as Wedge>::Bivector;

    fn drag_and_random_torque<R: Rng + ?Sized>(
        rotational_drag: &<Self::Rotation as RotationalMotionTypes>::RotationalDrag,
        moment_of_inertia: &<Self::Rotation as RotationalMotionTypes>::MomentOfInertia,
        angular_momentum: &<Self::Rotation as RotationalMotionTypes>::AngularMomentum,
        temperature: f64,
        delta_t: f64,
        rng: &mut R,
    ) -> (Self::NetTorque, Self::NetTorque) {
        // Calculate the drag torque
        let angular_velocity = angular_momentum / moment_of_inertia;
        let t_drag = angular_velocity * rotational_drag * -1.0;

        // Pick a random torque
        let uniform = Uniform::new_inclusive(-1.0, 1.0).unwrap();
        let magnitude = 
            if *moment_of_inertia > 0.0 {
                (6.0 * temperature * rotational_drag / delta_t).sqrt()
            } else {
                0.0
            };
        let t_rand = magnitude * uniform.sample(rng);

        (t_drag, t_rand)
    }
}

impl DragAndRandomTorque for Versor {
    type Rotation = Versor;
    type NetTorque = <Cartesian<3> as Wedge>::Bivector;

    fn drag_and_random_torque<R: Rng + ?Sized>(
        rotational_drag: &<Self::Rotation as RotationalMotionTypes>::RotationalDrag,
        moment_of_inertia: &<Self::Rotation as RotationalMotionTypes>::MomentOfInertia,
        angular_momentum: &<Self::Rotation as RotationalMotionTypes>::AngularMomentum,
        temperature: f64,
        delta_t: f64,
        rng: &mut R,
    ) -> (Self::NetTorque, Self::NetTorque) {
        // Calculate the drag torque
        let angular_velocity = Cartesian::<3>::from(
            core::array::from_fn(|i| angular_momentum[i] / moment_of_inertia[i])
        );
        let t_drag = Cartesian::<3>::from(
            core::array::from_fn(|i| angular_velocity[i] * rotational_drag[i] * -1.0)
        );

        // Pick a random torque
        let uniform = Uniform::new_inclusive(-1.0, 1.0).unwrap();
        let t_rand = Cartesian::<3>::from(
            core::array::from_fn(|i| {
                let magnitude = 
                    if moment_of_inertia[i] > 0.0 {
                        (6.0 * temperature * rotational_drag[i] / delta_t).sqrt()
                    } else {
                        0.0
                    };
                magnitude * uniform.sample(rng)
            })
        );

        (t_drag, t_rand)
    }
}

/// Langevin torques in 2 and 3-dimensional cartesian space.
impl Langevin {
    /// Apply drag and random torques to selected bodies in the microstate.
    /// 
    /// Drag torques are parameterized by `rotational_drag` and oppose the
    /// direction of motion. Random torques are uniform and have magnitudes that
    /// scale with drag and temperature. For details, see [above](Langevin).
    #[inline]
    pub fn apply_drag_and_random_torques<V, R, B, S, X, C, M, RNG, F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &self,
        rng: &mut RNG,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_update_body: F,
    )
    where
        V: Wedge,
        R: DragAndRandomTorque<Rotation = R, NetTorque = V::Bivector> + RotationalMotionTypes,
        B: Transform<S>
            + Clone
            + Orientation<Rotation = R>
            + AngularMomentum<AngularMomentum = <R as RotationalMotionTypes>::AngularMomentum>
            + MomentOfInertia<MomentOfInertia = <R as RotationalMotionTypes>::MomentOfInertia>
            + NetTorque<NetTorque = V::Bivector>
            + RotationalDrag<RotationalDrag = <R as RotationalMotionTypes>::RotationalDrag>,
        S: Position<Position = V> + Default,
        X: PointUpdate<V, SiteKey>,
        C: Wrap<B>
            + Wrap<S>
            + GenerateGhosts<S>,
        M: Temperature,
        RNG: Rng + ?Sized,
        V::Bivector: Add<Output = V::Bivector> + AddAssign<V::Bivector>,
    {       
        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_update_body(body) {
                continue;
            }
            let mut body_properties = body.item.properties.clone();

            let (t_drag, t_rand) = <R as DragAndRandomTorque>::drag_and_random_torque(
                body_properties.rotational_drag(),
                body_properties.moment_of_inertia(),
                body_properties.angular_momentum(),
                *macrostate.temperature(),
                self.delta_t,
                rng
            );

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
        + Drag
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
    /// This method is identical to `ConstantVolume::integrate_translation_half_step_one_with_filter`.
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
    /// `ConstantVolume::integrate_translation_half_step_two_with_filter`.
    fn integrate_translation_half_step_two_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();
        microstate.increment_substep();
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

impl<V, R, B, S, X, C, M> RotationalMotion<R, B, S, X, C, M> for Langevin
where
    V: Wedge + Copy,
    R: SymplecticIntegrateRotation<Rotation = R, NetTorque = V::Bivector>
        + DragAndRandomTorque<Rotation = R, NetTorque = V::Bivector>
        + RotationalMotionTypes
        + Clone,
    B: Copy
        + Transform<S>
        + Position<Position = V>
        + Orientation<Rotation = R>
        + AngularMomentum<AngularMomentum = <R as RotationalMotionTypes>::AngularMomentum>
        + MomentOfInertia<MomentOfInertia = <R as RotationalMotionTypes>::MomentOfInertia>
        + NetTorque<NetTorque = V::Bivector>
        + RotationalDrag<RotationalDrag = <R as RotationalMotionTypes>::RotationalDrag>,
    S: Position<Position = V> + Default,
    X: PointUpdate<V, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    M: Temperature,
    Microstate<B, S, X, C>: RotationalKineticEnergy<B, S>,
    <R as RotationalMotionTypes>::AngularMomentum: MulAssign<f64> + Clone,
    V::Bivector: Add<Output = V::Bivector> + AddAssign<V::Bivector>,
{
    /// Integrate selected body orientations forward a full step and their angular momenta forward a half step.
    ///
    /// This method is identical to `ConstantVolume::integrate_rotation_half_step_one_with_filter`.
    fn integrate_rotation_half_step_one_with_filter<
        F: Fn(&Tagged<Body<B, S>>) -> bool
    >(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        crate::method::constant_volume::integrate_rotation_half_step_one_with_filter::<V, R, B, S, X, C, NoThermostat, M, F>(
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
    /// identical to `ConstantVolume::integrate_rotation_half_step_two_with_filter`.
    fn integrate_rotation_half_step_two_with_filter<
        F: Fn(&Tagged<Body<B, S>>) -> bool
    >(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();
        microstate.increment_substep();
        self.apply_drag_and_random_torques(
            &mut rng,
            microstate,
            macrostate,
            &should_integrate_body,
        );

        crate::method::constant_volume::integrate_rotation_half_step_two_with_filter(
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
    use hoomd_geometry::shape::{Cuboid, Rectangle};
    use hoomd_interaction::{MaximumInteractionRange, PairwiseCutoff, Rigid, pairwise::Isotropic, univariate::LennardJones};
    use hoomd_microstate::{boundary::{MaximumAllowableInteractionRange, Periodic}, property::{DynamicOrientedPoint, Point}};
    use hoomd_simulation::macrostate::Isothermal;
    use hoomd_spatial::VecCell;
    use super::*;
    use crate::modify::{
        ThermalizeMomentum,
        ThermalizeAngularMomentum,
        ZeroCenterMomentum,
        ZeroCenterAngularMomentum
    };

    const R_CUT: f64 = 3.0;

    /// Make a simple LJ force using the constant R_CUT.
    fn make_lj() -> Rigid<PairwiseCutoff<Isotropic<LennardJones::<12, 6>>>> {
        let epsilon = 1.0;
        let sigma = 1.0;

        Rigid(PairwiseCutoff(Isotropic {
            interaction: LennardJones::<12, 6> { epsilon, sigma },
            r_cut: R_CUT,
        }))
    }

    /// Make a simple microstate with 2 bodies of a given template placed at given positions.
    fn make_microstate<const N: usize, G, E, B, S, X, C>(
        boundary_shape: G,
        positions: &mut [Cartesian<N>; 2],
        interaction_model: &E,
        body_template: Body<B, S>,
    ) -> anyhow::Result<Microstate<B, S, VecCell<SiteKey, N>, Periodic<G>>>
    where
        Body<B, S>: Clone,
        G: MaximumAllowableInteractionRange,
        B: Transform<S> + Position<Position = Cartesian<N>>,
        S: Position<Position = Cartesian<N>> + Default,
        E: MaximumInteractionRange,
        Periodic<G>: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    {
        let vec_cell = VecCell::builder()
            .nominal_search_radius(
                interaction_model.maximum_interaction_range().try_into()?,
            )
            .build();
        let boundary =
            Periodic::new(interaction_model.maximum_interaction_range(), boundary_shape)?;
        
        let mut microstate = Microstate::builder()
            .spatial_data(vec_cell)
            .boundary(boundary)
            .try_build()?;

        for p in positions {
            let mut body = body_template.clone();
            let mut body_position = body.properties.position_mut();
            body_position = p;
            microstate.add_body(body);
        }

        Ok(microstate)
    }

    #[test]
    fn test_langevin() {
        let delta_t = 0.001;
        let langevin = Langevin{ delta_t };
        assert_eq!(delta_t, langevin.delta_t);
    }

    /// Ensure that 2D cartesian bodies move around.
    #[test]
    fn test_langevin_bodies_move_cartesian2() -> anyhow::Result<()> {
        // Interaction model
        let interaction_model = make_lj();

        // Build microstate
        type BoxShape = Rectangle;
        let boundary_shape = BoxShape::with_equal_edges((2.5 * R_CUT).try_into()?);
        let mut positions = [
            Cartesian::<2>::from([-2.0, 0.0]),
            Cartesian::<2>::from([2.0, 0.0]),
        ];
        let body_template = Body::single_site(
            DynamicOrientedPoint::<_, Angle>::default(),
            Point::default(),
        );
        let mut microstate = make_microstate::<2, _, _, _, _, VecCell<SiteKey, 2>, Periodic<BoxShape>>(
            boundary_shape,
            &mut positions,
            &interaction_model,
            body_template
        ).unwrap();

        // Macrostate and related
        let temperature = 1.5;
        let macrostate = Isothermal { temperature };

        microstate.thermalize_momentum(temperature);
        microstate.thermalize_angular_momentum(temperature);
        microstate.zero_center_angular_momentum();
        microstate.zero_center_momentum();

        // Integrate.
        let mut langevin = Langevin{ delta_t: 0.001 };
        for _ in 0..5 {
            langevin.integrate_translation(
                &mut microstate,
                &macrostate,
                &interaction_model
            );
            microstate.increment_step();
        }

        // Ensure the positions have changed
        assert_ne!(positions[0], microstate.bodies()[0].item.properties.position().clone());
        assert_ne!(positions[1], microstate.bodies()[1].item.properties.position().clone());

        Ok(())
    }

    /// Ensure that custom 3D cartesian bodies move around.
    #[test]
    fn test_langevin_bodies_move_cartesian3() -> anyhow::Result<()> {
        // Interaction model
        let interaction_model = make_lj();

        // Build microstate
        type BoxShape = Cuboid;
        let boundary_shape = BoxShape::with_equal_edges((2.5 * R_CUT).try_into()?);
        let mut positions = [
            Cartesian::<3>::from([-2.0, 0.0, 0.0]),
            Cartesian::<3>::from([2.0, 0.0, 0.0]),
        ];
        let body_template = Body::single_site(
            DynamicOrientedPoint::<_, Versor>::default(),
            Point::default(),
        );
        let mut microstate = make_microstate::<3, _, _, _, _, VecCell<SiteKey, 3>, Periodic<BoxShape>>(
            boundary_shape,
            &mut positions,
            &interaction_model,
            body_template
        ).unwrap();

        // Macrostate and related
        let temperature = 1.5;
        let macrostate = Isothermal { temperature };

        microstate.thermalize_momentum(temperature);
        microstate.thermalize_angular_momentum(temperature);
        microstate.zero_center_angular_momentum();
        microstate.zero_center_momentum();

        // Integrate.
        let mut langevin = Langevin{ delta_t: 0.001 };
        for _ in 0..5 {
            langevin.integrate_translation(
                &mut microstate,
                &macrostate,
                &interaction_model
            );
            microstate.increment_step();
        }

        // Ensure the positions have changed
        assert_ne!(positions[0], microstate.bodies()[0].item.properties.position().clone());
        assert_ne!(positions[1], microstate.bodies()[1].item.properties.position().clone());

        Ok(())
    }
}

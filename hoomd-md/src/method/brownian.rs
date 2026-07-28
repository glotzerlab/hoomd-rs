// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `Brownian`.

use core::array::from_fn;

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
        Orientation,
        Position,
        RotationalMotionTypes
    }
};
use hoomd_simulation::macrostate::Temperature;
use hoomd_spatial::PointUpdate;
use hoomd_vector::{Angle, Cartesian, Quaternion, Rotate, Versor, Wedge};

use rand_distr::{Distribution, Normal, Uniform};

use crate::{RotationalKineticEnergy, RotationalMotion, TranslationalMotion, method::{Gamma, GammaR}};

/// Integrate bodies' degrees of freedom in the microstate according to
/// Brownian equations of motion.
pub struct Brownian {
    /// The time step size.
    pub delta_t: f64,
}

impl<const N: usize, B, S, X, C, M> TranslationalMotion<B, S, X, C, M> for Brownian
where
    B: Position<Position = Cartesian<N>>
        + Momentum<Momentum = Cartesian<N>>
        + NetForce<NetForce = Cartesian<N>>
        + Gamma
        + Mass
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<N>> + Default,
    X: PointUpdate<Cartesian<N>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    M: Temperature,
{
    /// Integrate selected bodies forward a whole step. [TODO]
    fn integrate_translation_half_step_one_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();

        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue;
            }
            
            let mut body_properties = body.item.properties.clone();

            // Pick a random force
            let g = body_properties.gamma();
            let uniform = Uniform::new_inclusive(-1.0, 1.0).unwrap();
            let magnitude = (6.0 * macrostate.temperature() * g / self.delta_t).sqrt();
            let f_rand = Cartesian::<N>::from(from_fn(|_| magnitude * uniform.sample(&mut rng)));

            // Update position using the net and random forces
            let net_force = body_properties.net_force().clone();
            *body_properties.position_mut() += (net_force + f_rand) * self.delta_t / g;

            // Pick a new random momentum
            let normal = Normal::new(
                0.0,
                (macrostate.temperature() * body_properties.mass()).sqrt(),
            ).unwrap();
            *body_properties.momentum_mut() = Cartesian::<N>::from(from_fn(|_| normal.sample(&mut rng)));

            microstate
                .update_body_properties(body_index, body_properties)
                .expect(
                    "Bodies and sites should remain in simulation boundary.\n
                Add interactions that prevent sites from moving outside the boundary.",
                );
        }
    }

    /// Do nothing. (There is no second step in brownian dynamics.)
    fn integrate_translation_half_step_two_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        _microstate: &mut Microstate<B, S, X, C>,
        _macrostate: &M,
        _should_integrate_body: F,
    ) {}
}

/// Rotational motion in 3-dimensional cartesian space.
impl<R, E, S, X, C, M> RotationalMotion<CustomBodyCartesian3<R, E>, S, X, C, M>
    for Brownian
where
    CustomBodyCartesian3<R, E>: Clone
        + Transform<S>
        + Position<Position = Cartesian<3>>
        + Orientation<Rotation = Versor>
        + AngularMomentum<AngularMomentum = <Versor as RotationalMotionTypes>::AngularMomentum>
        + MomentOfInertia<MomentOfInertia = <Versor as RotationalMotionTypes>::MomentOfInertia>
        + NetTorque<NetTorque = <Cartesian<3> as Wedge>::Bivector>,
    S: Position<Position = Cartesian<3>> + Default,
    X: PointUpdate<Cartesian<3>, SiteKey>,
    C: Wrap<CustomBodyCartesian3<R, E>>
        + Wrap<S>
        + GenerateGhosts<S>,
    M: Temperature,
    CustomBodyCartesian3<R, E>: Copy
        + Transform<S>
        + GammaR<GammaR = [f64; 3]>,
    Microstate<CustomBodyCartesian3<R, E>, S, X, C>: RotationalKineticEnergy<CustomBodyCartesian3<R, E>, S>
{
    /// Integrate selected bodies forward a whole step. [TODO]
    fn integrate_rotation_half_step_one_with_filter<
        F: Fn(&Tagged<Body<CustomBodyCartesian3<R, E>, S>>) -> bool
    >(
        &mut self,
        microstate: &mut Microstate<CustomBodyCartesian3<R, E>, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();

        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue;
            }
            
            let mut body_properties = body.item.properties.clone();

            // Pick a random torque in the body frame
            let g_r = body_properties.gamma_r();
            let moi = *body_properties.moment_of_inertia();
            let t_rand = Cartesian::<3>::from(from_fn(|i| {
                let normal = Normal::new(
                    0.0,
                    (2.0 * g_r[i] * macrostate.temperature() / self.delta_t).sqrt(),
                ).unwrap();
                let is_zero = if moi[i] == 0.0 { 0.0 } else { 1.0 };
                normal.sample(&mut rng) * is_zero
            }));

            // Rotate the torque to the system frame
            let t_rand_sys = body_properties.orientation().rotate(&t_rand);
            
            // Update orientation using the net and random torques
            // TODO: check this math
            let net_torque = *body_properties.net_torque();
            let dq_dt = Cartesian::<3>::from(from_fn(|i| {
                (t_rand_sys[i] + net_torque[i]) / g_r[i]
            }));
            *body_properties.orientation_mut() = (
                *body_properties.orientation().get()
                + (
                    *body_properties.orientation().get()
                    * Quaternion::pure(dq_dt)
                    * 0.5
                    * self.delta_t
                )
            ).to_versor_unchecked();

            // Pick a new random angular momentum
            *body_properties.angular_momentum_mut() = Cartesian::<3>::from(from_fn(|i| {
                let normal = Normal::new(
                    0.0,
                    (moi[i] * macrostate.temperature()).sqrt(),
                ).unwrap();
                let is_zero = if moi[i] == 0.0 { 0.0 } else { 1.0 };
                normal.sample(&mut rng) * is_zero
            }));

            microstate
                .update_body_properties(body_index, body_properties)
                .expect(
                    "Bodies and sites should remain in simulation boundary.\n
                Add interactions that prevent sites from moving outside the boundary.",
                );
        }
    }

    /// Do nothing. (There is no second step in brownian dynamics.)
    fn integrate_rotation_half_step_two_with_filter<
        F: Fn(&Tagged<Body<CustomBodyCartesian3<R, E>, S>>) -> bool
    >(
        &mut self,
        _microstate: &mut Microstate<CustomBodyCartesian3<R, E>, S, X, C>,
        _macrostate: &M,
        _should_integrate_body: F,
    ) {}
}

/// Rotational motion in 2-dimensional cartesian space.
impl<R, E, S, X, C, M> RotationalMotion<CustomBodyCartesian2<R, E>, S, X, C, M>
    for Brownian
where
    CustomBodyCartesian2<R, E>: Clone
        + Transform<S>
        + Position<Position = Cartesian<2>>
        + Orientation<Rotation = Angle>
        + AngularMomentum<AngularMomentum = <Angle as RotationalMotionTypes>::AngularMomentum>
        + MomentOfInertia<MomentOfInertia = <Angle as RotationalMotionTypes>::MomentOfInertia>
        + NetTorque<NetTorque = <Cartesian<2> as Wedge>::Bivector>,
    S: Position<Position = Cartesian<2>> + Default,
    X: PointUpdate<Cartesian<2>, SiteKey>,
    C: Wrap<CustomBodyCartesian2<R, E>>
        + Wrap<S>
        + GenerateGhosts<S>,
    M: Temperature,
    CustomBodyCartesian2<R, E>: Copy
        + Transform<S>
        + GammaR<GammaR = f64>,
    Microstate<CustomBodyCartesian2<R, E>, S, X, C>: RotationalKineticEnergy<CustomBodyCartesian2<R, E>, S>,
{
    /// Integrate selected bodies forward a whole step. [TODO]
    fn integrate_rotation_half_step_one_with_filter<
        F: Fn(&Tagged<Body<CustomBodyCartesian2<R, E>, S>>) -> bool
    >(
        &mut self,
        microstate: &mut Microstate<CustomBodyCartesian2<R, E>, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();

        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue;
            }
            
            let mut body_properties = body.item.properties.clone();

            // Pick a random torque in the body frame
            let g_r = body_properties.gamma_r();
            let moi = *body_properties.moment_of_inertia();
            let normal = Normal::new(
                0.0,
                (2.0 * g_r * macrostate.temperature() / self.delta_t).sqrt(),
            ).unwrap();
            let t_rand = if moi == 0.0 { 0.0 } else { normal.sample(&mut rng) };
            
            // Update orientation using the net and random torques
            // TODO: check math
            let net_torque = *body_properties.net_torque();
            let dq_dt = (t_rand + net_torque) / g_r;
            let current_theta = body_properties.orientation().theta;
            let new_theta = current_theta + (
                current_theta
                * 0.5
                * self.delta_t
                * dq_dt
            );
            *body_properties.orientation_mut() = Angle::from(new_theta).to_reduced();

            // Pick a new random angular momentum
            let normal = Normal::new(
                0.0,
                (moi * macrostate.temperature()).sqrt(),
            ).unwrap();
            *body_properties.angular_momentum_mut() = if moi == 0.0 { 0.0 } else { normal.sample(&mut rng) };

            microstate
                .update_body_properties(body_index, body_properties)
                .expect(
                    "Bodies and sites should remain in simulation boundary.\n
                Add interactions that prevent sites from moving outside the boundary.",
                );
        }
    }

    /// Do nothing. (There is no second step in brownian dynamics.)
    fn integrate_rotation_half_step_two_with_filter<
        F: Fn(&Tagged<Body<CustomBodyCartesian2<R, E>, S>>) -> bool
    >(
        &mut self,
        _microstate: &mut Microstate<CustomBodyCartesian2<R, E>, S, X, C>,
        _macrostate: &M,
        _should_integrate_body: F,
    ) {}
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
    fn test_brownian() {
        let delta_t = 0.001;
        let brownian = Brownian{ delta_t };
        assert_eq!(delta_t, brownian.delta_t);
    }

    /// Ensure that custom 2D cartesian bodies move around.
    #[test]
    fn test_brownian_bodies_move_cartesian2() -> anyhow::Result<()> {
        // Define the type that will hold the extra body properties
        #[derive(Clone)]
        struct ExtraBodyProperties {
            pub gamma: f64,
            pub gamma_r: f64,
        }

        // Type alias for custom body properties type
        type CustomBodyType = CustomBodyCartesian2<
            DynamicOrientedPoint<Cartesian<2>, Angle>,
            ExtraBodyProperties
        >;

        // Impl traits required for brownian on custom body properties type
        impl Gamma for CustomBodyType {
            fn gamma(&self) -> f64 {
                self.extra.gamma
            }
        }

        impl GammaR for CustomBodyType {
            type GammaR = f64;

            fn gamma_r(&self) -> Self::GammaR {
                self.extra.gamma_r
            }
        }

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
            CustomBodyType {
                required: DynamicOrientedPoint::default(),
                extra: ExtraBodyProperties { gamma: 1.0, gamma_r: 1.0 }
            },
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
        let mut brownian = Brownian{ delta_t: 0.001 };
        for _ in 0..5 {
            brownian.integrate_translation(
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
    fn test_brownian_bodies_move_cartesian3() -> anyhow::Result<()> {
        // Define the type that will hold the extra body properties
        #[derive(Clone)]
        struct ExtraBodyProperties {
            pub gamma: f64,
            pub gamma_r: [f64; 3],
        }

        // Type alias for custom body properties type
        type CustomBodyType = CustomBodyCartesian3<
            DynamicOrientedPoint<Cartesian<3>, Versor>,
            ExtraBodyProperties
        >;

        // Impl traits required for brownian on custom body properties type
        impl Gamma for CustomBodyType {
            fn gamma(&self) -> f64 {
                self.extra.gamma
            }
        }

        impl GammaR for CustomBodyType {
            type GammaR = [f64; 3];

            fn gamma_r(&self) -> Self::GammaR {
                self.extra.gamma_r
            }
        }

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
            CustomBodyType {
                required: DynamicOrientedPoint::default(),
                extra: ExtraBodyProperties { gamma: 1.0, gamma_r: [1.0; 3] }
            },
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
        let mut brownian = Brownian { delta_t: 0.001 };
        for _ in 0..5 {
            brownian.integrate_translation(
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

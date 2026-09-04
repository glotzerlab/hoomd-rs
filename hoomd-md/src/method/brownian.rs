// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `Brownian`.

use core::array::from_fn;
use std::ops::{Add, AddAssign};

use hoomd_microstate::{
    Body,
    Microstate,
    SiteKey,
    Tagged,
    Transform,
    boundary::{GenerateGhosts, Wrap},
    property::{
        AngularMomentum,
        Drag,
        Mass,
        MomentOfInertia,
        Momentum,
        NetForce,
        NetTorque,
        Orientation,
        Position,
        RotationalDrag,
        RotationalMotionTypes
    }
};
use hoomd_simulation::macrostate::Temperature;
use hoomd_spatial::PointUpdate;
use hoomd_vector::{Angle, Cartesian, Quaternion, Rotate, Versor, Wedge};

use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};

use crate::{RotationalMotion, TranslationalMotion};

/// Integrate bodies' degrees of freedom in the microstate according to Brownian equations of motion.
/// 
/// Brownian dynamics simulate Langevin equations of motion in
/// the overdamped or diffusive limit, where particle motion is dominated by
/// thermal fluctuations ("Brownian kicks") rather than classical Newtonian
/// dynamics. Position and orientation are still coupled to force and torque,
/// but momentum and angular momentum are not, being drawn instead from a
/// thermal distribution represented by the macrostate.
/// 
/// The `Brownian` implementation follows the integration scheme by [Snook 2007]
/// (Section 6.2.5). Integration occurs in a single step and is not symplectic.
/// See the implementations for [translational motion] and [rotational motion]
/// for governing equations and other details.
/// 
/// [Snook 2007]: https://dx.doi.org/10.1016/B978-0-444-52129-3.50028-6
/// [translational motion]: Self::integrate_translation_half_step_one_with_filter
/// [rotational motion]: Self::integrate_rotation_half_step_one_with_filter
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Brownian {
    /// The time step size.
    pub delta_t: f64,
}

/// Integrate rotational degrees of freedom according to Brownian dynamics.
/// 
/// This trait binds brownian rotational integration schemes to the types that
/// represent orientation and its associated quantities: angular momentum,
/// moment of inertia, rotational drag, and net torque. Implement this trait on
/// a type that represents body orientation to make a [`Microstate`] containing
/// such bodies integrateable with [`Brownian`].
///
/// [`Microstate`]: hoomd_microstate::Microstate
pub trait BrownianIntegrateRotation: RotationalMotionTypes {
    /// Type that represents a body's net torque.
    type NetTorque;

    /// Integrate orientation and angular momentum forward a full step.
    fn step<R: Rng + ?Sized>(
        delta_t: f64,
        net_torque: &Self::NetTorque,
        angular_momentum: &mut Self::AngularMomentum,
        orientation: &mut Self,
        moment_of_inertia: &Self::MomentOfInertia,
        rotational_drag: &Self::RotationalDrag,
        temperature: f64,
        rng: &mut R,
    );
}

/// Brownian rotational integration for bodies in 2-dimensional cartesian space.
impl BrownianIntegrateRotation for Angle {
    type NetTorque = <Cartesian<2> as Wedge>::Bivector;

    /// Integrate orientation forward a full step and pick a new random angular momentum.
    /// 
    /// The brownian integration procedure in 2-dimensional cartesian space is
    /// given by the equations below, which are applied to each body *i*. Bodies
    /// which have `moment_of_inertia = 0.0` are skipped.
    /// 
    /// 1. Pick a new random torque. This random torque is zero-centered
    /// 
    ///    ```math
    ///    \lang \tau_R \rang = 0
    ///    ```
    /// 
    ///    and normally distributed, with a variance of
    /// 
    ///    ```math
    ///    \lang \tau_R \cdot \tau_R \rang = 2 k T \gamma_R / \Delta t
    ///    ```
    /// 
    ///    where $` \gamma_R `$ is the rotational drag coefficient.
    /// 
    /// 2. Integrate orientation forward using the conventional and random
    ///    torques.
    /// 
    ///    ```math
    ///    \begin{align*}
    ///    \frac{d \theta_i}{dt} &= \frac{\tau_{C,i} + \tau_R}{\gamma_R} \\
    ///    \theta_i(t + \Delta t) &= \theta_i(t) + \frac{d \theta_i}{dt} \cdot \Delta t \\
    ///    \end{align*}
    ///    ```
    /// 
    /// 3. Pick a new random angular momentum. This random angular momentum is
    ///    zero-centered
    /// 
    ///    ```math
    ///    \lang L_i(t + \Delta t) \rang = 0
    ///    ```
    /// 
    ///    and normally distributed, with a variance of
    /// 
    ///    ```math
    ///    \lang L_i(t + \Delta t) \cdot L_i(t + \Delta t) \rang = k T I
    ///    ```
    fn step<R: Rng + ?Sized>(
        delta_t: f64,
        net_torque: &Self::NetTorque,
        angular_momentum: &mut Self::AngularMomentum,
        orientation: &mut Self,
        moment_of_inertia: &Self::MomentOfInertia,
        rotational_drag: &Self::RotationalDrag,
        temperature: f64,
        rng: &mut R,
    ) {
        // Pick a random torque in the body frame
        let normal = Normal::new(
            0.0,
            (2.0 * rotational_drag * temperature / delta_t).sqrt(),
        ).unwrap();
        let t_rand = if *moment_of_inertia == 0.0 { 0.0 } else { normal.sample(rng) };
        
        // Update orientation using the net and random torques
        // TODO: check math (HOOMD-blue had this all in quaternion form)
        let dq_dt = (t_rand + net_torque) / rotational_drag;
        let current_theta = orientation.theta;
        let new_theta = current_theta + (dq_dt * delta_t);
        *orientation = Angle::from(new_theta).to_reduced();

        // Pick a new random angular momentum
        let normal = Normal::new(
            0.0,
            (moment_of_inertia * temperature).sqrt(),
        ).unwrap();
        *angular_momentum = if *moment_of_inertia == 0.0 { 0.0 } else { normal.sample(rng) };
    }
}

/// Brownian rotational integration for bodies in 3-dimensional cartesian space.
impl BrownianIntegrateRotation for Versor {
    type NetTorque = <Cartesian<3> as Wedge>::Bivector;

    /// Integrate orientation forward a full step and pick a new random angular momentum.
    /// 
    /// The brownian integration procedure in 3-dimensional cartesian space is
    /// given by the equations below, which are applied to each body *i*.
    /// Rotational degrees of freedom with a moment of inertia component of zero
    /// are skipped.
    /// 
    /// 1. Pick a new random torque. This random torque is zero-centered
    /// 
    ///    ```math
    ///    \lang \vec{\tau}_R \rang = \vec{0}
    ///    ```
    /// 
    ///    and normally distributed, with a variance of
    /// 
    ///    ```math
    ///    \lang \tau_{R,j} \cdot \tau_{R,j} \rang = 2 k T \gamma_{R,j} / \Delta t
    ///    ```
    /// 
    ///    for each component $` j `$ of the torque bivector, where
    ///    $` \gamma_R `$ are the rotational drag coefficients.
    /// 
    /// 2. Integrate orientation forward using the conventional and random
    ///    torques.
    /// 
    ///    ```math
    ///    \begin{align*}
    ///    \frac{d \mathbf{q}_i}{dt} &= \frac{\tau_{C,i} + \tau_R}{\gamma_R} \\
    ///    \mathbf{q}_i(t + \Delta t) &= \mathbf{q}_i(t) + \frac{d \mathbf{q}_i}{dt} \cdot \Delta t \\
    ///    \end{align*}
    ///    ```
    /// 
    /// 3. Pick a new random angular momentum. This random angular momentum is
    ///    zero-centered
    /// 
    ///    ```math
    ///    \lang \vec{L}_i(t + \Delta t) \rang = 0
    ///    ```
    /// 
    ///    and normally distributed, with a variance of
    /// 
    ///    ```math
    ///    \lang L_{i,j}(t + \Delta t) \cdot L_{i,j}(t + \Delta t) \rang = k T I_j
    ///    ```
    ///    for each component $` j `$ of the angular momentum vector and the
    ///    diagonalized moment of inertia.
    fn step<R: Rng + ?Sized>(
        delta_t: f64,
        net_torque: &Self::NetTorque,
        angular_momentum: &mut Self::AngularMomentum,
        orientation: &mut Self,
        moment_of_inertia: &Self::MomentOfInertia,
        rotational_drag: &Self::RotationalDrag,
        temperature: f64,
        rng: &mut R,
    ) {
        // Pick a random torque in the body frame
        let t_rand = Cartesian::<3>::from(from_fn(|i| {
            let normal = Normal::new(
                0.0,
                (2.0 * rotational_drag[i] * temperature / delta_t).sqrt(),
            ).unwrap();
            let is_zero = if moment_of_inertia[i] == 0.0 { 0.0 } else { 1.0 };
            normal.sample(rng) * is_zero
        }));

        // Rotate the torque to the system frame
        let t_rand_sys = orientation.rotate(&t_rand);
        
        // Update orientation using the net and random torques
        // TODO: check this math
        let dq_dt = Cartesian::<3>::from(from_fn(|i| {
            (t_rand_sys[i] + net_torque[i]) / rotational_drag[i]
        }));
        *orientation = (
            *orientation.get()
            + (
                *orientation.get()
                * Quaternion::pure(dq_dt)
                * 0.5
                * delta_t
            )
        ).to_versor_unchecked();

        // Pick a new random angular momentum
        *angular_momentum = Cartesian::<3>::from(from_fn(|i| {
            let normal = Normal::new(
                0.0,
                (moment_of_inertia[i] * temperature).sqrt(),
            ).unwrap();
            let is_zero = if moment_of_inertia[i] == 0.0 { 0.0 } else { 1.0 };
            normal.sample(rng) * is_zero
        }));
    }
}

impl<const N: usize, B, S, X, C, M> TranslationalMotion<B, S, X, C, M> for Brownian
where
    B: Position<Position = Cartesian<N>>
        + Momentum<Momentum = Cartesian<N>>
        + NetForce<NetForce = Cartesian<N>>
        + Drag
        + Mass
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<N>> + Default,
    X: PointUpdate<Cartesian<N>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    M: Temperature,
{
    /// Integrate position forward a full step and pick a new random momentum.
    /// 
    /// The brownian integration procedure in cartesian space is given by the
    /// equations below, which are applied to each body *i*.
    /// 
    /// 1. Pick a new random force. This random force is zero-centered
    /// 
    ///    ```math
    ///    \lang \vec{F}_R \rang = \vec{0}
    ///    ```
    /// 
    ///    and normally distributed, with a variance of
    /// 
    ///    ```math
    ///    \lang \vec{F}_{R,j} \cdot \vec{F}_{R,j} \rang = 2 k T \gamma / \Delta t
    ///    ```
    /// 
    ///    for each component $` j `$ of the force vector, where $` \gamma `$ is
    ///    the drag coefficient.
    /// 
    /// 2. Integrate position forward using the conventional and random
    ///    forces.
    /// 
    ///    ```math
    ///    \begin{align*}
    ///    \frac{d \vec{r}_i}{dt} &= \frac{\vec{F}_{C,i} + \vec{F}_R}{\gamma} \\
    ///    \vec{r}_i(t + \Delta t) &= \vec{r}_i(t) + \frac{d \vec{r}_i}{dt} \cdot \Delta t \\
    ///    \end{align*}
    ///    ```
    /// 
    /// 3. Pick a new random momentum. This random momentum is zero-centered
    /// 
    ///    ```math
    ///    \lang \vec{p}_i(t + \Delta t) \rang = 0
    ///    ```
    /// 
    ///    and normally distributed, with a variance of
    /// 
    ///    ```math
    ///    \lang \vec{p}_{i,j}(t + \Delta t) \cdot \vec{p}_{i,j}(t + \Delta t) \rang = k T m
    ///    ```
    /// 
    ///    for each component $` j `$ of the momentum vector.
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
            let g = body_properties.drag().clone();
            let uniform = Uniform::new_inclusive(-1.0, 1.0).unwrap();
            let magnitude = (6.0 * macrostate.temperature() * g / self.delta_t).sqrt();
            let f_rand = Cartesian::<N>::from(
                from_fn(|_| magnitude * uniform.sample(&mut rng))
            );

            // Update position using the net and random forces
            let net_force = body_properties.net_force().clone();
            *body_properties.position_mut() += (net_force + f_rand) * self.delta_t / g;

            // Pick a new random momentum
            let normal = Normal::new(
                0.0,
                (macrostate.temperature() * body_properties.mass()).sqrt(),
            ).unwrap();
            *body_properties.momentum_mut() = Cartesian::<N>::from(
                from_fn(|_| normal.sample(&mut rng))
            );

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

impl<V, R, B, S, X, C, M> RotationalMotion<R, B, S, X, C, M> for Brownian
where
    V: Wedge + Copy,
    R: BrownianIntegrateRotation<NetTorque = V::Bivector>
        + Clone,
    B: Copy
        + Transform<S>
        + Position<Position = V>
        + Orientation<Rotation = R>
        + AngularMomentum<AngularMomentum = R::AngularMomentum>
        + MomentOfInertia<MomentOfInertia = R::MomentOfInertia>
        + NetTorque<NetTorque = V::Bivector>
        + RotationalDrag<RotationalDrag = R::RotationalDrag>,
    S: Position<Position = V> + Default,
    X: PointUpdate<V, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    M: Temperature,
    V::Bivector: Add<Output = V::Bivector>
        + AddAssign<V::Bivector>,
    R::AngularMomentum: Clone,
{
    /// Integrate selected body orientations and angular momenta forward a whole step.
    /// 
    /// For details, see the implementations for [`BrownianIntegrateRotation`].
    fn integrate_rotation_half_step_one_with_filter<
        F: Fn(&Tagged<Body<B, S>>) -> bool
    >(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    ) {
        let mut rng = microstate.counter().make_rng();
        microstate.increment_substep();

        for body_index in 0..microstate.bodies().len() {
            let body = &microstate.bodies()[body_index];
            if !should_integrate_body(body) {
                continue;
            }
            
            let mut body_properties = body.item.properties;

            let mut orientation = body_properties.orientation().clone();
            let mut angular_momentum = body_properties.angular_momentum().clone();
            <R as BrownianIntegrateRotation>::step(
                self.delta_t,
                body_properties.net_torque(),
                &mut angular_momentum,
                &mut orientation,
                body_properties.moment_of_inertia(),
                body_properties.rotational_drag(),
                *macrostate.temperature(),
                &mut rng,
            );

            *body_properties.angular_momentum_mut() = angular_momentum;
            *body_properties.orientation_mut() = orientation;

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
        F: Fn(&Tagged<Body<B, S>>) -> bool
    >(
        &mut self,
        _microstate: &mut Microstate<B, S, X, C>,
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
    use hoomd_derive::derive_dynamic_oriented_point;

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
            *body.properties.position_mut() = *p;
            microstate.add_body(body).unwrap();
        }

        Ok(microstate)
    }

    #[test]
    fn test_brownian() {
        let delta_t = 0.001;
        let brownian = Brownian{ delta_t };
        assert_eq!(delta_t, brownian.delta_t);
    }

    /// Ensure that simple 2D cartesian bodies move around.
    #[test]
    fn test_brownian_simple_bodies_move_cartesian2() -> anyhow::Result<()> {
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

    /// Ensure that custom 2D cartesian bodies move around.
    #[test]
    fn test_brownian_custom_bodies_move_cartesian2() -> anyhow::Result<()> {
        // Custom body properties type
        #[derive_dynamic_oriented_point(Cartesian::<2>, Angle)]
        struct CustomDynamicOrientedPoint<'a> {
            name: &'a str,
        }
        
        // Interaction model
        let interaction_model = make_lj();

        // Build microstate
        type BoxShape = Rectangle;
        let boundary_shape = BoxShape::with_equal_edges((2.5 * R_CUT).try_into()?);
        let mut positions = [
            Cartesian::from([-2.0, 0.0]),
            Cartesian::from([2.0, 0.0]),
        ];
        let body_template = Body::single_site(
            CustomDynamicOrientedPoint {
                name: "Jimothy",
                ..Default::default()
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

        // Ensure the extra data is still there
        assert_eq!("Jimothy", microstate.bodies()[0].item.properties.name);
        assert_eq!("Jimothy", microstate.bodies()[1].item.properties.name);

        Ok(())
    }

    /// Ensure that simple 3D cartesian bodies move around.
    #[test]
    fn test_brownian_simple_bodies_move_cartesian3() -> anyhow::Result<()> {
        // Interaction model
        let interaction_model = make_lj();

        // Build microstate
        type BoxShape = Cuboid;
        let boundary_shape = BoxShape::with_equal_edges((2.5 * R_CUT).try_into()?);
        let mut positions = [
            Cartesian::from([-2.0, 0.0, 0.0]),
            Cartesian::from([2.0, 0.0, 0.0]),
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

    /// Ensure that custom 3D cartesian bodies move around.
    #[test]
    fn test_brownian_custom_bodies_move_cartesian3() -> anyhow::Result<()> {
        // Custom body properties type
        #[derive_dynamic_oriented_point(Cartesian::<3>, Versor)]
        // #[derive(Clone)]
        struct CustomDynamicOrientedPoint<'a> {
            name: &'a str,
        }
        
        // Interaction model
        let interaction_model = make_lj();

        // Build microstate
        type BoxShape = Cuboid;
        let boundary_shape = BoxShape::with_equal_edges((2.5 * R_CUT).try_into()?);
        let mut positions = [
            Cartesian::from([-2.0, 0.0, 0.0]),
            Cartesian::from([2.0, 0.0, 0.0]),
        ];
        let body_template = Body::single_site(
            CustomDynamicOrientedPoint {
                name: "Jimothy",
                ..Default::default()
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

        // Ensure the extra data is still there
        assert_eq!("Jimothy", microstate.bodies()[0].item.properties.name);
        assert_eq!("Jimothy", microstate.bodies()[1].item.properties.name);

        Ok(())
    }
}

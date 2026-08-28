// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `ThermalizeAngularMomentum`

use super::ThermalizeAngularMomentum;
use hoomd_microstate::{
    Body,
    Microstate,
    SiteKey,
    Tagged,
    Transform,
    boundary::{GenerateGhosts, Wrap},
    property::{
        AngularMomentum,
        MomentOfInertia,
        Orientation,
        Position,
        RotationalMotionTypes
    },
};
use hoomd_spatial::PointUpdate;
use hoomd_vector::{Angle, Versor};
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Thermalize rotational degrees of freedom.
/// 
/// This trait binds rotational thermalization schemes to the types that
/// represent orientation and its associated quantities: angular momentum and
/// moment of inertia. Implement this trait on a type that represents body
/// orientation to make a [`Microstate`] containing such bodies thermalizeable
/// with [`ThermalizeAngularMomentum`].
pub trait ThermalizeRotation: RotationalMotionTypes {
    /// Draw a new angular momentum from the thermal distribution.
    fn thermalize<R: Rng + ?Sized>(
        temperature: &f64,
        moment_of_inertia: &Self::MomentOfInertia,
        angular_momentum: &mut Self::AngularMomentum,
        rng: &mut R
    );
}

/// Rotational thermalization for bodies in 2-dimensional cartesian space.
impl ThermalizeRotation for Angle {
    /// Draw a new angular momentum from the thermal distribution.
    /// 
    /// If the moment of inertia is zero, the angular momentum is not changed.
    /// 
    /// The new angular momentum is zero-centered
    /// 
    /// ```math
    /// \lang L \rang = 0
    /// ```
    /// 
    /// and normally distributed, with a variance of
    /// 
    /// ```math
    /// \lang L \cdot L \rang = k T I
    /// ```
    fn thermalize<R: Rng + ?Sized>(
        temperature: &f64,
        moment_of_inertia: &Self::MomentOfInertia,
        angular_momentum: &mut Self::AngularMomentum,
        rng: &mut R,
    ) {
        if *moment_of_inertia != 0.0 {
            let sigma = (temperature * moment_of_inertia).sqrt();
            let normal = Normal::new(0.0, sigma).expect("Normal distribution should be valid");
            *angular_momentum = normal.sample(rng);
        }
    }
}

/// Rotational thermalization for bodies in 3-dimensional cartesian space.
impl ThermalizeRotation for Versor {
    /// Draw a new angular momentum from the thermal distribution.
    /// 
    /// Rotational degrees of freedom with a moment of inertia component of zero
    /// are not changed.
    /// 
    /// The new angular momentum is zero-centered
    /// 
    /// ```math
    /// \lang \vec{L} \rang = \vec{0}
    /// ```
    /// 
    /// and normally distributed, with a variance of
    /// 
    /// ```math
    /// \lang L_j \cdot L_j \rang = k T I_j
    /// ```
    /// 
    /// for each component $` j `$ of the angular momentum vector and the
    /// diagonalized moment of inertia.
    fn thermalize<R: Rng + ?Sized>(
        temperature: &f64,
        moment_of_inertia: &Self::MomentOfInertia,
        angular_momentum: &mut Self::AngularMomentum,
        rng: &mut R
    ) {
        let x_nonzero = moment_of_inertia[0] > 0.0;
        let y_nonzero = moment_of_inertia[1] > 0.0;
        let z_nonzero = moment_of_inertia[2] > 0.0;
        
        let sigma_x = (temperature * moment_of_inertia[0]).sqrt();
        let sigma_y = (temperature * moment_of_inertia[1]).sqrt();
        let sigma_z = (temperature * moment_of_inertia[2]).sqrt();

        let normal_x = Normal::new(0.0, sigma_x).expect("Normal distribution should be valid.");
        let normal_y = Normal::new(0.0, sigma_y).expect("Normal distribution should be valid.");
        let normal_z = Normal::new(0.0, sigma_z).expect("Normal distribution should be valid.");

        if x_nonzero { angular_momentum[0] = normal_x.sample(rng) };
        if y_nonzero { angular_momentum[1] = normal_y.sample(rng) };
        if z_nonzero { angular_momentum[2] = normal_z.sample(rng) };
    }
}

/// Draw random angular momenta from a thermal distribution.
///
/// Angular momenta are drawn from the [Maxwell–Boltzmann distribution],
/// following a procedure that is bound to the type representing orientation
/// through the trait [`ThermalizeRotation`].
/// 
/// [Maxwell–Boltzmann distribution]: https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution
/// 
/// For example, in 3-dimensional cartesian space, where orientation is
/// represented by a [`Versor`], each component of the angular momentum
/// $` L_i `$ (aligned to the principal axes) is normally distributed with mean
/// 0 and variance $` \sigma^2 = I_i k T`$. The probability distribution is then
/// given by
/// 
/// ```math
///    f(L_i) = \frac{1}{\sqrt{2 \pi I_i k T}} \exp{\left( -\frac{L_i^2}{2 I_i k T} \right)}
/// ```
///
/// where $`I`$ is the diagonalized moment of inertia.
/// 
/// [`Versor`]: hoomd_vector::Versor
impl<V, R, B, S, X, C> ThermalizeAngularMomentum<B, S> for Microstate<B, S, X, C>
where
    V: Copy,
    R: ThermalizeRotation,
    B: Copy
        + Transform<S>
        + Position<Position = V>
        + Orientation<Rotation = R>
        + MomentOfInertia<MomentOfInertia = <R as RotationalMotionTypes>::MomentOfInertia>
        + AngularMomentum<AngularMomentum = <R as RotationalMotionTypes>::AngularMomentum>,
    S: Position<Position = V> + Default,
    X: PointUpdate<V, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    <R as RotationalMotionTypes>::AngularMomentum: Clone,
{
    #[inline]
    fn thermalize_angular_momentum_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        temperature: f64,
        should_thermalize_body: F,
    ) {
        let mut rng = self.counter().make_rng();

        for body_index in 0..self.bodies().len() {
            let body = &self.bodies()[body_index];
            if !should_thermalize_body(body) {
                continue;
            }

            let mut body_properties = body.item.properties;

            let moment_of_inertia = body_properties.moment_of_inertia();
            let mut angular_momentum = body_properties.angular_momentum().clone();

            <R as ThermalizeRotation>::thermalize(
                &temperature,
                moment_of_inertia,
                &mut angular_momentum,
                &mut rng
            );

            *body_properties.angular_momentum_mut() = angular_momentum;
            self.update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        self.increment_substep();
    }
}

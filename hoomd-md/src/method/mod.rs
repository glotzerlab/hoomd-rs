// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Integration methods.

mod constant_volume;
pub use constant_volume::{ConstantVolume, ConstantVolumeBuilder};

mod langevin;
use hoomd_microstate::property::NetForce;
use hoomd_vector::Cartesian;
pub use langevin::{Langevin, LangevinBuilder};

/// The translational drag coefficient.
/// 
/// `Gamma` describes a type that provides a float representing $` \gamma `$,
/// the translational drag coefficient used in [`Langevin`] and [`Brownian`]
/// integration. Implement this trait on a new type to assign different drag
/// coefficients to different sites.
/// 
/// The generic type names are:
/// * `B`: The [`Body::properties`](hoomd_microstate::Body) type.
pub trait Gamma<B> {
    /// Access the value for a site.
    fn value(&self, body_properties: &B) -> f64;

    /// Access the value for a site (mutable).
    fn value_mut(&mut self, body_properties: &B) -> &mut f64;
}

/// The rotational drag coefficients.
/// 
/// `GammaR` describes a type that provides an array of N floats representing
/// $` \gamma_R `$, the rotational drag coefficients for the rotational degrees
/// of freedom used in [`Langevin`] and [`Brownian`] integration. Implement this
/// trait on a new type to assign different sets of drag coefficients to
/// different sites.
/// The generic type names are:
/// * `N`: The number of dimensions. (TODO: revise)
/// * `B`: The [`Body::properties`](hoomd_microstate::Body) type.
pub trait GammaR<B> {
    /// gamma_r vector type.
    type GammaR;

    /// Access the value for a site
    fn value(&self, body_properties: &B) -> &Self::GammaR;

    /// Access the value for a site (mutable).
    fn value_mut(&mut self, body_properties: &B) -> &mut Self::GammaR;
}

impl<B> Gamma<B> for f64 {    
    fn value(&self, _: &B) -> f64 {
        *self
    }

    fn value_mut(&mut self, _: &B) -> &mut f64 {
        self
    }
}

impl<const N: usize, B> GammaR<B> for Cartesian<N> {
    type GammaR = Cartesian<N>;

    fn value(&self, _: &B) -> &Self::GammaR {
        self
    }
    
    fn value_mut(&mut self, _: &B) -> &mut Self::GammaR {
        self
    }
}

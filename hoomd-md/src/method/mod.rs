// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Integration methods.

mod constant_volume;
pub use constant_volume::{ConstantVolume, ConstantVolumeBuilder};

mod langevin;
use hoomd_microstate::property::{AngularMomentum, Momentum};
use hoomd_vector::Cartesian;
pub use langevin::{Langevin, LangevinBuilder};

/// The translational drag coefficient.
/// 
/// `Gamma` describes a type that provides a float representing $` \gamma `$,
/// the translational drag coefficient used in [`Langevin`] and [`Brownian`]
/// integration. Implement this trait on a new type to assign different drag
/// coefficients to different bodies.
/// 
/// The generic type names are:
/// * `B`: The [`Body::properties`](hoomd_microstate::Body) type.
pub trait Gamma<B: Momentum> {
    /// Access the value for a site.
    fn value(&self, body_properties: &B) -> f64;

    /// Access the value for a site (mutable).
    fn value_mut(&mut self, body_properties: &B) -> &mut f64;
}

impl<B: Momentum> Gamma<B> for f64 {    
    fn value(&self, _: &B) -> f64 {
        *self
    }

    fn value_mut(&mut self, _: &B) -> &mut f64 {
        self
    }
}

/// The rotational drag coefficients.
/// 
/// `GammaR` describes a type that provides an array of N floats representing
/// $` \gamma_R `$, the rotational drag coefficients for the rotational degrees
/// of freedom used in [`Langevin`] and [`Brownian`] integration. Implement this
/// trait on a new type to assign different sets of drag coefficients to
/// different bodies.
/// The generic type names are:
/// * `B`: The [`Body::properties`](hoomd_microstate::Body) type.
pub trait GammaR<B: AngularMomentum> {
    /// The type containing the rotational drag coefficient(s).
    type GammaR;

    /// Access the value for a site
    fn value(&self, body_properties: &B) -> &Self::GammaR;

    /// Access the value for a site (mutable).
    fn value_mut(&mut self, body_properties: &B) -> &mut Self::GammaR;
}

impl<B: AngularMomentum<AngularMomentum = f64>> GammaR<B> for f64 {
    type GammaR = f64;

    fn value(&self, _: &B) -> &f64 {
        self
    }
    
    fn value_mut(&mut self, _: &B) -> &mut f64 {
        self
    }
}

impl<B: AngularMomentum<AngularMomentum = Cartesian<3>>> GammaR<B> for [f64; 3] {
    type GammaR = [f64; 3];

    fn value(&self, _: &B) -> &[f64; 3] {
        self
    }
    
    fn value_mut(&mut self, _: &B) -> &mut [f64; 3] {
        self
    }
}

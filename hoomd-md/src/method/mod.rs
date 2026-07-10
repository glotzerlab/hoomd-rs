// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Integration methods.

mod constant_volume;
pub use constant_volume::{ConstantVolume, ConstantVolumeBuilder};

mod langevin;
pub use langevin::{Langevin, LangevinBuilder};

/// The translational drag coefficient.
/// 
/// `Gamma` describes a type that provides a float representing $` \gamma `$,
/// the translational drag coefficient used in [`Langevin`] and [`Brownian`]
/// integration. Implement this trait on a new type to assign different drag
/// coefficients to different sites.
pub trait Gamma {
    type BodyProperties;

    /// Access the value for a site.
    fn value(&self, site_properties: &Self::BodyProperties) -> f64;

    /// Access the value for a site (mutable).
    fn value_mut(&mut self, site_properties: &Self::BodyProperties) -> &mut f64;
}

/// The rotational drag coefficients.
/// 
/// `GammaR` describes a type that provides an array of N floats representing
/// $` \gamma_R `$, the rotational drag coefficients for the rotational degrees
/// of freedom used in [`Langevin`] and [`Brownian`] integration. Implement this
/// trait on a new type to assign different sets of drag coefficients to
/// different sites.
pub trait GammaR<const N: usize> {
    type BodyProperties;

    /// Access the value for a site
    fn value(&self, site_properties: &Self::BodyProperties) -> [f64; N];

    /// Access the value for a site (mutable).
    fn value_mut(&mut self, site_properties: &Self::BodyProperties) -> &mut [f64; N];
}

impl Gamma for f64 {
    type BodyProperties = usize;
    
    fn value(&self, _: &Self::BodyProperties) -> f64 {
        *self
    }

    fn value_mut(&mut self, _: &Self::BodyProperties) -> &mut f64 {
        self
    }
}

impl<const N: usize> GammaR<N> for [f64; N] {
    type BodyProperties = usize;
    
    fn value(&self, _: &Self::BodyProperties) -> [f64; N] {
        *self
    }
    
    fn value_mut(&mut self, _: &Self::BodyProperties) -> &mut [f64; N] {
        self
    }
}

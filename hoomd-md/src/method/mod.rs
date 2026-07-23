// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Integration methods.

mod constant_volume;
pub use constant_volume::{ConstantVolume, ConstantVolumeBuilder};

mod langevin;
pub use langevin::Langevin;

/// The translational drag coefficient.
/// 
/// `Gamma` describes a type that provides a float representing $` \gamma `$,
/// the translational drag coefficient used in [`Langevin`] and [`Brownian`]
/// integration. Implement this trait on a body properties object to assign a
/// specific drag coefficient to a specific body.
/// 
/// The generic type names are:
/// * `B`: The [`Body::properties`](hoomd_microstate::Body) type.
pub trait Gamma {
    /// Access the drag coefficient for the body.
    fn gamma(&self) -> f64;
}

/// The rotational drag coefficients.
/// 
/// `GammaR` describes a type that provides an array of N floats representing
/// $` \gamma_R `$, the rotational drag coefficients for the rotational degrees
/// of freedom used in [`Langevin`] and [`Brownian`] integration. Implement this
/// trait on a body properties object to assign specific drag coefficients to
/// specific bodies.
/// 
/// The generic type names are:
/// * `B`: The [`Body::properties`](hoomd_microstate::Body) type.
pub trait GammaR {
    /// The type containing the rotational drag coefficient(s).
    type GammaR;

    /// Access the value for a site
    fn gamma_r(&self) -> &Self::GammaR;
}

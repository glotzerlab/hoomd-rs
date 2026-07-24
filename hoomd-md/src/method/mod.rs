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
/// integration. Implement this trait on a body properties object to assign
/// translational drag coefficient values to bodies.
pub trait Gamma {
    /// Access the drag coefficient for the body.
    fn gamma(&self) -> f64;
}

/// The rotational drag coefficients.
/// 
/// `GammaR` describes a type that provides one or more floats representing
/// $` \gamma_R `$, the rotational drag coefficients for the rotational degrees
/// of freedom used in [`Langevin`] and [`Brownian`] integration. Implement this
/// trait on a body properties object to assign rotational drag coefficient
/// values to bodies.
/// 
/// <div class="warning">
/// For rotational integration in 2-dimensional cartesian space, the
/// associtated type of this trait must be <code>f64</code>. For 3-dimensional
/// cartesian space, the associated type must be <code>[f64; 3]</code>.
/// </div>
pub trait GammaR {
    /// The type containing the rotational drag coefficient(s).
    type GammaR;

    /// Access the value for a site
    fn gamma_r(&self) -> Self::GammaR;
}

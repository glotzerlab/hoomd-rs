// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Pairwise interactions.

use hoomd_vector::{Rotate, Vector};

pub mod angular_mask;
#[doc(inline)]
pub use angular_mask::AngularMask;

mod anisotropic;
mod approximate_shape_overlap;
mod hard_shape;
mod isotropic;

pub use anisotropic::Anisotropic;
pub use approximate_shape_overlap::ApproximateShapeOverlap;
pub use hard_shape::{HardShape, HardSphere};
pub use isotropic::Isotropic;

/// Computes pairwise energies between oriented particles.
///
/// An anisotropic pairwise energy is function of the relative position and
/// orientation of the *j* particle in *i's* reference frame:
/// ```math
/// U(\vec{r}_{ij}, \mathbf{o}_{ij})
/// ```
///
/// Implement [`AnisotropicEnergy`] on a custom type or use one of the provided
/// potentials in [`pairwise`](crate::pairwise) in MD or MC simulations.
pub trait AnisotropicEnergy<V: Vector, R: Rotate<V>> {
    /// Compute the pairwise energy between two oriented particles.
    /// ```math
    /// U(\vec{r}_{ij}, \mathbf{o}_{ij})
    /// ```
    #[must_use]
    fn energy(&self, r_ij: &V, o_ij: &R) -> f64;
}

// TODO: determine how to express the torque return type in a general way. Possibly use
// an associated type of Rotation.

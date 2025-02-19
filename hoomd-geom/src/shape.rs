// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use crate::sphere::Sphere;
/// General traits for shapes
use hoomd_vector::Vector;

// CLoneable but not copyable
// Intersects works on a centered reference to a shape?
// "Intersects" should take position and orientation as params 
// Even if Centered/Oriented doesnt work for HPMC, maybe we use something different for that

/// The N-hypervolume of a geometry. In 2D, this is area and in 3D this is Volume.
pub trait Volume {
    /// The N-hypervolume of a geometry
    #[must_use]
    fn volume(&self) -> f64;
}

/** A generalization of properties that are well defined for arbitrary shapes.

This trait requires a dimension `N` and a coordinate system defined by a [`Vector`] `V`.
*/
pub trait Shape<const N: usize, V: Vector> {
    /// Intertia Tensor
    // fn inertia_tensor(&self) -> Cartesian::<N, Cartesian<N>>;

    /// Bounding sphere. Maybe should be part of the Particle trait?
    fn bounding_sphere(&self) -> Sphere<N>; // NOT minimal bounding sphere: just a small one
}



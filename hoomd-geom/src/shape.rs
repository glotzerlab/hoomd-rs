// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use crate::sphere::Sphere;
/// General traits for shapes
use hoomd_vector::{Rotate, Vector, Cartesian};

// Cloneable but not copyable
// Intersects works on a centered reference to a shape?
// "Intersects" should take position and orientation as params
// Even if Centered/Oriented doesnt work for HPMC, maybe we use something different for that

/// The N-hypervolume of a geometry. In 2D, this is area and in 3D this is Volume.
pub trait Volume {
    /// The N-hypervolume of a geometry
    #[must_use]
    fn volume(&self) -> f64;

    // /// The (N-1)-hypervolume of a geometry
    // #[must_use]
    // fn surface_area(&self) -> f64;
}

/** A generalization of properties that are well defined for arbitrary shapes.

This trait requires a dimension `N` and a coordinate system defined by a [`Vector`] `V`.
*/
pub trait Shape<const N: usize, V: Vector> {
    /// The Euler Characteristic of the `Shape`
    fn euler_characteristic(&self) -> i32;
    /// Inertia Tensor
    // fn inertia_tensor(&self) -> Cartesian::<N, Cartesian<N>>;

    /// Bounding sphere. Maybe should be part of the Particle trait?
    fn bounding_sphere(&self) -> Sphere<N>; // NOT minimal bounding sphere: just a small one
    // NOTE: HPMC will often access the (centered) bounding sphere - should be cached?

    // fn is_inside(&self, v: V) -> bool;
}

/**
Definitions of the minimum distance between two `Shape`s. Will be zero if points are on
a boundary (within floating-point precision) and negative if the shapes are overlapping.

TODO: is it possible to have this return an intersection "depth"?
*/
pub trait MinDistance<const N: usize, V: Vector, R: Rotate<V>, S: Shape<N, V>> {
    /// Minimum distance between two `Shape`s in `N` dimensions
    fn min_distance(&self, other: &S, v_ij: &V, o_ij: R) -> f64;
}

pub trait SupportFn<const N: usize> {
    fn support(&self) -> Cartesian<N>;
}

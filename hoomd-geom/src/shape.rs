// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use crate::sphere::Sphere;
/// General traits for shapes
use hoomd_vector::Vector;

/// The N-hypervolume of a geometry. In 2D, this is area and in 3D this is Volume.
pub trait Volume {
    /// The N-hypervolume of a geometry
    #[must_use]
    fn volume(&self) -> f64;
}

/// Bounding trait for structs that are valid simulation particles
trait Particle {} // In different crate! - should be copyable (array based?)

/**
Trait bound to assert a shape is convex. This allows additional optimizations in e.g.
`Intersects` implementations.
*/
pub trait Convex {
    /// Whether a shape is convex.
    #[must_use]
    #[inline]
    fn is_convex(&self) -> bool {
        true
    }
    // Can be used to enable Xenocollide or something? provides a nice trait bound
}

/** A generalization of properties that are well defined for arbitrary shapes.

This trait requires a dimension `N` and a coordinate system defined by a [`Vector`] `V`.
*/
pub trait Shape<const N: usize, V: Vector> {
    /// Center of mass
    fn centroid(&self) -> V;
    /// Bounding sphere. Maybe should be part of the Particle trait?
    fn bounding_sphere(&self) -> Sphere<N, V>; // NOT minimal bounding sphere: just a small one
}

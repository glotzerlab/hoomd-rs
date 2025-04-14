// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! General traits for [`Shape`]s.*/
use crate::sphere::Sphere;
use hoomd_vector::{Rotate, Vector};

/// The N-hypervolume of a geometry. In 2D, this is area and in 3D this is Volume.
pub trait Volume {
    /// The N-hypervolume of a geometry
    #[must_use]
    fn volume(&self) -> f64;
}

/** A generalization of properties that are well defined for arbitrary shapes.

This trait requires a dimension `N` and a coordinate system defined by a [`Vector`] `V`.
*/
pub trait Shape<const N: usize> {
    /// Bounding sphere. Maybe should be part of the Particle trait?
    fn bounding_sphere(&self) -> Sphere<N>; // NOT minimal bounding sphere: just a small one
                                            // NOTE: HPMC will often access the (centered) bounding sphere - should be cached?

    // fn is_inside(&self, v: V) -> bool;
}

/**
Definitions of the minimum distance between two `Shape`s. Will be zero if points are on
a boundary (within floating-point precision) and negative if the shapes are overlapping.

*/

const XENOCOLLIDE_2D_MAX_ITERATIONS: usize = 1024;

pub trait MinDistance<const N: usize, V: Vector, R: Rotate<V>, S: Shape<N>> {
    /// Minimum distance between two `Shape`s in `N` dimensions
    fn min_distance(&self, other: &S, v_ij: &V, o_ij: R) -> f64;
}

/**
The support function of a geometry.
*/
pub trait SupportFn<V: Vector> {
    /// Center of mass of the shape
    /// Distances from the origin to each supporting hyperplane.
    fn support(&self, n: &V) -> V;
}

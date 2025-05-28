// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! General traits for [`Shape`]s.*/
use hoomd_vector::{Rotate, Vector};

/// The N-hypervolume of a geometry. In 2D, this is area and in 3D this is Volume.
pub trait Volume {
    /// The N-hypervolume of a geometry
    #[must_use]
    fn volume(&self) -> f64;
}

/**
Definitions of the minimum distance between two `Shape`s. Will be zero if points are on
a boundary (within floating-point precision) and negative if the shapes are overlapping.
*/
pub trait MinDistance<const N: usize, V: Vector, R: Rotate<V>, S> {
    /// Minimum distance between two `Shape`s in `N` dimensions
    fn min_distance(&self, other: &S, v_ij: &V, o_ij: R) -> f64;
}

/**
The support function of a geometry.

TODO: SupportFn should be called SupportMapping (fn typically returns dot product)
*/
pub trait SupportFn<V: Vector> {
    /// Center of mass of the shape
    /// Distances from the origin to each supporting hyperplane.
    fn support(&self, n: &V) -> V;
}

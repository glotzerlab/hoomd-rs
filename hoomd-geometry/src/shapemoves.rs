// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implementations of transformations of shapes
//!
//! This module provides trait implementations and helper utilities that
//! transform concrete shape types (scaling, shear, etc.).
//!
//! See [`crate::shape`] for the available shapes these impls target.

use crate::shape::{
    Capsule, Cylinder, Hypercuboid, Hyperellipsoid, Hyperparallelepiped, Hypersphere, Simplex3,
};
use hoomd_linear_algebra::{MatMul, SquareMatrix, matrix::Matrix};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Cartesian;

/// A shape that supports uniform scaling.
pub trait Scale {
    /// Uniformly scale the shape by the given positive factor.
    fn scale(&mut self, scale_factor: PositiveReal);
}

/// A shape that supports a shear transformation in `N` dimensions.
pub trait Shear<const N: usize> {
    /// Shear the shape by `angle` about the specified axes.
    ///
    /// `parallel_axis` defines the direction to shear along, and
    /// `perpendicular_axis` defines the direction in which the shear is applied.
    fn shear(
        &mut self,
        angle: f64,
        parallel_axis: &Cartesian<N>,
        perpendicular_axis: &Cartesian<N>,
    );
}

// pub trait Elongate{
//     fn scale(&mut self, ,scale_factor: PositiveReal);
// }

impl<const N: usize> Scale for Capsule<N> {
    /// Scale the capsule by scaling both its height and radius.
    #[inline]
    fn scale(&mut self, scale_factor: PositiveReal) {
        self.height *= scale_factor;
        self.radius *= scale_factor;
    }
}

impl Scale for Cylinder {
    /// Scale the cylinder by scaling both its height and radius.
    #[inline]
    fn scale(&mut self, scale_factor: PositiveReal) {
        self.height *= scale_factor;
        self.radius *= scale_factor;
    }
}

impl<const N: usize> Scale for Hypercuboid<N> {
    /// Scale the hypercuboid by scaling every edge length uniformly.
    #[inline]
    fn scale(&mut self, scale_factor: PositiveReal) {
        self.edge_lengths = self.edge_lengths.map(|v| v * scale_factor);
    }
}

impl<const N: usize> Scale for Hyperparallelepiped<N> {
    /// Scale the hyperparallelepiped by scaling each edge vector. The volume of the
    /// parallelepiped will scale by a factor of $`\alpha^N`$, where $`\alpha`$ is the scale factor.
    #[inline]
    fn scale(&mut self, scale_factor: PositiveReal) {
        self.edge_vectors = self.edge_vectors.map(|v| v * scale_factor);
    }
}

impl<const N: usize> Scale for Hypersphere<N> {
    /// Scale the hypersphere by scaling its radius.
    #[inline]
    fn scale(&mut self, scale_factor: PositiveReal) {
        self.radius *= scale_factor;
    }
}

impl<const N: usize> Scale for Hyperellipsoid<N> {
    /// Scale the hyperellipsoid by scaling each semi-axis.
    #[inline]
    fn scale(&mut self, scale_factor: PositiveReal) {
        *self = Hyperellipsoid::with_semi_axes(self.semi_axes().map(|v| v * scale_factor));
    }
}

impl Scale for Simplex3 {
    /// Scale the simplex by scaling every vertex coordinate.
    #[inline]
    fn scale(&mut self, scale_factor: PositiveReal) {
        for vertex in &mut self.vertices {
            *vertex *= scale_factor;
        }
    }
}

impl<const N: usize> Shear<N> for Hyperparallelepiped<N> {
    /// Apply a shear transformation to the hyperparallelepiped.
    ///
    /// The `parallel_axis` defines the direction along which the shear is
    /// performed, and `perpendicular_axis` defines the direction of displacement.
    #[inline]
    fn shear(
        &mut self,
        angle: f64,
        parallel_axis: &Cartesian<N>,
        perpendicular_axis: &Cartesian<N>,
    ) {
        let shear_matrix = Matrix::<N, N>::identity()
            + perpendicular_axis
                .to_column_matrix()
                .matmul(&parallel_axis.to_row_matrix())
                * angle.tan();
        self.edge_vectors = self.edge_vectors.map(|v| Cartesian {
            coordinates: v.to_row_matrix().matmul(&shear_matrix).rows[0],
        });
    }
}

#[cfg(test)]
#[expect(clippy::used_underscore_binding, reason = "Required for const tests.")]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    #[test]
    fn test_cuboid_scale() {
        let scale_factor: PositiveReal = 5.0.try_into().unwrap();
        let mut my_cuboid = Hypercuboid::<3> {
            edge_lengths: [1., 2., 1.].map(|x| x.try_into().unwrap()),
        };
        my_cuboid.scale(scale_factor);

        assert_eq!(
            my_cuboid.edge_lengths,
            [5., 10., 5.].map(|x| x.try_into().unwrap())
        );
    }

    #[test]
    fn test_parallelepiped_shear() {
        let mut my_box = Hyperparallelepiped::<3>::default();
        let parallel_axis = Cartesian {
            coordinates: [1., 0., 0.],
        };
        let perpendicular_axis = Cartesian {
            coordinates: [0., 0., 1.],
        };
        let angle = std::f64::consts::PI / 6.0;
        my_box.shear(angle, &parallel_axis, &perpendicular_axis);
        assert_relative_eq!(my_box.edge_vectors[0], [1., 0., 0.].into(), epsilon = 1e-14);
        assert_relative_eq!(my_box.edge_vectors[1], [0., 1., 0.].into(), epsilon = 1e-14);
        assert_relative_eq!(
            my_box.edge_vectors[2],
            [f64::sqrt(3.) / 3., 0., 1.].into(),
            epsilon = 1e-14
        );
    }
}

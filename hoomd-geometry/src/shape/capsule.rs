// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Capsule`] */

use crate::{BoundingSphere, SupportMapping};

use hoomd_vector::{Cartesian, Vector};

use super::Hypersphere;

/// All points less than or equal to a distance `r` along a line of length `h`.
#[derive(Clone, Copy, Debug)] // TODO: describe origin and orientation
pub struct Capsule<const N: usize> {
    /// Radius of of points that are considered enclosed in the shape.
    pub r: f64,
    /// Length of the line segment.
    pub h: f64,
}

impl<const N: usize> From<(f64, f64)> for Capsule<N> {
    #[inline]
    fn from(value: (f64, f64)) -> Self {
        Capsule {
            r: value.0,
            h: value.1,
        }
    }
}

impl<const N: usize> SupportMapping<Cartesian<N>> for Capsule<N> {
    #[inline]
    fn support_mapping(&self, n: &Cartesian<N>) -> Cartesian<N> {
        // Same support function as a ConvexPolyhedron with 2 vertices, plus the radius.
        let mut v_tip = [0.0; N];
        v_tip[N - 1] = self.h;
        let v_tip = v_tip.into();

        let mut v_base = [0.0; N];
        v_base[N - 1] = -self.h;
        let v_base = v_base.into();

        let (v_tip_dot_n, v_base_dot_n) = (n.dot(&v_tip), n.dot(&v_base));

        let rshift = *n * self.r * n.norm();
        if v_tip_dot_n > v_base_dot_n {
            v_tip / n.norm() + rshift
        } else {
            v_base / n.norm() + rshift
        }
    }
}

impl<const N: usize> BoundingSphere<N> for Capsule<N> {
    #[inline]
    fn bounding_sphere(&self) -> Hypersphere<N> {
        Hypersphere {
            r: self.h / 2.0 + self.r,
        }
    }
}

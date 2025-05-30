// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Capsule`] */

use crate::{BoundingSphere, SupportMapping, Volume};

use hoomd_vector::{Cartesian, Vector};

use super::Hypersphere;

/** All points less than or equal to a distance `r` along a line of length `h`.
This line is oriented along the `[0 0 ... 1]` direction, and has extents `+h/2`, `-h/2`
along that axis.
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Capsule<const N: usize> {
    /// Radius of of points that are considered enclosed in the shape.
    pub r: f64,
    /// Length of the line segment.
    pub h: f64,
}

impl<const N: usize> SupportMapping<Cartesian<N>> for Capsule<N> {
    #[inline]
    fn support_mapping(&self, n: &Cartesian<N>) -> Cartesian<N> {
        // Same support function as a ConvexPolyhedron with 2 vertices, plus the radius.
        let mut v_tip = [0.0; N];
        v_tip[N - 1] = self.h / 2.0;
        let v_tip = v_tip.into();

        let mut v_base = [0.0; N];
        v_base[N - 1] = -self.h / 2.0;
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

impl<const N: usize> Volume for Capsule<N> {
    #[inline]
    fn volume(&self) -> f64 {
        Hypersphere::<{ M }> { r: self.r }.volume() * self.h
            + Hypersphere::<N> { r: self.r }.volume()
    }
}

#[cfg(test)]
mod tests {

    use crate::shape::Sphere;

    use super::*;
    use approx::assert_relative_eq;
    use rstest::*;
    use std::marker::PhantomData;

    #[rstest]
    fn test_capsule_volume(#[values(0.0, 0.1, 1.0, 99.9)] r: f64) {
        assert_eq!(Capsule::<3> { r, h: 0.0 }.volume(), Sphere { r }.volume())
    }
}

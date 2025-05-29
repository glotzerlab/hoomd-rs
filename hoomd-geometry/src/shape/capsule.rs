// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Capsule`] */

use crate::SupportMapping;

use hoomd_vector::{Cartesian, Vector};

/// A [`Capsule`] in three dimensions.
#[derive(Clone, Copy, Debug)]
// pub struct Capsule<const N: usize> {
pub struct Capsule {
    /// Radius of the [`Capsule`]'s spherical caps.
    r: f64,
    /// Distance between the centers of the spherical caps.
    h: f64,
}

impl From<(f64, f64)> for Capsule {
    #[inline]
    fn from(value: (f64, f64)) -> Self {
        Capsule {
            r: value.0,
            h: value.1,
        }
    }
}

impl SupportMapping<Cartesian<3>> for Capsule {
    #[inline]
    fn support_mapping(&self, n: &Cartesian<3>) -> Cartesian<3> {
        /*Same support function as a ConvexPolyhedron with 2 vertices, plus the radius*/
        let (v_tip, v_base) = ([0.0, 0.0, self.h].into(), [0.0, 0.0, -self.h].into());

        let (v_tip_dot_n, v_base_dot_n) = (n.dot(&v_tip), n.dot(&v_base));

        let rshift = *n * self.r * n.norm();
        if v_tip_dot_n > v_base_dot_n {
            v_tip / n.norm() + rshift
        } else {
            v_base / n.norm() + rshift
        }
    }
}

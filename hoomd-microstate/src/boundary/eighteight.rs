// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement the {8,8} tiling of hyperbolic space
*/

use super::Boundary;
use crate::property::Point;

use hoomd_manifold::{FundamentalDomain, Hyperboloid};

/** The {8,8} tile of hyperbolic space
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EightEight {
    /// Skirt width of the hyperboloid
    pub skirt: f64,
}

impl Boundary<Hyperboloid<3>, Point<Hyperboloid<3>>, Point<Hyperboloid<3>>> for EightEight {
    #[inline]
    fn is_inside(&self, point: &Hyperboloid<3>) -> bool {
        point.distance_to_boundary() >= 0.0
    }
}

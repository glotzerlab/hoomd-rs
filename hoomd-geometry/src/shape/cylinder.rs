// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Cylinder`]

use super::Circle;
use crate::Volume;
use hoomd_utility::valid::PositiveReal;

/// A circle with normal `[0 0 1]` swept by `h/2` in the `+z` and `-z` directions.
///
/// # Example
///
/// [`Cylinder`] implements the [`Volume`] trait, which is equivalent to
/// $` \pi r^2 h `$.
/// ```
/// use hoomd_geometry::{Volume, shape::Cylinder};
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cyl = Cylinder {
///     radius: 2.0.try_into()?,
///     height: 3.0.try_into()?,
/// };
/// assert_eq!(cyl.volume(), PI * (2.0 * 2.0) * 3.0);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Cylinder {
    /// Radius of the [`Cylinder`]
    pub radius: PositiveReal,
    /// Height of the [`Cylinder`]
    pub height: PositiveReal,
}

impl Volume for Cylinder {
    #[inline]
    fn volume(&self) -> f64 {
        Circle {
            radius: self.radius,
        }
        .volume()
            * self.height.get()
    }
}

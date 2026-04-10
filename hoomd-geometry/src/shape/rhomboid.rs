// TODO: Oblique?

use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, Rotate, Rotation};

use crate::IntersectsAt;

/// An axis-aligned parallelogram defined by a 2 x 2 upper triangular matrix.
///
/// This shape is a general case of rhombus where pairs of sides are not equal.
pub struct Rhomboid {
    /// The extents [``L_x``, ``L_y``] of each edge of the Rhomboid along.
    extents: [PositiveReal; 2],
    /// The tilt factor TODO
    xy: f64,
}

impl<R> IntersectsAt<Rhomboid, Cartesian<2>, R> for Rhomboid
where
    R: Rotate<Cartesian<2>> + Rotation,
{
    #[inline]
    fn intersects_at(&self, other: &Rhomboid, v_ij: &Cartesian<2>, o_ij: &R) -> bool {
        true
    }
}

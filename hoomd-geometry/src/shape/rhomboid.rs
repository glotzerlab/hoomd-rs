// TODO: Oblique?

use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, Rotate, Rotation};

use crate::{BoundingSphereRadius, IntersectsAt};

/// An axis-aligned parallelogram defined by a 2 x 2 upper triangular matrix.
///
/// This shape is a general case of rhombus where pairs of sides are not equal.
/// We enforce the convention that the center of the shape is at the origin.
pub struct Rhomboid {
    /// The extents [``L_x``, ``L_y``] of each edge of the Rhomboid along.
    extents: [PositiveReal; 2],
    /// The shear applied to the shape in the x direction relative to ``L_y``
    xy: f64,
}

impl Rhomboid {
    #[inline(always)]
    pub fn lx(&self) -> PositiveReal {
        self.extents[0]
    }
    #[inline(always)]
    pub fn ly(&self) -> PositiveReal {
        self.extents[1]
    }
    #[inline(always)]
    pub fn xy(&self) -> f64 {
        self.xy
    }

    #[inline]
    #[must_use]
    pub fn vertices(&self) -> [Cartesian<2>; 4] {
        let ly_xy = self.ly().get() * self.xy();
        [
            [0.0, 0.0].into(),
            [self.lx().get(), 0.0].into(),
            [self.lx().get() + ly_xy, self.ly().get()].into(),
            [ly_xy, self.ly().get()].into(),
        ]
    }
}

impl BoundingSphereRadius for Rhomboid {
    #[inline]
    fn bounding_sphere_radius(&self) -> PositiveReal {
        f64::sqrt().into()
    }
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

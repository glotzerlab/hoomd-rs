// TODO: Oblique?

use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, Rotate, Rotation};

use crate::{BoundingSphereRadius, IntersectsAt, SupportMapping, Volume};

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

impl From<(PositiveReal, PositiveReal, f64)> for Rhomboid {
    #[inline]
    fn from(value: (PositiveReal, PositiveReal, f64)) -> Self {
        Rhomboid {
            extents: [value.0, value.1],
            xy: value.2,
        }
    }
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

    /// Compute the vertices of the Rhomboid assuming it is centered at the origin.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> [Cartesian<2>; 4] {
        let half_lx = self.lx().get() * 0.5;
        let half_ly = self.ly().get() * 0.5;
        let half_ly_xy = half_ly * self.xy();

        [
            [-half_lx - half_ly_xy, -half_ly].into(),
            [half_lx - half_ly_xy, -half_ly].into(),
            [half_lx + half_ly_xy, half_ly].into(),
            [-half_lx + half_ly_xy, half_ly].into(),
        ]
    }
    /// Represent a 2D triclinic box in the GSD box format.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rhomboid;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rhomb = Rhomboid::from((1.0.try_into(), 2.0.try_into(), 1.5))?;
    ///
    /// let gsd_box = triclinic.to_gsd_box();
    /// assert_eq!(gsd_box, [1.0, 2.0, 0.0, 1.5, 0.0, 0.0]);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use] // TODO: check
    pub fn to_gsd_box(&self) -> [f64; 6] {
        [
            self.extents[0].get(),
            self.extents[1].get(),
            0.0,
            self.xy,
            0.0,
            0.0,
        ]
    }
}

impl Volume for Rhomboid {
    #[inline]
    fn volume(&self) -> f64 {
        // When A is triangular, det(A) = det(diag(A))
        self.lx().get() * self.ly().get()
    }
}

impl SupportMapping<Cartesian<2>> for Rhomboid {
    #[inline]
    fn support_mapping(&self, n: &Cartesian<2>) -> Cartesian<2> {
        let mut iter = n
            .into_iter()
            .zip(self.extents)
            .map(|(n_i, l_i)| l_i.get() / 2.0 * n_i.signum());
        std::array::from_fn(|_| iter.next().expect("2==2")).into()
    }
}

impl BoundingSphereRadius for Rhomboid {
    #[inline]
    fn bounding_sphere_radius(&self) -> PositiveReal {
        // || maximal_extent || / 2.0 = { lx + ly * xy, ly } || / 2.0
        f64::sqrt((self.lx().get() + self.ly().get() * self.xy()).powi(2) + self.ly().get().powi(2))
            .try_into()
            .expect("Norm is always positive.")
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

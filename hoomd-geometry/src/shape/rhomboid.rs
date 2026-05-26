// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Rhomboid`]

use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, InnerProduct, Metric, Rotate, Rotation, RotationMatrix};
use serde::{Deserialize, Serialize};

use crate::{
    BoundingSphereRadius, IntersectsAt, IntersectsAtGlobal, IsPointInside, MapPoint, Scale,
    SupportMapping, Volume, shape::Hyperparallelepiped,
};

/// A non-orthogonal 2D box
///
/// A rhomboid is a 2D parallelogram shape defined by two edge lengths $`(L_x, L_y)`$
/// and a shear factor $`xy`$ that describes the shearing in the x-direction relative
/// to the y-extent.
///
/// The shape is centered at the origin, with the centroid at $`(0,0)`$.
///
/// # Construction
///
/// Rhomboids can be constructed using the `from_box_vector` method, which takes
/// an array of 3 values: `[lx, ly, xy]`. They can also be created from a 2D
/// parallelepiped using the `from_parallelogram` method, or directly from a tuple
/// of `(PositiveReal, PositiveReal, f64)`.
///
/// # Examples
///
/// Basic construction and methods:
/// ```
/// use hoomd_geometry::{Volume, shape::Rhomboid};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let rhomboid = Rhomboid::from_box_vector([10.0, 12.0, 1.0]);
/// assert_eq!(rhomboid.volume(), 120.0);
///
/// assert_eq!(rhomboid.lx().get(), 10.0);
/// assert_eq!(rhomboid.ly().get(), 12.0);
/// # Ok(())
/// # }
/// ```
///
/// Checking if a point is inside the rhomboid:
/// ```
/// use hoomd_geometry::{IsPointInside, shape::Rhomboid};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let rhomboid = Rhomboid::from_box_vector([6.0, 8.0, 0.5]);
///
/// assert!(rhomboid.is_point_inside(&[1.0, 1.0].into()));
/// assert!(!rhomboid.is_point_inside(&[4.0, 1.0].into()));
/// # Ok(())
/// # }
/// ```
///
/// Scaling the rhomboid:
/// ```
/// use hoomd_geometry::{Scale, shape::Rhomboid};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let rhomboid = Rhomboid::from_box_vector([10.0, 12.0, 1.0]);
///
/// let scaled = rhomboid.scale_length(2.0.try_into()?);
/// assert_eq!(scaled.lx().get(), 20.0);
/// assert_eq!(scaled.ly().get(), 24.0);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, PartialEq, Copy, Clone, Serialize, Deserialize)]
pub struct Rhomboid {
    /// The extents [``L_x``, ``L_y``] of each edge along the Cartesian axes ``x`` and ``y``.
    extents: [PositiveReal; 2],
    /// The shear applied to the shape in the x direction relative to ``L_y``
    xy: f64,
}

impl From<(PositiveReal, PositiveReal, f64)> for Rhomboid {
    /// Construct a rhomboid from its extents and shear factor.
    ///
    /// The tuple is interpreted as `(lx, ly, xy)`.
    #[inline]
    fn from(value: (PositiveReal, PositiveReal, f64)) -> Self {
        Rhomboid {
            extents: [value.0, value.1],
            xy: value.2,
        }
    }
}

impl Rhomboid {
    /// Construct a rhomboid from box dimensions.
    ///
    /// The dimensions array should contain [lx, ly, xy] where:
    /// - lx, ly are the edge lengths (must be positive)
    /// - xy is the shear factor
    ///
    /// # Panics
    ///
    /// Panics if any of lx, ly are not positive.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rhomboid;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rhomboid = Rhomboid::from_box_vector([10.0, 12.0, 1.0]);
    /// assert_eq!(rhomboid.lx().get(), 10.0);
    /// assert_eq!(rhomboid.xy(), 1.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn from_box_vector(box_dimensions: [f64; 3]) -> Self {
        Self {
            extents: [
                box_dimensions[0]
                    .try_into()
                    .expect("Extent lx must be positive"),
                box_dimensions[1]
                    .try_into()
                    .expect("Extent ly must be positive"),
            ],
            xy: box_dimensions[2],
        }
    }

    /// Construct a rhomboid from a 2D hyperparallelepiped.
    ///
    /// Computes the rhomboid box parameters from a parallelogram with edge vectors $`\vec{u}_i`$by computing
    /// the edge vectors and applying the transformation formulas:
    /// ```math
    ///     L_x = |\vec{u}_1|, \quad L_y = \sqrt{|\vec{u}_2|^2 - \frac{(\vec{u}_1 \cdot \vec{u}_2)^2}{|\vec{u}_1|^2}} \quad
    ///     xy = \frac{\vec{u}_1 \cdot \vec{u}_2}{|\vec{u}_1| L_y}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::shape::{Hyperparallelepiped, Rhomboid};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let parallelepiped = Hyperparallelepiped::new([
    ///     Cartesian::from([1.0, 0.0]),
    ///     Cartesian::from([0.5, 1.0]),
    /// ]);
    /// let rhomboid = Rhomboid::from_parallelogram(&parallelepiped);
    /// assert_relative_eq!(rhomboid.lx().get(), 1.0, epsilon = 1e-8);
    /// assert_relative_eq!(rhomboid.xy(), 0.5, epsilon = 1e-8);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn from_parallelogram(parallelepiped: &Hyperparallelepiped<2>) -> Self {
        let v1 = parallelepiped.edge_vectors[0];
        let v2 = parallelepiped.edge_vectors[1];

        let v1_mag = v1.norm();
        let v2_dot_v1 = v2.dot(&v1);

        let lx = v1_mag;
        let a2x = v2_dot_v1 / v1_mag;
        let ly = (v2.dot(&v2) - a2x * a2x).sqrt();
        let xy = a2x / ly;

        Self {
            extents: [
                lx.try_into().expect("lx must be positive"),
                ly.try_into().expect("ly must be positive"),
            ],
            xy,
        }
    }

    /// Returns the edge length in the x-direction (lx)
    #[inline]
    #[must_use]
    pub fn lx(&self) -> PositiveReal {
        self.extents[0]
    }

    /// Returns the edge length in the y-direction (ly)
    #[inline]
    #[must_use]
    pub fn ly(&self) -> PositiveReal {
        self.extents[1]
    }

    /// Returns the xy shear factor
    #[inline]
    #[must_use]
    pub fn xy(&self) -> f64 {
        self.xy
    }

    /// Convert a Cartesian vector to fractional (lattice) coordinates.
    ///
    /// Fractional coordinates express a point as coefficients of the edge
    /// vectors. If the edge vectors form the columns of matrix $`\mathbf{A}`$, then
    /// the fractional coordinate vector $`\vec{s}`$ satisfies $`\mathbf{A}\vec{s}=\vec{r}`$.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rhomboid;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rhomboid = Rhomboid::from_box_vector([4.0, 6.0, 0.0]);
    ///
    /// // A point at (1.0, 1.5) should have fractional coords (0.25, 0.25)
    /// let frac = rhomboid.to_fractional(&Cartesian::from([1.0, 1.5]));
    /// assert!((frac[0] - 0.25).abs() < 1e-10);
    /// assert!((frac[1] - 0.25).abs() < 1e-10);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn to_fractional(&self, pos: &Cartesian<2>) -> Cartesian<2> {
        let lx = self.lx().get();
        let ly = self.ly().get();
        let xy = self.xy();

        let s1 = (pos[0] - xy * pos[1]) / lx;
        let s2 = pos[1] / ly;

        Cartesian::from([s1, s2])
    }

    /// Convert fractional coordinates to absolute position.
    ///
    /// This is the inverse operation of `to_fractional`:
    /// ```math
    /// \begin{align*}
    ///     r_1 &= L_x s_1 + xy \cdot L_y s_2\\
    ///     r_2 &= L_y s_2
    /// \end{align*}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rhomboid;
    /// use hoomd_microstate::boundary::Periodic;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rhomboid = Rhomboid::from((2.0.try_into()?, 2.0.try_into()?, 0.0));
    /// let periodic = Periodic::new(1.0, rhomboid)?;
    ///
    /// let frac = Cartesian::from([0.5, 0.0]);
    /// let pos = periodic.to_absolute(&frac);
    /// assert_eq!(pos, Cartesian::from([1.0, 0.0]));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn to_absolute(&self, frac: &Cartesian<2>) -> Cartesian<2> {
        let lx = self.lx().get();
        let ly = self.ly().get();
        let xy = self.xy();

        let r1 = lx * frac[0] + xy * ly * frac[1];
        let r2 = ly * frac[1];

        Cartesian::from([r1, r2])
    }

    /// Compute the vertices of the Rhomboid assuming it is centered at the origin.
    ///
    /// Returns the four vertices of the rhomboid in counter-clockwise order,
    /// starting from the bottom-left corner.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rhomboid;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rhomboid = Rhomboid::from_box_vector([2.0, 3.0, 0.0]);
    /// let vertices = rhomboid.vertices();
    ///
    /// assert_eq!(vertices[0], Cartesian::from([-1.0, -1.5]));
    /// assert_eq!(vertices[1], Cartesian::from([1.0, -1.5]));
    /// assert_eq!(vertices[2], Cartesian::from([1.0, 1.5]));
    /// assert_eq!(vertices[3], Cartesian::from([-1.0, 1.5]));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> [Cartesian<2>; 4] {
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
            .map(|c| self.to_absolute(&Cartesian::from(c)))
    }

    /// Return the rhomboid edge vectors in Cartesian coordinates.
    ///
    /// The first edge vector is aligned with the x-axis and the second is
    /// sheared by the `xy` factor in the x direction.
    #[inline]
    #[must_use]
    pub fn get_edge_vectors(&self) -> [Cartesian<2>; 2] {
        let mut edge_vectors = [Cartesian::<2>::default(); 2];
        edge_vectors[0] = [self.lx().get(), 0.].into();
        edge_vectors[1] = [self.ly().get() * self.xy(), self.ly().get()].into();
        edge_vectors
    }

    /// Get the interior box angle of the rhomboid.
    ///
    /// The rhomboid has a single shear angle between its two edge vectors.
    /// This method returns the angle between the first edge vector aligned with
    /// the x-axis and the sheared second edge vector.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rhomboid;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rhomboid = Rhomboid::from_box_vector([2.0, 3.0, 1.0]);
    /// let angle = rhomboid.get_box_angle();
    /// assert!((angle - (1.0 / (1.0 + 1.0_f64).sqrt()).acos()).abs() < 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn get_box_angle(&self) -> f64 {
        (self.xy() / (1.0 + self.xy() * self.xy()).sqrt()).acos()
    }

    /// Get the perpendicular distances between parallel edges of the rhomboid.
    ///
    /// For a rhomboid, the distance between parallel edges is not simply
    /// the extent since it is sheared.
    ///
    /// Returns [`d_x`, `d_y`] where `d_i` is the width in direction i.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rhomboid;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rhomboid = Rhomboid::from_box_vector([2.0, 2.0, 0.0]);
    /// let distances = rhomboid.get_nearest_plane_distance();
    ///
    /// // For orthogonal rhomboid, distances are just extents
    /// assert_eq!(distances[0].get(), 2.0);
    /// assert_eq!(distances[1].get(), 2.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn get_nearest_plane_distance(&self) -> [PositiveReal; 2] {
        // Since V = A_ih_i, h_i = V/A_i. V = det(a_1, a_2), A = |a_j x a_k|.
        let mut dist = [PositiveReal::default(); 2];
        dist[0] = self.lx() / (f64::sqrt(1.0 + self.xy() * self.xy())).try_into().unwrap();
        dist[1] = self.ly();
        dist
    }

    /// Represent a 2D triclinic box in the GSD box format.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rhomboid;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rhomb = Rhomboid::from((1.0.try_into()?, 2.0.try_into()?, 1.5));
    ///
    /// let gsd_box = rhomb.to_gsd_box();
    /// assert_eq!(gsd_box, [1.0, 2.0, 0.0, 1.5, 0.0, 0.0]);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
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
    /// Calculate the area of the rhomboid.
    ///
    /// The area is computed as the product of the edge lengths:
    /// ```math
    /// A = L_x \times L_y
    /// ```
    #[inline]
    fn volume(&self) -> f64 {
        // When A is triangular, det(A) = det(diag(A))
        self.lx().get() * self.ly().get()
    }
}

impl Scale for Rhomboid {
    /// Construct a scaled rhomboid.
    ///
    /// The resulting rhomboid's edge lengths $` L_\mathrm{new} `$ are
    /// the original's $` L `$ scaled by $` v `$:
    /// ```math
    /// L_\mathrm{new} = v L
    /// ```
    ///
    /// The centroid remains at the origin.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::Rhomboid};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rhomboid = Rhomboid::from_box_vector([5.0, 6.0, 1.5]);
    ///
    /// let scaled_rhomboid = rhomboid.scale_length(0.5.try_into()?);
    ///
    /// assert_eq!(scaled_rhomboid.lx().get(), 2.5);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_length(&self, v: PositiveReal) -> Self {
        Rhomboid {
            extents: [self.extents[0] * v, self.extents[1] * v],
            xy: self.xy,
        }
    }

    /// Construct a scaled rhomboid.
    ///
    /// The resulting rhomboid's edge lengths $` L_\mathrm{new} `$ are
    /// the original's $` L `$ scaled by $` v^\frac{1}{2} `$:
    /// ```math
    /// L_\mathrm{new} = v^\frac{1}{2} L
    /// ```
    ///
    /// The centroid remains at the origin.
    ///
    /// # Examples
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::Rhomboid};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rhomboid = Rhomboid::from_box_vector([5.0, 6.0, 1.5]);
    ///
    /// let scaled_rhomboid = rhomboid.scale_volume(4.0.try_into()?);
    ///
    /// assert_eq!(scaled_rhomboid.lx().get(), 10.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_volume(&self, v: PositiveReal) -> Self {
        let v = v
            .get()
            .sqrt()
            .try_into()
            .expect("sqrt of positive real is positive");
        self.scale_length(v)
    }
}

impl IsPointInside<Cartesian<2>> for Rhomboid {
    /// Test if a point is inside the rhomboid.
    ///
    /// Uses the transformed coordinate system where the rhomboid
    /// becomes an axis-aligned rectangle.
    #[inline]
    fn is_point_inside(&self, point: &Cartesian<2>) -> bool {
        let [x, y] = point.coordinates;
        let ly_half = self.ly().get() / 2.0;
        if y < -ly_half || y >= ly_half {
            return false;
        }
        let lx_half = self.lx().get() / 2.0;
        let x_skew = x - self.xy() * y;
        if x_skew < -lx_half || x_skew >= lx_half {
            return false;
        }
        true
    }
}

impl SupportMapping<Cartesian<2>> for Rhomboid {
    /// Calculate the point furthest from the center in a given direction.
    #[inline]
    fn support_mapping(&self, n: &Cartesian<2>) -> Cartesian<2> {
        let d = self.to_fractional(n);
        let s = Cartesian::from([d[0].signum() * 0.5, d[1].signum() * 0.5]);
        self.to_absolute(&s)
    }
}

impl BoundingSphereRadius for Rhomboid {
    /// Calculate the radius of the bounding sphere.
    ///
    /// The bounding sphere is centered at the origin and encompasses
    /// all points of the rhomboid.
    #[inline]
    fn bounding_sphere_radius(&self) -> PositiveReal {
        // || maximal_extent || / 2.0 = { lx + ly * |xy|, ly } || / 2.0
        (0.5 * f64::sqrt(
            (self.lx().get() + self.ly().get() * self.xy().abs()).powi(2) + self.ly().get().powi(2),
        ))
        .try_into()
        .expect("Norm is always positive.")
    }
}

use rand::{
    Rng,
    distr::{Distribution, Uniform},
};

impl Distribution<Cartesian<2>> for Rhomboid {
    /// Generate points uniformly distributed in the rhomboid.
    ///
    /// # Example
    ///
    /// ```
    /// use rand::{SeedableRng, distr::Distribution, rngs::StdRng};
    ///
    /// use hoomd_geometry::{IsPointInside, shape::Rhomboid};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rhomboid = Rhomboid::from_box_vector([6.0, 8.0, 0.5]);
    /// let mut rng = StdRng::seed_from_u64(1);
    ///
    /// let point = rhomboid.sample(&mut rng);
    /// assert!(rhomboid.is_point_inside(&point));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Cartesian<2> {
        let uniform = Uniform::new(-0.5, 0.5).expect("");
        let x = uniform.sample(rng);
        let y = uniform.sample(rng);

        let scaled_x = self.lx().get() * x + self.ly().get() * self.xy() * y;
        let scaled_y = self.ly().get() * y;

        Cartesian {
            coordinates: [scaled_x, scaled_y],
        }
    }
}

impl<R> IntersectsAt<Rhomboid, Cartesian<2>, R> for Rhomboid
where
    R: Rotate<Cartesian<2>> + Rotation + Copy,
    RotationMatrix<2>: From<R>,
{
    /// Test rhomboid intersections using the separating axis theorem.
    ///
    /// Rhomboids have two unique edge directions each, giving four potential
    /// separating axes. All comparisons are scaled by 2 to avoid halving the
    /// projection radii.
    #[inline]
    fn intersects_at(&self, other: &Rhomboid, v_ij: &Cartesian<2>, o_ij: &R) -> bool {
        let o_j = RotationMatrix::from(*o_ij);
        let [c, s] = [o_j.rows()[0][0], o_j.rows()[1][0]];

        let (lx1, ly1, xy1) = (self.lx().get(), self.ly().get(), self.xy());
        let (lx2, ly2, xy2) = (other.lx().get(), other.ly().get(), other.xy());
        let [tx, ty] = [v_ij[0], v_ij[1]];

        // The SAT check projects the distance between centers onto the normals of each
        // shape's edges. For a rhomboid with extents (lx, ly) and shear xy, the
        // edge vectors are [lx, 0] and [ly*xy, ly]. The corresponding normals
        // are [0, 1] and [-1, xy].

        // Shared subexpression for projections involving the skewed edges of both shapes.
        let det_part = c * (xy1 - xy2) + s * (xy1 * xy2 + 1.0);

        // Axis 1: P1's horizontal edge normal [0, 1].
        let r2_on_n11 = (s * lx2).abs() + (s * xy2 + c).abs() * ly2;
        if 2.0 * ty.abs() > ly1 + r2_on_n11 {
            return false;
        }

        // Axis 3: P2's horizontal edge normal (rotated).
        let dist_n21 = ty * c - tx * s;
        let r1_on_n21 = (s * lx1).abs() + (c - s * xy1).abs() * ly1;
        if 2.0 * dist_n21.abs() > r1_on_n21 + ly2 {
            return false;
        }

        // Axis 2: P1's skewed edge normal [-1, xy1].
        let dist_n12 = xy1 * ty - tx;
        let r2_on_n12 = (c - s * xy1).abs() * lx2 + det_part.abs() * ly2;
        if 2.0 * dist_n12.abs() > lx1 + r2_on_n12 {
            return false;
        }

        // Axis 4: P2's skewed edge normal (rotated).
        let cross_term = tx * c + ty * s;
        let dist_n22 = xy2 * dist_n21 - cross_term;
        let r1_on_n22 = (c + s * xy2).abs() * lx1 + det_part.abs() * ly1;
        if 2.0 * dist_n22.abs() > r1_on_n22 + lx2 {
            return false;
        }

        true
    }
}

impl<R> IntersectsAtGlobal<Rhomboid, Cartesian<2>, R> for Rhomboid
where
    R: Rotate<Cartesian<2>> + Rotation + Copy,
    RotationMatrix<2>: From<R>,
{
    /// Test whether two rhomboids intersect in global coordinates.
    ///
    /// This first culls by bounding-sphere distance and then transforms the
    /// second rhomboid into the local frame of the first before calling the
    /// local intersection test.
    #[inline]
    fn intersects_at_global(
        &self,
        other: &Rhomboid,
        r_self: &Cartesian<2>,
        o_self: &R,
        r_other: &Cartesian<2>,
        o_other: &R,
    ) -> bool {
        let max_separation =
            self.bounding_sphere_radius().get() + other.bounding_sphere_radius().get();
        if r_self.distance_squared(r_other) >= max_separation.powi(2) {
            return false;
        }

        let (v_ij, o_ij) = hoomd_vector::pair_system_to_local(r_self, o_self, r_other, o_other);

        self.intersects_at(other, &v_ij, &o_ij)
    }
}

impl MapPoint<Cartesian<2>> for Rhomboid {
    /// Map a point from this rhomboid's coordinate system to another rhomboid.
    ///
    /// The point is first expressed in fractional coordinates relative to `self`
    /// and then transformed into the Cartesian coordinates of `other`.
    #[inline]
    fn map_point(&self, point: Cartesian<2>, other: &Self) -> Result<Cartesian<2>, crate::Error> {
        let fractional = self.to_fractional(&point);
        let mapped_coords = other.to_absolute(&fractional);
        Ok(mapped_coords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        IntersectsAt, IsPointInside,
        shape::{ConvexPolygon, ConvexSurfaceMesh2d, Hypercuboid},
    };
    use approxim::assert_relative_eq;
    use hoomd_vector::Angle;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use rstest::rstest;
    use rstest_reuse::{self, apply, template};
    use std::f64::consts::PI;

    /// Common rhomboid shapes used across multiple tests.
    #[template]
    #[rstest]
    #[case::unit_square(1.0, 1.0, 0.0)]
    #[case::square(2.0, 2.0, 0.0)]
    #[case::rectangle(3.0, 1.0, 0.0)]
    #[case::skinny(0.005, 5.0, 0.0)]
    #[case::mild_shear(2.0, 2.0, 0.5)]
    #[case::sheared(3.0, 2.0, 1.5)]
    #[case::sheared_2(1.0, 3.0, 1.0)]
    #[case::strong_shear(1.0, 5.0, 25.0)]
    #[case::negative_shear(2.0, 2.0, -5.5)]
    #[case::unit_shear(1.0, 1.0, 1.0)]
    fn rhomboid_shapes(#[case] lx: f64, #[case] ly: f64, #[case] xy: f64) {}

    fn random_rhomboid(rng: &mut StdRng) -> Rhomboid {
        let lx: f64 = rng.random_range(0.1..10.0);
        let ly: f64 = rng.random_range(0.1..10.0);
        let xy: f64 = rng.random_range(-2.0..2.0);
        (lx.try_into().unwrap(), ly.try_into().unwrap(), xy).into()
    }

    /// Check that the support value (dot product with direction) matches the polygon.
    /// This avoids tie-breaking differences when multiple vertices maximize the dot product.
    fn check_support_value(lx: f64, ly: f64, xy: f64, n: [f64; 2]) {
        let rhomboid: Rhomboid = (
            lx.try_into().expect("lx > 0"),
            ly.try_into().expect("ly > 0"),
            xy,
        )
            .into();

        let polygon = ConvexPolygon::with_vertices(rhomboid.vertices().to_vec())
            .expect("rhomboid vertices form a polygon");

        let r = rhomboid.support_mapping(&n.into());
        let p = polygon.support_mapping(&n.into());

        let support_rhomboid = r[0] * n[0] + r[1] * n[1];
        let support_polygon = p[0] * n[0] + p[1] * n[1];

        assert_relative_eq!(support_rhomboid, support_polygon, epsilon = 1e-12);
    }

    #[apply(rhomboid_shapes)]
    fn support_mapping_fixed_directions(#[case] lx: f64, #[case] ly: f64, #[case] xy: f64) {
        // Cardinal directions.
        check_support_value(lx, ly, xy, [1.0, 0.0]);
        check_support_value(lx, ly, xy, [0.0, 1.0]);
        check_support_value(lx, ly, xy, [-1.0, 0.0]);
        check_support_value(lx, ly, xy, [0.0, -1.0]);
        // Diagonal directions.
        check_support_value(lx, ly, xy, [1.0, 1.0]);
        check_support_value(lx, ly, xy, [-1.0, 1.0]);
        check_support_value(lx, ly, xy, [-1.0, -1.0]);
        check_support_value(lx, ly, xy, [1.0, -1.0]);
    }

    #[apply(rhomboid_shapes)]
    fn support_mapping_random_directions(#[case] lx: f64, #[case] ly: f64, #[case] xy: f64) {
        let mut rng = StdRng::seed_from_u64(42);
        let rhomboid: Rhomboid = (lx.try_into().unwrap(), ly.try_into().unwrap(), xy).into();
        let polygon =
            ConvexPolygon::with_vertices(rhomboid.vertices().to_vec()).expect("valid polygon");

        for _ in 0..1000 {
            let n: Cartesian<2> = rng.random();
            assert_relative_eq!(
                rhomboid.support_mapping(&n),
                polygon.support_mapping(&n),
                epsilon = 1e-12,
            );
        }
    }

    #[apply(rhomboid_shapes)]
    fn bounding_sphere_radius_matches_polygon(#[case] lx: f64, #[case] ly: f64, #[case] xy: f64) {
        let rhomboid: Rhomboid = (lx.try_into().unwrap(), ly.try_into().unwrap(), xy).into();
        let polygon = ConvexPolygon::with_vertices(rhomboid.vertices().to_vec())
            .expect("rhomboid vertices form a polygon");

        assert_relative_eq!(
            rhomboid.bounding_sphere_radius().get(),
            polygon.bounding_sphere_radius().get(),
            epsilon = 1e-12,
        );
    }

    /// Compare the Rhomboid SAT against the `ConvexSurfaceMesh2d` separating-planes
    /// implementation (an independently tested ground truth).
    fn check_sat_against_mesh(
        lx1: f64,
        ly1: f64,
        xy1: f64,
        lx2: f64,
        ly2: f64,
        xy2: f64,
        tx: f64,
        ty: f64,
        theta: f64,
    ) {
        let a: Rhomboid = (lx1.try_into().unwrap(), ly1.try_into().unwrap(), xy1).into();
        let b: Rhomboid = (lx2.try_into().unwrap(), ly2.try_into().unwrap(), xy2).into();

        let v_ij = Cartesian::from([tx, ty]);
        let o_ij = Angle::from(theta);

        let sat = a.intersects_at(&b, &v_ij, &o_ij);

        let mesh_a = ConvexSurfaceMesh2d::from_point_set(a.vertices().iter().copied()).unwrap();
        let mesh_b = ConvexSurfaceMesh2d::from_point_set(b.vertices().iter().copied()).unwrap();
        let mesh = mesh_a.intersects_at(&mesh_b, &v_ij, &o_ij);

        assert_eq!(
            sat, mesh,
            "SAT={sat}, mesh={mesh}\n\
             a=({lx1}, {ly1}, {xy1})\n\
             b=({lx2}, {ly2}, {xy2})\n\
             t=({tx}, {ty})\n\
             theta={theta}"
        );
    }

    /// Shape pairs for intersection tests with displacement and rotation.
    #[template]
    #[rstest]
    #[case::coincident(0.0, 0.0, 0.0)]
    #[case::half_overlap(0.5, 0.5, 0.0)]
    #[case::near_touch_x(1.999_999, 0.0, 0.0)]
    #[case::past_touch_x(2.000_001, 0.0, 0.0)]
    #[case::near_touch_y(0.0, 1.999_999, 0.0)]
    #[case::past_touch_y(0.0, 2.000_001, 0.0)]
    #[case::diagonal_45(1.0, 1.0, PI / 4.0)]
    #[case::rotated_60(1.3, 0.7, PI / 3.0)]
    #[case::rotated_90(0.0, 1.5, PI / 2.0)]
    fn square_displacements(#[case] tx: f64, #[case] ty: f64, #[case] theta: f64) {}

    #[apply(square_displacements)]
    fn intersects_at_identical_squares(#[case] tx: f64, #[case] ty: f64, #[case] theta: f64) {
        check_sat_against_mesh(2.0, 2.0, 0.0, 2.0, 2.0, 0.0, tx, ty, theta);
    }

    /// Displacements for mirror-sheared pairs (xy2 = -xy1).
    #[template]
    #[rstest]
    #[case::coincident(0.0, 0.0, 0.0)]
    #[case::shifted_x(1.0, 0.0, 0.0)]
    #[case::shifted_x2(2.0, 0.0, 0.0)]
    #[case::shifted_y(0.0, 1.0, 0.0)]
    #[case::diagonal(1.0, 1.0, 0.0)]
    #[case::rotated_45(0.5, 0.5, PI / 4.0)]
    #[case::rotated_60(1.0, 0.0, PI / 3.0)]
    fn mirror_shear_displacements(#[case] tx: f64, #[case] ty: f64, #[case] theta: f64) {}

    #[apply(mirror_shear_displacements)]
    fn intersects_at_mirror_sheared(#[case] tx: f64, #[case] ty: f64, #[case] theta: f64) {
        check_sat_against_mesh(2.0, 2.0, 1.0, 2.0, 2.0, -1.0, tx, ty, theta);
    }

    #[test]
    fn intersects_at_mixed_shapes() {
        // Different shapes, various displacements and rotations.
        check_sat_against_mesh(1.0, 3.0, 1.5, 2.0, 1.0, -0.5, 1.0, 0.5, 0.0);
        check_sat_against_mesh(1.0, 5.0, 2.0, 1.0, 5.0, 2.0, 1.0, 0.0, 0.0);
        check_sat_against_mesh(1.0, 3.0, 1.5, 2.0, 1.0, -0.5, 0.5, 0.5, PI / 6.0);
        check_sat_against_mesh(1.0, 5.0, 2.0, 1.0, 5.0, -2.0, 0.5, 0.0, PI / 2.0);
        check_sat_against_mesh(3.0, 1.0, -1.0, 1.0, 2.0, 0.5, 1.5, 0.3, PI / 5.0);
        check_sat_against_mesh(2.0, 1.0, 0.8, 1.5, 2.5, -0.3, 0.0, 1.0, PI / 8.0);
    }

    #[test]
    fn scale_preserves_aspect_ratio_and_volume() {
        let rhomboid: Rhomboid = (3.0.try_into().unwrap(), 2.0.try_into().unwrap(), 1.5).into();
        let original_volume = rhomboid.volume();
        let original_lx_over_ly = rhomboid.lx().get() / rhomboid.ly().get();

        let scaled = rhomboid.scale_length(2.0.try_into().unwrap());
        assert_relative_eq!(scaled.volume(), 4.0 * original_volume);
        assert_relative_eq!(scaled.lx().get() / scaled.ly().get(), original_lx_over_ly);
        assert_eq!(scaled.xy(), rhomboid.xy());

        let scaled = rhomboid.scale_volume(9.0.try_into().unwrap());
        assert_relative_eq!(scaled.volume(), 9.0 * original_volume);
        assert_relative_eq!(scaled.lx().get() / scaled.ly().get(), original_lx_over_ly);
        assert_eq!(scaled.xy(), rhomboid.xy());
    }

    /// Unsheared rhomboid shapes (xy=0) for rectangle comparison tests.
    #[template]
    #[rstest]
    #[case::unit_square(1.0, 1.0)]
    #[case::square(2.0, 2.0)]
    #[case::rectangle(3.0, 1.0)]
    #[case::skinny(0.005, 5.0)]
    fn unsheared_shapes(#[case] lx: f64, #[case] ly: f64) {}

    #[apply(unsheared_shapes)]
    fn is_point_inside_matches_rectangle(#[case] lx: f64, #[case] ly: f64) {
        let mut rng = StdRng::seed_from_u64(789);

        let rhomboid: Rhomboid = (lx.try_into().unwrap(), ly.try_into().unwrap(), 0.0).into();
        let rect = Hypercuboid {
            edge_lengths: [lx.try_into().unwrap(), ly.try_into().unwrap()],
        };

        for _ in 0..10_000 {
            let point: Cartesian<2> =
                rng.random::<Cartesian<2>>() * 20.0 - Cartesian::from([10.0; 2]);

            assert_eq!(
                rhomboid.is_point_inside(&point),
                rect.is_point_inside(&point),
                "Mismatch at ({}, {}) for lx={lx}, ly={ly}",
                point[0],
                point[1],
            );
        }
    }

    #[apply(rhomboid_shapes)]
    fn is_point_inside_area_fraction(#[case] lx: f64, #[case] ly: f64, #[case] xy: f64) {
        let mut rng = StdRng::seed_from_u64(1011);

        let rhomboid: Rhomboid = (lx.try_into().unwrap(), ly.try_into().unwrap(), xy).into();
        let area = rhomboid.volume();

        // Bounding box for sampling.
        let bx = lx + ly * xy.abs();
        let by = ly;
        let bbox_area = bx * by;

        let n_samples = 100_000_usize;
        let mut inside_count = 0_usize;

        for _ in 0..n_samples {
            let x = rng.random_range(-bx / 2.0..bx / 2.0);
            let y = rng.random_range(-by / 2.0..by / 2.0);
            if rhomboid.is_point_inside(&[x, y].into()) {
                inside_count += 1;
            }
        }

        let estimated_area = (inside_count as f64 / n_samples as f64) * bbox_area;
        assert_relative_eq!(estimated_area, area, max_relative = 0.02);
    }

    #[test]
    fn intersects_at_random() {
        let mut rng = StdRng::seed_from_u64(456);

        for i in 0..10_000 {
            let a = random_rhomboid(&mut rng);
            let b = random_rhomboid(&mut rng);

            let v_ij: Cartesian<2> =
                rng.random::<Cartesian<2>>() * 20.0 - Cartesian::from([10.0; 2]);
            let o_ij = Angle::from(rng.random_range(-std::f64::consts::PI..std::f64::consts::PI));

            let sat = a.intersects_at(&b, &v_ij, &o_ij);

            let mesh_a = ConvexSurfaceMesh2d::from_point_set(a.vertices().iter().copied()).unwrap();
            let mesh_b = ConvexSurfaceMesh2d::from_point_set(b.vertices().iter().copied()).unwrap();
            let mesh = mesh_a.intersects_at(&mesh_b, &v_ij, &o_ij);

            assert_eq!(
                sat,
                mesh,
                "Mismatch at iteration {i}\n\
                 a=({}, {}, {})\n\
                 b=({}, {}, {})\n\
                 t=({}, {})\n\
                 theta={}",
                a.lx().get(),
                a.ly().get(),
                a.xy(),
                b.lx().get(),
                b.ly().get(),
                b.xy(),
                v_ij[0],
                v_ij[1],
                o_ij.theta,
            );
        }
    }
}

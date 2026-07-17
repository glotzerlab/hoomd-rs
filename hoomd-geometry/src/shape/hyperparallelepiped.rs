// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Hyperparallelepiped`]
use std::array;

use hoomd_linear_algebra::matrix::{
    Matrix,
    qr::{self, get_r, get_r_inv},
};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, InnerProduct};
use rand::{
    Rng,
    distr::{Distribution, Uniform},
};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::{IsPointInside, MapPoint, Scale, SupportMapping, Volume};

/// An N-dimensional hyperparallelepiped defined by N edge vectors.
///
/// A hyperparallelepiped (also known as a parallelotope) is the N-dimensional
/// generalization of a parallelogram in 2D and parallelepiped in 3D. Points
/// $` \vec{r} `$ inside a hyperparallelepiped can be expressed by
/// ```math
/// \vec{r} = \sum_i \lambda_i \vec{e}_i
/// ```
/// where $` \vec{e}_i `$ are the edge vectors each $` \lambda_i `$ is in
/// the interval $` [ -0.5, 0.5 ) `$.
///
/// The shape can be used as the box geometry for simulations, but users should
/// prefer [Rhomboid] and [Triclinic] for 2 and 3-dimensional simulations,
/// respectively.
///
/// [Rhomboid]: crate::shape::Rhomboid
/// [Triclinic]: crate::shape::Triclinic
///
/// # Example
///
/// ```
/// use hoomd_geometry::shape::Hyperparallelepiped;
/// use hoomd_vector::Cartesian;
///
/// let hyperparallelepiped = Hyperparallelepiped::new([
///     Cartesian::from([10.0, 3.0, 0.0]),
///     Cartesian::from([-2.0, 12.0, -3.0]),
///     Cartesian::from([1.0, -4.0, 14.0]),
/// ]);
/// ```
#[serde_as]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hyperparallelepiped<const N: usize> {
    /// The N edge vectors that define the shape. Each vector spans one
    /// edge of the parallelotope.
    #[serde_as(as = "[_; N]")]
    pub(crate) edge_vectors: [Cartesian<N>; N],

    /// Cached (condensed) QR factorization of the column matrix formed by the
    /// edge vectors.
    qr: Matrix<N, N>,
}

impl<const N: usize> Hyperparallelepiped<N> {
    /// Construct a new hyperparallelepiped from N edge vectors.
    ///
    /// Hyperparallelepiped construction is expensive because it computes the
    /// QR decomposition of a matrix. Once construction is complete, operations
    /// such as [`fractional`], [`is_point_inside`], and [`map_point`] are very
    /// efficient.
    ///
    /// [`fractional`]: Self::fractional
    /// [`is_point_inside`]: IsPointInside::is_point_inside
    /// [`map_point`]: MapPoint::map_point
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Hyperparallelepiped;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let parallelepiped = Hyperparallelepiped::new([
    ///     Cartesian::from([10.0, 0.0, 0.0]),
    ///     Cartesian::from([0.0, 12.0, 0.0]),
    ///     Cartesian::from([0.0, 0.0, 14.0]),
    /// ]);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn new(edge_vectors: [Cartesian<N>; N]) -> Self {
        Self {
            edge_vectors,
            qr: Self::calculate_qr(&edge_vectors),
        }
    }

    /// Get the edge vectors.
    #[inline]
    pub fn edge_vectors(&self) -> &[Cartesian<N>; N] {
        &self.edge_vectors
    }

    /// Compute and cache the QR factorization of the edge-vector matrix.
    ///
    /// The edge vectors are assembled into an N×N matrix $`\mathbf{A}`$ whose *columns*
    /// are the edge vectors, and the result is stored in `self.qr`. This
    /// factorization is later used by [`fractional`](Self::fractional),
    /// [`is_point_inside`](IsPointInside::is_point_inside), and
    /// [`map_point`](MapPoint::map_point) to solve the linear system
    /// $`\mathbf{A} \vec{s} = \vec{r}`$.
    #[inline]
    fn calculate_qr(edge_vectors: &[Cartesian<N>; N]) -> Matrix<N, N> {
        let box_matrix = Matrix {
            rows: std::array::from_fn(|r| {
                std::array::from_fn(|c| edge_vectors[c].coordinates[r])
            }),
        };

        let (qr, _) = box_matrix.qr();
        qr
    }

    /// Determine the maximal extents of the hyperparallelepiped along each
    /// Cartesian axis.
    ///
    /// For each axis `k`, the maximal extent is half the sum of the absolute
    /// values of the k-th component across all edge vectors. This gives the
    /// smallest axis-aligned bounding box (AABB) that contains the shape.
    ///
    /// # Returns
    ///
    /// An array `[f64; N]` where entry `k` is the largest coordinate
    /// along axis `k` that the shape reaches.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Hyperparallelepiped;
    /// use hoomd_vector::Cartesian;
    ///
    /// let unit_square = Hyperparallelepiped::new([
    ///     Cartesian::from([1.0, 0.0]),
    ///     Cartesian::from([0.0, 1.0]),
    /// ]);
    /// assert_eq!(unit_square.maximal_extents(), [0.5, 0.5]);
    /// ```
    #[inline]
    pub fn maximal_extents(&self) -> [f64; N] {
        (0.5 * self
            .edge_vectors
            .iter()
            .fold(Cartesian::<N>::default(), |acc, v| v.map(f64::abs) + acc))
        .coordinates
    }

    /// Determine the minimal extents of the hyperparallelepiped along each
    /// Cartesian axis.
    ///
    /// This is the negation of [`maximal_extents`],
    /// representing the most-negative reachable coordinate along each axis.
    ///
    /// [`maximal_extents`]: Self::maximal_extents
    ///
    /// # Returns
    ///
    /// An array `[f64; N]` where entry `k` is the most-negative coordinate
    /// along axis `k` that the shape reaches.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Hyperparallelepiped;
    /// use hoomd_vector::Cartesian;
    ///
    /// let unit_square = Hyperparallelepiped::new([
    ///     Cartesian::from([1.0, 0.0]),
    ///     Cartesian::from([0.0, 1.0]),
    /// ]);
    /// assert_eq!(unit_square.minimal_extents(), [-0.5, -0.5]);
    /// ```
    #[inline]
    pub fn minimal_extents(&self) -> [f64; N] {
        self.maximal_extents().map(|x| -x)
    }

    /// Convert a Cartesian vector to fractional coordinates.
    ///
    /// Fractional coordinates express a point as coefficients of the edge
    /// vectors. Let the edge vectors form the columns of matrix $`\mathbf{A}`$. Then
    /// the fractional coordinate vector $`\vec{s}`$ satisfies $`\mathbf{A}\vec{s}=\vec{r}`$.
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::shape::Hyperparallelepiped;
    /// use hoomd_vector::Cartesian;
    ///
    /// let hyperparallelepiped = Hyperparallelepiped::new([
    ///     Cartesian::from([4.0, 0.0]),
    ///     Cartesian::from([0.0, 6.0]),
    /// ]);
    ///
    /// let fractional = hyperparallelepiped.fractional(Cartesian::from([1.0, 1.5]));
    /// assert_relative_eq!(fractional[0], 0.25);
    /// assert_relative_eq!(fractional[1], 0.25);
    /// ```
    #[inline]
    pub fn fractional(&self, absolute: Cartesian<N>) -> Cartesian<N> {
        Cartesian::from_column_matrix(&qr::qr_solve(
            &self.qr,
            &absolute.to_column_matrix(),
        ))
    }

    /// Convert fractional coordinates to Cartesian coordinates.
    ///
    /// This is the inverse of [`fractional`]. Given a vector of fractional
    /// coefficients $`\vec{s}`$, the Cartesian point is:
    ///
    /// ```math
    /// \vec{r} = \sum_{i=0}^{N-1} s_i \vec{a}_i
    /// ```
    /// where $`\vec{a}_i`$ are the edge vectors.
    ///
    /// [`fractional`]: Self::fractional
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::shape::Hyperparallelepiped;
    /// use hoomd_vector::Cartesian;
    ///
    /// let hyperparallelepiped = Hyperparallelepiped::new([
    ///     Cartesian::from([4.0, 0.0]),
    ///     Cartesian::from([0.0, 6.0]),
    /// ]);
    ///
    /// let absolute = hyperparallelepiped.absolute(Cartesian::from([0.25, 0.25]));
    /// assert_relative_eq!(absolute[0], 1.0);
    /// assert_relative_eq!(absolute[1], 1.5);
    /// ```
    #[inline]
    pub fn absolute(&self, fractional: Cartesian<N>) -> Cartesian<N> {
        let mut absolute = Cartesian::<N>::default();
        for (edge_vector, f) in self.edge_vectors.iter().zip(fractional) {
            absolute += f * *edge_vector;
        }
        absolute
    }

    /// Computes the perpendicular distances between each of the N pairs of bounding
    /// hyperplanes of the parallelotope.
    ///
    /// # Mathematical Background
    ///
    /// The perpendicular distance between faces can be found using the reciprocal
    /// lattice construction. Let $`\vec{b}_i`$ be a normal vector to the face spanned
    /// by all edge vectors except $`\vec{a}_i`$. Then
    /// $`h_k = \lVert \operatorname{proj}_{b_k}(\vec{a}_k) \rVert`$.
    ///
    /// For the edge-vector matrix $`\mathbf{A}`$, write the QR decomposition
    /// $`\mathbf{A} = \mathbf{Q}\mathbf{R}`$. Since $`\mathbf{Q}`$ is orthogonal,
    /// right-multiplying by $`\mathbf{Q}^T`$ preserves the Euclidean norm of each
    /// row. Therefore the norm of the $`k`$-th row of $`\mathbf{R}^{-1}`$ is the
    /// same as the norm of the corresponding row of $`\mathbf{A}^{-1}`$.
    ///
    /// Hence,
    /// ```math
    /// h_k = \frac{1}{\left\lVert (\mathbf{R}^{-1})_k \right\rVert},
    /// ```
    /// where $`\lVert (\mathbf{R}^{-1})_k \rVert`$ denotes the Euclidean norm of the
    /// $`k`$-th row of $`\mathbf{R}^{-1}`$.
    ///
    /// # Returns
    ///
    /// An array of N [`PositiveReal`] values $`[h_0, h_1, \dots, h_{N-1}]`$, where $`h_k`$ is
    /// the perpendicular distance from the origin to the $`k`$-th bounding hyperplane.
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::shape::Hyperparallelepiped;
    /// use hoomd_vector::Cartesian;
    ///
    /// let hyperparallelepiped = Hyperparallelepiped::new([
    ///     Cartesian::from([4.0, 0.0]),
    ///     Cartesian::from([0.0, 6.0]),
    /// ]);
    ///
    /// let nearest_plane_distance = hyperparallelepiped.nearest_plane_distances();
    ///
    /// assert_relative_eq!(nearest_plane_distance[0].get(), 4.0);
    /// assert_relative_eq!(nearest_plane_distance[1].get(), 6.0);
    /// ```
    #[inline]
    #[expect(
        clippy::missing_panics_doc,
        reason = "Panic would occur due to a bug in hoomd-rs."
    )]
    pub fn nearest_plane_distances(&self) -> [PositiveReal; N] {
        let r_inv = get_r_inv(&self.qr);
        let distances: [PositiveReal; N] = std::array::from_fn(|i| {
            let row = r_inv.get_row(i);
            let inv_norm = 1.0 / row.as_slice().iter().map(|&x| x * x).sum::<f64>().sqrt();
            inv_norm.try_into().expect("row norm should be positive")
        });
        distances
    }
}

impl<const N: usize> Volume for Hyperparallelepiped<N> {
    /// Compute the N-dimensional hypervolume of the hyperparallelepiped.
    ///
    /// The volume is the absolute product of the diagonal elements of the upper
    /// triangular matrix $`\mathbf{R}`$ from the QR decomposition of the edge-vector
    /// matrix $`\mathbf{A}=\mathbf{Q}\mathbf{R}`$.
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::{Volume, shape::Hyperparallelepiped};
    /// use hoomd_vector::Cartesian;
    ///
    /// let hyperparallelepiped = Hyperparallelepiped::new([
    ///     Cartesian::from([4.0, 0.0]),
    ///     Cartesian::from([0.0, 6.0]),
    /// ]);
    ///
    /// let volume = hyperparallelepiped.volume();
    ///
    /// assert_relative_eq!(volume, 24.0);
    /// ```
    #[inline]
    fn volume(&self) -> f64 {
        let r = get_r(&self.qr);
        r.diagonal().elements.iter().product::<f64>().abs()
    }
}

impl<const N: usize> Scale for Hyperparallelepiped<N> {
    /// Produce a scaled hyperparallelepiped by uniformly scaling each edge vector.
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::{Scale, Volume, shape::Hyperparallelepiped};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let hyperparallelepiped = Hyperparallelepiped::new([
    ///     Cartesian::from([4.0, 0.0]),
    ///     Cartesian::from([0.0, 6.0]),
    /// ]);
    ///
    /// let scaled = hyperparallelepiped.scale_length(2.0.try_into()?);
    ///
    /// assert_relative_eq!(scaled.volume(), 96.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_length(&self, v: PositiveReal) -> Self {
        let edge_vectors = self.edge_vectors.map(|ev| ev * v);
        let qr = Self::calculate_qr(&edge_vectors);

        Self {
            edge_vectors,
            qr,
        }
    }

    /// Produce a scaled hyperparallelepiped by uniformly scaling volume.
    ///
    /// Each edge vector is scaled by `v^(1/N)` so that the N-dimensional
    /// volume scales by exactly `v`.
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::{Scale, Volume, shape::Hyperparallelepiped};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let hyperparallelepiped = Hyperparallelepiped::new([
    ///     Cartesian::from([4.0, 0.0]),
    ///     Cartesian::from([0.0, 6.0]),
    /// ]);
    ///
    /// let scaled = hyperparallelepiped.scale_volume(2.0.try_into()?);
    ///
    /// assert_relative_eq!(scaled.volume(), 48.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_volume(&self, v: PositiveReal) -> Self {
        let v_linear = v
            .get()
            .powf(1.0 / N as f64)
            .try_into()
            .expect("v^(1/N) must be positive");
        self.scale_length(v_linear)
    }
}

impl<const N: usize> SupportMapping<Cartesian<N>> for Hyperparallelepiped<N> {
    /// Compute the support mapping of the hyperparallelepiped in a given direction.
    ///
    /// The support mapping returns the point on (or inside) the shape that has
    /// the greatest dot product with the query direction $`d`$. For a
    /// hyperparallelepiped this is computed by choosing, for each edge vector
    /// $`\vec{a}_i`$, the vertex $`\pm 1/2  \vec{a}_i`$ whose sign matches the sign of
    /// $`\vec{a}_i \cdot \vec{d} `$ and summing the contributions:
    ///
    /// ```math
    /// h(\mathbf{d}) = \frac{1}{2} \sum_{i=0}^{N-1}
    ///     \operatorname{sgn}(\mathbf{a}_i \cdot \mathbf{d})\, \mathbf{a}_i
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{SupportMapping, shape::Hyperparallelepiped};
    /// use hoomd_vector::Cartesian;
    ///
    /// let unit_square = Hyperparallelepiped::new([
    ///     Cartesian::from([1.0, 0.0]),
    ///     Cartesian::from([0.0, 1.0]),
    /// ]);
    ///
    /// let s = unit_square.support_mapping(&Cartesian::from([1.0, 1.0]));
    /// assert_eq!(s, Cartesian::from([0.5, 0.5]));
    /// ```
    #[inline]
    fn support_mapping(&self, n: &Cartesian<N>) -> Cartesian<N> {
        0.5 * self
            .edge_vectors
            .iter()
            .fold(Cartesian::<N>::default(), |acc, v| {
                v.dot(n).signum() * *v + acc
            })
    }
}

impl<const N: usize> IsPointInside<Cartesian<N>> for Hyperparallelepiped<N> {
    /// Check whether a Cartesian point lies inside the hyperparallelepiped.
    ///
    /// The test converts the point to fractional coordinates and checks
    /// that every coordinate lies in the half-open interval `[-0.5, 0.5)`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{IsPointInside, shape::Hyperparallelepiped};
    /// use hoomd_vector::Cartesian;
    ///
    /// let hyperparallelepiped = Hyperparallelepiped::new([
    ///     Cartesian::from([6.0, 0.0]),
    ///     Cartesian::from([0.0, 8.0]),
    /// ]);
    ///
    /// assert!( hyperparallelepiped.is_point_inside(&Cartesian::from([ 2.5, -3.5])));
    /// assert!( hyperparallelepiped.is_point_inside(&Cartesian::from([-3.0,  0.0])));
    /// assert!(!hyperparallelepiped.is_point_inside(&Cartesian::from([ 3.0, -3.5])));
    /// assert!(!hyperparallelepiped.is_point_inside(&Cartesian::from([ 4.0, -3.5])));
    /// ```
    #[inline]
    fn is_point_inside(&self, point: &Cartesian<N>) -> bool {
        let fractional = qr::qr_solve(
            &self.qr,
            &point.to_column_matrix(),
        );

        fractional
            .rows
            .into_iter()
            .all(|x| -0.5 <= x[0] && x[0] < 0.5)
    }
}

impl<const N: usize> MapPoint<Cartesian<N>> for Hyperparallelepiped<N> {
    /// Map a point from one hyperparallelepiped to another via linear transformation.
    ///
    /// Converts `point` (expressed in `self`'s Cartesian frame) to fractional
    /// coordinates relative to `self`, then evaluates those same fractional
    /// coordinates in `other`'s frame. This is used to rescale or deform a
    /// simulation box while preserving the relative positions of particles.
    ///
    /// # Returns
    ///
    /// The corresponding Cartesian coordinate in `other`'s frame.
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::{MapPoint, shape::Hyperparallelepiped};
    /// use hoomd_vector::Cartesian;
    ///
    /// let source = Hyperparallelepiped::new([
    ///     Cartesian::from([4.0, 0.0]),
    ///     Cartesian::from([0.0, 4.0]),
    /// ]);
    ///
    /// let destination = Hyperparallelepiped::new([
    ///     Cartesian::from([8.0, 0.0]),
    ///     Cartesian::from([0.0, 8.0]),
    /// ]);
    ///
    /// let mapped = source.map_point(Cartesian::from([1.0, 1.0]), &destination).unwrap();
    /// assert_relative_eq!(mapped[0], 2.0);
    /// assert_relative_eq!(mapped[1], 2.0);
    /// ```
    #[inline]
    fn map_point(&self, point: Cartesian<N>, other: &Self) -> Result<Cartesian<N>, crate::Error> {
        let fractional = self.fractional(point);
        let mapped_coords = other.absolute(fractional);
        Ok(mapped_coords)
    }
}

impl<const N: usize> Distribution<Cartesian<N>> for Hyperparallelepiped<N> {
    /// Generate points uniformly distributed in the hyperparallelepiped.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{IsPointInside, shape::Hyperparallelepiped};
    /// use hoomd_vector::Cartesian;
    /// use rand::{SeedableRng, distr::Distribution, rngs::StdRng};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let hyperparallelepiped = Hyperparallelepiped::new([
    ///     Cartesian::from([6.0, 0.0]),
    ///     Cartesian::from([0.0, 8.0]),
    /// ]);
    ///
    /// let mut rng = StdRng::seed_from_u64(1);
    ///
    /// let point = hyperparallelepiped.sample(&mut rng);
    /// assert!(hyperparallelepiped.is_point_inside(&point));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Cartesian<N> {
        let uniform = Uniform::new(-0.5, 0.5).expect("hard coded distribution should be valid");
        let fractional: [f64; N] = array::from_fn(|_| uniform.sample(rng));
        self.absolute(Cartesian::from(fractional))
    }
}

#[cfg(test)]
mod tests {
    use approxim::assert_relative_eq;
    use super::*;
    use hoomd_utility::valid::PositiveReal;

    fn ortho_box_2d(lx: f64, ly: f64) -> Hyperparallelepiped<2> {
        Hyperparallelepiped::new([Cartesian::from([lx, 0.0]), Cartesian::from([0.0, ly])])
    }

    fn unit_box_2d() -> Hyperparallelepiped<2> {
        Hyperparallelepiped::new([Cartesian::from([1.0, 0.0]), Cartesian::from([0.0, 1.0])])
    }

    fn ortho_box_3d(lx: f64, ly: f64, lz: f64) -> Hyperparallelepiped<3> {
        Hyperparallelepiped::new([
            Cartesian::from([lx, 0.0, 0.0]),
            Cartesian::from([0.0, ly, 0.0]),
            Cartesian::from([0.0, 0.0, lz]),
        ])
    }

    #[test]
    fn default_2d_is_unit_square() {
        let b = unit_box_2d();
        assert_eq!(b.edge_vectors[0], Cartesian::from([1.0, 0.0]));
        assert_eq!(b.edge_vectors[1], Cartesian::from([0.0, 1.0]));

        assert_relative_eq!(b.volume(), 1.0);

        let nearest_plane_distances = b.nearest_plane_distances();
        assert_relative_eq!(nearest_plane_distances[0].get(), 1.0);
        assert_relative_eq!(nearest_plane_distances[1].get(), 1.0);
    }

    #[test]
    fn default_3d_is_unit_cube() {
        let b = Hyperparallelepiped::new([
            Cartesian::from([1.0, 0.0, 0.0]),
            Cartesian::from([0.0, 1.0, 0.0]),
            Cartesian::from([0.0, 0.0, 1.0]),
        ]);
        assert_eq!(b.edge_vectors[0], Cartesian::from([1.0, 0.0, 0.0]));
        assert_eq!(b.edge_vectors[1], Cartesian::from([0.0, 1.0, 0.0]));
        assert_eq!(b.edge_vectors[2], Cartesian::from([0.0, 0.0, 1.0]));

        let nearest_plane_distances = b.nearest_plane_distances();
        assert_relative_eq!(nearest_plane_distances[0].get(), 1.0);
        assert_relative_eq!(nearest_plane_distances[1].get(), 1.0);
        assert_relative_eq!(nearest_plane_distances[2].get(), 1.0);
    }

    #[test]
    fn maximal_extents_unit_square() {
        let b = unit_box_2d();
        assert_eq!(b.maximal_extents(), [0.5, 0.5]);
    }

    #[test]
    fn minimal_extents_unit_square() {
        let b = unit_box_2d();
        assert_eq!(b.minimal_extents(), [-0.5, -0.5]);
    }

    #[test]
    fn maximal_extents_rectangular_box() {
        let b = ortho_box_3d(10.0, 12.0, 14.0);
        assert_eq!(b.maximal_extents(), [5.0, 6.0, 7.0]);
    }

    #[test]
    fn minimal_extents_rectangular_box() {
        let b = ortho_box_3d(10.0, 12.0, 14.0);
        assert_eq!(b.minimal_extents(), [-5.0, -6.0, -7.0]);
    }

    #[test]
    fn maximal_extents_tilted_2d() {
        // a1 = (2, 0),  a2 = (1, 3)
        // max_x = 0.5*(2 + 1) = 1.5,  max_y = 0.5*(0 + 3) = 1.5
        let b =
            Hyperparallelepiped::new([Cartesian::from([2.0, 0.0]), Cartesian::from([1.0, 3.0])]);
        let ext = b.maximal_extents();
        assert_relative_eq!(ext[0], 1.5);
        assert_relative_eq!(ext[1], 1.5);
    }

    #[test]
    fn fractional_round_trip_ortho_2d() {
        let b = ortho_box_2d(4.0, 6.0);
        let original = Cartesian::from([1.0, 1.5]);
        let frac = b.fractional(original);
        let back = b.absolute(frac);
        assert_relative_eq!(back, original);
    }

    #[test]
    fn fractional_round_trip_ortho_3d() {
        let b = ortho_box_3d(10.0, 12.0, 14.0);
        let original = Cartesian::from([3.0, -4.0, 6.5]);
        let frac = b.fractional(original);
        let back = b.absolute(frac);
        assert_relative_eq!(back, original);
    }

    #[test]
    fn fractional_round_trip_tilted_2d() {
        // Triclinic-like 2D box
        let b =
            Hyperparallelepiped::new([Cartesian::from([3.0, 0.0]), Cartesian::from([1.0, 4.0])]);
        let original = Cartesian::from([0.5, 1.0]);
        let frac = b.fractional(original);
        let back = b.absolute(frac);
        assert_relative_eq!(back, original);
    }

    #[test]
    fn fractional_known_values_ortho() {
        // For a 4×6 box, the point (1, 1.5) should have fractional coords (0.25, 0.25)
        let b = ortho_box_2d(4.0, 6.0);
        let frac = b.fractional(Cartesian::from([1.0, 1.5]));
        assert_relative_eq!(frac[0], 0.25);
        assert_relative_eq!(frac[1], 0.25);
    }

    #[test]
    fn absolute_origin_maps_to_origin() {
        let b = ortho_box_3d(5.0, 7.0, 9.0);
        let origin = Cartesian::from([0.0, 0.0, 0.0]);
        let result = b.absolute(origin);
        assert_relative_eq!(result, origin);
    }

    #[test]
    fn center_is_inside_ortho_2d() {
        let b = ortho_box_2d(6.0, 8.0);
        assert!(b.is_point_inside(&Cartesian::from([0.0, 0.0])));
    }

    #[test]
    fn interior_point_is_inside() {
        let b = ortho_box_2d(6.0, 8.0);
        assert!(b.is_point_inside(&Cartesian::from([2.5, -3.5])));
    }

    /// Points exactly on the minimum face (coordinate = −L/2) are inside.
    #[test]
    fn min_face_is_inside() {
        let b = ortho_box_2d(6.0, 8.0);
        assert!(b.is_point_inside(&Cartesian::from([-3.0, 0.0])));
        assert!(b.is_point_inside(&Cartesian::from([0.0, -4.0])));
    }

    /// Points exactly on the maximum face (coordinate = +L/2) are outside.
    #[test]
    fn max_face_is_outside() {
        let b = ortho_box_2d(6.0, 8.0);
        assert!(!b.is_point_inside(&Cartesian::from([3.0, 0.0])));
        assert!(!b.is_point_inside(&Cartesian::from([0.0, 4.0])));
    }

    #[test]
    fn outside_point_is_not_inside() {
        let b = ortho_box_2d(6.0, 8.0);
        assert!(!b.is_point_inside(&Cartesian::from([4.0, -3.5])));
    }

    #[test]
    fn is_point_inside_3d() {
        let b = ortho_box_3d(10.0, 12.0, 14.0);
        assert!(b.is_point_inside(&Cartesian::from([4.9, 5.9, 6.9])));
        assert!(!b.is_point_inside(&Cartesian::from([5.0, 0.0, 0.0]))); // on +x face
        assert!(!b.is_point_inside(&Cartesian::from([0.0, 6.0, 0.0]))); // on +y face
        assert!(!b.is_point_inside(&Cartesian::from([0.0, 0.0, 7.0]))); // on +z face
        assert!(b.is_point_inside(&Cartesian::from([-5.0, 0.0, 0.0]))); // on −x face (inside)
    }

    #[test]
    fn is_point_inside_tilted_2d() {
        let b =
            Hyperparallelepiped::new([Cartesian::from([4.0, 0.0]), Cartesian::from([1.0, 4.0])]);
        // Fractional origin maps to Cartesian (0,0) — must be inside
        assert!(b.is_point_inside(&Cartesian::from([0.0, 0.0])));
    }

    // ------------------------------------------------------------------
    // SupportMapping
    // ------------------------------------------------------------------

    #[test]
    fn support_mapping_axis_aligned_2d() {
        let b = unit_box_2d();
        // Direction (1, 1) → top-right corner (0.5, 0.5)
        let s = b.support_mapping(&Cartesian::from([1.0, 1.0]));
        assert_eq!(s[0], 0.5);
        assert_eq!(s[1], 0.5);
    }

    #[test]
    fn support_mapping_negative_direction() {
        let b = unit_box_2d();
        // Direction (−1, −1) → bottom-left corner (−0.5, −0.5)
        let s = b.support_mapping(&Cartesian::from([-1.0, -1.0]));
        assert_eq!(s[0], -0.5);
        assert_eq!(s[1], -0.5);
    }

    #[test]
    fn support_mapping_mixed_direction_2d() {
        let b = ortho_box_2d(4.0, 6.0);
        // Direction (1, −1) → (2.0, −3.0)
        let s = b.support_mapping(&Cartesian::from([1.0, -1.0]));
        assert_eq!(s[0], 2.0);
        assert_eq!(s[1], -3.0);
    }

    #[test]
    fn support_mapping_3d() {
        let b = ortho_box_3d(2.0, 4.0, 6.0);
        // All-positive direction → corner (1.0, 2.0, 3.0)
        let s = b.support_mapping(&Cartesian::from([1.0, 1.0, 1.0]));
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], 2.0);
        assert_eq!(s[2], 3.0);
    }

    #[test]
    fn scale_length_scales_volume_by_nth_power() {
        let b = ortho_box_2d(2.0, 3.0);
        let scaled = b.scale_length(PositiveReal::try_from(2.0).unwrap());

        // scaling length by 2 in 2D multiplies area by 2^2 = 4
        assert_relative_eq!(scaled.volume(), 24.0);
    }

    #[test]
    fn scale_volume_scales_box_volume() {
        let b = Hyperparallelepiped::new([
            Cartesian::from([2.0, 0.0, 0.0]),
            Cartesian::from([0.0, 3.0, 0.0]),
            Cartesian::from([0.0, 0.0, 4.0]),
        ]);

        assert_relative_eq!(b.volume(), 24.0);

        let scaled = b.scale_volume(PositiveReal::try_from(8.0).unwrap());
        assert_relative_eq!(scaled.volume(), 192.0);
    }

    #[test]
    fn map_point_identity() {
        let b = ortho_box_2d(4.0, 4.0);
        let p = Cartesian::from([1.0, -1.0]);
        let mapped = b.map_point(p, &b).unwrap();
        assert_relative_eq!(mapped, p);
    }

    #[test]
    fn map_point_scales_uniformly() {
        // Mapping from a 4×4 box to an 8×8 box should double all coordinates.
        let src = ortho_box_2d(4.0, 4.0);
        let dst = ortho_box_2d(8.0, 8.0);
        let p = Cartesian::from([1.0, 1.0]);
        let mapped = src.map_point(p, &dst).unwrap();
        assert_relative_eq!(mapped, Cartesian::from([2.0, 2.0]));
    }

    #[test]
    fn map_point_anisotropic_scaling() {
        // x-axis doubles, y-axis stays the same
        let src = ortho_box_2d(4.0, 6.0);
        let dst = ortho_box_2d(8.0, 6.0);
        let p = Cartesian::from([1.0, 1.5]);
        let mapped = src.map_point(p, &dst).unwrap();
        assert_relative_eq!(mapped, Cartesian::from([2.0, 1.5]));
    }

    #[test]
    fn map_point_3d() {
        let src = ortho_box_3d(10.0, 12.0, 14.0);
        let dst = ortho_box_3d(20.0, 24.0, 28.0);
        let p = Cartesian::from([3.0, -4.0, 6.0]);
        let mapped = src.map_point(p, &dst).unwrap();
        assert_relative_eq!(mapped, Cartesian::from([6.0, -8.0, 12.0]));
    }

    #[test]
    fn nearest_plane_distance_triclinic_box() {
        let b = Hyperparallelepiped::new([
            Cartesian::from([4.0, 0.0, 0.0]),
            Cartesian::from([2.0, 4.0, 0.0]),
            Cartesian::from([2.0, 1.0, 4.0]),
        ]);

        let distances = b.nearest_plane_distances();
        let expected = [3.39199, 3.88057, 4.];

        for i in 0..3 {
            assert_relative_eq!(expected[i], distances[i].get(), epsilon = 1.0e-4);
        }
    }

    #[test]
    fn nearest_plane_distance_rotated_box_matrix() {
        // Take the previous box and rotate it. The distances shouldn't change.
        let b = Hyperparallelepiped::new([
            Cartesian::from([1.33333, 3.64273, -0.976_068]),
            Cartesian::from([-0.309_401, 3.1547, 3.1547]),
            Cartesian::from([4.06538, 1.17863, 1.75598]),
        ]);
        let distances = b.nearest_plane_distances();
        let expected = [3.39199, 3.88057, 4.];

        for i in 0..3 {
            assert_relative_eq!(expected[i], distances[i].get(), epsilon = 1.0e-4);
        }
    }
}

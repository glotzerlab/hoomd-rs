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

use crate::{IsPointInside, MapPoint, Scale, SupportMapping, Volume};

/// An N-dimensional hyperparallelepiped defined by N edge vectors.
///
/// A hyperparallelepiped (also known as a parallelotope) is the N-dimensional generalization of a parallelogram in 2D
/// and parallelepiped in 3D. It is the set of all points that can be expressed as a
/// linear combination of its edge vectors with coefficients in `[-0.5, 0.5)`.
///
/// The shape can be used as the box geometry for simulations, but users should prefer [Rhomboid](crate::shape::Rhomboid)
/// and [Triclinic](crate::shape::Triclinic) for 2 and 3-dimensional simulations, respectively. The QR
/// factorization of the edge vector matrix is cached in `_qr` to accelerate repeated coordinate
/// conversions between Cartesian and fractional frames.
///
/// # Type aliases
///
/// - [`Parallelogram`] — 2D specialization
/// - [`Parallelepiped`] — 3D specialization
///
/// # Example
///
/// ```
/// use hoomd_geometry::shape::Hyperparallelepiped;
/// use hoomd_vector::Cartesian;
///
/// // Construct an orthorhombic box with dimensions 10 x 12 x 14
/// let mut box3d = Hyperparallelepiped::new([
///     Cartesian::from([10.0, 0.0, 0.0]),
///     Cartesian::from([0.0, 12.0, 0.0]),
///     Cartesian::from([0.0, 0.0, 14.0]),
/// ]);
/// box3d.calc_qr();
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Hyperparallelepiped<const N: usize> {
    /// The N edge vectors that define the shape. Each vector spans one
    /// edge of the parallelotope.
    pub edge_vectors: [Cartesian<N>; N],

    /// Cached (condensed) QR factorization of the column matrix formed by the
    /// edge vectors. This is `None` until [`calc_qr`](Self::calc_qr) is called,
    /// and must be computed before any method that converts coordinates between
    /// absolute and fractional positions (e.g. [`to_fractional`](Self::to_fractional),
    /// [`is_point_inside`](IsPointInside::is_point_inside), and
    /// [`map_point`](MapPoint::map_point)).
    pub qr: Option<Matrix<N, N>>,
}

/// A 2D sheared box
pub type Parallelogram = Hyperparallelepiped<2>;

/// A 3D sheared box
pub type Parallelepiped = Hyperparallelepiped<3>;

impl<const N: usize> Default for Hyperparallelepiped<N> {
    /// Returns the N-dimensional unit hypercube: edge vectors are the standard
    /// Cartesian basis vectors `e_0, e_1, …, e_{N-1}` with unit length.
    #[inline]
    fn default() -> Self {
        Self {
            edge_vectors: std::array::from_fn(|i| {
                std::array::from_fn(|j| if i == j { 1. } else { 0. }).into()
            }),
            qr: None,
        }
    }
}

impl<const N: usize> Hyperparallelepiped<N> {
    /// Construct a new hyperparallelepiped from N edge vectors.
    ///
    /// The QR cache is not computed at construction time; call
    /// [`calc_qr`](Self::calc_qr) before performing any coordinate
    /// conversions or point-in-shape tests.
    ///
    /// # Arguments
    ///
    /// * `edge_vectors` — An array of N [`Cartesian`] vectors. The i-th vector
    ///   spans the edge of the parallelotope.
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
    #[must_use]
    pub fn new(edge_vectors: [Cartesian<N>; N]) -> Self {
        Self {
            edge_vectors,
            qr: None,
        }
    }

    /// Compute and cache the QR factorization of the edge-vector matrix.
    ///
    /// The edge vectors are assembled into an N×N matrix $`\mathbf{A}`$ whose *columns*
    /// are the edge vectors, and the result is stored in `self.qr`. This
    /// factorization is later used by [`to_fractional`](Self::to_fractional),
    /// [`is_point_inside`](IsPointInside::is_point_inside), and
    /// [`map_point`](MapPoint::map_point) to solve the linear system
    /// $`\mathbf{A} \vec{s} = \vec{r}`$.
    ///
    /// This method must be called once after construction (or after modifying
    /// `edge_vectors`) before any coordinate conversion is attempted. Calling
    /// it multiple times is safe — it simply recomputes the cache.
    #[inline]
    pub fn calc_qr(&mut self) {
        let box_matrix = Matrix {
            rows: std::array::from_fn(|r| {
                std::array::from_fn(|c| self.edge_vectors[c].coordinates[r])
            }),
        };

        let (qr_mat, _taus) = box_matrix.qr();
        self.qr = Some(qr_mat);
    }

    /// Determine the maximal extents of the hyperparallelepiped along each
    /// Cartesian axis. That is, the furthest the box spans in each coordinate direction.
    ///
    /// For each axis `k`, the maximal extent is half the sum of the absolute
    /// values of the k-th component across all edge vectors. This gives the
    /// smallest axis-aligned bounding box (AABB) that contains the shape.
    ///
    /// # Returns
    ///
    /// An array `[f64; N]` where entry `k` is the largest positive coordinate
    /// along axis `k` that the shape can reach.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Hyperparallelepiped;
    /// use hoomd_vector::Cartesian;
    ///
    /// let unit_square = Hyperparallelepiped::<2>::default();
    /// assert_eq!(unit_square.maximal_extents(), [0.5, 0.5]);
    /// ```
    #[inline]
    #[must_use]
    pub fn maximal_extents(&self) -> [f64; N] {
        (0.5 * self
            .edge_vectors
            .iter()
            .fold(Cartesian::<N>::default(), |acc, v| v.map(f64::abs) + acc))
        .into()
    }

    /// Determine the minimal extents of the hyperparallelepiped along each
    /// Cartesian axis.
    ///
    /// This is simply the negation of [`maximal_extents`](Self::maximal_extents),
    /// representing the most-negative reachable coordinate along each axis.
    ///
    /// # Returns
    ///
    /// An array `[f64; N]` where entry `k` is the most-negative coordinate
    /// along axis `k` that the shape can reach.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Hyperparallelepiped;
    ///
    /// let unit_square = Hyperparallelepiped::<2>::default();
    /// assert_eq!(unit_square.minimal_extents(), [-0.5, -0.5]);
    /// ```
    #[inline]
    #[must_use]
    pub fn minimal_extents(&self) -> [f64; N] {
        self.maximal_extents().map(|x| -x)
    }

    /// Convert a Cartesian vector to fractional (lattice) coordinates.
    ///
    /// Fractional coordinates express a point as coefficients of the edge
    /// vectors. If the edge vectors form the columns of matrix $`\mathbf{A}`$, then
    /// the fractional coordinate vector $`\vec{s}`$ satisfies $`\mathbf{A}\vec{s}=\vec{r}`$,
    /// solved here using the cached QR factorization for numerical stability.
    ///
    /// # Panics
    ///
    /// Panics if the QR decomposition has not been computed using [`calc_qr`](Self::calc_qr).
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Hyperparallelepiped;
    /// use hoomd_vector::Cartesian;
    ///
    /// let mut box2d = Hyperparallelepiped::new([
    ///     Cartesian::from([4.0, 0.0]),
    ///     Cartesian::from([0.0, 6.0]),
    /// ]);
    /// box2d.calc_qr();
    ///
    /// // A point at (1.0, 1.5) should have fractional coords (0.25, 0.25)
    /// let frac = box2d.to_fractional(Cartesian::from([1.0, 1.5]));
    /// assert!((frac[0] - 0.25).abs() < 1e-10);
    /// assert!((frac[1] - 0.25).abs() < 1e-10);
    /// ```
    #[inline]
    #[must_use]
    pub fn to_fractional(&self, v: Cartesian<N>) -> Cartesian<N> {
        Cartesian::from_col_matrix(&qr::qr_solve(
            self.qr
                .as_ref()
                .expect("qr attribute is not computed; call calc_qr() first"),
            &v.to_column_matrix(),
        ))
    }

    /// Convert fractional (lattice) coordinates to Cartesian coordinates.
    ///
    /// This is the inverse of [`to_fractional`](Self::to_fractional). Given a
    /// vector of fractional coefficients $`\vec{s}`$, the Cartesian point is:
    ///
    /// ```math
    /// \vec{r} = \sum_{i=0}^{N-1} s_i \, \vec{a}_i
    /// ```
    ///
    /// where $`\vec{a}_i`$ are the edge vectors.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Hyperparallelepiped;
    /// use hoomd_vector::Cartesian;
    ///
    /// let mut box2d = Hyperparallelepiped::new([
    ///     Cartesian::from([4.0, 0.0]),
    ///     Cartesian::from([0.0, 6.0]),
    /// ]);
    /// box2d.calc_qr();
    ///
    /// // Fractional (0.25, 0.25) should map back to Cartesian (1.0, 1.5)
    /// let cart = box2d.to_absolute(Cartesian::from([0.25, 0.25]));
    /// assert!((cart[0] - 1.0).abs() < 1e-10);
    /// assert!((cart[1] - 1.5).abs() < 1e-10);
    /// ```
    #[inline]
    #[must_use]
    pub fn to_absolute(&self, f: Cartesian<N>) -> Cartesian<N> {
        let mut absolute = Cartesian::<N>::default();
        for (i, edge_vector) in self.edge_vectors.iter().enumerate() {
            absolute += f[i] * *edge_vector;
        }
        absolute
    }

    /// Computes the perpendicular distances from the origin to each of the N bounding
    /// hyperplanes of the parallelotope.
    ///
    /// # Mathematical Background
    ///
    /// The perpendicular distance between faces can be found using the reciprocal
    /// lattice construction. Let $\vec{b}_i$ be a normal vector to the face spanned
    /// by all edge vectors except $\vec{a}_i$. Then
    /// $h_k = \lVert \operatorname{proj}_{b_k}(\vec{a}_k) \rVert$.
    ///
    /// For the edge-vector matrix $\mathbf{A}$, write the QR decomposition
    /// $\mathbf{A} = \mathbf{Q}\mathbf{R}$. Since $\mathbf{Q}$ is orthogonal,
    /// right-multiplying by $\mathbf{Q}^T$ preserves the Euclidean norm of each
    /// row. Therefore the norm of the $k$-th row of $\mathbf{R}^{-1}$ is the
    /// same as the norm of the corresponding row of $\mathbf{A}^{-1}$.
    ///
    /// Hence,
    /// ```math
    /// h_k = \frac{1}{\left\lVert (\mathbf{R}^{-1})_k \right\rVert},
    /// ```
    /// where $\lVert (\mathbf{R}^{-1})_k \rVert$ denotes the Euclidean norm of the
    /// $k$-th row of $\mathbf{R}^{-1}$.
    ///
    /// # Returns
    ///
    /// An array of N [`PositiveReal`] values $`[h_0, h_1, \dots, h_{N-1}]`$, where $`h_k`$ is
    /// the perpendicular distance from the origin to the $`k`$-th bounding hyperplane.
    ///
    /// # Panics
    ///
    /// Panics if the QR decomposition has not been computed using [`calc_qr`](Self::calc_qr).
    #[inline]
    #[must_use]
    pub fn get_nearest_plane_distance(&self) -> [PositiveReal; N] {
        let r_inv = get_r_inv(
            self.qr
                .as_ref()
                .expect("qr attribute is not computed; call calc_qr() first"),
        );
        let distances: [PositiveReal; N] = std::array::from_fn(|i| {
            let row = r_inv.get_row(i);
            let inv_norm = 1.0 / row.as_slice().iter().map(|&x| x * x).sum::<f64>().sqrt();
            inv_norm.try_into().expect("row norm must be positive")
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
    /// If the QR decomposition has not been calculated, calucales the volume using the
    /// determinant of the box matrix.
    #[inline]
    fn volume(&self) -> f64 {
        let matrix = Matrix::<N, N> {
            rows: std::array::from_fn(|r| {
                std::array::from_fn(|c| self.edge_vectors[c].coordinates[r])
            }),
        };

        if let Some(qr_mat) = self.qr.as_ref() {
            let r = get_r(qr_mat);
            r.diagonal().elements.iter().product::<f64>().abs()
        } else {
            matrix.determinant().abs()
        }
    }
}

impl<const N: usize> Scale for Hyperparallelepiped<N> {
    /// Produce a scaled hyperparallelepiped by uniformly scaling each edge vector.
    ///
    /// The QR cache is recomputed on the returned value so that it is immediately
    /// usable for coordinate conversions and ghost generation.
    #[inline]
    fn scale_length(&self, v: PositiveReal) -> Self {
        let mut scaled = Self {
            edge_vectors: self.edge_vectors.map(|ev| ev * v),
            qr: None,
        };
        scaled.calc_qr();
        scaled
    }

    /// Produce a scaled hyperparallelepiped by uniformly scaling volume.
    ///
    /// Each edge vector is scaled by `v^(1/N)` so that the N-dimensional
    /// volume scales by exactly `v`.
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
    /// Compute the support point of the hyperparallelepiped in a given direction.
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
    ///
    /// # Arguments
    ///
    /// * `direction` — The query direction (need not be normalised).
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{SupportMapping, shape::Hyperparallelepiped};
    /// use hoomd_vector::Cartesian;
    ///
    /// let unit_square = Hyperparallelepiped::<2>::default();
    ///
    /// // Querying along +x should return the top-right corner (0.5, 0.5)
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
    /// # Panics
    ///
    /// Panics if the QR decomposition has not been computed using [`calc_qr`](Self::calc_qr).
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{IsPointInside, shape::Hyperparallelepiped};
    /// use hoomd_vector::Cartesian;
    ///
    /// let mut box2d = Hyperparallelepiped::new([
    ///     Cartesian::from([6.0, 0.0]),
    ///     Cartesian::from([0.0, 8.0]),
    /// ]);
    /// box2d.calc_qr();
    ///
    /// assert!( box2d.is_point_inside(&Cartesian::from([ 2.5, -3.5])));  // interior
    /// assert!( box2d.is_point_inside(&Cartesian::from([-3.0,  0.0]))); // on min face (inside)
    /// assert!(!box2d.is_point_inside(&Cartesian::from([ 3.0, -3.5]))); // on max face (outside)
    /// assert!(!box2d.is_point_inside(&Cartesian::from([ 4.0, -3.5]))); // beyond max
    /// ```
    #[inline]
    fn is_point_inside(&self, point: &Cartesian<N>) -> bool {
        let fractional = qr::qr_solve(
            self.qr
                .as_ref()
                .expect("qr attribute is not computed; call calc_qr() first"),
            &point.to_column_matrix(),
        );

        fractional
            .rows
            .into_iter()
            .all(|x| -1.0 / 2.0 <= x[0] && x[0] < 1.0 / 2.0)
    }
}

impl<const N: usize> MapPoint<Cartesian<N>> for Hyperparallelepiped<N> {
    /// Map a point from one hyperparallelepiped to another. The same linear transformation to convert one box to another is used to transform the point within the box.
    ///
    /// Converts `point` (expressed in `self`'s Cartesian frame) to fractional
    /// coordinates relative to `self`, then evaluates those same fractional
    /// coordinates in `other`'s frame. This is used to rescale or deform a
    /// simulation box while preserving the relative positions of particles.
    ///
    /// # Arguments
    ///
    /// * `point` — A Cartesian coordinate in `self`'s frame.
    /// * `other` — The target hyperparallelepiped.
    ///
    /// # Returns
    ///
    /// The corresponding Cartesian coordinate in `other`'s frame.
    ///
    /// # Panics
    ///
    /// Panics if the QR decomposition has not been computed using [`calc_qr`](Self::calc_qr). (needed for [`to_fractional`](Hyperparallelepiped::to_fractional)).
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{MapPoint, shape::Hyperparallelepiped};
    /// use hoomd_vector::Cartesian;
    ///
    /// // Map from a 4×4 box to an 8×8 box — coordinates should double.
    /// let mut src = Hyperparallelepiped::new([
    ///     Cartesian::from([4.0, 0.0]),
    ///     Cartesian::from([0.0, 4.0]),
    /// ]);
    /// src.calc_qr();
    ///
    /// let dst = Hyperparallelepiped::new([
    ///     Cartesian::from([8.0, 0.0]),
    ///     Cartesian::from([0.0, 8.0]),
    /// ]);
    ///
    /// let mapped = src.map_point(Cartesian::from([1.0, 1.0]), &dst).unwrap();
    /// assert!((mapped[0] - 2.0).abs() < 1e-10);
    /// assert!((mapped[1] - 2.0).abs() < 1e-10);
    /// ```
    #[inline]
    fn map_point(&self, point: Cartesian<N>, other: &Self) -> Result<Cartesian<N>, crate::Error> {
        let fractional = self.to_fractional(point);
        let mapped_coords = other.to_absolute(fractional);
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
    /// let mut box2d = Hyperparallelepiped::new([
    ///     Cartesian::from([6.0, 0.0]),
    ///     Cartesian::from([0.0, 8.0]),
    /// ]);
    /// box2d.calc_qr();
    /// let mut rng = StdRng::seed_from_u64(1);
    ///
    /// let point = box2d.sample(&mut rng);
    /// assert!(box2d.is_point_inside(&point));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Cartesian<N> {
        let uniform = Uniform::new(-0.5, 0.5).expect("");
        let fractional: [f64; N] = array::from_fn(|_| uniform.sample(rng));
        self.to_absolute(Cartesian::from(fractional))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_ulps_eq;
    use hoomd_utility::valid::PositiveReal;

    fn assert_approx_eq_cartesian<const N: usize>(a: Cartesian<N>, b: Cartesian<N>, tol: f64) {
        for i in 0..N {
            assert!(
                (a[i] - b[i]).abs() < tol,
                "coordinate {i}: expected {}, got {} (diff {})",
                b[i],
                a[i],
                (a[i] - b[i]).abs()
            );
        }
    }

    fn ortho_box_2d(lx: f64, ly: f64) -> Hyperparallelepiped<2> {
        let mut b =
            Hyperparallelepiped::new([Cartesian::from([lx, 0.0]), Cartesian::from([0.0, ly])]);
        b.calc_qr();
        b
    }

    fn ortho_box_3d(lx: f64, ly: f64, lz: f64) -> Hyperparallelepiped<3> {
        let mut b = Hyperparallelepiped::new([
            Cartesian::from([lx, 0.0, 0.0]),
            Cartesian::from([0.0, ly, 0.0]),
            Cartesian::from([0.0, 0.0, lz]),
        ]);
        b.calc_qr();
        b
    }

    #[test]
    fn default_2d_is_unit_square() {
        let b = Hyperparallelepiped::<2>::default();
        assert_eq!(b.edge_vectors[0], Cartesian::from([1.0, 0.0]));
        assert_eq!(b.edge_vectors[1], Cartesian::from([0.0, 1.0]));
        assert!(b.qr.is_none());
    }

    #[test]
    fn default_3d_is_unit_cube() {
        let b = Hyperparallelepiped::<3>::default();
        assert_eq!(b.edge_vectors[0], Cartesian::from([1.0, 0.0, 0.0]));
        assert_eq!(b.edge_vectors[1], Cartesian::from([0.0, 1.0, 0.0]));
        assert_eq!(b.edge_vectors[2], Cartesian::from([0.0, 0.0, 1.0]));
    }

    #[test]
    fn new_stores_edge_vectors() {
        let vecs = [Cartesian::from([2.0, 1.0]), Cartesian::from([-1.0, 3.0])];
        let b = Hyperparallelepiped::new(vecs);
        assert_eq!(b.edge_vectors[0], Cartesian::from([2.0, 1.0]));
        assert_eq!(b.edge_vectors[1], Cartesian::from([-1.0, 3.0]));
        assert!(b.qr.is_none());
    }

    #[test]
    fn calc_qr_populates_cache() {
        let b = ortho_box_2d(4.0, 6.0);
        assert!(b.qr.is_some());
    }

    #[test]
    fn maximal_extents_unit_square() {
        let b = Hyperparallelepiped::<2>::default();
        assert_eq!(b.maximal_extents(), [0.5, 0.5]);
    }

    #[test]
    fn minimal_extents_unit_square() {
        let b = Hyperparallelepiped::<2>::default();
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
        assert_ulps_eq!(ext[0], 1.5, epsilon = 1.0e-12);
        assert_ulps_eq!(ext[1], 1.5, epsilon = 1.0e-12);
    }

    #[test]
    fn fractional_round_trip_ortho_2d() {
        let b = ortho_box_2d(4.0, 6.0);
        let original = Cartesian::from([1.0, 1.5]);
        let frac = b.to_fractional(original);
        let back = b.to_absolute(frac);
        assert_approx_eq_cartesian(back, original, 1e-10);
    }

    #[test]
    fn fractional_round_trip_ortho_3d() {
        let b = ortho_box_3d(10.0, 12.0, 14.0);
        let original = Cartesian::from([3.0, -4.0, 6.5]);
        let frac = b.to_fractional(original);
        let back = b.to_absolute(frac);
        assert_approx_eq_cartesian(back, original, 1e-10);
    }

    #[test]
    fn fractional_round_trip_tilted_2d() {
        // Triclinic-like 2D box
        let mut b =
            Hyperparallelepiped::new([Cartesian::from([3.0, 0.0]), Cartesian::from([1.0, 4.0])]);
        b.calc_qr();
        let original = Cartesian::from([0.5, 1.0]);
        let frac = b.to_fractional(original);
        let back = b.to_absolute(frac);
        assert_approx_eq_cartesian(back, original, 1e-10);
    }

    #[test]
    fn to_fractional_known_values_ortho() {
        // For a 4×6 box, the point (1, 1.5) should have fractional coords (0.25, 0.25)
        let b = ortho_box_2d(4.0, 6.0);
        let frac = b.to_fractional(Cartesian::from([1.0, 1.5]));
        assert!((frac[0] - 0.25).abs() < 1e-10);
        assert!((frac[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn to_absolute_origin_maps_to_origin() {
        let b = ortho_box_3d(5.0, 7.0, 9.0);
        let origin = Cartesian::from([0.0, 0.0, 0.0]);
        let result = b.to_absolute(origin);
        assert_approx_eq_cartesian(result, origin, 1e-12);
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
        let mut b =
            Hyperparallelepiped::new([Cartesian::from([4.0, 0.0]), Cartesian::from([1.0, 4.0])]);
        b.calc_qr();
        // Fractional origin maps to Cartesian (0,0) — must be inside
        assert!(b.is_point_inside(&Cartesian::from([0.0, 0.0])));
    }

    // ------------------------------------------------------------------
    // SupportMapping
    // ------------------------------------------------------------------

    #[test]
    fn support_mapping_axis_aligned_2d() {
        let b = Hyperparallelepiped::<2>::default();
        // Direction (1, 1) → top-right corner (0.5, 0.5)
        let s = b.support_mapping(&Cartesian::from([1.0, 1.0]));
        assert_ulps_eq!(s[0], 0.5, epsilon = 1.0e-12);
        assert_ulps_eq!(s[1], 0.5, epsilon = 1.0e-12);
    }

    #[test]
    fn support_mapping_negative_direction() {
        let b = Hyperparallelepiped::<2>::default();
        // Direction (−1, −1) → bottom-left corner (−0.5, −0.5)
        let s = b.support_mapping(&Cartesian::from([-1.0, -1.0]));
        assert_ulps_eq!(s[0], -0.5, epsilon = 1.0e-12);
        assert_ulps_eq!(s[1], -0.5, epsilon = 1.0e-12);
    }

    #[test]
    fn support_mapping_mixed_direction_2d() {
        let b = ortho_box_2d(4.0, 6.0);
        // Direction (1, −1) → (2.0, −3.0)
        let s = b.support_mapping(&Cartesian::from([1.0, -1.0]));
        assert_ulps_eq!(s[0], 2.0, epsilon = 1.0e-12);
        assert_ulps_eq!(s[1], -3.0, epsilon = 1.0e-12);
    }

    #[test]
    fn support_mapping_3d() {
        let b = ortho_box_3d(2.0, 4.0, 6.0);
        // All-positive direction → corner (1.0, 2.0, 3.0)
        let s = b.support_mapping(&Cartesian::from([1.0, 1.0, 1.0]));
        assert_ulps_eq!(s[0], 1.0, epsilon = 1.0e-12);
        assert_ulps_eq!(s[1], 2.0, epsilon = 1.0e-12);
        assert_ulps_eq!(s[2], 3.0, epsilon = 1.0e-12);
    }

    #[test]
    fn scale_length_scales_volume_by_nth_power() {
        let b = ortho_box_2d(2.0, 3.0);
        let scaled = b.scale_length(PositiveReal::try_from(2.0).unwrap());

        // scaling length by 2 in 2D multiplies area by 2^2 = 4
        assert_ulps_eq!(scaled.volume(), 24.0, epsilon = 1.0e-12);
    }

    #[test]
    fn scale_volume_scales_box_volume() {
        let b = Hyperparallelepiped::new([
            Cartesian::from([2.0, 0.0, 0.0]),
            Cartesian::from([0.0, 3.0, 0.0]),
            Cartesian::from([0.0, 0.0, 4.0]),
        ]);

        assert_ulps_eq!(b.volume(), 24.0, epsilon = 1.0e-12);

        let scaled = b.scale_volume(PositiveReal::try_from(8.0).unwrap());
        assert_ulps_eq!(scaled.volume(), 192.0, epsilon = 1.0e-12);
    }

    #[test]
    fn map_point_identity() {
        let b = ortho_box_2d(4.0, 4.0);
        let p = Cartesian::from([1.0, -1.0]);
        let mapped = b.map_point(p, &b).unwrap();
        assert_approx_eq_cartesian(mapped, p, 1e-10);
    }

    #[test]
    fn map_point_scales_uniformly() {
        // Mapping from a 4×4 box to an 8×8 box should double all coordinates.
        let src = ortho_box_2d(4.0, 4.0);
        let dst = ortho_box_2d(8.0, 8.0);
        let p = Cartesian::from([1.0, 1.0]);
        let mapped = src.map_point(p, &dst).unwrap();
        assert_approx_eq_cartesian(mapped, Cartesian::from([2.0, 2.0]), 1e-10);
    }

    #[test]
    fn map_point_anisotropic_scaling() {
        // x-axis doubles, y-axis stays the same
        let src = ortho_box_2d(4.0, 6.0);
        let dst = ortho_box_2d(8.0, 6.0);
        let p = Cartesian::from([1.0, 1.5]);
        let mapped = src.map_point(p, &dst).unwrap();
        assert_approx_eq_cartesian(mapped, Cartesian::from([2.0, 1.5]), 1e-10);
    }

    #[test]
    fn map_point_3d() {
        let src = ortho_box_3d(10.0, 12.0, 14.0);
        let dst = ortho_box_3d(20.0, 24.0, 28.0);
        let p = Cartesian::from([3.0, -4.0, 6.0]);
        let mapped = src.map_point(p, &dst).unwrap();
        assert_approx_eq_cartesian(mapped, Cartesian::from([6.0, -8.0, 12.0]), 1e-10);
    }

    #[test]
    fn nearest_plane_distance_triclinic_box() {
        let mut b = Hyperparallelepiped::new([
            Cartesian::from([4.0, 0.0, 0.0]),
            Cartesian::from([2.0, 4.0, 0.0]),
            Cartesian::from([2.0, 1.0, 4.0]),
        ]);
        b.calc_qr();

        let distances = b.get_nearest_plane_distance();
        let expected = [3.39199, 3.88057, 4.];

        for i in 0..3 {
            assert_ulps_eq!(expected[i], distances[i].get(), epsilon = 1.0e-4);
        }
    }

    #[test]
    fn nearest_plane_distance_rotated_box_matrix() {
        // Take the previous box and rotate it. The distances shouldn't change.
        let mut b = Hyperparallelepiped::new([
            Cartesian::from([1.33333, 3.64273, -0.976_068]),
            Cartesian::from([-0.309_401, 3.1547, 3.1547]),
            Cartesian::from([4.06538, 1.17863, 1.75598]),
        ]);
        b.calc_qr();
        let distances = b.get_nearest_plane_distance();
        let expected = [3.39199, 3.88057, 4.];

        for i in 0..3 {
            assert_ulps_eq!(expected[i], distances[i].get(), epsilon = 1.0e-4);
        }
    }
}

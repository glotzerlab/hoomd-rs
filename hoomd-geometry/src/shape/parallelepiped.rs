// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use hoomd_linear_algebra::matrix::{
    Matrix,
    qr::{self, get_r_inv},
};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, InnerProduct};

use crate::{IsPointInside, MapPoint, SupportMapping};

/// An N-dimensional hyperparallelepiped (parallelotope) defined by N edge vectors.
///
/// A hyperparallelepiped is the N-dimensional generalization of a parallelogram (2D)
/// and parallelepiped (3D). It is the set of all points that can be expressed as a
/// linear combination of the edge vectors with coefficients in `[-0.5, 0.5)`, i.e.
/// the shape is centered at the origin.
///
/// The shape can be used as the box geometry for simulations, but users should prefer [Rhomboid]
/// and [`Triclinic`] for 2 and 3-dimensional simulations, respectively. The QR
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
    /// The N edge vectors that define the shape. Each vector points along one
    /// edge of the parallelotope emanating from the origin.
    pub edge_vectors: [Cartesian<N>; N],

    /// Cached (condensed) QR factorization of the column matrix formed by the
    /// edge vectors. This is `None` until [`calc_qr`](Self::calc_qr) is called,
    /// and must be computed before any method that converts coordinates between
    /// Cartesian and fractional frames (e.g. [`to_fractional`](Self::to_fractional),
    /// [`is_point_inside`](IsPointInside::is_point_inside), and
    /// [`map_point`](MapPoint::map_point)).
    pub _qr: Option<Matrix<N, N>>,
}

/// A 2D hyperparallelepiped (parallelogram). Prefer rhomboid for 2D sheared
/// boundary conditions.
pub type Parallelogram = Hyperparallelepiped<2>;

/// A 3D hyperparallelepiped (parallelepiped). Prefer rhomboid for 3D sheared
/// boundary conditions.
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
            _qr: None,
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
    ///   is the edge of the parallelotope that starts at the origin along the
    ///   i-th lattice direction.
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
    pub fn new(edge_vectors: [Cartesian<N>; N]) -> Self {
        Self {
            edge_vectors,
            _qr: None,
        }
    }

    /// Compute and cache the QR factorization of the edge-vector matrix.
    ///
    /// The edge vectors are assembled into an N×N matrix **A** whose *columns*
    /// are the edge vectors, and the result is stored in `self._qr`. This
    /// factorization is later used by [`to_fractional`](Self::to_fractional),
    /// [`is_point_inside`](IsPointInside::is_point_inside), and
    /// [`map_point`](MapPoint::map_point) to solve the linear system
    /// **A** **f** = **v** efficiently.
    ///
    /// This method must be called once after construction (or after modifying
    /// `edge_vectors`) before any coordinate conversion is attempted. Calling
    /// it multiple times is safe — it simply recomputes the cache.
    ///
    /// # Note
    ///
    /// The method takes `&mut self` so that the computed factorization is
    /// stored back into the struct.
    pub fn calc_qr(&mut self) {
        self._qr = Some(
            Matrix::<N, N> {
                rows: self.edge_vectors.map(|v| v.coordinates),
            }
            .transpose(),
        );
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
    /// vectors. If the edge vectors form the columns of matrix **A**, then
    /// the fractional coordinate vector **f** satisfies **A** **f** = **v**,
    /// solved here via the cached QR factorization.
    ///
    /// A point is inside the hyperparallelepiped when all fractional
    /// coordinates lie in `[-0.5, 0.5)`.
    ///
    /// # Panics
    ///
    /// Panics if [`calc_qr`](Self::calc_qr) has not been called yet.
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
    pub fn to_fractional(&self, v: Cartesian<N>) -> Cartesian<N> {
        Cartesian::from_col_matrix(qr::qr_solve(
            self._qr
                .as_ref()
                .expect("_qr attribute is not computed; call calc_qr() first"),
            v.to_column_matrix(),
        ))
    }

    /// Convert fractional (lattice) coordinates to Cartesian coordinates.
    ///
    /// This is the inverse of [`to_fractional`](Self::to_fractional). Given a
    /// vector of fractional coefficients **f**, the Cartesian point is:
    ///
    /// ```math
    /// \mathbf{v} = \sum_{i=0}^{N-1} f_i \, \mathbf{a}_i
    /// ```
    ///
    /// where **a**_i are the edge vectors.
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
    pub fn to_absolute(&self, f: Cartesian<N>) -> Cartesian<N> {
        let mut absolute = Cartesian::<N>::default();
        for (i, edge_vector) in self.edge_vectors.iter().enumerate() {
            absolute += f[i] * *edge_vector;
        }
        absolute
    }

    /// Computes the perpendicular distances from the origin to each of the `N` bounding
    /// hyperplanes of the parallelotope.
    ///
    /// # Mathematical Background
    ///
    /// The perpendicular distance (height) $h_k$ to the $k$-th face is derived from the
    /// generalization of the fact that the volume of a prism is equal to the area of the base time the height,
    /// $`V = A \cdot h`$, rearranged as:
    ///
    /// ```math
    /// h_k = \frac{V}{A_k}
    /// ```
    ///
    /// where $V$ is the volume of the parallelotope and $A_k$ is the area of its $k$-th face.
    /// Expressing both via their Gramians yields:
    ///
    /// ```math
    /// h_k = \frac{\det(A)}{\det\!\left(\sqrt{A_k^T A_k}\right)}
    ///      = \frac{1}{\lVert A_k^{-1} \rVert}
    ///      = \frac{1}{\lVert (R^{-1})_k \rVert}
    /// ```
    ///
    /// where $A = QR$ is the QR decomposition of the matrix whose columns are the edge vectors
    /// of the parallelotope, and $(R^{-1})_k$ denotes the $k$-th row of $R^{-1}$.
    ///
    /// That is, each nearest-plane distance is the reciprocal of the norm of the
    /// corresponding row of $R^{-1}$.
    ///
    ///
    /// # Returns
    ///
    /// An array of `N` [`PositiveReal`] values $[h_0, h_1, \dots, h_{N-1}]$, where $h_k$ is
    /// the perpendicular distance from the origin to the $k$-th bounding hyperplane.
    ///
    /// # Panics
    ///
    /// Panics if the QR decomposition has not been computed (i.e. the internal `_qr` field is
    /// `None`)
    pub fn get_nearest_plane_distance(&self) -> [PositiveReal; N] {
        // Since V = A_ih_i, h_i = V/A_i.
        let r_inv = get_r_inv(self._qr.as_ref().unwrap());
        println!("{:?}", self._qr);
        let distances: [PositiveReal; N] = std::array::from_fn(|i| {
            let row = r_inv.get_row(i);
            let inv_norm = 1.0 / row.as_slice().iter().map(|&x| x * x).sum::<f64>().sqrt();
            inv_norm.try_into().expect("row norm must be positive")
        });
        distances
    }
}

impl<const N: usize> SupportMapping<Cartesian<N>> for Hyperparallelepiped<N> {
    /// Compute the support point of the hyperparallelepiped in a given direction.
    ///
    /// The support mapping returns the point on (or inside) the shape that has
    /// the greatest dot product with the query direction. For a
    /// hyperparallelepiped this is computed by choosing, for each edge vector
    /// **a**_i, the vertex ±½ **a**_i whose sign matches the sign of
    /// **a**_i · **d** and summing the contributions:
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
    fn support_mapping(&self, direction: &Cartesian<N>) -> Cartesian<N> {
        0.5 * self
            .edge_vectors
            .iter()
            .fold(Cartesian::<N>::default(), |acc, v| {
                v.dot(direction).signum() * *v + acc
            })
    }
}

impl<const N: usize> IsPointInside<Cartesian<N>> for Hyperparallelepiped<N> {
    /// Check whether a Cartesian point lies inside the hyperparallelepiped.
    ///
    /// The test converts the point to fractional coordinates and checks
    /// that every coordinate lies in the half-open interval `[-0.5, 0.5)`.
    /// This convention — closed on the lower bound, open on the upper bound
    /// — is standard in periodic boundary condition implementations and
    /// ensures that each point belongs to exactly one image of the box
    /// when the lattice is tiled.
    ///
    /// # Panics
    ///
    /// Panics if [`calc_qr`](Hyperparallelepiped::calc_qr) has not been
    /// called yet.
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
    /// assert!( box2d.is_point_inside(&Cartesian::from([-3.0,  4.0]))); // on min face (inside)
    /// assert!(!box2d.is_point_inside(&Cartesian::from([ 3.0, -3.5]))); // on max face (outside)
    /// assert!(!box2d.is_point_inside(&Cartesian::from([ 4.0, -3.5]))); // beyond max
    /// ```
    #[inline]
    fn is_point_inside(&self, point: &Cartesian<N>) -> bool {
        let fractional = qr::qr_solve(
            self._qr
                .as_ref()
                .expect("_qr attribute is not computed; call calc_qr() first"),
            point.to_column_matrix(),
        );

        fractional
            .rows
            .into_iter()
            .all(|x| -1.0 / 2.0 <= x[0] && x[0] < 1.0 / 2.0)
    }
}

impl<const N: usize> MapPoint<Cartesian<N>> for Hyperparallelepiped<N> {
    /// Map a point from one hyperparallelepiped's frame to another's.
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
    /// The corresponding Cartesian coordinate in `other`'s frame, or a
    /// [`crate::Error`] if the conversion fails.
    ///
    /// # Panics
    ///
    /// Panics if [`calc_qr`](Hyperparallelepiped::calc_qr) has not been called
    /// on `self` (needed for [`to_fractional`](Hyperparallelepiped::to_fractional)).
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
    fn map_point(&self, point: Cartesian<N>, other: &Self) -> Result<Cartesian<N>, crate::Error> {
        let fractional = self.to_fractional(point);
        let mapped_coords = other.to_absolute(fractional);
        Ok(mapped_coords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(b._qr.is_none());
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
        assert!(b._qr.is_none());
    }

    #[test]
    fn calc_qr_populates_cache() {
        let b = ortho_box_2d(4.0, 6.0);
        assert!(b._qr.is_some());
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
        assert!((ext[0] - 1.5).abs() < 1e-12);
        assert!((ext[1] - 1.5).abs() < 1e-12);
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
        assert_approx_eq_cartesian(s, Cartesian::from([0.5, 0.5]), 1e-12);
    }

    #[test]
    fn support_mapping_negative_direction() {
        let b = Hyperparallelepiped::<2>::default();
        // Direction (−1, −1) → bottom-left corner (−0.5, −0.5)
        let s = b.support_mapping(&Cartesian::from([-1.0, -1.0]));
        assert_approx_eq_cartesian(s, Cartesian::from([-0.5, -0.5]), 1e-12);
    }

    #[test]
    fn support_mapping_mixed_direction_2d() {
        let b = ortho_box_2d(4.0, 6.0);
        // Direction (1, −1) → (2.0, −3.0)
        let s = b.support_mapping(&Cartesian::from([1.0, -1.0]));
        assert_approx_eq_cartesian(s, Cartesian::from([2.0, -3.0]), 1e-12);
    }

    #[test]
    fn support_mapping_3d() {
        let b = ortho_box_3d(2.0, 4.0, 6.0);
        // All-positive direction → corner (1.0, 2.0, 3.0)
        let s = b.support_mapping(&Cartesian::from([1.0, 1.0, 1.0]));
        assert_approx_eq_cartesian(s, Cartesian::from([1.0, 2.0, 3.0]), 1e-12);
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
            Cartesian::from([0.5, 4.0, 0.0]),
            Cartesian::from([0.5, 0.25, 4.0]),
        ]);
        b.calc_qr();

        let distances = b.get_nearest_plane_distance();
        println!("{:?}", distances);
        let expected = [3.39199, 3.88057, 4.];

        for i in 0..3 {
            assert!(
                (distances[i].get() - expected[i]).abs() < 1e-6,
                "distance[{i}] expected {} got {}",
                expected[i],
                distances[i]
            );
        }
    }

    #[test]
    fn nearest_plane_distance_rotated_box_matrix() {
        let mut b = Hyperparallelepiped::new([
            Cartesian::from([1.33333, -0.309401, 4.06538]),
            Cartesian::from([3.64273, 3.1547, 1.17863]),
            Cartesian::from([-0.976068, 3.1547, 1.75598]),
        ]);
        b.calc_qr();

        let distances = b.get_nearest_plane_distance();
        let expected = [3.39199, 3.88057, 4.];

        for i in 0..3 {
            assert!(
                (distances[i].get() - expected[i]).abs() < 1e-6,
                "distance[{i}] expected {} got {}",
                expected[i],
                distances[i]
            );
        }
    }
}

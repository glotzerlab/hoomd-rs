// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! N-Dimensional generalization of a convex polyhedron.

use std::f64::consts::SQRT_2;

use serde::{Deserialize, Serialize};

use crate::{BoundingSphereRadius, Error, Scale, SupportMapping};
use arrayvec::ArrayVec;
use hoomd_utility::{positive_real, valid::PositiveReal};
use hoomd_vector::{Cartesian, InnerProduct};
use itertools::Itertools;

/// A faceted solid defined by the convex hull of a set of points.
///
/// [`ConvexPolytope`] stores the given point set without any modification.
/// Therefore, it can be constructed quickly. The *implicit* convex
/// hull is formed by [`SupportMapping`] during intersection tests of
/// `Convex(ConvexPolytope)` with other `Convex(_)` types.
///
/// Every vertex in the convex hull is an elements of [`vertices`]. [`vertices`]
/// may also include duplicate, collinear, coplanar, and/or interior points.
/// They are exactly the points given at construction.
///
/// [`vertices`]: Self::vertices
///
/// # Examples
///
/// Construction and basic methods:
/// ```
/// use approxim::assert_relative_eq;
/// use hoomd_geometry::{BoundingSphereRadius, shape::ConvexPolyhedron};
///
/// # fn main() -> Result<(), hoomd_geometry::Error> {
/// let tetrahedron = ConvexPolyhedron::with_vertices([
///     [1.0, 1.0, 1.0].into(),
///     [1.0, -1.0, -1.0].into(),
///     [-1.0, 1.0, -1.0].into(),
///     [-1.0, -1.0, 1.0].into(),
/// ])?;
///
/// let bounding_radius = tetrahedron.bounding_sphere_radius();
///
/// assert_relative_eq!(bounding_radius.get(), 3.0_f64.sqrt());
/// # Ok(())
/// # }
/// ```
///
/// Intersection tests:
/// ```
/// use hoomd_geometry::{Convex, IntersectsAt, shape::ConvexPolygon};
/// use hoomd_vector::{Angle, Cartesian};
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), hoomd_geometry::Error> {
/// let rectangle = ConvexPolygon::with_vertices([
///     [-2.0, -1.0].into(),
///     [2.0, -1.0].into(),
///     [2.0, 1.0].into(),
///     [-2.0, 1.0].into(),
/// ])?;
/// let rectangle = Convex(rectangle);
///
/// assert!(!rectangle.intersects_at(
///     &rectangle,
///     &[0.0, 2.1].into(),
///     &Angle::default()
/// ));
/// assert!(rectangle.intersects_at(
///     &rectangle,
///     &[0.0, 2.1].into(),
///     &Angle::from(PI / 2.0)
/// ));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConvexPolytope<const N: usize, const MAX_VERTICES: usize = 64> {
    /// The vertices of the shape.
    vertices: ArrayVec<Cartesian<N>, MAX_VERTICES>,
    /// The radius of a bounding sphere of the geometry.
    bounding_radius: PositiveReal,
}

/// A faceted convex body in two dimensions.
///
/// ```rust
/// use hoomd_geometry::shape::ConvexPolygon;
///
/// # fn main() -> Result<(), hoomd_geometry::Error> {
/// let hexagon = ConvexPolygon::regular(6);
/// let square = ConvexPolygon::with_vertices([
///     [-1.0, -1.0].into(),
///     [1.0, -1.0].into(),
///     [1.0, 1.0].into(),
///     [-1.0, 1.0].into(),
/// ])?;
/// # Ok(())
/// # }
/// ```
pub type ConvexPolygon = ConvexPolytope<2, 32>;

/// A faceted convex body in three dimensions.
///
/// # Example
///
/// ```
/// use hoomd_geometry::shape::{ConvexPolyhedron, Simplex3};
/// # fn main() -> Result<(), hoomd_geometry::Error> {
/// let poly = ConvexPolyhedron::with_vertices([
///     [1.0, 1.0, 1.0].into(),
///     [1.0, -1.0, -1.0].into(),
///     [-1.0, 1.0, -1.0].into(),
///     [-1.0, -1.0, 1.0].into(),
/// ])?;
///
/// assert_eq!(poly.vertices(), Simplex3::default().vertices());
/// # Ok(())
/// # }
/// ```
pub type ConvexPolyhedron = ConvexPolytope<3, 32>;

impl<const MAX_VERTICES: usize> ConvexPolytope<2, MAX_VERTICES> {
    /// Create a regular *n*-gon with *n* vertices and circumradius 0.5.
    ///
    /// # Example
    /// ```
    /// use hoomd_geometry::shape::ConvexPolygon;
    ///
    /// let equilateral_triangle = ConvexPolygon::regular(3);
    /// ```
    #[inline]
    #[must_use]
    pub fn regular(n: usize) -> ConvexPolytope<2, MAX_VERTICES> {
        ConvexPolytope {
            vertices: (0..n)
                .map(|x| {
                    let theta = 2.0 * std::f64::consts::PI * (x as f64) / (n as f64);
                    Cartesian::from([0.5 * f64::cos(theta), 0.5 * f64::sin(theta)])
                })
                .collect(),
            bounding_radius: positive_real!(0.5),
        }
    }
}

impl<const N: usize, const MAX_VERTICES: usize> ConvexPolytope<N, MAX_VERTICES> {
    /// Create an `N`-polytope with the given vertices.
    ///
    /// # Example
    /// ```
    /// use hoomd_geometry::shape::ConvexPolytope;
    ///
    /// # fn main() -> Result<(), hoomd_geometry::Error> {
    /// let equilateral_triangle = ConvexPolytope::<2>::with_vertices([
    ///     [1.0, 0.0].into(),
    ///     [0.5, f64::sqrt(3.0) / 2.0].into(),
    ///     [-0.5, f64::sqrt(3.0) / 2.0].into(),
    /// ])?;
    /// # Ok(())
    /// # }
    /// ```
    /// # Errors
    ///
    /// [`Error::DegeneratePolytope`] when no vertices are provided.
    #[inline]
    pub fn with_vertices<I>(vertices: I) -> Result<ConvexPolytope<N, MAX_VERTICES>, Error>
    where
        I: IntoIterator<Item = Cartesian<N>>,
    {
        let mut array_vec: ArrayVec<Cartesian<N>, MAX_VERTICES> = ArrayVec::new();
        for v in vertices {
            array_vec.try_push(v).map_err(|_| Error::TooManyVertices)?;
        }

        if array_vec.is_empty() {
            return Err(Error::DegeneratePolytope);
        }

        Ok(ConvexPolytope {
            bounding_radius: Self::bounding_radius(&array_vec),
            vertices: array_vec,
        })
    }

    /// The vertices of the shape.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> &[Cartesian<N>] {
        &self.vertices
    }

    /// Compute the bounding radius.
    pub(crate) fn bounding_radius(vertices: &[Cartesian<N>]) -> PositiveReal {
        vertices
            .iter()
            .map(Cartesian::norm_squared)
            .fold(0.0, f64::max)
            .sqrt()
            .try_into()
            .expect("convex polytope should have a positive bounding radius")
    }

    /// Build the N-dimensional generalization of an octahedron with unit
    /// volume.
    ///
    /// This shape, also referred to as the hyperoctahedron or cross polytope, has `2N`
    /// vertices, one pair on each coordinate axis.
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::{
    ///     BoundingSphereRadius,
    ///     shape::{ConvexPolyhedron, ConvexPolytope},
    /// };
    ///
    /// # fn main() -> Result<(), hoomd_geometry::Error> {
    /// let diamond = ConvexPolytope::<2>::orthoplex();
    /// // A 2D orthoplex is a square, with edges rotated 45° relative to the x axis.
    /// assert_relative_eq!(
    ///     diamond.bounding_sphere_radius().get(),
    ///     f64::sqrt(2.0) / 2.0
    /// );
    /// assert_eq!(diamond.vertices().len(), 2 * 2);
    ///
    /// let octahedron = ConvexPolyhedron::orthoplex();
    /// assert_relative_eq!(
    ///     octahedron.bounding_sphere_radius().get(),
    ///     6.0_f64.cbrt() / 2.0 // (3!)^(1/3) / 2
    /// );
    /// assert_eq!(octahedron.vertices().len(), 2 * 3);
    ///
    /// // Cross polytopes always have 2*N vertices
    /// let octachoron = ConvexPolytope::<4, 8>::orthoplex();
    /// assert_relative_eq!(
    ///     octachoron.bounding_sphere_radius().get(),
    ///     24.0_f64.sqrt().sqrt() / 2.0 // (4!)^(1/4) / 2
    /// );
    /// assert_eq!(octachoron.vertices().len(), 2 * 4);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Panics
    /// If `N=0` or `N > MAX_VERTICES/2`.
    #[inline]
    #[must_use]
    pub fn orthoplex() -> Self {
        assert!(
            N != 0,
            "An orthoplex is not well-defined in zero dimensions!"
        );
        // (2r)^N / N! == 1
        let n_factorial = (1..=N).map(|k| k as f64).product::<f64>();
        let circumradius = n_factorial.powf((N as f64).recip()) / 2.0;
        let vertices = (0..N).flat_map(|nonzero_index| {
            let coord = Cartesian::<N>::from(std::array::from_fn(|i| {
                f64::from(i == nonzero_index) * circumradius
            }));
            [coord, -coord]
        });

        Self {
            vertices: ArrayVec::<_, MAX_VERTICES>::from_iter(vertices),
            bounding_radius: circumradius
                .try_into()
                .expect("the circumradius of a unit-volume orthoplex is positive"),
        }
    }

    /// Build the N-dimensional generalization of a tetrahedron with unit
    /// volume.
    ///
    /// A regular simplex is the convex hull of `N+1` mutually equidistant points:
    /// an equilateral triangle in 2D, a tetrahedron in 3D, and so on.
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::{
    ///     BoundingSphereRadius,
    ///     shape::{ConvexPolygon, ConvexPolyhedron, ConvexPolytope},
    /// };
    ///
    /// # fn main() -> Result<(), hoomd_geometry::Error> {
    /// let triangle = ConvexPolygon::simplex();
    /// assert_relative_eq!(
    ///     triangle.bounding_sphere_radius().get(),
    ///     2.0 / 3.0_f64.powf(0.75) // 2 / 3^(3/4)
    /// );
    ///
    /// let tetrahedron = ConvexPolyhedron::simplex();
    /// assert_relative_eq!(
    ///     tetrahedron.bounding_sphere_radius().get(),
    ///     (9.0 * f64::sqrt(3.0) / 8.0).cbrt() // (9 sqrt(3) / 8)^(1/3)
    /// );
    ///
    /// // Simplices always have N+1 vertices
    /// let pentachoron = ConvexPolytope::<4, 5>::simplex();
    /// assert_relative_eq!(
    ///     pentachoron.bounding_sphere_radius().get(),
    ///     (384.0 / (25.0 * f64::sqrt(5.0))).powf(0.25) // (384 / (25 sqrt(5)))^(1/4)
    /// );
    /// assert_eq!(pentachoron.vertices().len(), 5);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Panics
    /// If `N+1 > MAX_VERTICES`.
    #[inline]
    #[must_use]
    pub fn simplex() -> Self {
        // https://en.wikipedia.org/wiki/Simplex#Cartesian_coordinates_for_a_regular_n-dimensional_simplex_in_Rn
        let inv_sqrt2 = SQRT_2.recip();

        let mut vertices = ArrayVec::<_, MAX_VERTICES>::new();

        // Un-centered: N vertices at scaled standard basis positions + 1 shared vertex
        for k in 0..N {
            vertices.push(std::array::from_fn(|i| f64::from(i == k) * inv_sqrt2).into());
        }

        let c = (1.0 - f64::sqrt(N as f64 + 1.0)) / (f64::sqrt(2.0) * N as f64);
        vertices.push([c; N].into());

        // Center by subtracting centroid
        let center = Cartesian::from([(inv_sqrt2 + c) / (N as f64 + 1.0); N]);
        vertices.iter_mut().for_each(|vertex| *vertex -= center);

        // Rescale the unit-edge simplex to unit volume:
        // s = (2^(N/2) N! / sqrt(N+1))^(1/N).
        let n_factorial = (1..=N).map(|k| k as f64).product::<f64>();
        let scale = (2.0_f64.powf(N as f64 / 2.0) * n_factorial / f64::sqrt(N as f64 + 1.0))
            .powf((N as f64).recip());
        vertices.iter_mut().for_each(|vertex| *vertex *= scale);

        Self {
            vertices,
            bounding_radius: (scale * f64::sqrt(N as f64 / (2.0 * (N as f64 + 1.0))))
                .try_into()
                .expect("sqrt of positive is positive"),
        }
    }

    /// Build the N-dimensional generalization of a cube with unit volume.
    ///
    /// This shape, also referred to as the hypercube or orthotope, has `2^N` vertices
    /// at the signed combinations of `{±0.5}`.
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::{
    ///     BoundingSphereRadius,
    ///     shape::{ConvexPolygon, ConvexPolyhedron, ConvexPolytope},
    /// };
    ///
    /// # fn main() -> Result<(), hoomd_geometry::Error> {
    /// let square = ConvexPolygon::hypercube();
    /// assert_relative_eq!(
    ///     square.bounding_sphere_radius().get(),
    ///     f64::sqrt(2.0) / 2.0
    /// );
    ///
    /// let cube = ConvexPolyhedron::hypercube();
    /// assert_relative_eq!(
    ///     cube.bounding_sphere_radius().get(),
    ///     f64::sqrt(3.0) / 2.0
    /// );
    ///
    /// // Hypercubes have 2^N vertices
    /// let hypercube = ConvexPolytope::<4, 16>::hypercube();
    /// assert_eq!(hypercube.vertices().len(), 16);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Panics
    /// If `N=0` or `2^N > MAX_VERTICES`.
    #[inline]
    #[must_use]
    pub fn hypercube() -> Self {
        assert!(
            N != 0,
            "A hypercube is not well-defined in zero dimensions!"
        );
        let bounding_radius = (f64::sqrt(N as f64) / 2.0)
            .try_into()
            .expect("sqrt(positive) is positive.");

        Self {
            vertices: ArrayVec::<_, MAX_VERTICES>::from_iter(
                sign_variants([0.5; N]).map(Cartesian::from),
            ),
            bounding_radius,
        }
    }
}

impl<const MAX_VERTICES: usize> ConvexPolytope<3, MAX_VERTICES> {
    /// Create an icosahedron with edge length 2.
    ///
    /// # Example
    /// ```
    /// use hoomd_geometry::shape::ConvexPolyhedron;
    ///
    /// let icosahedron = ConvexPolyhedron::icosahedron();
    ///
    /// assert_eq!(icosahedron.vertices().len(), 12);
    /// ```
    /// # Panics
    ///
    /// If the shape is initialized with `MAX_VERTICES` < 12.
    #[inline]
    #[must_use]
    pub fn icosahedron() -> ConvexPolytope<3, MAX_VERTICES> {
        ConvexPolytope::with_vertices(
            permutations([0.0, 1.0, std::f64::consts::GOLDEN_RATIO], true).map(Cartesian::from),
        )
        .expect("an icosahedron requires at least 12 vertices")
    }

    /// Create a dodecahedron with edge length `2 / phi`.
    ///
    /// # Example
    /// ```
    /// use hoomd_geometry::shape::ConvexPolyhedron;
    ///
    /// let dodecahedron = ConvexPolyhedron::dodecahedron();
    ///
    /// assert_eq!(dodecahedron.vertices().len(), 20);
    /// ```
    ///
    /// # Panics
    ///
    /// If the shape is initialized with `MAX_VERTICES` < 20.
    #[inline]
    pub fn dodecahedron() -> ConvexPolytope<3, MAX_VERTICES> {
        let phi = std::f64::consts::GOLDEN_RATIO;
        // An edge-2 cube together with the cyclic permutations of `(0, ±1/phi, ±phi)`.
        let hypercube = Self::hypercube().scale_length(positive_real!(2.0));
        let vertices = hypercube
            .vertices()
            .iter()
            .copied()
            .chain(permutations([0.0, phi.recip(), phi], true).map(Cartesian::from));
        ConvexPolytope::with_vertices(vertices)
            .expect("a dodecahedron requires at least 20 vertices")
    }
}

impl<const MAX_VERTICES: usize> ConvexPolytope<4, MAX_VERTICES> {
    /// Build a 600-cell with unit volume.
    ///
    /// The 600-cell, also called the hypericosahedron or hexacosichoron, is the
    /// four-dimensional analogue of the icosahedron, with 120 vertices and 600
    /// 600 tetrahedral cells. Its vertices are those of a tesseract and a 16-cell
    /// together with the even coordinate permutations of `(phi, 1, 1/phi, 0)/2`.
    /// Equivalently, its vertices form the *icosians*, a 120-element set of points on
    /// the three-sphere that, when interpreted as [`Versor`]s rather than unit
    /// [`Cartesian<4>`] points, form a uniform mesh on the three-dimensional group of
    /// rotations `SO(3)`.
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::{BoundingSphereRadius, shape::ConvexPolytope};
    ///
    /// let hypericosahedron = ConvexPolytope::<4, 120>::hypericosahedron();
    ///
    /// assert_eq!(hypericosahedron.vertices().len(), 120);
    /// assert_relative_eq!(
    ///     hypericosahedron.bounding_sphere_radius().get(),
    ///     (8.0 / (25.0 * (5.0_f64.sqrt() - 1.0))).powf(0.25)
    /// );
    /// ```
    ///
    /// # Panics
    /// If `MAX_VERTICES < 120`.
    #[inline]
    #[must_use]
    pub fn hypericosahedron() -> Self {
        let phi = std::f64::consts::GOLDEN_RATIO;
        let tesseract = Self::hypercube();
        let orthoplex = Self::orthoplex();
        let sixteen_cell = orthoplex.scale_length(orthoplex.bounding_sphere_radius().recip());
        let vertices = tesseract
            .vertices()
            .iter()
            .copied()
            .chain(sixteen_cell.vertices().iter().copied())
            .chain(
                permutations([phi, 1.0, phi.recip(), 0.0].map(|x| x / 2.0), true)
                    .map(Cartesian::from),
            );
        let volume = 25.0 * (5.0_f64.sqrt() - 1.0) / 8.0;
        Self::with_vertices(vertices)
            .expect("a hypericosahedron requires at least 120 vertices")
            .scale_volume(volume.recip().try_into().expect("volume is positive"))
    }

    /// Build a 120-cell with unit volume.
    ///
    /// The 120-cell, also called the hyperdodecahedron or icosachoron, is the
    /// four-dimensional analogue of the dodecahedron, with 600 vertices and 120
    /// dodecahedral cells.
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::{BoundingSphereRadius, shape::ConvexPolytope};
    ///
    /// let hyperdodecahedron = ConvexPolytope::<4, 600>::hyperdodecahedron();
    ///
    /// assert_eq!(hyperdodecahedron.vertices().len(), 600);
    /// assert_relative_eq!(
    ///     hyperdodecahedron.bounding_sphere_radius().get(),
    ///     8.0_f64.sqrt() / (120.0 * 5.0_f64.sqrt()).powf(0.25)
    /// );
    /// ```
    ///
    /// # Panics
    /// If `MAX_VERTICES < 600`.
    #[inline]
    #[must_use]
    pub fn hyperdodecahedron() -> Self {
        let phi = std::f64::consts::GOLDEN_RATIO;
        let inv_phi = phi.recip();
        let vertices = permutations([2.0, 2.0, 0.0, 0.0], false)
            .chain(permutations([phi, phi, phi, inv_phi.powi(2)], false))
            .chain(permutations([1.0, 1.0, 1.0, 5f64.sqrt()], false))
            .chain(permutations(
                [inv_phi, inv_phi, inv_phi, phi.powi(2)],
                false,
            ))
            .chain(permutations([0.0, inv_phi, phi, 5f64.sqrt()], true))
            .chain(permutations([0.0, inv_phi.powi(2), 1.0, phi.powi(2)], true))
            .chain(permutations([inv_phi, 1.0, phi, 2.0], true))
            .map(Cartesian::from);
        let volume = 120.0 * 5f64.sqrt();
        Self::with_vertices(vertices)
            .expect("a hyperdodecahedron requires at least 600 vertices")
            .scale_volume(volume.recip().try_into().expect("volume is positive"))
    }
}

/// Every way to flip the signs of the nonzero entries of `v`.
#[inline]
fn sign_variants<const N: usize>(v: [f64; N]) -> impl Iterator<Item = [f64; N]> {
    let zeros = v
        .iter()
        .enumerate()
        .fold(0, |m, (i, &x)| m | usize::from(x == 0.0) << i);
    (0..1_usize << N)
        .filter(move |bits| bits & zeros == 0)
        .map(move |bits| std::array::from_fn(|i| if bits >> i & 1 == 1 { -v[i] } else { v[i] }))
}

/// Whether a permutation rearranges an even number of entry pairs.
#[inline]
fn is_even(permutation: &[usize]) -> bool {
    permutation
        .iter()
        .array_combinations()
        .filter(|[a, b]| a > b)
        .count()
        % 2
        == 0
}

/// The distinct coordinate permutations of `±v`, even ones only if `even`.
///
/// An *even* permutation is a product of an even number of pairwise swaps. In three
/// dimensions, even permutations are equivalent to cyclic permutations.
#[inline]
fn permutations<const N: usize>(v: [f64; N], even: bool) -> impl Iterator<Item = [f64; N]> {
    (0..N)
        .permutations(N)
        .filter(move |p| !even || is_even(p))
        // Permutations that only swap equal entries would result in duplicates, so
        // we filter these out.
        .filter(move |p| (0..N).all(|i| (i + 1..N).all(|j| !(v[p[i]] == v[p[j]] && p[i] > p[j]))))
        .flat_map(move |p| sign_variants(std::array::from_fn(|i| v[p[i]])))
}

impl<const N: usize, const MAX_VERTICES: usize> Scale for ConvexPolytope<N, MAX_VERTICES> {
    /// Construct a scaled polytope.
    ///
    /// The resulting polytope's vertices $` v_\mathrm{new} `$ are the
    /// original's $` v `$ scaled uniformly:
    /// ```math
    /// v_\mathrm{new_i} = v_\mathrm{old_i} \cdot v
    /// ```
    ///
    /// The centroid remains at the origin.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::ConvexPolygon};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let square = ConvexPolygon::hypercube();
    ///
    /// let scaled_square = square.scale_length(2.0.try_into()?);
    ///
    /// assert_eq!(scaled_square.vertices()[0], [1.0, 1.0].into());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_length(&self, v: PositiveReal) -> Self {
        Self {
            vertices: self.vertices.iter().map(|vertex| *vertex * v).collect(),
            bounding_radius: self.bounding_radius * v,
        }
    }

    /// Construct a polytope scaled to a new N-volume.
    ///
    /// The resulting polytope's N-volume is the original's scaled by $` v `$.
    #[inline]
    fn scale_volume(&self, v: PositiveReal) -> Self {
        let v = v.get().powf(1.0 / N as f64);
        self.scale_length(v.try_into().expect("v^{1/N} should be a positive real"))
    }
}

/// Compute the matrix-vector multiplication of an `ArrayVec` against a `Cartesian<N>`.
///
/// This returns an `ExactSizeIterator` of f64 values with `lhs.len()` elements.
#[inline(always)]
fn matrix_vector_multiply<const MAX_VERTICES: usize, const N: usize>(
    lhs: &ArrayVec<Cartesian<N>, MAX_VERTICES>,
    rhs: Cartesian<N>, // Copy appears to be compiled out, and this lets us elide '_
) -> impl ExactSizeIterator<Item = f64> + '_ {
    lhs.iter()
        .map(move |vertex| (0..N).map(|m| vertex[m] * rhs[m]).sum())
}

impl<const N: usize, const MAX_VERTICES: usize> SupportMapping<Cartesian<N>>
    for ConvexPolytope<N, MAX_VERTICES>
{
    #[inline]
    fn support_mapping(&self, n: &Cartesian<N>) -> Cartesian<N> {
        match N {
            0 => Cartesian::<N>::default(),
            1 => self.vertices[0],
            _ => {
                let scalars = matrix_vector_multiply(&self.vertices, *n);

                let (mut argmax, mut max_val) = (0, f64::NEG_INFINITY);
                scalars.enumerate().for_each(|(i, x)| {
                    if x > max_val {
                        argmax = i;
                        max_val = x;
                    }
                });
                self.vertices[argmax]
            }
        }
    }
}

impl<const N: usize, const MAX_VERTICES: usize> BoundingSphereRadius
    for ConvexPolytope<N, MAX_VERTICES>
{
    #[inline]
    fn bounding_sphere_radius(&self) -> PositiveReal {
        self.bounding_radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Convex, IntersectsAt};
    use hoomd_vector::{Angle, Cartesian, InnerProduct, Rotate, Rotation, Versor};

    use approxim::assert_relative_eq;
    use rstest::*;
    use std::f64::consts::{FRAC_1_SQRT_2, PI};

    #[fixture]
    fn simplex3() -> ConvexPolyhedron {
        ConvexPolyhedron::with_vertices([
            [1.0, 1.0, 1.0].into(),
            [1.0, -1.0, -1.0].into(),
            [-1.0, 1.0, -1.0].into(),
            [-1.0, -1.0, 1.0].into(),
        ])
        .unwrap()
    }

    #[fixture]
    fn equilateral_triangle() -> ConvexPolygon {
        ConvexPolytope::with_vertices([
            [1.0, 0.0].into(),
            [0.5, f64::sqrt(3.0) / 2.0].into(),
            [-0.5, f64::sqrt(3.0) / 2.0].into(),
        ])
        .unwrap()
    }

    #[rstest]
    fn test_bounding_radius_computed(
        simplex3: ConvexPolyhedron,
        equilateral_triangle: ConvexPolygon,
    ) {
        assert_eq!(simplex3.bounding_radius.get(), f64::sqrt(3.0));
        assert_eq!(equilateral_triangle.bounding_radius.get(), f64::sqrt(1.0));
    }

    #[rstest]
    fn test_bounding_radius_regular_polygons(#[values(1, 3, 8, 32)] n: usize) {
        assert_eq!(ConvexPolygon::regular(n).bounding_radius.get(), 0.5);
        assert_eq!(
            ConvexPolytope::<2, 32>::regular(n).bounding_radius.get(),
            0.5
        );
    }

    #[test]
    fn degenerate_polytope() {
        let result = ConvexPolytope::<3>::with_vertices([]);
        assert_eq!(result, Err(Error::DegeneratePolytope));
    }

    #[test]
    fn support_mapping_2d() {
        let cuboid = ConvexPolygon::with_vertices([
            [-1.0, -2.0].into(),
            [1.0, -2.0].into(),
            [1.0, 2.0].into(),
            [-1.0, 2.0].into(),
        ])
        .expect("hard-coded vertices form a polygon");

        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([1.0, 0.1])),
            [1.0, 2.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([1.0, -0.1])),
            [1.0, -2.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([-0.1, 1.0])),
            [-1.0, 2.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([-0.1, -1.0])),
            [-1.0, -2.0].into()
        );
    }

    #[test]
    fn support_mapping_3d() {
        let cuboid = ConvexPolyhedron::with_vertices([
            [-1.0, -2.0, 3.0].into(),
            [1.0, -2.0, 3.0].into(),
            [1.0, 2.0, 3.0].into(),
            [-1.0, 2.0, 3.0].into(),
            [-1.0, -2.0, -3.0].into(),
            [1.0, -2.0, -3.0].into(),
            [1.0, 2.0, -3.0].into(),
            [-1.0, 2.0, -3.0].into(),
        ])
        .expect("hard-coded vertices form a polygon");

        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([1.0, 0.1, 0.1])),
            [1.0, 2.0, 3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([1.0, 0.1, -0.1])),
            [1.0, 2.0, -3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([1.0, -0.1, 0.1])),
            [1.0, -2.0, 3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([1.0, -0.1, -0.1])),
            [1.0, -2.0, -3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([-1.0, 0.1, 0.1])),
            [-1.0, 2.0, 3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([-1.0, 0.1, -0.1])),
            [-1.0, 2.0, -3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([-1.0, -0.1, 0.1])),
            [-1.0, -2.0, 3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([-1.0, -0.1, -0.1])),
            [-1.0, -2.0, -3.0].into()
        );
    }

    // ConvexPolygon tests from hoomd-blue's test_convex_polygon.cc

    #[fixture]
    fn square() -> Convex<ConvexPolygon> {
        Convex(
            ConvexPolygon::with_vertices([
                [-0.5, -0.5].into(),
                [0.5, -0.5].into(),
                [0.5, 0.5].into(),
                [-0.5, 0.5].into(),
            ])
            .expect("hard-coded vertices form a valid polygon"),
        )
    }

    #[fixture]
    fn triangle() -> Convex<ConvexPolygon> {
        Convex(
            ConvexPolygon::with_vertices([
                [-0.5, -0.5].into(),
                [0.5, -0.5].into(),
                [0.5, 0.5].into(),
            ])
            .expect("hard-coded vertices form a valid polygon"),
        )
    }

    #[rstest]
    fn square_no_rot(square: Convex<ConvexPolygon>) {
        let a = Angle::identity();
        assert!(!square.intersects_at(&square, &[10.0, 0.0].into(), &a));
        assert!(!square.intersects_at(&square, &[-10.0, 0.0].into(), &a));

        assert!(!square.intersects_at(&square, &[1.1, 0.0].into(), &a));
        assert!(!square.intersects_at(&square, &[-1.1, 0.0].into(), &a));
        assert!(!square.intersects_at(&square, &[0.0, 1.1].into(), &a));
        assert!(!square.intersects_at(&square, &[0.0, -1.1].into(), &a));

        assert!(square.intersects_at(&square, &[0.9, 0.2].into(), &a));
        assert!(square.intersects_at(&square, &[-0.9, 0.2].into(), &a));
        assert!(square.intersects_at(&square, &[-0.2, 0.9].into(), &a));
        assert!(square.intersects_at(&square, &[-0.2, -0.9].into(), &a));

        assert!(square.intersects_at(&square, &[1.0, 0.2].into(), &a));
    }

    #[rstest]
    fn square_rot(square: Convex<ConvexPolygon>) {
        let a = Angle::from(PI / 4.0);

        assert!(!square.intersects_at(&square, &[10.0, 0.0].into(), &a));
        assert!(!square.intersects_at(&square, &[-10.0, 0.0].into(), &a));

        assert!(!square.intersects_at(&square, &[1.3, 0.0].into(), &a));
        assert!(!square.intersects_at(&square, &[-1.3, 0.0].into(), &a));
        assert!(!square.intersects_at(&square, &[0.0, 1.3].into(), &a));
        assert!(!square.intersects_at(&square, &[0.0, -1.3].into(), &a));

        assert!(!square.intersects_at(&square, &[1.3, 0.2].into(), &a));
        assert!(!square.intersects_at(&square, &[-1.3, 0.2].into(), &a));
        assert!(!square.intersects_at(&square, &[-0.2, 1.3].into(), &a));
        assert!(!square.intersects_at(&square, &[-0.2, -1.3].into(), &a));

        assert!(square.intersects_at(&square, &[1.2, 0.2].into(), &a));
        assert!(square.intersects_at(&square, &[-1.2, 0.2].into(), &a));
        assert!(square.intersects_at(&square, &[-0.2, 1.2].into(), &a));
        assert!(square.intersects_at(&square, &[-0.2, -1.2].into(), &a));
    }

    fn test_overlap<A, B, R, const N: usize>(
        r_ab: Cartesian<N>,
        a: &A,
        b: &B,
        o_a: R,
        o_b: &R,
    ) -> bool
    where
        R: Rotation + Rotate<Cartesian<N>>,
        A: IntersectsAt<B, Cartesian<N>, R>,
    {
        let r_a_inverted = o_a.inverted();
        let v_ij = r_a_inverted.rotate(&r_ab);
        let o_ij = o_b.combine(&r_a_inverted);
        a.intersects_at(b, &v_ij, &o_ij)
    }

    fn assert_symmetric_overlap<A, B, R, const N: usize>(
        r_ab: Cartesian<N>,
        a: &A,
        b: &B,
        r_a: R,
        r_b: R,
        expected: bool,
    ) where
        R: Rotation + Rotate<Cartesian<N>>,
        A: IntersectsAt<B, Cartesian<N>, R>,
        B: IntersectsAt<A, Cartesian<N>, R>,
    {
        assert_eq!(test_overlap(r_ab, a, b, r_a, &r_b), expected);
        assert_eq!(test_overlap(-r_ab, b, a, r_b, &r_a), expected);
    }

    #[rstest]
    fn square_triangle(square: Convex<ConvexPolygon>, triangle: Convex<ConvexPolygon>) {
        let r_square = Angle::from(-PI / 4.0);
        let r_triangle = Angle::from(PI);

        assert_symmetric_overlap(
            [10.0, 0.0].into(),
            &square,
            &triangle,
            r_square,
            r_triangle,
            false,
        );

        assert_symmetric_overlap(
            [1.3, 0.0].into(),
            &square,
            &triangle,
            r_square,
            r_triangle,
            false,
        );

        assert_symmetric_overlap(
            [-1.3, 0.0].into(),
            &square,
            &triangle,
            r_square,
            r_triangle,
            false,
        );

        assert_symmetric_overlap(
            [0.0, 1.3].into(),
            &square,
            &triangle,
            r_square,
            r_triangle,
            false,
        );

        assert_symmetric_overlap(
            [0.0, -1.3].into(),
            &square,
            &triangle,
            r_square,
            r_triangle,
            false,
        );

        assert_symmetric_overlap(
            [1.2, 0.2].into(),
            &square,
            &triangle,
            r_square,
            r_triangle,
            true,
        );

        assert_symmetric_overlap(
            [-0.7, -0.2].into(),
            &square,
            &triangle,
            r_square,
            r_triangle,
            true,
        );

        assert_symmetric_overlap(
            [0.4, 1.1].into(),
            &square,
            &triangle,
            r_square,
            r_triangle,
            true,
        );

        assert_symmetric_overlap(
            [-0.2, -1.2].into(),
            &square,
            &triangle,
            r_square,
            r_triangle,
            true,
        );
    }

    #[fixture]
    fn octahedron() -> Convex<ConvexPolyhedron> {
        Convex(
            ConvexPolyhedron::with_vertices([
                [-0.5, -0.5, 0.0].into(),
                [0.5, -0.5, 0.0].into(),
                [0.5, 0.5, 0.0].into(),
                [-0.5, 0.5, 0.0].into(),
                [0.0, 0.0, FRAC_1_SQRT_2].into(),
                [0.0, 0.0, -FRAC_1_SQRT_2].into(),
            ])
            .expect("hard-coded vertices form a valid polyhedron"),
        )
    }

    #[fixture]
    fn cube() -> Convex<ConvexPolyhedron> {
        Convex(
            ConvexPolyhedron::with_vertices([
                [-0.5, -0.5, -0.5].into(),
                [0.5, -0.5, -0.5].into(),
                [0.5, 0.5, -0.5].into(),
                [-0.5, 0.5, -0.5].into(),
                [-0.5, -0.5, 0.5].into(),
                [0.5, -0.5, 0.5].into(),
                [0.5, 0.5, 0.5].into(),
                [-0.5, 0.5, 0.5].into(),
            ])
            .expect("hard-coded vertices form a valid polyhedron"),
        )
    }

    #[rstest]
    fn overlap_octahedron_no_rot(octahedron: Convex<ConvexPolyhedron>) {
        let q = Versor::identity();

        assert_symmetric_overlap([0.0, 0.0, 0.0].into(), &octahedron, &octahedron, q, q, true);

        assert_symmetric_overlap(
            [10.0, 0.0, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            false,
        );

        assert_symmetric_overlap(
            [1.1, 0.0, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            false,
        );

        assert_symmetric_overlap(
            [0.0, 1.1, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            false,
        );

        assert_symmetric_overlap(
            [1.1, 0.2, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            false,
        );

        assert_symmetric_overlap(
            [-1.1, 0.2, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            false,
        );

        assert_symmetric_overlap(
            [-0.2, 1.1, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            false,
        );

        assert_symmetric_overlap(
            [-0.2, -1.1, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            false,
        );

        assert_symmetric_overlap([0.9, 0.2, 0.0].into(), &octahedron, &octahedron, q, q, true);

        assert_symmetric_overlap(
            [-0.9, 0.2, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            true,
        );

        assert_symmetric_overlap(
            [-0.2, 0.9, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            true,
        );

        assert_symmetric_overlap(
            [-0.2, -0.9, 0.0].into(),
            &octahedron,
            &octahedron,
            q,
            q,
            true,
        );

        assert_symmetric_overlap([1.0, 0.2, 0.0].into(), &octahedron, &octahedron, q, q, true);
    }

    #[rstest]
    fn overlap_cube_no_rot(cube: Convex<ConvexPolyhedron>) {
        let q = Versor::identity();

        assert_symmetric_overlap([0.0, 0.0, 0.0].into(), &cube, &cube, q, q, true);
        assert_symmetric_overlap([10.0, 0.0, 0.0].into(), &cube, &cube, q, q, false);

        assert_symmetric_overlap([1.1, 0.0, 0.0].into(), &cube, &cube, q, q, false);
        assert_symmetric_overlap([0.0, 1.1, 0.0].into(), &cube, &cube, q, q, false);
        assert_symmetric_overlap([0.0, 0.0, 1.1].into(), &cube, &cube, q, q, false);
        assert_symmetric_overlap([1.1, 0.2, 0.0].into(), &cube, &cube, q, q, false);

        assert_symmetric_overlap([-1.1, 0.2, 0.0].into(), &cube, &cube, q, q, false);
        assert_symmetric_overlap([-0.2, 1.1, 0.0].into(), &cube, &cube, q, q, false);
        assert_symmetric_overlap([-0.2, -1.1, 0.0].into(), &cube, &cube, q, q, false);

        assert_symmetric_overlap([0.9, 0.2, 0.0].into(), &cube, &cube, q, q, true);
        assert_symmetric_overlap([-0.9, 0.2, 0.0].into(), &cube, &cube, q, q, true);
        assert_symmetric_overlap([-0.2, 0.9, 0.0].into(), &cube, &cube, q, q, true);
        assert_symmetric_overlap([-0.2, -0.9, 0.0].into(), &cube, &cube, q, q, true);

        assert_symmetric_overlap([0.2, 0.0, 0.0].into(), &cube, &cube, q, q, true);
        assert_symmetric_overlap([0.2, 0.00001, 0.00001].into(), &cube, &cube, q, q, true);
        assert_symmetric_overlap([0.1, 0.2, 0.1].into(), &cube, &cube, q, q, true);
        assert_symmetric_overlap([1.0, 0.2, 0.0].into(), &cube, &cube, q, q, true);
    }

    #[rstest]
    fn overlap_cube_rot1(cube: Convex<ConvexPolyhedron>) {
        let q_a = Versor::identity();
        let q_b = Versor::from_axis_angle(
            [0.0, 0.0, 1.0]
                .try_into()
                .expect("hard-coded vector is non-zero"),
            PI / 4.0,
        );

        assert_symmetric_overlap([10.0, 0.0, 0.0].into(), &cube, &cube, q_a, q_b, false);

        assert_symmetric_overlap([1.3, 0.0, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([-1.3, 0.0, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([0.0, 1.3, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([0.0, -1.3, 0.0].into(), &cube, &cube, q_a, q_b, false);

        assert_symmetric_overlap([1.3, 0.2, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([-1.3, 0.2, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([-0.2, 1.3, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([-0.2, -1.3, 0.0].into(), &cube, &cube, q_a, q_b, false);

        assert_symmetric_overlap([1.2, 0.2, 0.0].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([-1.2, 0.2, 0.0].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([-0.2, 1.2, 0.0].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([-0.2, -1.2, 0.0].into(), &cube, &cube, q_a, q_b, true);
    }

    #[rstest]
    fn overlap_cube_rot3(cube: Convex<ConvexPolyhedron>) {
        let q_a = Versor::identity();
        let q1 = Versor::from_axis_angle(
            [1.0, 0.0, 0.0]
                .try_into()
                .expect("hard-coded vector is non-zero"),
            PI / 4.0,
        );
        let q2 = Versor::from_axis_angle(
            [0.0, 0.0, 1.0]
                .try_into()
                .expect("hard-coded vector is non-zero"),
            PI / 4.0,
        );
        let q_b = q2.combine(&q1);

        assert_symmetric_overlap([10.0, 0.0, 0.0].into(), &cube, &cube, q_a, q_b, false);

        assert_symmetric_overlap([1.4, 0.0, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([-1.4, 0.0, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([0.0, 1.4, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([0.0, -1.4, 0.0].into(), &cube, &cube, q_a, q_b, false);

        assert_symmetric_overlap([1.4, 0.2, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([-1.4, 0.2, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([-0.2, 1.4, 0.0].into(), &cube, &cube, q_a, q_b, false);
        assert_symmetric_overlap([-0.2, -1.4, 0.0].into(), &cube, &cube, q_a, q_b, false);

        assert_symmetric_overlap([0.0, 1.2, 0.0].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([0.0, 1.2, 0.1].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([0.1, 1.2, 0.1].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([1.2, 0.0, 0.0].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([1.2, 0.1, 0.0].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([1.2, 0.1, 0.1].into(), &cube, &cube, q_a, q_b, true);

        assert_symmetric_overlap([-0.9, 0.9, 0.0].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([-0.9, 0.899, 0.001].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([0.9, -0.9, 0.0].into(), &cube, &cube, q_a, q_b, true);
        assert_symmetric_overlap([-0.9, 0.9, 0.1].into(), &cube, &cube, q_a, q_b, true);
    }
    /// Each platonic solid has the expected number of vertices, all at a common
    /// distance from the center and separated by a known edge length.
    #[rstest]
    #[case::icosahedron(ConvexPolyhedron::icosahedron(), 12, 2.0)]
    #[case::dodecahedron(
        ConvexPolyhedron::dodecahedron(),
        20,
        2.0 / std::f64::consts::GOLDEN_RATIO
    )]
    #[case::octahedron(ConvexPolyhedron::orthoplex(), 6, (3.0 / 2f64.sqrt()).cbrt())]
    #[case::cube(ConvexPolyhedron::hypercube(), 8, 1.0)]
    #[case::tetrahedron(ConvexPolyhedron::simplex(), 4, 2f64.sqrt() * 3f64.cbrt())]
    fn platonic_solids(#[case] solid: ConvexPolyhedron, #[case] n: usize, #[case] edge: f64) {
        assert_eq!(solid.vertices().len(), n);

        for vertex in solid.vertices() {
            assert_relative_eq!(
                vertex.norm_squared(),
                solid.vertices()[0].norm_squared(),
                max_relative = 5e-16
            );
        }

        // The shortest vertex-to-vertex distance is the edge length.
        let mut min_distance = f64::INFINITY;
        for (i, vertex) in solid.vertices().iter().enumerate() {
            for other in &solid.vertices()[i + 1..] {
                min_distance = min_distance.min((*vertex - *other).norm());
            }
        }
        assert_relative_eq!(min_distance, edge, max_relative = 1e-13);
    }

    #[rstest]
    #[case::orthoplex_2(ConvexPolytope::<2>::orthoplex(), 4, 1.0)]
    #[case::orthoplex_3(ConvexPolytope::<3>::orthoplex(), 6, 6f64.cbrt() / SQRT_2)]
    #[case::orthoplex_4(ConvexPolytope::<4>::orthoplex(), 8, 24f64.sqrt().sqrt() / SQRT_2)]
    #[case::orthoplex_5(ConvexPolytope::<5>::orthoplex(), 10, 120f64.powf(0.2) / SQRT_2)]
    #[case::orthoplex_6(
        ConvexPolytope::<6>::orthoplex(),
        12,
        720f64.powf(1.0 / 6.0) / SQRT_2
    )]
    #[case::hypercube_2(ConvexPolytope::<2>::hypercube(), 4, 1.0)]
    #[case::hypercube_3(ConvexPolytope::<3>::hypercube(), 8, 1.0)]
    #[case::hypercube_4(ConvexPolytope::<4>::hypercube(), 16, 1.0)]
    #[case::hypercube_5(ConvexPolytope::<5>::hypercube(), 32, 1.0)]
    #[case::hypercube_6(ConvexPolytope::<6>::hypercube(), 64, 1.0)]
    #[case::simplex_2(
        ConvexPolytope::<2>::simplex(),
        3,
        (4.0 / 3f64.sqrt()).sqrt()
    )]
    #[case::simplex_3(ConvexPolytope::<3>::simplex(), 4, 2f64.sqrt() * 3f64.cbrt())]
    #[case::simplex_4(
        ConvexPolytope::<4>::simplex(),
        5,
        (96.0 / 5f64.sqrt()).sqrt().sqrt()
    )]
    #[case::simplex_5(
        ConvexPolytope::<5>::simplex(),
        6,
        (480.0 / 3f64.sqrt()).powf(1.0 / 5.0)
    )]
    #[case::simplex_6(
        ConvexPolytope::<6>::simplex(),
        7,
        (5760.0 / 7f64.sqrt()).powf(1.0 / 6.0)
    )]
    fn hyperplatonic_solids<const N: usize>(
        #[case] solid: ConvexPolytope<N>,
        #[case] n: usize,
        #[case] edge: f64,
    ) {
        assert_eq!(solid.vertices().len(), n);

        for vertex in solid.vertices() {
            assert_relative_eq!(
                vertex.norm_squared(),
                solid.vertices()[0].norm_squared(),
                max_relative = 5e-16
            );
        }

        // The stored bounding radius matches the vertices.
        assert_relative_eq!(
            solid.bounding_sphere_radius().get(),
            solid.vertices()[0].norm(),
            max_relative = 1e-13
        );

        // The shortest vertex-to-vertex distance is the edge length.
        let mut min_distance = f64::INFINITY;
        for (i, vertex) in solid.vertices().iter().enumerate() {
            for other in &solid.vertices()[i + 1..] {
                min_distance = min_distance.min((*vertex - *other).norm());
            }
        }
        assert_relative_eq!(min_distance, edge, max_relative = 1e-13);
    }

    /// The two H4 regular 4-polytopes have the vertex counts, circumradii,
    /// edge lengths, and vertex degrees documented on the Wikipedia
    /// 600-cell and 120-cell pages (validated in `polytope-h4-solids.wl` in
    /// the workspace root).
    #[rstest]
    #[case::hypericosahedron(
        ConvexPolytope::<4, 600>::hypericosahedron(),
        120,
        (8.0 / (25.0 * (5.0_f64.sqrt() - 1.0))).powf(0.25),
        std::f64::consts::GOLDEN_RATIO.recip() * (8.0 / (25.0 * (5.0_f64.sqrt() - 1.0))).powf(0.25),
        12,
        720
    )]
    #[case::hyperdodecahedron(
        ConvexPolytope::<4, 600>::hyperdodecahedron(),
        600,
        8.0_f64.sqrt() / (120.0 * 5.0_f64.sqrt()).powf(0.25),
        (3.0 - 5.0_f64.sqrt()) / (120.0 * 5.0_f64.sqrt()).powf(0.25),
        4,
        1200
    )]
    fn h4_solids(
        #[case] solid: ConvexPolytope<4, 600>,
        #[case] n: usize,
        #[case] circumradius: f64,
        #[case] edge: f64,
        #[case] degree: usize,
        #[case] n_edges: usize,
    ) {
        assert_eq!(solid.vertices().len(), n);

        for vertex in solid.vertices() {
            assert_relative_eq!(
                vertex.norm_squared(),
                solid.vertices()[0].norm_squared(),
                max_relative = 5e-16
            );
        }
        assert_relative_eq!(
            solid.bounding_sphere_radius().get(),
            circumradius,
            max_relative = 1e-13
        );

        // The shortest vertex-to-vertex distance is the edge length, and
        // exactly `n_edges` pairs sit at it, `degree` per vertex.
        let mut min_distance = f64::INFINITY;
        for (i, vertex) in solid.vertices().iter().enumerate() {
            for other in &solid.vertices()[i + 1..] {
                min_distance = min_distance.min((*vertex - *other).norm());
            }
        }
        assert_relative_eq!(min_distance, edge, max_relative = 1e-13);

        let mut incident = vec![0_usize; n];
        let mut edges = 0;
        for (i, vertex) in solid.vertices().iter().enumerate() {
            for (j, other) in solid.vertices().iter().enumerate().skip(i + 1) {
                if ((*vertex - *other).norm() - min_distance).abs() < 1e-9 {
                    incident[i] += 1;
                    incident[j] += 1;
                    edges += 1;
                }
            }
        }
        assert_eq!(edges, n_edges);
        assert!(incident.iter().all(|&k| k == degree));
    }

    /// Scaling length by `s` and volume by `s^N` produce the same shape.
    #[test]
    fn scale_length_and_volume() {
        let dodecahedron = ConvexPolyhedron::dodecahedron();
        let by_length = dodecahedron.scale_length(positive_real!(2.0));
        let by_volume = dodecahedron.scale_volume(positive_real!(8.0));

        for (scaled, expected) in by_volume.vertices().iter().zip(by_length.vertices()) {
            assert_relative_eq!(scaled, expected);
        }
        assert_relative_eq!(
            by_length.vertices()[0].norm(),
            2.0 * dodecahedron.vertices()[0].norm()
        );
    }
}

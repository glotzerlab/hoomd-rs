// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Common geometric primitives that implement only a small number of operations.*/

use std::array;

use crate::{IntersectsAt, SupportFn};
use hoomd_vector::{Cartesian, Cross, Rotate, Rotation, RotationMatrix, Unit, Vector};

/// A [`Cylinder`] in three dimensions.
#[derive(Clone, Copy, Debug)]
pub struct Cylinder {
    /// Radius of the [`Cylinder`]
    r: f64,
    /// Height of the [`Cylinder`]
    h: f64,
}
/// A [`Capsule`] in three dimensions.
#[derive(Clone, Copy, Debug)]
// pub struct Capsule<const N: usize> {
pub struct Capsule {
    /// Radius of the [`Capsule`]'s spherical caps.
    r: f64,
    /// Distance between the centers of the spherical caps.
    h: f64,
}

impl From<(f64, f64)> for Capsule {
    #[inline]
    fn from(value: (f64, f64)) -> Self {
        Capsule {
            r: value.0,
            h: value.1,
        }
    }
}

#[allow(clippy::expect_used)]
impl SupportFn<Cartesian<3>> for Capsule {
    #[inline]
    fn support(&self, n: &Cartesian<3>) -> Cartesian<3> {
        /*Same support function as a ConvexPolyhedron with 2 vertices, plus the radius*/
        let (v_tip, v_base) = ([0.0, 0.0, self.h].into(), [0.0, 0.0, -self.h].into());

        let (v_tip_dot_n, v_base_dot_n) = (n.dot(&v_tip), n.dot(&v_base));

        let rshift = *n * self.r * n.norm();
        if v_tip_dot_n > v_base_dot_n {
            v_tip / n.norm() + rshift
        } else {
            v_base / n.norm() + rshift
        }
    }
}

/// Closest point on a line segment bounded by `a` and `b` to a sphere at point `p`
fn closest_point_on_line_segment(
    a: Cartesian<3>,
    b: Cartesian<3>,
    p: Cartesian<3>,
) -> Cartesian<3> {
    let ab = b - a;
    let t = (p - a).dot(&ab) / ab.norm_squared();
    ab * (a + 1f64.min(0f64.max(t)))
}

// impl<S, V: Vector, R: Rotate<Cartesian<3>>> IntersectsAt<S, V, R> for Capsule {
impl<R: Rotate<Cartesian<3>> + Rotation> IntersectsAt<Capsule, Cartesian<3>, R> for Capsule {
    #[inline]
    fn intersects_at(&self, other: &Capsule, r_ij: &Cartesian<3>, o_ij: &R) -> bool {
        /*

        Capsule-Capsule intersection can be though of as a two step process:
        1. Find the closest pair of spheres from A and B
        2. Check if those spheres overlap

        This implementation is based on code from the following link:
        https://wickedengine.net/2020/04/capsule-collision-detection/
        */
        // First capsule is axis-aligned by convention
        let a_b = Cartesian::from([0.0, 0.0, -self.r]) + self.h / 2.0;
        let a_a = -a_b;
        // let a_a = Cartesian::from([0.0, 0.0, -self.r]) + self.h / 2.0;

        let b_line_end_offset = o_ij.rotate(&(*r_ij + Cartesian::from([0.0, 0.0, other.r])));
        let b_b = -b_line_end_offset + other.h / 2.0;
        let b_a = -b_b;

        // Squared distances between line endpoints
        let d0 = (b_a - a_a).norm_squared();
        let d1 = (b_b - a_a).norm_squared();
        let d2 = (b_a - a_b).norm_squared();
        let d3 = (b_b - a_b).norm_squared();

        // select best potential endpoint on capsule A:
        let best_a = if d2 < d0 || d2 < d1 || d3 < d0 || d3 < d1 {
            a_b
        } else {
            a_a
        };

        // Select the point on capsule B's primary axis that is nearest to the best potential endpoing on capsule A
        let best_b = closest_point_on_line_segment(b_a, b_b, best_a);

        // Repeat for the primary axis of capsule A
        let best_a = closest_point_on_line_segment(a_a, a_b, best_b);

        // Now we have the closest points, just do a sphere intersection at those pts.
        let penetration_depth = self.r + other.r - (best_a - best_b).norm();

        penetration_depth > 0.0

        // TODO: test against XenoCollide
    }
}

/// An N-Dimensional [`HyperEllipsoid`] defined by its semi-major axes.
#[derive(Clone, Copy, Debug)]
pub struct HyperEllipsoid<const N: usize> {
    /// The principle semi-axes of the [`HyperEllipsoid`] along each direction.
    axes: Cartesian<N>,
}
impl<const N: usize> IntoIterator for HyperEllipsoid<N> {
    type Item = f64;
    type IntoIter = <[f64; N] as IntoIterator>::IntoIter;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.axes.into_iter()
    }
}

impl<const N: usize> SupportFn<Cartesian<N>> for HyperEllipsoid<N> {
    #[inline]
    fn support(&self, n: &Cartesian<N>) -> Cartesian<N> {
        let mut denominator = self.into_iter().zip(*n).map(|(r, n)| r * n);
        let denominator: f64 = Cartesian::<N>::from(std::array::from_fn(|_| {
            denominator.next().unwrap_or_default()
        }))
        .norm();
        let mut iter = n
            .into_iter()
            .zip(self.axes)
            .map(|(r, n)| r.powi(2) * n / denominator);
        std::array::from_fn(|_| iter.next().unwrap_or_default()).into()
    }
}

impl HyperEllipsoid<3> {
    #[inline]
    #[must_use]
    /// Compute a matrix representation of the ellipsoid.
    #[allow(clippy::many_single_char_names, dead_code)]
    fn compute_ellipsoid_matrix<R>(&self, r_ij: &Cartesian<3>, o_ij: &R) -> Cartesian<10>
    where
        RotationMatrix<3>: From<R>,
        R: Copy,
    {
        // See the HOOMD-Blue ShapeEllipsoid.h for the original source code.
        let r = RotationMatrix::from(*o_ij);
        let a = 1.0 / self.axes[0].powi(2);
        let b = 1.0 / self.axes[1].powi(2);
        let c = 1.0 / self.axes[2].powi(2);

        let mut m = Cartesian::default();

        // ...rotation part
        // M[i][j] = a * R[i][0] * R[j][0] + b * R[i][1] * R[j][1] + c * R[i][2] * R[j][2];
        m[0] = a * r.rows()[0][0] * r.rows()[0][0]
            + b * r.rows()[0][1] * r.rows()[0][1]
            + c * r.rows()[0][2] * r.rows()[0][2];
        m[1] = a * r.rows()[1][0] * r.rows()[0][0]
            + b * r.rows()[1][1] * r.rows()[0][1]
            + c * r.rows()[1][2] * r.rows()[0][2];
        m[2] = a * r.rows()[1][0] * r.rows()[1][0]
            + b * r.rows()[1][1] * r.rows()[1][1]
            + c * r.rows()[1][2] * r.rows()[1][2];
        m[3] = a * r.rows()[2][0] * r.rows()[0][0]
            + b * r.rows()[2][1] * r.rows()[0][1]
            + c * r.rows()[2][2] * r.rows()[0][2];
        m[4] = a * r.rows()[2][0] * r.rows()[1][0]
            + b * r.rows()[2][1] * r.rows()[1][1]
            + c * r.rows()[2][2] * r.rows()[1][2];
        m[5] = a * r.rows()[2][0] * r.rows()[2][0]
            + b * r.rows()[2][1] * r.rows()[2][1]
            + c * r.rows()[2][2] * r.rows()[2][2];

        // calculateTranslationPart(x, m);
        // precalculation
        let m0x0 = m[0] * r_ij[0];
        let m1x0 = m[1] * r_ij[0];
        let m1x1 = m[1] * r_ij[1];
        let m2x1 = m[2] * r_ij[1];
        let m3x0 = m[3] * r_ij[0];
        let m3x2 = m[3] * r_ij[2];
        let m4x1 = m[4] * r_ij[1];
        let m4x2 = m[4] * r_ij[2];
        let m5x2 = m[5] * r_ij[2];

        // ...translation part
        // m[i][3] = m[3][i] = -m[i][0] * x[0] - m[i][1] * x[1] - m[i][2] * x[2];
        m[6] = -m0x0 - m1x1 - m3x2;
        m[7] = -m1x0 - m2x1 - m4x2;
        m[8] = -m3x0 - m4x1 - m5x2;
        // ...mixed part
        // m[3][3] = -1.0 + m[0][0] * x[0] * x[0] + m[1][1] * x[1] * x[1] + m[2][2] * x[2] * x[2] +
        //           2.0 * (m[0][1] * x[0] * x[1] + m[1][2] * x[1] * x[2] + m[2][0] * x[2] * x[0]);
        m[9] = -1.0
            + r_ij[0] * (m0x0 + 2.0 * m1x1)
            + r_ij[1] * (m2x1 + 2.0 * m4x2)
            + r_ij[2] * (m5x2 + 2.0 * m3x0);

        m
    }
}

// impl IntersectsAt for HyperEllipsoid<3> {
//     fn intersects_at(&self, other: &S, r_ij: &V, o_ij: &R) -> bool {}
// }

impl<const N: usize> HyperEllipsoid<N> {} // TODO matrix form and IntersectsAt

/// The simplest three-dimensional geometry.
#[derive(Clone, Copy, Debug)]
pub struct Simplex3 {
    /// Vertices of the simplex
    vertices: [Cartesian<3>; 4], // NOT public, to force orientation on construction
}

impl SupportFn<Cartesian<3>> for Simplex3 {
    #[allow(clippy::expect_used)]
    #[inline]
    fn support(&self, n: &Cartesian<3>) -> Cartesian<3> {
        *self
            .vertices
            .iter()
            .max_by(|a, b| {
                a.dot(n)
                    .partial_cmp(&b.dot(n))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("Support function not valid with 0 vertices!")
    }
}

impl From<[Cartesian<3>; 4]> for Simplex3 {
    fn from(vertices: [Cartesian<3>; 4]) -> Self {
        let s = Simplex3 { vertices };
        s.orient()
    }
}
const EDGES: [(usize, usize); 5] = [(1, 0), (2, 0), (3, 0), (2, 1), (3, 2)];

impl Simplex3 {
    /// Get the edges of the tetrahedron as edge endpoint coordinates. In vertex index
    /// form, this returns values in the order [(1, 0), (2, 0), (3, 0), (2, 1), (3, 2)]
    #[inline]
    pub fn get_edges(&self) -> [[Cartesian<3>; 2]; 5] {
        [
            [self.b(), self.a()],
            [self.c(), self.a()],
            [self.d(), self.a()],
            [self.c(), self.b()],
            [self.d(), self.b()],
        ]
    }
    #[inline]
    pub(crate) fn a(&self) -> Cartesian<3> {
        self.vertices[0]
    }
    #[inline]
    pub(crate) fn b(&self) -> Cartesian<3> {
        self.vertices[1]
    }
    #[inline]
    pub(crate) fn c(&self) -> Cartesian<3> {
        self.vertices[2]
    }
    #[inline]
    pub(crate) fn d(&self) -> Cartesian<3> {
        self.vertices[3]
    }
    /// Orient the vertices of a simplex such that the fourth vertex is on the opposite
    /// side of the plane defined by the first three points.
    #[inline]
    fn orient_in_place(&mut self) {
        *self = self.orient()
    }
    /// Return the vertices of an oriented tetrahedron. Users should call ``orient_in_place``
    #[inline]
    pub(crate) fn orient(&self) -> Simplex3 {
        let dp = (self.d() - self.a()).dot(&((self.b() - self.a()).cross(&(self.c() - self.a()))));
        if dp < 0.0 {
            self.vertices.into()
        } else {
            [self.a(), self.c(), self.b(), self.d()].into()
        }
    }

    /// Compute a bitmask for a sequence of four affine coordinates
    #[inline]
    pub(crate) fn compute_mask(&self, aff: [f64; 4]) -> [bool; 4] {
        /*
        if aff[0] > 0.0 {1} else {0}
        if aff[1] > 0.0 {2} else {0}
        if aff[2] > 0.0 {4} else {0}
        if aff[3] > 0.0 {8} else {0}
        bit or'd together. Not hard, but unnecessary optimization
        */
        let mut iter = aff.iter().map(|&x| x > 0.0);
        array::from_fn::<_, 4, _>(|_| iter.next().unwrap_or_default())
    }
    /**Check the faces of tetrahedron 0, returning a vector of bitmasks*/
    fn check_faces_a(&self, deltas: [Cartesian<3>; 4], q: &Simplex3) {
        let edges = self.get_edges();

        let p = self.b(); // Reference point on simplex 0
        // let n = ea.cross(&eb);
    }
}

impl<R: Rotate<Cartesian<3>> + Rotation> IntersectsAt<Simplex3, Cartesian<3>, R> for Simplex3 {
    #[inline]
    fn intersects_at(&self, other: &Simplex3, r_ij: &Cartesian<3>, o_ij: &R) -> bool {
        /* Based on the implementation from the following publication:
            http://vcg.isti.cnr.it/Publications/2003/GPR03/fast_tetrahedron_tetrahedron_overlap_algorithm.pdf
        */
        // p, q = self, other. Oriented by default, as all constructors MUST call orient

        // let masks = self.check_faces_a(deltas, /*self.get_edges()*/, other.vertices,);

        false // TODO: use this as test case. Not round, so should be
    }
}

// TODO: tolerance check

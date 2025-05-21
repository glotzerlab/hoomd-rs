// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Common geometric primitives that implement only a small number of operations.*/
use crate::{IntersectsAt, SupportFn, xenocollide::collide3d};

use hoomd_vector::{Cartesian, Rotate, Rotation, RotationMatrix, Vector};

/// A [`Cylinder`] in three dimensions.
#[expect(dead_code, reason = "End users will make use of this struct.")]
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
    #[expect(
        clippy::many_single_char_names,
        dead_code,
        reason = "Ported from HOOMD-Blue, with variable names maintained for consistency."
    )]
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

impl<S: SupportFn<Cartesian<3>>, R: Copy + Rotation + Rotate<Cartesian<3>>>
    IntersectsAt<S, Cartesian<3>, R> for HyperEllipsoid<3>
where
    RotationMatrix<3>: From<R>,
{
    #[inline]
    fn intersects_at(&self, other: &S, v_ij: &Cartesian<3>, o_ij: &R) -> bool {
        collide3d(self, other, v_ij, o_ij)
    } // TODO: implement fast ellipsoid overlap
}

impl<const N: usize> HyperEllipsoid<N> {} // TODO matrix form and IntersectsAt

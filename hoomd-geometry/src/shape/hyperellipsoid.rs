// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Hyperellipsoid`] */

use crate::{IntersectsAt, SupportFn, xenocollide::collide3d};

use hoomd_vector::{Cartesian, Rotate, Rotation, RotationMatrix, Vector};

/// An N-Dimensional [`Hyperellipsoid`] defined by its semi-major axes.
#[derive(Clone, Copy, Debug)]
pub struct Hyperellipsoid<const N: usize> {
    /// The principle semi-axes of the [`Hyperellipsoid`] along each direction.
    axes: Cartesian<N>,
}

/**A two-dimensional ellipse.*/
type Ellipse = Hyperellipsoid<2>;
/**A three-dimensional ellipsoid.*/
type Ellipsoid = Hyperellipsoid<3>;

impl<const N: usize> IntoIterator for Hyperellipsoid<N> {
    type Item = f64;
    type IntoIter = <[f64; N] as IntoIterator>::IntoIter;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.axes.into_iter()
    }
}

impl<const N: usize> SupportFn<Cartesian<N>> for Hyperellipsoid<N> {
    #[inline]
    fn support(&self, n: &Cartesian<N>) -> Cartesian<N> {
        let denominator = Cartesian::<N>::from(std::array::from_fn(|i| self.axes[i] * n[i])).norm();
        std::array::from_fn(|i| n[i] * self.axes[i].powi(2) / denominator).into()
    }
}

impl Hyperellipsoid<3> {
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
    IntersectsAt<S, Cartesian<3>, R> for Hyperellipsoid<3>
where
    RotationMatrix<3>: From<R>,
{
    #[inline]
    fn intersects_at(&self, other: &S, v_ij: &Cartesian<3>, o_ij: &R) -> bool {
        collide3d(self, other, v_ij, o_ij)
    } // TODO: implement fast ellipsoid overlap
}

impl<const N: usize> Hyperellipsoid<N> {} // TODO matrix form and IntersectsAt

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Sphere;
    use ::approx::assert_relative_eq;
    use rstest::*;
    use std::marker::PhantomData;

    #[rstest]
    #[case(PhantomData::<Sphere<0>>)]
    #[case(PhantomData::<Sphere<1>>)]
    #[case(PhantomData::<Sphere<2>>)]
    #[case(PhantomData::<Sphere<3>>)]
    #[case(PhantomData::<Sphere<4>>)]
    #[case(PhantomData::<Sphere<5>>)]
    fn test_support_hyperellipsoid<const N: usize>(
        #[case] _n: PhantomData<Sphere<N>>,
        #[values(0.1, 1.0, 33.3)] r: f64,
    ) {
        let s = Sphere::<N>::from(r);
        let he = Hyperellipsoid {
            axes: [r; N].into(),
        };
        let v = [1.0; N].into();
        assert_relative_eq!(he.support(&v), s.support(&v));
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Hyperellipsoid`].


```rust
use hoomd_geometry::shape::{Hyperellipsoid, Ellipse, Ellipsoid, Sphere};
use hoomd_geometry::Volume;

let ellipse = Hyperellipsoid {axes: [1.0, 2.0]};

assert_eq!(ellipse.volume(), Ellipse {axes: [2.0, 1.0]}.volume());
assert_eq!(Ellipsoid{ axes: [1.0, 1.0, 1.0] }.volume(), Sphere {radius: 1.0 }.volume());

```
*/

use super::sphere::sphere_volume_prefactor;
use crate::{BoundingSphereRadius, IntersectsAt, SupportMapping, Volume};
use hoomd_vector::{Cartesian, Rotate, Rotation, RotationMatrix, Vector};
use std::ops::{Add, Mul};
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SquareMatrix<const N: usize> {
    /// The elements of the matrix
    pub(crate) rows: [[f64; N]; N],
    // diagonal: bool,
    // symmetry: ???
}

impl<const N: usize> From<RotationMatrix<N>> for SquareMatrix<N> {
    fn from(value: RotationMatrix<N>) -> Self {
        Self {
            rows: value.rows().map(|arr| arr.coordinates),
        }
    }
}

impl<const N: usize> Default for SquareMatrix<N> {
    /** Create an identity matrix.

    ```math
    \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
    ```
    ,
    ```math
    \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}
    ```
    , and so on.

    # Example

    ```
    use hoomd_vector::SquareMatrix;

    let identity = SquareMatrix::<3>::default();
    ```
    */
    #[inline]
    fn default() -> SquareMatrix<N> {
        SquareMatrix {
            rows: std::array::from_fn(|i| {
                std::array::from_fn(|j| if i == j { 1.0 } else { 0.0 }).into()
            }),
        }
    }
}

impl<const N: usize> SquareMatrix<N> {
    /// Extract the diagonal elements of the matrix
    #[inline]
    fn diag(&self) -> [f64; N] {
        std::array::from_fn(|i| self.rows[i][i])
    }
    #[inline]
    fn from_diag(other: &[f64; N]) -> Self {
        SquareMatrix {
            rows: std::array::from_fn(|i| {
                std::array::from_fn(|j| if i == j { other[i] } else { 0.0 }).into()
            }),
        }
    }

    /// Multiply a [`SquareMatrix`] by a diagonal matrix on the right hand side
    #[inline]
    fn mul_diagonal(&self, diag: &[f64; N]) -> Self {
        let mut rows = [[0f64; N]; N];
        for i in 0..N {
            for j in 0..N {
                rows[i][j] = self.rows[i][j] * diag[j];
            }
        }
        Self { rows }
    }

    /// Transpose the matrix
    #[inline]
    fn transpose(&self) -> Self {
        Self {
            rows: std::array::from_fn(|i| std::array::from_fn::<_, N, _>(|j| self.rows[j][i])),
        }
    }
    /// (Naive) Matrix multiplication of two square matrixes
    #[inline]
    fn matmul(&self, other: &Self) -> Self {
        let mut result = Self {
            rows: [[0.0; N]; N],
        };
        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    result.rows[i][j] += self.rows[i][k] * other.rows[k][j];
                }
            }
        }

        result
    }
    #[inline]
    fn compute_quadratic_form(&self, other: &[f64; N]) -> f64 {
        let mut result = 0.0;

        for i in 0..N {
            for j in 0..N {
                result += other[i] * self.rows[i][j] * other[j];
            }
        }
        result
    }
}
impl<const N: usize> Mul<f64> for SquareMatrix<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            rows: self.rows.map(|r| r.map(|x| x * rhs)),
        }
    }
}
impl<const N: usize> Add<Self> for SquareMatrix<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            rows: std::array::from_fn(|i| {
                std::array::from_fn(|j| self.rows[i][j] + rhs.rows[i][j])
            }),
        }
    }
}

impl SquareMatrix<2> {
    /// TODO
    #[inline]
    fn det(&self) -> f64 {
        self.rows[0][0] * self.rows[1][1] - self.rows[1][0] * self.rows[0][1]
    }
    /// TODO
    #[inline]
    fn inverse(&self) -> Self {
        let inv_det = self.det().recip();
        Self {
            rows: [
                [inv_det * self.rows[1][1], inv_det * -self.rows[0][1]],
                [inv_det * -self.rows[1][0], inv_det * self.rows[0][0]],
            ],
        }
    }
}
impl SquareMatrix<3> {
    /// TODO
    #[inline]
    fn det(&self) -> f64 {
        let m = &self.rows;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
}

/// An n-[`Hyperellipsoid`] defined by its semi-major axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hyperellipsoid<const N: usize> {
    /// The principle semi-axes of the [`Hyperellipsoid`] along each direction.
    pub axes: [f64; N],
}

/**An ellipse in two dimensions.*/
pub type Ellipse = Hyperellipsoid<2>;
/**An ellipsoid in three dimensions.*/
pub type Ellipsoid = Hyperellipsoid<3>;

impl<const N: usize> SupportMapping<Cartesian<N>> for Hyperellipsoid<N> {
    #[inline]
    fn support_mapping(&self, n: &Cartesian<N>) -> Cartesian<N> {
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

impl<const N: usize> BoundingSphereRadius for Hyperellipsoid<N> {
    #[inline]
    fn bounding_sphere_radius(&self) -> f64 {
        self.axes.into_iter().fold(f64::NAN, f64::max)
    }
}
impl<const N: usize> Volume for Hyperellipsoid<N> {
    #[inline]
    fn volume(&self) -> f64 {
        self.axes
            .into_iter()
            .fold(sphere_volume_prefactor(N), |prod, x| prod * x)
    }
}

// TODO: https://www.iri.upc.edu/files/scidoc/1852-New-algebraic-conditions-for-the-identification-of-the-relative-position-of-two-coplanar-ellipses.pdf

///TODO
const _INV_PHI: f64 = 0.618_033_988_7f64;

impl<R: Copy + Rotate<Cartesian<2>>> IntersectsAt<Hyperellipsoid<2>, Cartesian<2>, R>
    for Hyperellipsoid<2>
where
    RotationMatrix<2>: From<R>,
{
    #[inline]
    fn intersects_at(&self, other: &Hyperellipsoid<2>, v_ij: &Cartesian<2>, o_ij: &R) -> bool {
        let a_inv = other.axes.map(|x| x.powi(2));

        let rot = RotationMatrix::<2>::from(*o_ij);
        let rot_transpose = rot.inverted();

        let b_inv = SquareMatrix::from(rot)
            .mul_diagonal(&self.axes.map(|x| x.powi(2)))
            .matmul(&rot_transpose.into());
        // EVERYTHING BEFORE THIS LINE IS OK

        let v_ij = &v_ij.coordinates;
        let a_inv = SquareMatrix::from_diag(&a_inv);

        // Compute three values for golden ratio method
        let (mut b, mut a) = (0.999_999_999, 0.000_000_001); // TODO: cannot evaluate at bounds?
        while (b - a) > 1e-15 {
            // TODO: sane tolerance
            let c = b - (b - a) * _INV_PHI;
            let d = a + (b - a) * _INV_PHI;

            // TODO: reuse computed k values between loops
            let k_c = k_lambda(a_inv, b_inv, c, v_ij);
            if k_c <= 0.0 {
                println!("kc < 0.0");
                return false;
            }
            let k_d = k_lambda(a_inv, b_inv, d, v_ij);
            if k_d <= 0.0 {
                println!("kd < 0.0");
                return false;
            }
            if k_c < k_d {
                b = d;
            } else {
                a = c;
            }
        }
        println!("Exited iteration");
        true // If we did not detect a negative value, the shapes OVERLAP

        // HOWEVER: we can exit early, so an existing package like argmin may be slower

        // FOR NOW: simple secant method from k[0.5]
    }
}

/// TODO
#[inline]
fn k_lambda(a_inv: SquareMatrix<2>, b_inv: SquareMatrix<2>, l: f64, v_ij: &[f64; 2]) -> f64 {
    let m = b_inv * ((1.0 - l).recip()) + (a_inv * l.recip());

    1.0 - m.inverse().compute_quadratic_form(v_ij)
}

#[expect(
    clippy::used_underscore_binding,
    reason = "Used for const parameterization."
)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Convex, shape::Hypersphere};
    use ::approx::assert_relative_eq;
    use hoomd_vector::{Angle, Unit, Versor};
    use rstest::*;
    use std::marker::PhantomData;

    #[rstest]
    #[case(PhantomData::<Hypersphere<0>>)]
    #[case(PhantomData::<Hypersphere<1>>)]
    #[case(PhantomData::<Hypersphere<2>>)]
    #[case(PhantomData::<Hypersphere<3>>)]
    #[case(PhantomData::<Hypersphere<4>>)]
    #[case(PhantomData::<Hypersphere<5>>)]
    fn test_support_hyperellipsoid<const N: usize>(
        #[case] _n: PhantomData<Hypersphere<N>>,
        #[values(0.1, 1.0, 33.3)] radius: f64,
    ) {
        let s = Hypersphere::<N> { radius };
        let he = Hyperellipsoid { axes: [radius; N] };
        let v = [1.0; N].into();
        assert_relative_eq!(he.support_mapping(&v), s.support_mapping(&v));
    }
    #[rstest]
    #[case(PhantomData::<Hypersphere<0>>)]
    #[case(PhantomData::<Hypersphere<1>>)]
    #[case(PhantomData::<Hypersphere<2>>)]
    #[case(PhantomData::<Hypersphere<3>>)]
    #[case(PhantomData::<Hypersphere<4>>)]
    #[case(PhantomData::<Hypersphere<5>>)]
    fn test_volume_hyperellipsoid<const N: usize>(
        #[case] _n: PhantomData<Hypersphere<N>>,
        #[values(0.1, 1.0, 33.3)] radius: f64,
    ) {
        let s = Hypersphere::<N> { radius };
        let he = Hyperellipsoid { axes: [radius; N] };
        assert_relative_eq!(he.volume(), s.volume());
    }

    #[rstest]
    fn test_ellipse_overlap_along_axis(
        #[values([0.0, 0.0], [1.0,0.0], [1.999_999, 0.0], [2.000_001, 0.0], [2.1, 0.0])]
        v_ij: [f64; 2],
    ) {
        let el0 = Ellipse { axes: [1.0, 4.0] };
        let el1 = Ellipse { axes: [1.0, 4.0] };

        assert_eq!(
            el0.intersects_at(&el1, &v_ij.into(), &Angle::default()),
            Convex(el0).intersects_at(&Convex(el1), &v_ij.into(), &Angle::default())
        );
    }
}

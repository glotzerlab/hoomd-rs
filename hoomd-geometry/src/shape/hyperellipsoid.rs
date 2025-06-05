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

# Example

Rapid ellipse-ellipse intersection testing is possible with hoomd-geometry. This check
is based on a result from algebraic geometry, with the precise approach documented
within the code.

```rust
use hoomd_geometry::shape::Ellipse;
use hoomd_geometry::IntersectsAt;
use hoomd_vector::Angle;

let long_ellipse = Ellipse {axes: [0.5, 3.0]};
let round_ellipse = Ellipse {axes: [1.0, 2.0]};

let v_ij = [0.0, long_ellipse.axes[1]+round_ellipse.axes[1]-0.1].into();

assert_eq!(
    long_ellipse.intersects_at(&round_ellipse, &v_ij, &Angle::from(0.0)),
    true
);

```
*/

use super::sphere::sphere_volume_prefactor;
use crate::{BoundingSphereRadius, IntersectsAt, SupportMapping, Volume};
use hoomd_vector::{Cartesian, Rotate, RotationMatrix, Vector};
use std::ops::{Add, Mul};
/// TODO: temp
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

/// The inverse of the Golden ratio, used for a golden section solver
const _INV_PHI: f64 = 0.618_033_988_749_894_9_f64;
/** Precision within which ellipsoids are considered to be overlapping.

This is 1000x more precise than HOOMD-Blue.
*/
const _ELLIPSOID_OVERLAP_PRECISION: f64 = 1e-9;
/// Max bound of the root search for an ellipsoid characteristic polynomial.
const _ELLIPSOID_K_MAX_BOUND: f64 = 1.0 - _ELLIPSOID_OVERLAP_PRECISION;
/// Min bound of the root search for an ellipsoid characteristic polynomial.
const _ELLIPSOID_K_MIN_BOUND: f64 = _ELLIPSOID_OVERLAP_PRECISION;

impl<R: Copy + Rotate<Cartesian<2>>> IntersectsAt<Hyperellipsoid<2>, Cartesian<2>, R>
    for Hyperellipsoid<2>
where
    RotationMatrix<2>: From<R>,
{
    #[inline]
    fn intersects_at(&self, other: &Hyperellipsoid<2>, v_ij: &Cartesian<2>, o_ij: &R) -> bool {
        /*

        This approach is derived from "A Robust Computational Test for Overlap of Two
        Arbitrary-dimensional Ellipsoids in Fault-Detection of Kalman Filters".
        Rather than generalize over dimension (which results in significant performance
        losses, even with efficient linear algebra libraries), we choose to implement
        the special case of ellipses in 2d. This derivation is far from rigorous, but
        aims to inform how the method actually works.

        We begin with Remark 1 from the above paper. In essence, we are defininig a
        convex, one dimensional function K(λ) that represents the intersection of our
        two ellipsoids, which is also a conic section. If this intersection function has
        any real roots (in the domain (0, 1)), our ellipsoids must intersect. To compute
        this function, we represent our ellipsoids as matrixes $A$ and $B$. $K(λ)$ is...

        $$
            K(λ) = 1 - v^T @ (1/(1-λ) B^-1 + 1/λ A^-1)^-1
        $$

        The "natural" matrix form of an ellipse is $diag(1/axes_i**2)$. However, we must
        transform one of our matrixes for the intersection calculation. We choose B, as
        it simplifies our future calculations. With R as the rotation matrix of o_ij:

        $$
            B = (R^-1).T @ diag(1/axes_B**2) @ R^-1
        $$
        The inverse of a rotation matrix is its transpose, so this simplifies. However,
        because we actually desire A_inverse and B_inverse (and A and B are diagonal),
        we can simplify further:
        $$
            A^-1 = diag(axes_A**2)
            B^-1 = R @ diag(axes_B**2) @ R^T
        $$

        Both these results can be cached and reused for evaluation of [`k_lambda`].
        Note that our equation is really of the quadratic form $K(λ)=1 - v.T @ M @ v$,
        with $M = (1/(1-λ) B^-1 + 1/λ A^-1)^-1$. This final inversion makes evaluating
        K(λ) in this form undesirable in the general case, but in 2D we have a simple
        closed form for the inverse.

        Recall that our ellipsoids do not overlap if there is a real root of Κ(λ) on
        (0, 1). Rather than searching for such a root, we can instead query for a
        negative element in the codomain. Our function is extremely well behaved
        (numerical instability notwithstanding), so we can use a simple golden section
        search for such an element. In most cases, we can exit within a single iteration
        although the method converges linearly in general. If the search does NOT find a
        negative element in the codomain, the ellipsoids intersect (within a tolerance).
        */
        let a_inv = other.axes.map(|x| x.powi(2));

        let rot = RotationMatrix::<2>::from(*o_ij);
        let rot_transpose = rot.inverted();

        let b_inv = SquareMatrix::from(rot)
            .mul_diagonal(&self.axes.map(|x| x.powi(2)))
            .matmul(&rot_transpose.into());

        let v_ij = &v_ij.coordinates;
        let a_inv = SquareMatrix::from_diag(&a_inv);

        // Golden section solver for minimizing K(λ)
        let (mut b, mut a) = (_ELLIPSOID_K_MAX_BOUND, _ELLIPSOID_K_MIN_BOUND);
        while (b - a) > _ELLIPSOID_OVERLAP_PRECISION {
            let c = b - (b - a) * _INV_PHI;
            let d = a + (b - a) * _INV_PHI;

            // Could reuse computed k values between loops for better performance?
            let k_c = k_lambda(a_inv, b_inv, c, v_ij);
            if k_c <= 0.0 {
                return false;
            }
            let k_d = k_lambda(a_inv, b_inv, d, v_ij);
            if k_d <= 0.0 {
                return false;
            }
            if k_c < k_d {
                b = d;
            } else {
                a = c;
            }
        }
        true // If we did not detect a negative value of K(λ), the shapes overlap
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
    use crate::{
        Convex,
        shape::{Circle, Hypersphere},
    };
    use ::approx::assert_relative_eq;
    use hoomd_vector::{Angle, Unit, Versor};
    use rand::{Rng, SeedableRng, rngs::StdRng};
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
    #[rstest]
    fn test_random_sphere_ellipse_overlaps() {
        let mut rng = StdRng::seed_from_u64(2);
        for _ in 0..10_000 {
            let (a, c) = StdRng::random(&mut rng);
            let el0 = Ellipse { axes: [a, a] };
            let el1 = Ellipse { axes: [c, c] };

            let v_ij = StdRng::random::<Cartesian<2>>(&mut rng) * 10.0;
            let angle = Angle::from(
                rng.random_range((-2.0 * std::f64::consts::PI)..(2.0 * std::f64::consts::PI)),
            );
            assert_eq!(
                el0.intersects_at(&el1, &v_ij, &angle),
                Circle { radius: a }.intersects_at(&Circle { radius: c }, &v_ij, &angle),
            );
        }
    }

    #[rstest]
    fn test_random_ellipsoids() {
        // Xenocollide precision becomes an issue! So only a few tests are possible
        // Inspecting failing cases in Ovito & HOOMD shows we are correct
        let mut rng = StdRng::seed_from_u64(2);
        for _ in 0..10 {
            let (a, b, c, d) = StdRng::random(&mut rng);
            let el0 = Ellipse { axes: [a, b] };
            let el1 = Ellipse { axes: [c, d] };

            let v_ij = StdRng::random::<Cartesian<2>>(&mut rng) * 10.0;
            let angle = Angle::from(
                rng.random_range((-2.0 * std::f64::consts::PI)..(2.0 * std::f64::consts::PI)),
            );
            assert_eq!(
                el0.intersects_at(&el1, &v_ij, &angle),
                Convex(el0).intersects_at(&Convex(el1), &v_ij, &angle),
                "(a,b,c,d)= ({a}, {b}, {c}, {d})\nangle= {angle}\nv_ij= {v_ij}"
            );
        }
    }
}

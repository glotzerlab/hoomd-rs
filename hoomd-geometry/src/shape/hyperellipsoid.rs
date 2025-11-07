// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Hyperellipsoid`].

use super::sphere::sphere_volume_prefactor;
use crate::{BoundingSphereRadius, IntersectsAt, SupportMapping, Volume};
use hoomd_linear_algebra::{
    Invertible, MatMul, QuadraticForm,
    matrix::{DiagonalMatrix, Matrix22},
};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, InnerProduct, Metric, Rotate, Rotation, RotationMatrix};

use std::ops::Mul;

/// The geometry resulting from an Hypersphere that is scaled along the Cartesian axes.
///
/// See [`Ellipse`] and [`Ellipsoid`] for special cases in 2 and 3 dimensions.
///
/// # Examples
///
/// Basic construction and methods:
/// ```
/// use approxim::assert_relative_eq;
/// use hoomd_geometry::{BoundingSphereRadius, Volume, shape::Hyperellipsoid};
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let ellipse =
///     Hyperellipsoid::with_semi_axes([1.0.try_into()?, 2.0.try_into()?]);
/// let bounding_radius = ellipse.bounding_sphere_radius();
/// let volume = ellipse.volume();
///
/// assert_eq!(bounding_radius.get(), 2.0);
/// assert_relative_eq!(volume, PI * 1.0 * 2.0);
///
/// let sphere = Hyperellipsoid::with_semi_axes([
///     2.0.try_into()?,
///     2.0.try_into()?,
///     2.0.try_into()?,
/// ]);
/// let bounding_radius = sphere.bounding_sphere_radius();
/// let volume = sphere.volume();
///
/// assert_eq!(bounding_radius.get(), 2.0);
/// assert_eq!(volume, 4.0 / 3.0 * PI * 2.0_f64.powi(3));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Hyperellipsoid<const N: usize> {
    /// The principle semi-axes of the [`Hyperellipsoid`] along each Cartesian direction.
    semi_axes: [PositiveReal; N],

    /// The bounding sphere radius.
    bounding_sphere_radius: PositiveReal,
}

impl<const N: usize> Hyperellipsoid<N> {
    /// Construct a new Hyperellipsoid with the given semi-axes along each Cartesian direction.
    #[expect(
        clippy::missing_panics_doc,
        reason = "Panic would occur due to a bug in hoomd-rs."
    )]
    #[must_use]
    #[inline]
    pub fn with_semi_axes(semi_axes: [PositiveReal; N]) -> Self {
        let bounding_sphere_radius = semi_axes
            .iter()
            .map(PositiveReal::get)
            .reduce(f64::max)
            .expect("N must be greater than or equal to 1")
            .try_into()
            .expect("expression evaluates to a positive real");

        Self {
            semi_axes,
            bounding_sphere_radius,
        }
    }

    /// Get the semi axes.
    #[must_use]
    #[inline]
    pub fn semi_axes(&self) -> &[PositiveReal; N] {
        &self.semi_axes
    }
}

/// A circle scaled along the x and y axes.
///
/// # Examples
///
/// Basic construction and methods:
/// ```
/// use approxim::assert_relative_eq;
/// use hoomd_geometry::{BoundingSphereRadius, Volume, shape::Ellipse};
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let ellipse = Ellipse::with_semi_axes([1.0.try_into()?, 2.0.try_into()?]);
/// let bounding_radius = ellipse.bounding_sphere_radius();
/// let volume = ellipse.volume();
///
/// assert_eq!(bounding_radius.get(), 2.0);
/// assert_relative_eq!(volume, PI * 1.0 * 2.0);
/// # Ok(())
/// # }
/// ```
///
/// Rapid ellipse-ellipse intersection testing is possible with hoomd-geometry. This check
/// is based on a result from algebraic geometry, with the precise approach documented
/// within the code:
/// ```
/// use hoomd_geometry::{IntersectsAt, shape::Ellipse};
/// use hoomd_vector::Angle;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let long_ellipse =
///     Ellipse::with_semi_axes([0.5.try_into()?, 3.0.try_into()?]);
/// let round_ellipse =
///     Ellipse::with_semi_axes([1.0.try_into()?, 2.0.try_into()?]);
///
/// let v_ij = [
///     0.0,
///     long_ellipse.semi_axes()[1].get() + round_ellipse.semi_axes()[1].get()
///         - 0.1,
/// ]
/// .into();
///
/// assert_eq!(
///     long_ellipse.intersects_at(&round_ellipse, &v_ij, &Angle::from(0.0)),
///     true
/// );
/// # Ok(())
/// # }
/// ```
pub type Ellipse = Hyperellipsoid<2>;

/// A sphere scaled along the x, y, and z axes.
///
/// # Examples
///
/// Basic construction and methods:
/// ```
/// use approxim::assert_relative_eq;
/// use hoomd_geometry::{BoundingSphereRadius, Volume, shape::Ellipsoid};
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let sphere = Ellipsoid::with_semi_axes([
///     2.0.try_into()?,
///     2.0.try_into()?,
///     2.0.try_into()?,
/// ]);
/// let bounding_radius = sphere.bounding_sphere_radius();
/// let volume = sphere.volume();
///
/// assert_eq!(bounding_radius.get(), 2.0);
/// assert_eq!(volume, 4.0 / 3.0 * PI * 2.0_f64.powi(3));
/// # Ok(())
/// # }
/// ```
///
/// Test for intersections using [`Convex`](crate::Convex):
/// ```
/// use hoomd_geometry::{Convex, IntersectsAt, shape::Ellipsoid};
/// use hoomd_vector::Versor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let ellipsoid = Convex(Ellipsoid::with_semi_axes([
///     1.0.try_into()?,
///     2.0.try_into()?,
///     3.0.try_into()?,
/// ]));
/// let q = Versor::default();
///
/// assert_eq!(
///     ellipsoid.intersects_at(&ellipsoid, &[0.9, 0.0, 0.0].into(), &q),
///     true
/// );
/// assert_eq!(
///     ellipsoid.intersects_at(&ellipsoid, &[1.1, 0.0, 0.0].into(), &q),
///     true
/// );
/// assert_eq!(
///     ellipsoid.intersects_at(&ellipsoid, &[0.0, 1.9, 0.0].into(), &q),
///     true
/// );
/// assert_eq!(
///     ellipsoid.intersects_at(&ellipsoid, &[0.0, 2.1, 0.0].into(), &q),
///     true
/// );
/// # Ok(())
/// # }
/// ```
pub type Ellipsoid = Hyperellipsoid<3>;

impl<const N: usize> SupportMapping<Cartesian<N>> for Hyperellipsoid<N> {
    #[inline]
    fn support_mapping(&self, n: &Cartesian<N>) -> Cartesian<N> {
        let denominator =
            Cartesian::<N>::from(std::array::from_fn(|i| self.semi_axes[i].get() * n[i])).norm();
        std::array::from_fn(|i| n[i] * self.semi_axes[i].get().powi(2) / denominator).into()
    }
}

impl<const N: usize> BoundingSphereRadius for Hyperellipsoid<N> {
    #[inline]
    fn bounding_sphere_radius(&self) -> PositiveReal {
        self.bounding_sphere_radius
    }
}
impl<const N: usize> Volume for Hyperellipsoid<N> {
    #[inline]
    fn volume(&self) -> f64 {
        self.semi_axes
            .iter()
            .map(PositiveReal::get)
            .fold(sphere_volume_prefactor(N), f64::mul)
    }
}

/// The inverse of the Golden ratio, used for a golden section solver
const _INV_PHI: f64 = 0.618_033_988_749_894_9_f64;
/// Precision within which ellipsoids are considered to be overlapping.
///
/// This is 1000x more precise than HOOMD-Blue.
const _ELLIPSOID_OVERLAP_PRECISION: f64 = 1e-9;
/// Max bound of the root search for an ellipsoid characteristic polynomial.
const _ELLIPSOID_K_MAX_BOUND: f64 = 1.0 - _ELLIPSOID_OVERLAP_PRECISION;
/// Min bound of the root search for an ellipsoid characteristic polynomial.
const _ELLIPSOID_K_MIN_BOUND: f64 = _ELLIPSOID_OVERLAP_PRECISION;

impl<R> IntersectsAt<Hyperellipsoid<2>, Cartesian<2>, R> for Hyperellipsoid<2>
where
    R: Rotation + Rotate<Cartesian<2>>,
    RotationMatrix<2>: From<R>,
{
    #[inline]
    fn intersects_at(&self, other: &Hyperellipsoid<2>, v_ij: &Cartesian<2>, o_ij: &R) -> bool {
        // This approach is derived from "A Robust Computational Test for Overlap of Two
        // Arbitrary-dimensional Ellipsoids in Fault-Detection of Kalman Filters".
        // Rather than generalize over dimension (which results in significant performance
        // losses, even with efficient linear algebra libraries), we choose to implement
        // the special case of ellipses in 2d. This derivation is far from rigorous, but
        // aims to inform how the method actually works.
        //
        // We begin with Remark 1 from the above paper. In essence, we are defininig a
        // convex, one dimensional function K(λ) that represents the intersection of our
        // two ellipsoids, which is also a conic section. If this intersection function has
        // any real roots (in the domain (0, 1)), our ellipsoids must intersect. To compute
        // this function, we represent our ellipsoids as matrixes $A$ and $B$. $K(λ)$ is...
        //
        // $$
        // K(λ) = 1 - v^T @ (1/(1-λ) B^-1 + 1/λ A^-1)^-1
        // $$
        //
        // The "natural" matrix form of an ellipse is $diag(1/axes_i**2)$. However, we must
        // transform one of our matrixes for the intersection calculation. We choose B, as
        // it simplifies our future calculations. With R as the rotation matrix of o_ij:
        //
        // $$
        // B = (R^-1).T @ diag(1/axes_B**2) @ R^-1
        // $$
        // The inverse of a rotation matrix is its transpose, so this simplifies. However,
        // because we actually desire A_inverse and B_inverse (and A and B are diagonal),
        // we can simplify further:
        // $$
        // A^-1 = diag(axes_A**2)
        // B^-1 = R @ diag(axes_B**2) @ R^T
        // $$
        //
        // Both these results can be cached and reused for evaluation of [`k_lambda`].
        // Note that our equation is really of the quadratic form $K(λ)=1 - v.T @ M @ v$,
        // with $M = (1/(1-λ) B^-1 + 1/λ A^-1)^-1$. This final inversion makes evaluating
        // K(λ) in this form undesirable in the general case, but in 2D we have a simple
        // closed form for the inverse.
        //
        // Recall that our ellipsoids do not overlap if there is a real root of Κ(λ) on
        // (0, 1). Rather than searching for such a root, we can instead query for a
        // negative element in the codomain. Our function is extremely well behaved
        // (numerical instability notwithstanding), so we can use a simple golden section
        // search for such an element. In most cases, we can exit within a single iteration
        // although the method converges linearly in general. If the search does NOT find a
        // negative element in the codomain, the ellipsoids intersect (within a tolerance).
        let a_inv = other.semi_axes.map(|x| x.get().powi(2));

        let rot = RotationMatrix::<2>::from(*o_ij);
        let rot_transpose = rot.inverted();

        let b_inv = Matrix22::from(rot)
            .matmul(&DiagonalMatrix {
                elements: self.semi_axes.map(|x| x.get().powi(2)),
            })
            .matmul(&Matrix22::from(rot_transpose));

        let v_ij = &v_ij.coordinates;
        let a_inv = Matrix22::with_diagonal(a_inv);

        // Golden section solver for minimizing K(λ)
        let (mut b, mut a) = (_ELLIPSOID_K_MAX_BOUND, _ELLIPSOID_K_MIN_BOUND);
        while (b - a) > _ELLIPSOID_OVERLAP_PRECISION {
            let c = b - (b - a) * _INV_PHI;
            let d = a + (b - a) * _INV_PHI;

            // Could reuse computed k values between loops for better performance?
            let k_c = k_lambda::<2, Matrix22>(&a_inv, &b_inv, c, v_ij);
            if k_c <= 0.0 {
                return false;
            }
            let k_d = k_lambda::<2, Matrix22>(&a_inv, &b_inv, d, v_ij);
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

    #[inline]
    fn intersects_at_global(
        &self,
        other: &Hyperellipsoid<2>,
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

/// Solve the characteristic equation of two ellipses.
#[inline]
fn k_lambda<const N: usize, M>(a_inv: &M, b_inv: &M, l: f64, v_ij: &[f64; N]) -> f64
where
    M: Invertible + Copy + QuadraticForm<N>,
{
    let m = *b_inv * ((1.0 - l).recip()) + (*a_inv * l.recip());

    1.0 - m
        .inverse()
        .expect("Matrix is not invertible - overlap check would return NaN.")
        .compute_quadratic_form(v_ij)
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
    use approxim::assert_relative_eq;
    use hoomd_vector::Angle;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rstest::*;
    use std::marker::PhantomData;

    #[rstest]
    #[case(PhantomData::<Hypersphere<1>>)]
    #[case(PhantomData::<Hypersphere<2>>)]
    #[case(PhantomData::<Hypersphere<3>>)]
    #[case(PhantomData::<Hypersphere<4>>)]
    #[case(PhantomData::<Hypersphere<5>>)]
    fn test_support_hyperellipsoid<const N: usize>(
        #[case] _n: PhantomData<Hypersphere<N>>,
        #[values(0.1, 1.0, 33.3)] radius: f64,
    ) {
        let s = Hypersphere::<N> {
            radius: radius.try_into().expect("test value is a positive real"),
        };
        let he = Hyperellipsoid::with_semi_axes(
            [radius.try_into().expect("test value is a positive real"); N],
        );
        let v = [1.0; N].into();
        assert_relative_eq!(he.support_mapping(&v), s.support_mapping(&v));
    }
    #[rstest]
    #[case(PhantomData::<Hypersphere<1>>)]
    #[case(PhantomData::<Hypersphere<2>>)]
    #[case(PhantomData::<Hypersphere<3>>)]
    #[case(PhantomData::<Hypersphere<4>>)]
    #[case(PhantomData::<Hypersphere<5>>)]
    fn test_volume_hyperellipsoid<const N: usize>(
        #[case] _n: PhantomData<Hypersphere<N>>,
        #[values(0.1, 1.0, 33.3)] radius: f64,
    ) {
        let s = Hypersphere::<N> {
            radius: radius.try_into().expect("test value is a positive real"),
        };
        let he = Hyperellipsoid::with_semi_axes(
            [radius.try_into().expect("test value is a positive real"); N],
        );
        assert_relative_eq!(he.volume(), s.volume());
    }

    #[rstest]
    fn test_ellipse_overlap_along_axis(
        #[values([0.0, 0.0], [1.0,0.0], [1.999_999, 0.0], [2.000_001, 0.0], [2.1, 0.0])]
        v_ij: [f64; 2],
    ) {
        let el0 = Ellipse::with_semi_axes([
            1.0.try_into().expect("test value is a positive real"),
            4.0.try_into().expect("test value is a positive real"),
        ]);
        let el1 = Ellipse::with_semi_axes([
            1.0.try_into().expect("test value is a positive real"),
            4.0.try_into().expect("test value is a positive real"),
        ]);

        assert_eq!(
            el0.intersects_at(&el1, &v_ij.into(), &Angle::default()),
            Convex(el0).intersects_at(&Convex(el1), &v_ij.into(), &Angle::default())
        );
    }
    #[rstest]
    fn test_random_sphere_ellipse_overlap() {
        let mut rng = StdRng::seed_from_u64(2);
        for _ in 0..10_000 {
            let (a, c): (f64, f64) = StdRng::random(&mut rng);
            let a = a.try_into().expect("test value is a positive real");
            let c = c.try_into().expect("test value is a positive real");
            let el0 = Ellipse::with_semi_axes([a, a]);
            let el1 = Ellipse::with_semi_axes([c, c]);

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
    fn test_random_ellipsoids_overlap() {
        // Xenocollide precision becomes an issue! So only a few tests are possible
        // Inspecting failing cases in Ovito & HOOMD shows we are correct
        let mut rng = StdRng::seed_from_u64(2);
        for _ in 0..10 {
            let (a, b, c, d): (f64, f64, f64, f64) = StdRng::random(&mut rng);
            let a = a.try_into().expect("test value is a positive real");
            let b = b.try_into().expect("test value is a positive real");
            let c = c.try_into().expect("test value is a positive real");
            let d = d.try_into().expect("test value is a positive real");

            let el0 = Ellipse::with_semi_axes([a, b]);
            let el1 = Ellipse::with_semi_axes([c, d]);

            let v_ij = StdRng::random::<Cartesian<2>>(&mut rng) * 10.0;
            let angle = Angle::from(
                rng.random_range((-2.0 * std::f64::consts::PI)..(2.0 * std::f64::consts::PI)),
            );
            assert_eq!(
                el0.intersects_at(&el1, &v_ij, &angle),
                Convex(el0).intersects_at(&Convex(el1), &v_ij, &angle),
                "(a,b,c,d)= ({}, {}, {}, {})\nangle= {angle}\nv_ij= {v_ij}",
                a.get(),
                b.get(),
                c.get(),
                d.get()
            );
        }
    }
}

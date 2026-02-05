// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Hyperellipsoid`].

use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::ops::Mul;

use super::sphere::sphere_volume_prefactor;
use crate::{BoundingSphereRadius, IntersectsAt, IntersectsAtGlobal, SupportMapping, Volume};
use hoomd_linear_algebra::{
    QuadraticForm,
    matrix::{DiagonalMatrix, Matrix22},
};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Angle, Cartesian, InnerProduct, Metric, Rotate, Rotation, RotationMatrix};

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
#[serde_as]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hyperellipsoid<const N: usize> {
    /// The principle semi-axes of the [`Hyperellipsoid`] along each Cartesian direction.
    #[serde_as(as = "[_; N]")]
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

impl<R> IntersectsAt<Hyperellipsoid<2>, Cartesian<2>, R> for Hyperellipsoid<2>
where
    R: Rotation + Rotate<Cartesian<2>>,
    Angle: From<R>,
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
        // transform one of our matrixes for the intersection calculation. We choose A, as
        // it simplifies our future calculations. With R as the rotation matrix of o_ij:
        //
        // $$
        // A = (R^-1).T @ diag(1/axes_A**2) @ R^-1
        // $$
        // The inverse of a rotation matrix is its transpose, so this simplifies. However,
        // because we actually desire A_inverse and B_inverse (and A and B are diagonal),
        // we can simplify further:
        // $$
        // A^-1 = R @ diag(axes_A**2) @ R^T
        // B^-1 = diag(axes_B**2)
        // $$
        //
        // Both these results can be cached and reused for evaluation of [`k_lambda`].
        // Note that our equation is really of the quadratic form $K(λ)=1 - v.T @ M @ v$,
        // with $M = (1/(1-λ) B^-1 + 1/λ A^-1)^-1$. This final inversion makes evaluating
        // K(λ) in this form undesirable in the general case, but in 2D we have a simple
        // closed form for the inverse.
        //
        // Recall that our ellipsoids do not overlap if there is a real root of Κ(λ) on
        // (0, 1). Rather than searching for such a root, we can instead query for its
        // existence analytically. We can show that K(λ) <= 0 if and only if its numerator
        // polynomial P(λ) <= 0. P(λ) is a $n+1$-degree (cubic) polynomial:
        //
        // P(λ) = c3 λ^3 + c2 λ^2 + c1 λ + c0
        //
        // Where the coefficients are derived from the determinants and adjoints of A^-1
        // and B^-1. Since P(0) > 0 and P(1) > 0, the ellipsoids are separated if and only
        // if P(λ) has a local minimum in (0, 1) where P(λ) <= 0.

        // b^-1 = R @ (B^-1) @ R.T, where B^-1 is in the local frame and ∴ diagonal
        // Simplified, we get the following (provided B^-1=diag([a_sq, b_sq]))
        // {{Cos[θ]^2 a_sq + b_sq Sin[θ]^2, Cos[θ] (a_sq - b_sq) Sin[θ]},
        //  {Cos[θ] (a_sq - b_sq) Sin[θ], Cos[θ]^2 b_sq + a_sq Sin[θ]^2}}
        let theta = Angle::from(*o_ij).theta;
        let (sin, cos) = theta.sin_cos();
        let (s_sq, c_sq) = (sin.powi(2), cos.powi(2));
        let a = other.semi_axes[0].get();
        let b = other.semi_axes[1].get();

        let a_inv = DiagonalMatrix {
            elements: self.semi_axes.map(|x| x.get().powi(2)),
        };
        let diagonal = cos * (a.powi(2) - b.powi(2)) * sin;
        let b_inv_m_a_inv = Matrix22 {
            rows: [
                [
                    c_sq * a.powi(2) + b.powi(2) * s_sq - a_inv[(0, 0)],
                    diagonal,
                ],
                [
                    diagonal,
                    c_sq * b.powi(2) + a.powi(2) * s_sq - a_inv[(1, 1)],
                ],
            ],
        };

        let det_a_inv = a_inv[(0, 0)] * a_inv[(1, 1)];
        let det_b_inv_m_a_inv = b_inv_m_a_inv.determinant();

        let adj_a_inv = Matrix22 {
            rows: [[a_inv[(1, 1)], 0.0], [0.0, a_inv[(0, 0)]]],
        };

        let adj_b_inv_m_a_inv = Matrix22 {
            rows: [
                [b_inv_m_a_inv.rows[1][1], -b_inv_m_a_inv.rows[0][1]],
                [-b_inv_m_a_inv.rows[1][0], b_inv_m_a_inv.rows[0][0]],
            ],
        };
        let q0 = adj_a_inv.compute_quadratic_form(&v_ij.coordinates);
        let q1 = adj_b_inv_m_a_inv.compute_quadratic_form(&v_ij.coordinates);

        // d1 = tr(adj(A_inv) @ d).
        // Note the off diagonal terms would be 0 and are excluded
        let d1 = f64::mul_add(
            adj_a_inv[(0, 0)],
            b_inv_m_a_inv[(0, 0)],
            adj_a_inv[(1, 1)] * b_inv_m_a_inv[(1, 1)],
        );

        let (c3, c2, c1, c0) = (q1, det_b_inv_m_a_inv - q1 + q0, d1 - q0, det_a_inv);

        // Early exit with an IVT check
        let p0 = c0; // P(0) = c0
        let p1 = c3 + c2 + c1 + c0; // P(1) = c3 + c2 + c1 + c0

        // If endpoints have different signs we definitely have a root
        if p0 * p1 <= 0.0 {
            return false;
        }

        // Both endpoints are negative -- no overlap is possible
        if (p0 < 0.0) & (p1 < 0.0) {
            return false;
        }

        // Bézier Clipping check
        // Control points b0 = p0, b3 = p1, so all we need are b1 and b2
        let b1 = c0 + c1 / 3.0;
        let b2 = c0 + f64::mul_add(2.0, c1, c2) / 3.0;

        // Check if the hull is strictly separated from 0.
        // Since we know p0 and p1 have the same sign (from step 1),
        // we just need to check if b1 and b2 share that sign.
        if p0 > 0.0 {
            // Entire hull is positive -> No root possible.
            if b1 > 0.0 && b2 > 0.0 {
                return true;
            }
        } else {
            // Entire hull is negative -> No root possible.
            if b1 < 0.0 && b2 < 0.0 {
                return true;
            }
        }

        // Find local extrema of P(λ) = c3*λ^3 + c2*λ^2 + c1*λ + c0
        // P'(lambda) = 3*c3*λ^2 + 2*c2*λ + c1
        // Polynomial is effectively quadratic
        if c3.abs() < 1e-15 {
            // Polynomial is *not* effectively linear
            if c2.abs() > 1e-15 {
                let l_star = -c1 / (2.0 * c2);
                if (0.0..1.0).contains(&l_star) {
                    let p_star = (c2 * l_star + c1) * l_star + c0;
                    if p0 > 0.0 && p_star <= 0.0 {
                        return false;
                    }
                    if p0 < 0.0 && p_star >= 0.0 {
                        return false;
                    }
                }
            }
        } else {
            let delta = f64::mul_add(c2, c2, (-3.0 * c3) * c1);
            if delta >= 0.0 {
                let sqrt_delta = delta.sqrt();
                let l1 = (-c2 - sqrt_delta) / (3.0 * c3);
                let l2 = (-c2 + sqrt_delta) / (3.0 * c3);

                for l in [l1, l2] {
                    if (0.0..1.0).contains(&l) {
                        let p = ((c3 * l + c2) * l + c1) * l + c0;
                        if p0 > 0.0 && p <= 0.0 {
                            return false;
                        }
                        if p0 < 0.0 && p >= 0.0 {
                            return false;
                        }
                    }
                }
            }
        }
        true // If we did not detect a non-positive value of P(λ), the shapes overlap
    }
}
impl<R> IntersectsAtGlobal<Hyperellipsoid<2>, Cartesian<2>, R> for Hyperellipsoid<2>
where
    R: Rotation + Rotate<Cartesian<2>>,
    Angle: From<R>,
    RotationMatrix<2>: From<R>,
{
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
        #[values(0.0, 2.0 * std::f64::consts::PI)] o_ij: f64,
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
            el0.intersects_at(&el1, &v_ij.into(), &Angle::from(o_ij)),
            Convex(el0).intersects_at(&Convex(el1), &v_ij.into(), &Angle::from(o_ij))
        );
    }
    // This data was generated by root finding in mathematica, and cases have been
    // visually verified. I tried to annotate why the cases are named as they are, but
    // to be totally honest they don't fully describe the system. "large" gaps indicate
    // spacing of ~1e-1 and "small" gaps indicate <1e-3. Much smaller than this and my
    // mathematica script starts to disagree with rust, but I will note that xenocollide
    // and our custom implementation DO agree in all tested cases.
    #[rstest]
    #[case::small_gap_nearly_axis_aligned([1.0595, 1.9480], [0.8127, 1.8536], [1.8784, 0.0441], 3.1083, false)]
    #[case::large_gap_near_45_degrees([1.1006, 1.7573], [1.1109, 0.5128], [0.5199, -2.8111], 0.8937, false)]
    #[case::oblate([1.6074, 0.8916], [1.7323, 1.4077], [2.3685, 1.5793], 2.0076, false)]
    #[case::oblate_near_spherical_modest_gap([1.1498, 0.6598], [1.4868, 1.4980], [2.4987, 0.9798], 3.0069, false)]
    #[case::oblate_near_spherical_overlap([1.1498, 0.6598], [1.4868, 1.4980], [2.3453, 0.9196], 3.0069, true)]
    #[case::very_oblate_modest_gap([1.9115, 0.5322], [1.8442, 1.3827], [-0.5431, -2.2782], 2.6297, false)]
    #[case::very_oblate_modest_overlap([1.9115, 0.5322], [1.8442, 1.3827], [-0.4628, -1.9416], 2.6297, true)]
    #[case::nearly_orthogonal_large_gap([0.7574, 1.5755], [0.6234, 1.5001], [2.0806, -1.1887], 1.6139, false)]
    #[case::nearly_orthogonal_overlap([0.7574, 1.5755], [0.6234, 1.5001], [1.9758, -1.1289], 1.6139, true)]
    #[case::nothing_special([0.6577, 1.0500], [1.4852, 1.0473], [-0.3930, -2.2628], 0.5831, false)]
    #[case::nothing_special_overlaps([0.6577, 1.0500], [1.4852, 1.0473], [-0.3812, -2.1947], 0.5831, true)]
    #[case::quite_close([0.6166, 1.4486], [1.4183, 0.9930], [-1.7161, -1.0606], 2.6675, false)]
    #[case::quite_close_overlaps([0.6166, 1.4486], [1.4183, 0.9930], [-1.6661, -1.0297], 2.6675, true)]
    #[case::near_sphere_very_close([1.1526, 1.9197], [1.8130, 1.7356], [2.8098, -0.8908], 1.4848, false)]
    #[case::near_sphere_overlaps_very_close([1.1526, 1.9197], [1.8130, 1.7356], [2.7958, -0.8864], 1.4848, true)]
    #[case::near_matching_oblate_small_gap([1.4806, 0.6951], [0.6211, 1.8280], [2.5104, -0.5142], 0.6163, false)]
    #[case::near_matching_oblate_overlaps([1.4806, 0.6951], [0.6211, 1.8280], [2.4964, -0.5114], 0.6163, true)]
    #[case::size_disparity_very_small_gap([1.0641, 0.7902], [1.7608, 0.6606], [0.4238, 1.5632], 0.2742, false)]
    #[case::size_disparity_overlaps([1.0641, 0.7902], [1.7608, 0.6606], [0.4206, 1.5515], 0.2742, true)]
    #[case::very_skinny_nearly_orthogonal_very_close([0.2442, 7.1313], [9.5758, 0.9664], [-6.7470, 7.7818], 0.0056, false)]
    #[case::very_skinny_nearly_orthogonal_very_close([0.2442, 7.1313], [9.5758, 0.9664], [-6.7367, 7.7700], 0.0056, true)]
    #[case::very_skinny_very_close([6.7374, 0.2257], [6.6506, 1.9512], [-9.0568, -1.4038], -0.2483, false)]
    #[case::very_skinny_overlap([6.7374, 0.2257], [6.6506, 1.9512], [-8.9931, -1.3939], -0.2483, false)]
    #[case::nearly_orthogonal([4.1355, 7.7023], [1.0113, 7.0499], [-5.0217, 3.9708], -0.0012, false)]
    #[case::nearly_orthogonal_overlaps([4.1355, 7.7023], [1.0113, 7.0499], [-5.0070, 3.9591], -0.0012, true)]
    #[case::big_sphere_very_close([0.7540, 2.0375], [6.5678, 6.8224], [4.9170, 6.6840], 0.1307, false)]
    #[case::big_sphere_overlap([0.7540, 2.0375], [6.5678, 6.8224], [4.9041, 6.6665], 0.1307, true)]
    fn test_ellipse_overlap_known_cases(
        #[case] e0: [f64; 2],
        #[case] e1: [f64; 2],
        #[case] v_ij: [f64; 2],
        #[case] o_ij: f64,
        #[case] does_overlap: bool,
    ) {
        let el0 = Ellipse::with_semi_axes([
            e0[0].try_into().expect("test value is a positive real"),
            e0[1].try_into().expect("test value is a positive real"),
        ]);
        let el1 = Ellipse::with_semi_axes([
            e1[0].try_into().expect("test value is a positive real"),
            e1[1].try_into().expect("test value is a positive real"),
        ]);

        assert_eq!(
            el0.intersects_at(&el1, &v_ij.into(), &Angle::from(o_ij)),
            does_overlap
        );
        assert_eq!(
            Convex(el0).intersects_at(&Convex(el1), &v_ij.into(), &Angle::from(o_ij)),
            does_overlap
        );
    }

    #[rstest]
    #[case::six_one_aligned([4.2952, -2.1351], -0.0880, false)]
    #[case::six_one_aligned([4.2597, -2.1174], -0.0880, false)]
    #[case::six_one_aligned([-1.9937, -2.1883], -0.2352, false)]
    #[case::six_one_aligned([-1.9787, -2.1719], -0.2352, true)]
    #[case::six_one_aligned([-0.7759, 2.0001], 0.0095, false)]
    #[case::six_one_aligned([-0.7688, 1.9816], 0.0095, true)]
    #[case::six_one_aligned([2.1407, 1.9930], -0.1539, true)]
    #[case::six_one_aligned([6.4962, -1.7588], 0.0426, false)]
    #[case::six_one_aligned([6.2927, -1.7037], 0.0426, false)]
    #[case::six_one_aligned([2.0187, -3.0287], -0.3221, false)]
    #[case::six_one_aligned([-5.9183, -1.3618], -0.2930, false)]
    #[case::six_one_aligned([1.0047, -2.0353], 0.0426, false)]
    #[case::six_one_aligned([0.9783, -1.9817], 0.0426, true)]
    #[case::six_one_aligned([11.2492, 0.8228], 0.0199, false)]
    #[case::six_one_aligned([11.0, 0.8208], 0.0199, true)]
    #[case::tip_to_tail([-11.9985, -0.0165], 0.0085, false)]
    #[case::tip_to_tail([-11.9864, -0.0165], 0.0085, true)]
    fn test_ellipse_overlap_likely_cases(
        #[case] v_ij: [f64; 2],
        #[case] o_ij: f64,
        #[case] does_overlap: bool,
    ) {
        let el0 = Ellipse::with_semi_axes([
            6.0.try_into().expect("test value is a positive real"),
            1.0.try_into().expect("test value is a positive real"),
        ]);
        let el1 = Ellipse::with_semi_axes([
            6.0.try_into().expect("test value is a positive real"),
            1.0.try_into().expect("test value is a positive real"),
        ]);

        assert_eq!(
            el0.intersects_at(&el1, &Cartesian::from(v_ij), &Angle::from(o_ij)),
            does_overlap
        );
        assert_eq!(
            Convex(el0).intersects_at(&Convex(el1), &v_ij.into(), &Angle::from(o_ij)),
            does_overlap
        );
    }

    #[rstest]
    fn test_random_sphere_ellipse_overlap() {
        let mut rng = StdRng::seed_from_u64(2);
        for _ in 0..100_000 {
            let (a, c): (f64, f64) = StdRng::random(&mut rng);
            let a = (a * 5.0).try_into().expect("test value is a positive real");
            let c = (c * 5.0).try_into().expect("test value is a positive real");
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

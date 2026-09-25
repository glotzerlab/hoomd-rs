// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! A robust four-dimensional orientation predicate.
//!
//! [`orient4d`] determines on which side of the hyperplane through four
//! points a fifth point lies, by the sign of the determinant of the
//! $` 4 \times 4 `$ matrix of their differences:
//!
//! ```math
//! \det\begin{bmatrix}
//!     \vec{a} - \vec{e} \\
//!     \vec{b} - \vec{e} \\
//!     \vec{c} - \vec{e} \\
//!     \vec{d} - \vec{e}
//! \end{bmatrix}
//! ```
//!
//! The determinant is antisymmetric in the five points and zero exactly when
//! they lie on a common hyperplane. For example, the points at the origin and
//! the four coordinate axes in the order `(o, x, y, z, w)` evaluate to `+1`.
//!
//! The predicate is evaluated in two stages:
//!
//! 1. A plain f64 prefilter evaluates the determinant by a Laplace expansion, together
//!    with a rigorous bound on its roundoff error. When the magnitude of the computed
//!    determinant exceeds the bound, its sign is known and can be returned.
//! 2. Otherwise, [`orient4d_exact`] evaluates the determinant exactly: the
//!    coordinates are converted to integers with a common power-of-two
//!    scale, and the determinant is computed with 512-bit integer
//!    arithmetic ([`I512`]). The sign is then exact for the coordinates as
//!    stored.
//!
//! The exact stage accommodates coordinates whose exponents span up to 72
//! powers of two (roughly 21 orders of magnitude, e.g. values in
//! $` [10^{-16}, 10^{3}] `$). Note the span is measured on the raw coordinates: a
//! point set clustered near a large offset (small coordinate differences,
//! large coordinates) is rejected even though its differences span little.
//! Point sets with a wider dynamic range, or with non-finite coordinates,
//! cannot be resolved and produce
//! [`Error::NumericallyAmbiguousPolytope`].

use crate::Error;
use hoomd_linear_algebra::matrix::Matrix44;
use hoomd_utility::dyad::Dyad;
use hoomd_vector::Cartesian;
use i256::I512;

/// The unit roundoff of f64, `2⁻⁵³`.
const UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;

/// The smallest positive subnormal f64, `2⁻¹⁰⁷⁴`.
///
/// Follows the precomputed constants of Shewchuk's `predicates.c` and the `robust`.
const MIN_SUBNORMAL: f64 = 5e-324;

/// Higham's `γₖ = ku / (1 - ku)`: the bound on the relative error of a
/// floating-point expression evaluated with `k` rounding operations, relative
/// to the sum of the absolute values of its terms.
///
/// Source: N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
/// SIAM, 2002, Lemma 3.1
const fn roundoff_bound(k: u32) -> f64 {
    let k = k as f64;
    k * UNIT_ROUNDOFF / (1.0 - k * UNIT_ROUNDOFF)
}

/// The bound on the rounding error of the filtered evaluation, relative to `Λ̂`, the
/// permanent of the elementwise absolute value of the input matrix.
///
/// Two sources of rounding combine into one bound over the same sum:
///
/// * Evaluating the expansion: `Matrix44::determinant` uses 45 roundings (six
///   2×2 minors at 3 roundings each; four groups of a row entry times a
///   three-term minor combination at 6 roundings each; three final combines).
///   Following one determinant term along its path to the result, it passes
///   through at most 9 of them: one rounding of its product inside a 2×2
///   minor, the minor's subtraction, the multiplication by a second-row
///   entry, the two combines within its group's parentheses, the
///   multiplication by the first-row entry, and the three final combines.
///   Each of the 24 terms enters the expanded sum exactly once (the six
///   shared minors feed distinct terms of the two groups they enter), so
///   `|fl(det(M)) - det(M)| ≤ γ₉·Λ ≤ γ₄₅·Λ` by Higham Lemma 3.1.
/// * Rounding the differences: each entry `mᵢⱼ = fl(dᵢⱼ) = dᵢⱼ(1 + δᵢⱼ)` with
///   `|δᵢⱼ| ≤ u` makes each determinant term of the exact matrix `D` equal to
///   the corresponding term of `M` times a product of four `(1 + δᵢⱼ)⁻¹`
///   factors, i.e. four more roundings per term, again by Lemma 3.1 (which
///   admits negative exponents). Thus `|det(M) - det(D)| ≤ γ₄·Λ`.
///
/// The leading 2 covers the rounding of `Λ̂` itself.
///
/// Sources: N. J. Higham, *Accuracy and Stability of Numerical Algorithms*
const ROUNDING_ERROR: f64 = 2.0 * roundoff_bound(49);

/// The absolute floor for subnormal intermediates in the determinant evaluation,
/// scaled by the square of the largest entry.
///
/// A subnormal intermediate has absolute error up to `2⁻¹⁰⁷⁵` (half the smallest
/// subnormal spacing) instead of a relatively bounded one. Any intermediate of
/// `Matrix44::determinant` is multiplied by at most two further entries on its path
/// to the determinant, and there are 45 operations: `45·2⁻¹⁰⁷⁵·max² < 64·2⁻¹⁰⁷⁴·max²`.
const EVALUATION_UNDERFLOW_ERROR: f64 = 64.0 * MIN_SUBNORMAL;

/// The absolute floor for subnormal differences, scaled by the greatest entry cubed.
///
/// When a rounded difference `mᵢⱼ` is subnormal or zero, the relative bound in
/// [`ROUNDING_ERROR`] fails: its absolute error is at most `2⁻¹⁰⁷⁵`. Replacing one
/// entry perturbs a determinant term by that much times the product of the other three
/// entries, a term contains at most four subnormal entries, and there are 24 terms:
/// `4·24·2⁻¹⁰⁷⁵·max³ = 48·2⁻¹⁰⁷⁴·max³`. We keep a margin for the rounding of max³ too.
const DIFFERENCE_UNDERFLOW_ERROR: f64 = 256.0 * MIN_SUBNORMAL;

/// The absolute floor for matrices so tiny that the scaled floors above underflow to 0.
const UNDERFLOW_ERROR: f64 = 32.0 * MIN_SUBNORMAL;

/// The largest span of coordinate exponents the exact orientation predicate can handle.
///
/// Every nonzero coordinate is `m · 2^e` with `|m| ≤ 2⁵³ - 1 < 2⁵³`. Scaling the
/// coordinates to a common exponent can raise the magnitude by at most `2^span`, and
/// differencing two coordinates can double it, so a matrix entry is bounded by `|d| <
/// 2^(54 + span)`. The determinant terms and their sum then follow from the exponents:
///
/// ```text
/// λ = product of four entries < (2^(54 + span))⁴ = 2^(4·54 + 4·span) = 2^(216 + 4·span)
/// |det| ≤ Σ₂₄ |λ| < 2⁵ · 2^(216 + 4·span) = 2^(216 + 5 + 4·span) = 2^(221 + 4·span)
/// ```
///
/// with `216 = 4·54` the four-entry product exponent and `221 = 216 + 5` the sum
/// exponent. Requiring that bound to fit the signed 512-bit range, `|x| ≤ 2⁵¹¹` gives
/// `221 + 4·span ≤ 511`, i.e. `span ≤ ⌊290/4⌋ = 72`: a dynamic range of 72 powers of 2
///
/// The same argument bounds every intermediate of the Laplace evaluation (each is a
/// partial sum of at most 24 terms bounded as above), so with this budget enforced the
/// [`I512`] arithmetic of the exact stage cannot overflow and is used unchecked.
pub const MAX_EXPONENT_SPAN: i32 = 72;

/// Compute the sign of the four-dimensional orientation determinant.
///
/// Returns `1` or `-1` when the fifth point lies strictly on either side of the
/// hyperplane through the other four, and `0` when all five points are hypercoplanar.
///
/// # Errors
///
/// Returns [`Error::NumericallyAmbiguousPolytope`] when the determinant
/// cannot be resolved exactly. See `MAX_EXPONENT_SPAN` for the allowed range
/// of values, and ensure your data does not contain subnormal or nonfinite
/// numbers.
///
/// # Example
///
/// ```
/// use hoomd_geometry::orient4d::orient4d;
/// use hoomd_vector::Cartesian;
///
/// let origin = Cartesian::from([0.0, 0.0, 0.0, 0.0]);
/// let x = Cartesian::from([1.0, 0.0, 0.0, 0.0]);
/// let y = Cartesian::from([0.0, 1.0, 0.0, 0.0]);
/// let z = Cartesian::from([0.0, 0.0, 1.0, 0.0]);
/// let w = Cartesian::from([0.0, 0.0, 0.0, 1.0]);
///
/// // The apex w lies off the hyperplane through o, x, y and z.
/// assert_eq!(orient4d(origin, x, y, z, w), Ok(1));
///
/// // A point in that hyperplane gives exactly zero.
/// let in_plane = Cartesian::from([1.0, 1.0, 0.0, 0.0]);
/// assert_eq!(orient4d(origin, x, y, z, in_plane), Ok(0));
/// ```
#[inline]
pub fn orient4d(
    pa: Cartesian<4>,
    pb: Cartesian<4>,
    pc: Cartesian<4>,
    pd: Cartesian<4>,
    pe: Cartesian<4>,
) -> Result<i64, Error> {
    if let Some(sign) = orient4d_filtered(&differences(pa, pb, pc, pd, pe)) {
        return Ok(sign);
    }

    orient4d_exact(pa, pb, pc, pd, pe)
}

/// Compute the exact sign of the four-dimensional orientation determinant.
///
/// The coordinates are converted to large integers scaled by a common power of two,
/// and the determinant is evaluated with [`I512`] arithmetic so the returned
/// sign is exact for the coordinates as stored. The budget of [`MAX_EXPONENT_SPAN`]
/// bounds every intermediate below the 512-bit range, so the arithmetic cannot overflow
///
/// # Errors
///
/// Returns [`Error::NumericallyAmbiguousPolytope`] when any coordinate is not finite,
/// or when the exponents of the coordinates span too many powers of two. This is
/// uncommon for normal geometries, but can occur with very large or very inaccurate
/// point sets.
#[inline]
pub fn orient4d_exact(
    pa: Cartesian<4>,
    pb: Cartesian<4>,
    pc: Cartesian<4>,
    pd: Cartesian<4>,
    pe: Cartesian<4>,
) -> Result<i64, Error> {
    let coordinates = decode_points(&[pa, pb, pc, pd, pe])?;

    // Zero mantissas do not constrain the common scale; if every coordinate
    // is zero, all five points coincide.
    let Some((min_exponent, max_exponent)) = exponent_range(&coordinates) else {
        return Ok(0);
    };
    if max_exponent - min_exponent > MAX_EXPONENT_SPAN {
        return Err(Error::NumericallyAmbiguousPolytope);
    }

    // Scale the coordinates exactly, then difference the points.
    let mut scaled = [[I512::ZERO; 4]; 5];
    for (row, coordinates) in scaled.iter_mut().zip(&coordinates) {
        for (entry, &coordinate) in row.iter_mut().zip(coordinates) {
            *entry = coordinate.scale_to(min_exponent);
        }
    }

    let mut matrix = [[I512::ZERO; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            matrix[i][j] = scaled[i][j] - scaled[4][j];
        }
    }

    Ok(det44i(&matrix).signum().as_i64())
}

/// Compute the sign of the determinant of a 4×4 matrix of differences, if f64 is safe.
///
/// The determinant is evaluated by a Laplace expansion, and its total error
/// relative to the determinant of the *exact* differences is bounded by
///
/// ```text
/// errbound = R·Λ̂ + E·max² + D·max³ + F
/// ```
///
/// with `Λ̂` the computed sum of the absolute values of the 24 determinant terms
/// (`lambda4`) and `max` the largest absolute entry of the matrix. `R`
/// ([`ROUNDING_ERROR`]) covers all rounding, of the evaluation and of the differences
/// alike; `E`, `D` and `F` are absolute floors for subnormal intermediates, subnormal
/// differences and fully subnormal matrices ([`EVALUATION_UNDERFLOW_ERROR`],
/// [`DIFFERENCE_UNDERFLOW_ERROR`], [`UNDERFLOW_ERROR`]). When `|det|` exceeds the
/// bound, its sign is certified; otherwise the caller must fall back to an exact
/// evaluation. `Λ̂` or a floor may overflow to infinity, which only inflates the bound.
#[inline]
fn orient4d_filtered(m: &Matrix44) -> Option<i64> {
    let det = m.determinant();
    if !det.is_finite() {
        return None;
    }

    let max = max_absolute_value(m);
    let errbound = ROUNDING_ERROR * permanent_abs(&m.rows)
        + EVALUATION_UNDERFLOW_ERROR * max.powi(2)
        + DIFFERENCE_UNDERFLOW_ERROR * max.powi(3)
        + UNDERFLOW_ERROR;

    if det > errbound {
        Some(1)
    } else if det < -errbound {
        Some(-1)
    } else {
        None
    }
}

/// The matrix of differences of four points from a fifth, by rows.
#[inline]
fn differences(
    pa: Cartesian<4>,
    pb: Cartesian<4>,
    pc: Cartesian<4>,
    pd: Cartesian<4>,
    pe: Cartesian<4>,
) -> Matrix44 {
    let row = |p: Cartesian<4>| (p - pe).coordinates;
    Matrix44 {
        rows: [row(pa), row(pb), row(pc), row(pd)],
    }
}

/// Permanent of the element-wise absolute value matrix.
///
/// For N = 3 this is the sum of the absolute values of the 6 determinant terms;
/// for N = 4 it is the sum of the absolute values of the 24 determinant terms.
#[inline]
fn permanent_abs<const N: usize>(m: &[[f64; N]; N]) -> f64 {
    #[inline]
    fn recurr<const N: usize>(m: &[[f64; N]; N], row: usize, used: u64) -> f64 {
        if row == N {
            return 1.0;
        }
        (0..N)
            .filter(|&j| used & (1 << j) == 0)
            .map(|j| m[row][j].abs() * recurr(m, row + 1, used | (1 << j)))
            .sum()
    }
    recurr(m, 0, 0)
}

/// The largest absolute entry of a 4×4 matrix.
#[inline]
fn max_absolute_value(m: &Matrix44) -> f64 {
    m.iter_elements()
        .fold(0.0_f64, |max, entry| max.max(entry.abs()))
}

/// Decompose every coordinate of five points into an exact dyadic rational.
///
/// # Errors
///
/// Returns [`Error::NumericallyAmbiguousPolytope`] when any coordinate is not finite.
fn decode_points(points: &[Cartesian<4>; 5]) -> Result<[[Dyad; 4]; 5], Error> {
    let mut decoded = [[Dyad::ZERO; 4]; 5];
    for (row, point) in decoded.iter_mut().zip(points) {
        for (entry, &value) in row.iter_mut().zip(&point.coordinates) {
            *entry = Dyad::try_from_f64(value).ok_or(Error::NumericallyAmbiguousPolytope)?;
        }
    }
    Ok(decoded)
}

/// The range of exponents of the nonzero mantissas, or [`None`] when all of them are 0.
fn exponent_range(coordinates: &[[Dyad; 4]; 5]) -> Option<(i32, i32)> {
    coordinates
        .iter()
        .flatten()
        .filter(|dyad| !dyad.is_zero())
        .map(|dyad| dyad.exponent())
        .fold(None, |range, exponent| {
            Some(match range {
                Some((min, max)) => (min.min(exponent), max.max(exponent)),
                None => (exponent, exponent),
            })
        })
}

/// Scale a dyadic coordinate to the common exponent of a point set.
trait ScaleTo {
    /// The value scaled by `2^(exponent - min_exponent)`, as an [`I512`].
    fn scale_to(self, min_exponent: i32) -> I512;
}

impl ScaleTo for Dyad {
    #[inline]
    fn scale_to(self, min_exponent: i32) -> I512 {
        if self.is_zero() {
            return I512::ZERO;
        }

        let shift = u32::try_from(self.exponent() - min_exponent)
            .expect("nonzero mantissas are at or above the minimum exponent");
        I512::from_i64(self.mantissa()) << shift
    }
}

/// The determinant of a 4x4 matrix of [`I512`] values, by Laplace expansion.
#[expect(clippy::many_single_char_names, reason = "clarity")]
fn det44i(mat: &[[I512; 4]; 4]) -> I512 {
    let [[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]] = *mat;

    a * (f * (k * p - l * o) - g * (j * p - l * n) + h * (j * o - k * n))
        - b * (e * (k * p - l * o) - g * (i * p - l * m) + h * (i * o - k * m))
        + c * (e * (j * p - l * n) - f * (i * p - l * m) + h * (i * n - j * m))
        - d * (e * (j * o - k * n) - f * (i * o - k * m) + g * (i * n - j * m))
}

#[cfg(test)]
mod tests {
    use super::*;

    const O: Cartesian<4> = Cartesian {
        coordinates: [0.0, 0.0, 0.0, 0.0],
    };
    const X: Cartesian<4> = Cartesian {
        coordinates: [1.0, 0.0, 0.0, 0.0],
    };
    const Y: Cartesian<4> = Cartesian {
        coordinates: [0.0, 1.0, 0.0, 0.0],
    };
    const Z: Cartesian<4> = Cartesian {
        coordinates: [0.0, 0.0, 1.0, 0.0],
    };
    const W: Cartesian<4> = Cartesian {
        coordinates: [0.0, 0.0, 0.0, 1.0],
    };

    #[test]
    fn sign_convention() {
        assert_eq!(orient4d(O, X, Y, Z, W), Ok(1));
    }

    #[test]
    fn transposing_two_points_flips_the_sign() {
        assert_eq!(orient4d(X, O, Y, Z, W), Ok(-1));
    }

    #[test]
    fn hypercoplanar_points_give_zero() {
        let in_plane = Cartesian::from([1.0, 1.0, 0.0, 0.0]);
        assert_eq!(orient4d(O, X, Y, Z, in_plane), Ok(0));
    }

    #[test]
    fn unresolvable_coordinates_error() {
        // An exponent span of 73 exceeds the exact stage's budget of 72.
        // (The prefilter may resolve this one, so query the exact stage.)
        let tiny = Cartesian::from([0.0, 0.0, 0.0, 2.0_f64.powi(-73)]);
        assert_eq!(
            orient4d_exact(O, X, Y, Z, tiny),
            Err(Error::NumericallyAmbiguousPolytope)
        );

        let nan = Cartesian::from([f64::NAN, 0.0, 0.0, 0.0]);
        assert_eq!(
            orient4d(O, X, Y, Z, nan),
            Err(Error::NumericallyAmbiguousPolytope)
        );
    }
}

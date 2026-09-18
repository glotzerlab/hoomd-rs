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
        I512::from_i128(self.mantissa() << shift)
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
    use crate::shape::ConvexPolyhedron;
    use assert2::check;
    use hoomd_vector::InnerProduct;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use rstest::*;

    /// Convert integer coordinates to points.
    fn to_point(x: &[i64; 4]) -> Cartesian<4> {
        Cartesian::from([x[0] as f64, x[1] as f64, x[2] as f64, x[3] as f64])
    }

    /// Convert a 5-tuple of integer coordinates to points.
    fn integer_arrs_to_cart(x: &[[i64; 4]; 5]) -> [Cartesian<4>; 5] {
        x.map(|v| to_point(&v))
    }

    /// Exact, simple (ish) orientation predicate for integer coordinates.
    fn orient4d_i128(points: &[[i64; 4]; 5]) -> i64 {
        let mut m = [[0_i128; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] = i128::from(points[i][j] - points[4][j]);
            }
        }

        let det3 = |m: &[[i128; 3]; 3]| -> i128 {
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        };

        let mut det = 0_i128;
        for j in 0..4 {
            let mut submatrix = [[0_i128; 3]; 3];
            for (i, row) in m.iter().enumerate().skip(1) {
                for (k, &entry) in row.iter().enumerate() {
                    if k != j {
                        let c = if k > j { k - 1 } else { k };
                        submatrix[i - 1][c] = entry;
                    }
                }
            }
            let term = m[0][j] * det3(&submatrix);
            det = if j % 2 == 0 { det + term } else { det - term };
        }

        det.signum() as i64
    }

    /// The 16 vertices of a tesseract.
    fn tesseract() -> Vec<[i64; 4]> {
        let mut points = Vec::with_capacity(16);
        for x in [-1, 1] {
            for y in [-1, 1] {
                for z in [-1, 1] {
                    for w in [-1, 1] {
                        points.push([x, y, z, w]);
                    }
                }
            }
        }
        points
    }

    /// The 8 vertices of a 16-cell.
    fn hexadecachoron() -> Vec<[i64; 4]> {
        let mut points = Vec::with_capacity(8);
        for axis in 0..4 {
            for sign in [-1, 1] {
                let mut p = [0_i64; 4];
                p[axis] = sign;
                points.push(p);
            }
        }
        points
    }

    /// The 24 vertices of a 24-cell: permutations of (+-1, +-1, 0, 0).
    fn icositetrachoron() -> Vec<[i64; 4]> {
        let mut points = Vec::with_capacity(24);
        for i in 0..4 {
            for j in (i + 1)..4 {
                for a in [-1_i64, 1] {
                    for b in [-1_i64, 1] {
                        let mut p = [0_i64; 4];
                        p[i] = a;
                        p[j] = b;
                        points.push(p);
                    }
                }
            }
        }
        points
    }

    /// The 5 vertices of a regular 5-cell.
    fn pentachoron() -> Vec<Cartesian<4>> {
        vec![
            Cartesian::from([1.0, 1.0, 1.0, 1.0]),
            Cartesian::from([1.0, -1.0, -1.0, 1.0]),
            Cartesian::from([-1.0, 1.0, -1.0, 1.0]),
            Cartesian::from([-1.0, -1.0, 1.0, 1.0]),
            Cartesian::from([0.0, 0.0, 0.0, 1.0 + f64::sqrt(5.0)]),
        ]
    }

    /// A standard normal variate (Box-Muller).
    fn gaussian(rng: &mut StdRng) -> f64 {
        let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        let u2 = rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// TODO: use ``DoubleVersor``
    /// A random orthonormal basis of R^4, by Gram-Schmidt over gaussian
    /// vectors. The basis vectors are the columns of the transformation.
    fn random_rotation(rng: &mut StdRng) -> [Cartesian<4>; 4] {
        let mut basis = [Cartesian::<4>::default(); 4];
        for i in 0..4 {
            loop {
                let mut v = Cartesian::from(std::array::from_fn(|_| gaussian(rng)));
                for previous in &basis[..i] {
                    v = v - *previous * v.dot(previous);
                }
                let norm_squared = v.norm_squared();
                if norm_squared > 1e-4 && norm_squared.is_finite() {
                    basis[i] = v / norm_squared.sqrt();
                    break;
                }
            }
        }
        basis
    }

    /// The sign of the determinant of a rotation basis (+1 or -1).
    fn rotation_sign(basis: &[Cartesian<4>; 4]) -> i64 {
        let det = Matrix44 {
            rows: std::array::from_fn::<_, 4, _>(|i| basis[i].coordinates),
        }
        .determinant();
        assert!(
            det.abs() > 0.9,
            "the sampled basis is not orthonormal (det {det})"
        );
        if det < 0.0 { -1 } else { 1 }
    }

    /// Apply a rotation given by columns to a point.
    fn rotate(basis: &[Cartesian<4>; 4], p: &[i64; 4]) -> Cartesian<4> {
        let q = to_point(p);
        basis[0] * q[0] + basis[1] * q[1] + basis[2] * q[2] + basis[3] * q[3]
    }

    #[rstest]
    fn test_axis_convention() {
        let points = integer_arrs_to_cart(&[
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]);

        check!(orient4d(points[0], points[1], points[2], points[3], points[4]) == Ok(1));
        check!(
            orient4d_i128(&[
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]) == 1
        );

        // A transposition of two points flips the sign.
        check!(orient4d(points[1], points[0], points[2], points[3], points[4]) == Ok(-1));

        // A point in the hyperplane of the first four gives exactly zero.
        let in_plane = Cartesian::from([1.0, 1.0, 0.0, 0.0]);
        check!(orient4d(points[0], points[1], points[2], points[3], in_plane) == Ok(0));
    }

    #[rstest]
    fn test_antisymmetry(#[values(0, 1, 2, 3)] seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);

        for _ in 0..200 {
            let points: [[i64; 4]; 5] = std::array::from_fn(|_| {
                std::array::from_fn(|_| i64::from(rng.random_range(-9_i32..=9)))
            });
            let cartesian = integer_arrs_to_cart(&points);
            let base = orient4d(
                cartesian[0],
                cartesian[1],
                cartesian[2],
                cartesian[3],
                cartesian[4],
            )
            .expect("integer coordinates are resolvable");

            // Fisher-Yates with a parity count.
            let mut permutation = [0_usize; 5];
            let identity = [0_usize, 1, 2, 3, 4];
            let mut order: Vec<usize> = identity.to_vec();
            let mut parity = 1_i64;
            for i in (1..5).rev() {
                let j = rng.random_range(0..=i);
                if i != j {
                    order.swap(i, j);
                    parity = -parity;
                }
            }
            permutation.copy_from_slice(&order);

            let permuted: [Cartesian<4>; 5] = std::array::from_fn(|i| cartesian[permutation[i]]);
            let sign = orient4d(
                permuted[0],
                permuted[1],
                permuted[2],
                permuted[3],
                permuted[4],
            )
            .expect("integer coordinates are resolvable");

            // The determinant is antisymmetric: the sign flips with the
            // parity of the permutation (and zero stays zero).
            if base == 0 {
                check!(sign == 0);
            } else {
                check!(sign == parity * base);
            }
        }
    }

    #[rstest]
    fn test_integer_oracle_stress(#[values(11, 22, 33, 44)] seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut zeros = 0;
        for _ in 0..10_000 {
            let raw: [[i64; 4]; 5] = std::array::from_fn(|_| {
                if rng.random::<f64>() < 0.05 {
                    // Occasionally force hyper-coplanarity (w = 0).
                    std::array::from_fn(|j| {
                        if j == 3 {
                            0
                        } else {
                            i64::from(rng.random_range(-16_i32..=16))
                        }
                    })
                } else {
                    std::array::from_fn(|_| i64::from(rng.random_range(-16_i32..=16)))
                }
            });
            let mut points = raw;
            if rng.random::<f64>() < 0.1 {
                // Occasionally duplicate a point (exact zero orientation).
                points[4] = points[0];
            }

            let oracle = orient4d_i128(&points);
            let cartesian = integer_arrs_to_cart(&points);
            let sign = orient4d(
                cartesian[0],
                cartesian[1],
                cartesian[2],
                cartesian[3],
                cartesian[4],
            )
            .expect("integer coordinates are in range");

            check!(sign == oracle);
            if oracle == 0 {
                zeros += 1;
            }
        }
        // The stress set contains both zero and nonzero cases.
        check!(zeros > 100);
    }

    #[rstest]
    fn test_exact_hyper_coplanar() {
        // Five vertices of one cube facet of the tesseract (x = 1) are
        // hyper-coplanar.
        let facet: Vec<[i64; 4]> = tesseract().into_iter().filter(|p| p[0] == 1).collect();
        assert_eq!(facet.len(), 8);
        for subset in subsets_of_5(&facet) {
            let p = integer_arrs_to_cart(&subset);
            check!(
                orient4d(p[0], p[1], p[2], p[3], p[4]) == Ok(0),
                "facet subset {subset:?}"
            );
        }

        // Five vertices of one octahedral facet of the 24-cell
        // (x0 + x1 + x2 + x3 = 2, all nonnegative) are hyper-coplanar.
        let facet: Vec<[i64; 4]> = icositetrachoron()
            .into_iter()
            .filter(|p| p.iter().all(|&x| x >= 0) && p.iter().sum::<i64>() == 2)
            .collect();
        assert_eq!(facet.len(), 6);
        for subset in subsets_of_5(&facet) {
            let p = integer_arrs_to_cart(&subset);
            check!(orient4d(p[0], p[1], p[2], p[3], p[4]) == Ok(0));
        }

        // An exact affine combination of four points lies in their span.
        let base = [
            [3_i64, 1, -2, 5],
            [-4, 7, 1, 1],
            [2, -3, 6, -7],
            [5, 5, 5, 5],
        ];
        // The average of the four points is an exact affine combination
        // (integer sums and division by four are exact in f64).
        let combination =
            (to_point(&base[0]) + to_point(&base[1]) + to_point(&base[2]) + to_point(&base[3]))
                / 4.0;
        let p: Vec<Cartesian<4>> = base.iter().map(to_point).chain([combination]).collect();
        let p: [Cartesian<4>; 5] = p.try_into().expect("five points");
        check!(orient4d(p[0], p[1], p[2], p[3], p[4]) == Ok(0));
    }

    /// All 5-element subsets of a small point set.
    fn subsets_of_5(points: &[[i64; 4]]) -> Vec<[[i64; 4]; 5]> {
        let n = points.len();
        let mut subsets = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                for c in (b + 1)..n {
                    for d in (c + 1)..n {
                        for e in (d + 1)..n {
                            subsets.push([points[a], points[b], points[c], points[d], points[e]]);
                        }
                    }
                }
            }
        }
        subsets
    }

    #[rstest]
    fn test_rotated_hyperplatonic(#[values(101, 202, 303, 404)] seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut cases = 0;
        for _ in 0..100 {
            let basis = random_rotation(&mut rng);
            let rotation = rotation_sign(&basis);

            for vertices in [tesseract(), hexadecachoron(), icositetrachoron()] {
                // Random 5-subsets of the vertices.
                for _ in 0..16 {
                    let mut subset: Vec<[i64; 4]> = Vec::with_capacity(5);
                    while subset.len() < 5 {
                        let i = rng.random_range(0..vertices.len());
                        let candidate = vertices[i];
                        if !subset.contains(&candidate) {
                            subset.push(candidate);
                        }
                    }
                    let integer_points: [[i64; 4]; 5] = subset.try_into().expect("five points");

                    let axis = orient4d_i128(&integer_points);
                    let rotated: [Cartesian<4>; 5] =
                        std::array::from_fn(|i| rotate(&basis, &integer_points[i]));
                    let sign = orient4d(rotated[0], rotated[1], rotated[2], rotated[3], rotated[4])
                        .expect("rotated coordinates are in range");

                    if axis == 0 {
                        // The subset is hyper-coplanar: after a generic
                        // rotation the determinant is dominated by the
                        // roundoff of the rotated coordinates, so only
                        // antisymmetry is guaranteed; skip.
                        continue;
                    }
                    cases += 1;
                    check!(sign == rotation * axis);
                }
            }
        }
        check!(cases > 2500, "only {cases} non-degenerate cases");
    }

    #[rstest]
    fn test_rotated_pentachoron(#[values(7, 8)] seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let vertices = pentachoron();

        let base = orient4d(
            vertices[0],
            vertices[1],
            vertices[2],
            vertices[3],
            vertices[4],
        )
        .expect("the 5-cell spans four dimensions");
        check!(base != 0, "the 5-cell must span four dimensions");

        for _ in 0..200 {
            let basis = random_rotation(&mut rng);
            let rotation = rotation_sign(&basis);
            let rotated: [Cartesian<4>; 5] = std::array::from_fn(|i| {
                basis[0] * vertices[i][0]
                    + basis[1] * vertices[i][1]
                    + basis[2] * vertices[i][2]
                    + basis[3] * vertices[i][3]
            });

            let sign = orient4d(rotated[0], rotated[1], rotated[2], rotated[3], rotated[4])
                .expect("the rotated 5-cell is resolvable");
            check!(sign == rotation * base);
        }
    }

    #[rstest]
    fn test_supported_dynamic_range(#[values(0, 1, 2)] seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);

        for _ in 0..1000 {
            // Log-uniform magnitudes in [1e-16, 1e3] with random signs: the
            // full range the exact stage is expected to support.
            let points: [Cartesian<4>; 5] = std::array::from_fn(|_| {
                Cartesian::from(std::array::from_fn(|_| {
                    let magnitude = 10.0_f64.powf(rng.random_range(-16.0..3.0)).max(1e-16);
                    let sign = if rng.random::<f64>() < 0.5 { -1.0 } else { 1.0 };
                    sign * magnitude
                }))
            });

            let filtered = orient4d(points[0], points[1], points[2], points[3], points[4])
                .expect("the dynamic range is supported");
            let exact = orient4d_exact(points[0], points[1], points[2], points[3], points[4])
                .expect("the dynamic range is supported");
            check!(filtered == exact);
        }
    }

    #[rstest]
    fn test_scale_and_translation_invariance(#[values(5, 6)] seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let points: [[i64; 4]; 5] = std::array::from_fn(|_| {
            std::array::from_fn(|_| i64::from(rng.random_range(-9_i32..=9)))
        });
        let base = orient4d_i128(&points);

        for scale in [2.0_f64.powi(-60), 1e-6, 1.0, 1e6, 2.0_f64.powi(60)] {
            for offset in [0.0, 1.0, 1e3, 1e5] {
                // An offset much larger than the scaled geometry rounds the
                // points together in the coordinates; such combinations do
                // not represent the translated point set.
                if offset > scale * 1e10 {
                    continue;
                }
                let offset = Cartesian::from([offset, -offset, offset, offset]);
                let scaled: [Cartesian<4>; 5] =
                    std::array::from_fn(|i| to_point(&points[i]) * scale + offset);

                let sign = orient4d(scaled[0], scaled[1], scaled[2], scaled[3], scaled[4])
                    .expect("scaled and translated coordinates are in range");
                check!(sign == base);
            }
        }
    }

    #[rstest]
    fn test_unresolvable_configurations() {
        // The exponents of 1.0 and 2^-73 span 73 powers of two.
        let points = integer_arrs_to_cart(&[
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ]);
        let tiny = Cartesian::from([0.0, 0.0, 0.0, 2.0_f64.powi(-73)]);
        check!(
            orient4d_exact(points[0], points[1], points[2], points[3], tiny)
                == Err(Error::NumericallyAmbiguousPolytope)
        );

        // A span of exactly 72 is supported.
        let tiny = Cartesian::from([0.0, 0.0, 0.0, 2.0_f64.powi(-72)]);
        check!(orient4d_exact(points[0], points[1], points[2], points[3], tiny) == Ok(1));

        // Non-finite coordinates.
        let infinite = Cartesian::from([f64::INFINITY, 0.0, 0.0, 0.0]);
        check!(
            orient4d(points[0], points[1], points[2], points[3], infinite)
                == Err(Error::NumericallyAmbiguousPolytope)
        );
        let nan = Cartesian::from([f64::NAN, 0.0, 0.0, 0.0]);
        check!(
            orient4d(points[0], points[1], points[2], points[3], nan)
                == Err(Error::NumericallyAmbiguousPolytope)
        );

        // An exactly hyper-coplanar set whose coordinates span more than 72
        // powers of two: the prefilter cannot certify zero and the exact
        // stage must decline.
        let wide = [
            Cartesian::from([0.0, 0.0, 0.0, 0.0]),
            Cartesian::from([1.0, 0.0, 0.0, 0.0]),
            Cartesian::from([0.0, 2.0_f64.powi(-80), 0.0, 0.0]),
            Cartesian::from([0.0, 0.0, 2.0_f64.powi(-80), 0.0]),
            Cartesian::from([2.0_f64.powi(-80), 2.0_f64.powi(-80), 0.0, 0.0]),
        ];
        check!(
            orient4d(wide[0], wide[1], wide[2], wide[3], wide[4])
                == Err(Error::NumericallyAmbiguousPolytope)
        );
    }

    #[rstest]
    fn test_prefilter_consistency(#[values(17, 18, 19)] seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut resolved = 0;
        let mut total = 0;
        for _ in 0..2000 {
            let points: [Cartesian<4>; 5] = std::array::from_fn(|_| {
                Cartesian::from(std::array::from_fn(|_| rng.random_range(-9.0_f64..9.0)))
            });

            let m = differences(points[0], points[1], points[2], points[3], points[4]);
            let exact = orient4d_exact(points[0], points[1], points[2], points[3], points[4])
                .expect("generic coordinates are in range");

            if let Some(sign) = orient4d_filtered(&m) {
                // A certified prefilter result must agree with the exact one.
                check!(sign == exact);
                resolved += 1;
            }
            total += 1;
        }

        // The filter resolves the overwhelming majority of generic inputs.
        let rate = f64::from(resolved) / f64::from(total);
        check!(rate > 0.95, "the prefilter resolved only {rate:.3}");
    }
    /// Classification of how a predicate call was resolved.
    #[derive(Debug, Default, Clone, Copy)]
    struct ResolutionStats {
        total: usize,
        /// Resolved by the f64 prefilter's error bound.
        certified: usize,
        /// Resolved by the exact 512-bit fallback.
        fallback: usize,
        /// Not resolved at all.
        errors: usize,
        /// Certified by the prefilter with the wrong sign (must never occur).
        mismatches: usize,
        /// Fallback cases where the naive f64 determinant is exactly zero
        /// and the configuration is truly degenerate: the naive predicate
        /// would be right by luck.
        fallback_zero_ok: usize,
        /// Fallback cases where the naive f64 sign happens to equal the
        /// exact sign: the naive predicate would be right by luck.
        fallback_naive_ok: usize,
        /// Fallback cases where the naive f64 sign differs from the exact
        /// sign: the naive predicate would be wrong.
        fallback_naive_wrong: usize,
    }

    impl ResolutionStats {
        /// Resolve one configuration through both stages, classifying the
        /// outcome and cross-checking the prefilter against the exact one.
        fn resolve(&mut self, points: &[Cartesian<4>; 5]) -> i64 {
            self.total += 1;

            let m = differences(points[0], points[1], points[2], points[3], points[4]);
            let naive = m.determinant();
            let certified = orient4d_filtered(&m);
            let exact = orient4d_exact(points[0], points[1], points[2], points[3], points[4])
                .expect("the configuration is resolvable");

            if let Some(sign) = certified {
                self.certified += 1;
                if sign != exact {
                    self.mismatches += 1;
                }
            } else {
                self.fallback += 1;
                let naive_sign = if naive > 0.0 {
                    1_i64
                } else if naive < 0.0 {
                    -1
                } else {
                    0
                };
                if exact == 0 {
                    self.fallback_zero_ok += usize::from(naive_sign == 0);
                    self.fallback_naive_wrong += usize::from(naive_sign != 0);
                } else if naive_sign == exact {
                    self.fallback_naive_ok += 1;
                } else {
                    self.fallback_naive_wrong += 1;
                }
            }

            exact
        }
    }

    /// Print the header line of a resolution sweep.
    #[allow(
        clippy::print_stdout,
        reason = "the sweep prints its resolution statistics"
    )]
    fn print_sweep_header() {
        println!(
            "{:<10} {:>10} {:>10} {:>10} {:>10}",
            "delta", "certified", "fallback", "mismatches", "errors"
        );
    }

    /// Print the summary line of a resolution sweep.
    #[allow(
        clippy::print_stdout,
        reason = "the sweep prints its resolution statistics"
    )]
    fn print_sweep_row(
        delta: f64,
        certified: usize,
        fallback: usize,
        mismatches: usize,
        errors: usize,
    ) {
        println!("{delta:<10.0e} {certified:>10} {fallback:>10} {mismatches:>10} {errors:>10}");
    }

    /// Print the summary of a facet resolution run.
    #[allow(
        clippy::print_stdout,
        reason = "the test prints its resolution statistics"
    )]
    fn print_facet_stats(stats: &ResolutionStats) {
        println!("facet (rotated hypercube): {stats:?}");
    }

    #[rstest]
    fn test_facet_resolution(#[values(0x51, 0x52, 0x53)] seed: u64) {
        // One facet of a randomly oriented hypercube: eight exactly coplanar
        // vertices whose coplanarity is broken by the rounding of the
        // rotation (and later by any translation). The predicate cannot
        // certify such tiny determinants with the f64 prefilter and must
        // fall back to the exact stage, which always resolves them.
        let tesseract_i = tesseract();
        let facet_indices: Vec<usize> = (0..16).filter(|&i| tesseract_i[i][0] == 1).collect();

        let mut rng = StdRng::seed_from_u64(seed);
        let mut stats = ResolutionStats::default();

        for _ in 0..64 {
            let basis = random_rotation(&mut rng);
            let rotated: Vec<Cartesian<4>> =
                tesseract_i.iter().map(|p| rotate(&basis, p)).collect();

            // All five-point subsets of the eight facet vertices.
            let n = facet_indices.len();
            for a in 0..n {
                for b in (a + 1)..n {
                    for c in (b + 1)..n {
                        for d in (c + 1)..n {
                            for e in (d + 1)..n {
                                let subset = [
                                    rotated[facet_indices[a]],
                                    rotated[facet_indices[b]],
                                    rotated[facet_indices[c]],
                                    rotated[facet_indices[d]],
                                    rotated[facet_indices[e]],
                                ];
                                stats.resolve(&subset);
                            }
                        }
                    }
                }
            }
        }

        print_facet_stats(&stats);

        // The exact stage always resolves facet configurations within the
        // supported range, and the prefilter is sound when it certifies.
        check!(stats.errors == 0);
        check!(stats.mismatches == 0);
        // Facet determinants are rounding-scale, far below the error bound:
        // nearly every call falls back to exact arithmetic.
        check!(
            stats.fallback as f64 / stats.total as f64 > 0.95,
            "the fallback rate was only {stats:?}"
        );
    }

    #[rstest]
    fn test_facet_perturbation_sweep(#[values(0x61, 0x62)] seed: u64) {
        // The facet points of the unrotated tesseract are exactly coplanar;
        // perturbing them by `delta` produces determinants of order delta.
        // The prefilter's bound is relative to the sum of the absolute term
        // values, which scales the same way, so the certified fraction is
        // roughly delta-independent down to the representation limit: a
        // perturbation below half an ulp of the +-1 coordinates does not
        // survive rounding, leaving exactly coplanar stored points. This
        // sweep checks soundness at every magnitude, full resolution above
        // the representation limit, and full decline below it.
        let mut rng = StdRng::seed_from_u64(seed);
        let facet: Vec<Cartesian<4>> = tesseract()
            .into_iter()
            .filter(|p| p[0] == 1)
            .take(5)
            .map(|p| to_point(&p))
            .collect();

        print_sweep_header();

        for &delta in &[
            1e-18_f64, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-8, 1e-6, 1e-4,
        ] {
            let mut stats = ResolutionStats::default();
            for _ in 0..200 {
                let perturbed: [Cartesian<4>; 5] = std::array::from_fn(|i| {
                    let jitter =
                        Cartesian::from(std::array::from_fn(|_| gaussian(&mut rng) * delta));
                    facet[i] + jitter
                });
                stats.resolve(&perturbed);
            }

            print_sweep_row(
                delta,
                stats.certified,
                stats.fallback,
                stats.mismatches,
                stats.errors,
            );

            // The prefilter never mis-certifies, whatever the perturbation.
            check!(stats.mismatches == 0);
            check!(stats.errors == 0);

            if delta <= 1e-18 {
                // Below the representation limit the stored points are
                // exactly the coplanar ones: zero determinants, declined.
                check!(
                    stats.certified == 0,
                    "certified at delta {delta:.0e}: {stats:?}"
                );
            }
            if delta >= 1e-15 {
                // Above the representation limit the bound, scaling with the
                // determinant terms, resolves the majority.
                check!(
                    stats.certified * 2 > stats.total,
                    "only {} of {} certified at delta {delta:.0e}",
                    stats.certified,
                    stats.total
                );
            }
        }
    }
    /// One dodecahedral cell embedded in R^4 (all vertices in w = 0), as
    /// found in the 120-cell: twenty vertices, twelve pentagonal faces.
    fn embedded_dodecahedron() -> Vec<Cartesian<4>> {
        ConvexPolyhedron::dodecahedron()
            .vertices()
            .iter()
            .map(|v| Cartesian::from([v[0], v[1], v[2], 0.0]))
            .collect()
    }

    /// The twelve face normals of the canonical dodecahedron: sign
    /// combinations of the cyclic permutations of `(1, 0, phi)`.
    fn dodecahedron_face_normals() -> Vec<[f64; 3]> {
        let phi = f64::midpoint(1.0, f64::sqrt(5.0));
        let mut normals = Vec::with_capacity(12);
        for (x, y, z) in [(1.0, 0.0, phi), (phi, 1.0, 0.0), (0.0, phi, 1.0)] {
            // Only the two nonzero components are signed; negating the zero
            // component would duplicate normals.
            for s1 in [-1.0, 1.0] {
                for s2 in [-1.0, 1.0] {
                    let mut n = [x, y, z];
                    let mut k = 0;
                    for c in &mut n {
                        if *c != 0.0 {
                            *c *= [s1, s2][k];
                            k += 1;
                        }
                    }
                    normals.push(n);
                }
            }
        }
        normals
    }

    #[rstest]
    fn test_oriented_dodecahedral_faces() {
        let vertices = embedded_dodecahedron();

        // Each of the twelve pentagonal faces is the set of vertices
        // maximizing its supporting plane, has exactly five members, and
        // supports the solid: all other vertices lie strictly inside.
        let mut face_count = 0;
        let mut vertex_slots = 0;
        for normal in dodecahedron_face_normals() {
            let level: f64 = vertices
                .iter()
                .map(|v| v[0] * normal[0] + v[1] * normal[1] + v[2] * normal[2])
                .fold(f64::NEG_INFINITY, f64::max);

            let face: Vec<usize> = (0..vertices.len())
                .filter(|&i| {
                    vertices[i][0] * normal[0]
                        + vertices[i][1] * normal[1]
                        + vertices[i][2] * normal[2]
                        == level
                })
                .collect();
            check!(
                face.len() == 5,
                "face {normal:?} has {} vertices",
                face.len()
            );

            for &i in &face {
                for (k, v) in vertices.iter().enumerate() {
                    let side = v[0] * normal[0] + v[1] * normal[1] + v[2] * normal[2];
                    if k == i {
                        check!(side == level);
                    } else if !face.contains(&k) {
                        check!(side < level, "vertex {k} outside face {normal:?}");
                    }
                }
                let _ = i;
            }
            vertex_slots += face.len();

            // Orient the pentagon counter-clockwise around its outward
            // normal and check every consecutive triple turns the same way.
            let normal_length =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            let n = [
                normal[0] / normal_length,
                normal[1] / normal_length,
                normal[2] / normal_length,
            ];
            let helper = if n[0].abs() < 0.9 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let dot = helper[0] * n[0] + helper[1] * n[1] + helper[2] * n[2];
            let u = [
                helper[0] - dot * n[0],
                helper[1] - dot * n[1],
                helper[2] - dot * n[2],
            ];
            let u_length = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            let u = [u[0] / u_length, u[1] / u_length, u[2] / u_length];
            let v = [
                n[1] * u[2] - n[2] * u[1],
                n[2] * u[0] - n[0] * u[2],
                n[0] * u[1] - n[1] * u[0],
            ];

            let angle = |&i: &usize| -> f64 {
                let p = &vertices[i];
                let (x, y, z) = (p[0], p[1], p[2]);
                let (ru, rv) = (
                    x * u[0] + y * u[1] + z * u[2],
                    x * v[0] + y * v[1] + z * v[2],
                );
                f64::atan2(rv, ru)
            };
            let mut ordered = face.clone();
            ordered.sort_by(|a, b| angle(a).total_cmp(&angle(b)));

            for i in 0..5 {
                let (p, q, r) = (
                    &vertices[ordered[i]],
                    &vertices[ordered[(i + 1) % 5]],
                    &vertices[ordered[(i + 2) % 5]],
                );
                let cross = [
                    (q[1] - p[1]) * (r[2] - q[2]) - (q[2] - p[2]) * (r[1] - q[1]),
                    (q[2] - p[2]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[2] - q[2]),
                    (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0]),
                ];
                let out = cross[0] * n[0] + cross[1] * n[1] + cross[2] * n[2];
                check!(out > 0.0, "face {normal:?} is not convexly oriented");
            }

            // The five oriented face vertices are hyper-coplanar: the
            // predicate must return exactly zero, in any order.
            let ordered_points: [Cartesian<4>; 5] = std::array::from_fn(|i| vertices[ordered[i]]);
            check!(
                orient4d(
                    ordered_points[0],
                    ordered_points[1],
                    ordered_points[2],
                    ordered_points[3],
                    ordered_points[4],
                ) == Ok(0)
            );
            let reversed = ordered_points;
            let reversed = [
                reversed[0],
                reversed[2],
                reversed[1],
                reversed[4],
                reversed[3],
            ];
            check!(
                orient4d(
                    reversed[0],
                    reversed[1],
                    reversed[2],
                    reversed[3],
                    reversed[4],
                ) == Ok(0)
            );
            face_count += 1;
        }

        assert_eq!(face_count, 12, "expected 12 faces");
        // Twenty vertices, three faces per vertex: 60 vertex slots.
        assert_eq!(vertex_slots, 60, "vertex slots");

        // Every five-subset of the cell's twenty vertices is hyper-coplanar.
        let n = vertices.len();
        let mut subset_count = 0_usize;
        for a in 0..n {
            for b in (a + 1)..n {
                for c in (b + 1)..n {
                    for d in (c + 1)..n {
                        for e in (d + 1)..n {
                            let subset = [
                                vertices[a],
                                vertices[b],
                                vertices[c],
                                vertices[d],
                                vertices[e],
                            ];
                            check!(
                                orient4d(subset[0], subset[1], subset[2], subset[3], subset[4],)
                                    == Ok(0)
                            );
                            subset_count += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(subset_count, 15_504);
    }

    /// A random five-subset of a randomly rotated dodecahedral cell.
    fn rotated_dodecahedral_subset(cell: &[Cartesian<4>], rng: &mut StdRng) -> [Cartesian<4>; 5] {
        let basis = random_rotation(rng);
        let picked: Vec<usize> = {
            let mut indices: Vec<usize> = (0..cell.len()).collect();
            for i in (1..5).rev() {
                indices.swap(i, rng.random_range(0..=i));
            }
            indices.truncate(5);
            indices
        };
        std::array::from_fn(|k| {
            let p = cell[picked[k]];
            basis[0] * p[0] + basis[1] * p[1] + basis[2] * p[2] + basis[3] * p[3]
        })
    }

    #[rstest]
    #[allow(
        clippy::print_stdout,
        reason = "the test prints its resolution statistics"
    )]
    fn test_rotated_dodecahedral_cells(#[values(0x73, 0x74)] seed: u64) {
        // After a random rotation the cell's facets are no longer exactly
        // hyper-coplanar; the predicate must still agree with the exact
        // stage on every such configuration.
        let mut rng = StdRng::seed_from_u64(seed);
        let cell = embedded_dodecahedron();

        let mut stats = ResolutionStats::default();
        for _ in 0..2000 {
            let subset = rotated_dodecahedral_subset(&cell, &mut rng);
            stats.resolve(&subset);
        }

        check!(stats.mismatches == 0);
        check!(stats.errors == 0);
        print_rotated_cells_stats(&stats);
    }

    /// Print the summary of a rotated dodecahedral cell resolution run.
    #[allow(
        clippy::print_stdout,
        reason = "the test prints its resolution statistics"
    )]
    fn print_rotated_cells_stats(stats: &ResolutionStats) {
        println!("rotated dodecahedral cells: {stats:?}");
    }

    #[test]
    #[ignore = "extended stress: 10^7 cases in 100 batches of 10^5; run with `cargo test --release -- --ignored`"]
    #[allow(
        clippy::print_stdout,
        reason = "the stress test reports progress and final statistics"
    )]
    fn test_extended_stress_10m() {
        const BATCHES: usize = 100;
        const BATCH_SIZE: usize = 100_000;

        let mut rng = StdRng::seed_from_u64(0x00c0_ffee);
        let cell = embedded_dodecahedron();
        let tesseract_i = tesseract();

        let mut stats = ResolutionStats::default();
        let mut zeros = 0_usize;
        let mut oracle_checks = 0_usize;

        let mut basis = random_rotation(&mut rng);
        for batch in 0..BATCHES {
            if batch % 10 == 0 && batch > 0 {
                basis = random_rotation(&mut rng);
            }

            for case in 0..BATCH_SIZE {
                let pick = (case + batch) % 6;
                match pick {
                    // Random integer configuration, checked against the
                    // independent i128 oracle.
                    0..=2 => {
                        let integer: [[i64; 4]; 5] = std::array::from_fn(|_| {
                            std::array::from_fn(|_| i64::from(rng.random_range(-16_i32..=16)))
                        });
                        let oracle = orient4d_i128(&integer);
                        let p = integer_arrs_to_cart(&integer);
                        let sign = orient4d(p[0], p[1], p[2], p[3], p[4])
                            .expect("integer coordinates are in range");
                        check!(sign == oracle);
                        oracle_checks += 1;
                        stats.resolve(&p);
                        if sign == 0 {
                            zeros += 1;
                        }
                    }
                    // Exactly hyper-coplanar integer configurations (w = 0).
                    3 => {
                        let integer: [[i64; 4]; 5] = std::array::from_fn(|_| {
                            let mut p =
                                std::array::from_fn(|_| i64::from(rng.random_range(-16_i32..=16)));
                            p[3] = 0;
                            p
                        });
                        let p = integer_arrs_to_cart(&integer);
                        let sign = orient4d(p[0], p[1], p[2], p[3], p[4])
                            .expect("integer coordinates are in range");
                        check!(sign == 0);
                        zeros += 1;
                        stats.resolve(&p);
                        let _ = &stats;
                    }
                    // Five vertices of one facet of a randomly oriented
                    // tesseract.
                    4 => {
                        let rotated: [Cartesian<4>; 5] = std::array::from_fn(|i| {
                            let mut p = to_point(&tesseract_i[(i * 3 + batch) % 16]);
                            p[0] = 1.0;
                            basis[0] * p[0] + basis[1] * p[1] + basis[2] * p[2] + basis[3] * p[3]
                        });
                        stats.resolve(&rotated);
                    }
                    // Five vertices of a randomly rotated dodecahedral cell.
                    _ => {
                        if case % 1000 == 0 {
                            basis = random_rotation(&mut rng);
                        }
                        let subset = rotated_dodecahedral_subset(&cell, &mut rng);
                        stats.resolve(&subset);
                    }
                }
            }

            if (batch + 1) % 10 == 0 {
                println!(
                    "batch {}: {stats:?}, zeros {zeros}, oracle checks {oracle_checks}",
                    batch + 1
                );
            }
        }

        println!("final: {stats:?}, zeros {zeros}, oracle checks {oracle_checks}");

        check!(stats.total == BATCHES * BATCH_SIZE);
        check!(stats.mismatches == 0);
        check!(stats.errors == 0);
        check!(zeros > 1_000_000);
        check!(oracle_checks > 4_000_000);
    }
    #[rstest]
    fn test_dynamic_range_err_census(#[values(0xBEEF_u64, 0xBEEF + 1, 0xBEEF + 2)] seed: u64) {
        // Census of the exact stage across coordinate exponent spans. The
        // 512-bit budget supports spans up to 72 powers of two: spans at or
        // below the budget must never produce errors, and spans beyond it
        // must always be rejected.
        let mut rng = StdRng::seed_from_u64(seed);

        let mut total = [0_usize; 130];
        let mut errs = [0_usize; 130];
        let mut other_errs = 0_usize;

        for _ in 0..5000 {
            let span = rng.random_range(0..=110_usize);
            let emin = -64_i32;
            let points: [Cartesian<4>; 5] = std::array::from_fn(|_| {
                Cartesian::from(std::array::from_fn(|_| {
                    let sign = if rng.random::<f64>() < 0.5 { -1.0 } else { 1.0 };
                    let offset =
                        u32::try_from(rng.random_range(0..=span)).expect("the span fits u32");
                    let exponent = emin + i32::try_from(offset).expect("fits i32");
                    let magnitude = (1.0 + rng.random::<f64>()) * 2.0_f64.powi(exponent);
                    sign * magnitude
                }))
            });

            let decoded: Vec<Dyad> = points
                .iter()
                .flat_map(|p| {
                    p.coordinates
                        .iter()
                        .map(|&x| Dyad::try_from_f64(x).expect("finite"))
                })
                .collect();
            let (min_e, max_e) = decoded
                .iter()
                .filter(|dyad| !dyad.is_zero())
                .fold((i32::MAX, i32::MIN), |(a, b), dyad| {
                    (a.min(dyad.exponent()), b.max(dyad.exponent()))
                });
            let actual_span = if min_e > max_e {
                0
            } else {
                usize::try_from(max_e - min_e).expect("nonnegative")
            };

            let bin = actual_span.min(129);
            total[bin] += 1;
            match orient4d_exact(points[0], points[1], points[2], points[3], points[4]) {
                Ok(_) => {}
                Err(Error::NumericallyAmbiguousPolytope) => errs[bin] += 1,
                Err(_) => other_errs += 1,
            }
        }

        // Every configuration within the budget resolves; everything past it
        // is rejected; nothing produces a different error.
        for (span, (&count, &errors)) in total.iter().zip(errs.iter()).enumerate() {
            if span <= MAX_EXPONENT_SPAN as usize {
                check!(errors == 0, "errors at span {span}");
            } else {
                check!(
                    errors == count,
                    "span {span}: {} of {count} not rejected",
                    count - errors
                );
            }
        }
        check!(other_errs == 0);
        // The random spans exercise the whole supported envelope.
        let in_budget: usize = total[..=MAX_EXPONENT_SPAN as usize].iter().sum();
        check!(in_budget > 2_000, "only {in_budget} in-budget samples");
    }
    #[rstest]
    fn test_zero_coordinates_large_scale(#[values(0_u32, 1, 30, 52, 53, 71, 72)] exponent: u32) {
        // A configuration whose nonzero coordinates all decode with
        // exponent `exponent` (values >= 2^52 + exponent > 2^53 for large
        // exponents) and which contains exact zero coordinates. The zeros
        // have no decoded scale of their own and must never be rejected
        // (regression test: with min_exponent > 0 the zero scaling used to
        // compute a negative shift and error out).
        let base = 2.0_f64.powi(i32::try_from(52 + exponent).expect("the exponent fits i32"));
        let points = [
            Cartesian::from([base, base, base, 0.0]),
            Cartesian::from([base, base, 0.0, base]),
            Cartesian::from([base, 0.0, base, base]),
            Cartesian::from([0.0, base, base, base]),
            Cartesian::from([base, base, base, base]),
        ];

        let exact = orient4d_exact(points[0], points[1], points[2], points[3], points[4])
            .expect("zero coordinates must not be rejected");

        // The four difference rows lie along the four coordinate axes: the
        // determinant is nonzero, and the two stages must agree.
        check!(exact != 0);
        check!(
            orient4d(points[0], points[1], points[2], points[3], points[4])
                .expect("zeros must not be rejected")
                == exact
        );
    }
    #[test]
    #[ignore = "exhaustive ternary sweep: 3^20 = 3,486,784,401 cases; set O4D_START/O4D_COUNT"]
    #[allow(
        clippy::print_stdout,
        reason = "the sweep prints its per-range statistics"
    )]
    fn test_exhaustive_ternary() {
        use std::hint::black_box;

        let total_cases = 3_u64.pow(20);
        let start = std::env::var("O4D_START")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        let count = std::env::var("O4D_COUNT")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(total_cases - start);
        let end = start.saturating_add(count).min(total_cases);

        let mut mismatches = 0_u64;
        let mut zeros = 0_u64;
        let mut sign_sum = 0_i128;
        let mut idx = start;

        while idx < end {
            // Decode the case index into 20 base-3 digits: point i,
            // coordinate j lives at digit 4i + j.
            let mut c = idx;
            let mut integer_points = [[0_i64; 4]; 5];
            let mut points: [Cartesian<4>; 5] = Default::default();
            for (row, point) in integer_points.iter_mut().enumerate() {
                for coordinate in point.iter_mut() {
                    // Base-3 digits are 0, 1 or 2: centered to -1, 0, 1.
                    #[allow(clippy::cast_possible_wrap, reason = "base-3 digits are 0, 1 or 2")]
                    {
                        *coordinate = (c % 3) as i64 - 1;
                    }
                    c /= 3;
                    let _ = black_box(*coordinate);
                }
                points[row] = Cartesian::from([
                    point[0] as f64,
                    point[1] as f64,
                    point[2] as f64,
                    point[3] as f64,
                ]);
            }

            let oracle = orient4d_i128(&integer_points);
            let sign = orient4d(points[0], points[1], points[2], points[3], points[4])
                .expect("ternary coordinates are in range");

            if sign != oracle {
                mismatches += 1;
            }
            if oracle == 0 {
                zeros += 1;
            }
            sign_sum += i128::from(sign);
            idx += 1;
        }

        println!(
            "range [{start}, {end}): {count} cases, {mismatches} mismatches, \
{zeros} degenerate, sign sum {sign_sum}"
        );
        let _ = count;
        assert_eq!(mismatches, 0);
        let _ = black_box(sign_sum);
    }
    #[rstest]
    fn test_hardcoded_constants() {
        // The hard-coded values are fixed by IEEE 754 binary64; pin them so
        // that neither the literals nor the doc-stated values drift.
        check!(MIN_SUBNORMAL == f64::from_bits(1));
        check!(MIN_SUBNORMAL == 5e-324);
        check!(UNIT_ROUNDOFF == 1.110_223_024_625_156_5e-16); // = georust EPSILON
        check!(roundoff_bound(1) == 1.110_223_024_625_156_8e-16);
        check!(ROUNDING_ERROR == 1.088_018_564_132_659_4e-14);
        check!(EVALUATION_UNDERFLOW_ERROR == 64.0 * MIN_SUBNORMAL);
        check!(DIFFERENCE_UNDERFLOW_ERROR == 256.0 * MIN_SUBNORMAL);
        check!(UNDERFLOW_ERROR == 32.0 * MIN_SUBNORMAL);
    }
}

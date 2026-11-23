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
//! [`orient4d`] returns the floating-point determinant's sign when its magnitude
//! exceeds the error bound. Otherwise, [`orient4d_exact`] scales the stored coordinates
//! to integers with a common power-of-two scale and computes the determinant using
//! [`I512`] arithmetic.
//!
//! The exact stage returns [`Error::NumericallyAmbiguousPolytope`] for nonfinite
//! coordinates or when the exponents of the nonzero coordinates differ by more
//! than [`MAX_EXPONENT_SPAN`]. The span is measured before subtracting points.

use crate::Error;
use hoomd_linear_algebra::matrix::Matrix44;
use hoomd_utility::dyad::Dyad;
use hoomd_vector::Cartesian;
use i256::I512;

/// The unit roundoff of f64, `2⁻⁵³`.
const UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;

/// The smallest positive subnormal f64, `2⁻¹⁰⁷⁴`.
const MIN_SUBNORMAL: f64 = f64::from_bits(1);

/// Roundoff bound relative to the sum of absolute determinant terms.
///
/// Use Higham's `γₖ = ku / (1 - ku)` with `k = 49`, allowing all 45 operations
/// of `Matrix44::determinant` plus four rounded coordinate differences per
/// determinant term. The factor of two allows for rounding in the computed sum.
/// See Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Lemma 3.1.
const ROUNDING_ERROR: f64 = 2.0 * (49.0 * UNIT_ROUNDOFF / (1.0 - 49.0 * UNIT_ROUNDOFF));

/// Underflow allowance for determinant intermediates, scaled by the largest entry squared.
///
/// Each error is at most half a subnormal spacing and can be multiplied by at
/// most two more entries. Counting shared minors twice gives 63 operations.
const EVALUATION_UNDERFLOW_ERROR: f64 = 64.0 * MIN_SUBNORMAL;

/// Underflow allowance for coordinate differences, scaled by the largest entry cubed.
///
/// The bound allows half a subnormal spacing for each of four factors in each
/// of the 24 determinant terms, with margin for rounding the scale factor.
const DIFFERENCE_UNDERFLOW_ERROR: f64 = 256.0 * MIN_SUBNORMAL;

/// Underflow allowance when all entries have magnitude below one.
const UNDERFLOW_ERROR: f64 = 32.0 * MIN_SUBNORMAL;

/// The largest span of coordinate exponents the exact orientation predicate can handle.
///
/// After scaling and subtraction, each entry has magnitude below `2^(54 + span)`
/// (53 mantissa bits, up to `span` shift bits, and one bit for subtraction).
/// Each intermediate is a sum of at most 24 products of up to four entries,
/// so its magnitude is below `2^(4 · (54 + span) + 5)`. Keeping that exponent at
/// most 511 gives a span of 72 and guarantees that [`I512`] arithmetic fits.
pub const MAX_EXPONENT_SPAN: i32 = 72;

/// Compute the sign of the four-dimensional orientation determinant.
///
/// Returns `1` or `-1` when the fifth point lies strictly on either side of the
/// hyperplane through the other four, and `0` when all five points are hypercoplanar.
///
/// # Errors
///
/// Returns [`Error::NumericallyAmbiguousPolytope`] when the filter is inconclusive
/// and the coordinates fall outside the range supported by [`orient4d_exact`].
///
#[inline]
pub fn orient4d(
    pa: Cartesian<4>,
    pb: Cartesian<4>,
    pc: Cartesian<4>,
    pd: Cartesian<4>,
    pe: Cartesian<4>,
) -> Result<i64, Error> {
    let differences = Matrix44 {
        rows: [pa, pb, pc, pd].map(|point| (point - pe).coordinates),
    };
    if let Some(sign) = orient4d_filtered(&differences) {
        return Ok(sign);
    }

    orient4d_exact(pa, pb, pc, pd, pe)
}

/// Compute the exact sign of the four-dimensional orientation determinant.
///
/// The coordinates are converted to large integers scaled by a common power of two,
/// and the determinant is evaluated with [`I512`] arithmetic so the returned
/// sign is exact for the coordinates as stored.
///
/// # Errors
/// Returns [`Error::NumericallyAmbiguousPolytope`] for nonfinite coordinates
/// or when the exponents of nonzero coordinates differ by more than [`MAX_EXPONENT_SPAN`].
#[inline]
pub fn orient4d_exact(
    pa: Cartesian<4>,
    pb: Cartesian<4>,
    pc: Cartesian<4>,
    pd: Cartesian<4>,
    pe: Cartesian<4>,
) -> Result<i64, Error> {
    let mut coordinates = [[Dyad::ZERO; 4]; 5];
    let values = [pa, pb, pc, pd, pe]
        .into_iter()
        .flat_map(|point| point.coordinates);
    let (min_exponent, max_exponent) = coordinates.iter_mut().flatten().zip(values).try_fold(
        (i32::MAX, i32::MIN),
        |(min, max), (coordinate, value)| -> Result<_, Error> {
            *coordinate = Dyad::try_from(value).map_err(|_| Error::NumericallyAmbiguousPolytope)?;
            if coordinate.is_zero() {
                return Ok((min, max));
            }
            let exponent = coordinate.exponent();
            Ok((min.min(exponent), max.max(exponent)))
        },
    )?;
    // Only nonzero coordinates constrain the scale. An empty range means all are zero.
    if min_exponent > max_exponent {
        return Ok(0);
    }
    if max_exponent - min_exponent > MAX_EXPONENT_SPAN {
        return Err(Error::NumericallyAmbiguousPolytope);
    }

    let scale = |coordinate: Dyad| {
        if coordinate.is_zero() {
            return I512::ZERO;
        }
        let shift = coordinate.exponent().abs_diff(min_exponent);
        I512::from_i64(coordinate.mantissa()) << shift
    };
    let [rows @ .., anchor] = coordinates;
    let anchor = anchor.map(scale);
    let matrix = rows.map(|row| std::array::from_fn(|j| scale(row[j]) - anchor[j]));

    Ok(det44i(&matrix).signum().as_i64())
}

/// Compute the sign of the determinant of a 4×4 matrix of differences, if f64 is safe.
///
/// A nonfinite determinant or a magnitude within the error bound requires exact evaluation.
#[inline]
fn orient4d_filtered(m: &Matrix44) -> Option<i64> {
    let det = m.determinant();
    if !det.is_finite() {
        return None;
    }

    let max = m
        .iter_elements()
        .fold(0.0_f64, |max, entry| max.max(entry.abs()));
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

/// Sum of the absolute values of the 24 determinant terms.
#[inline]
fn permanent_abs(m: &[[f64; 4]; 4]) -> f64 {
    #[inline]
    fn sum_terms(m: &[[f64; 4]; 4], row: usize, used: u64) -> f64 {
        if row == 4 {
            return 1.0;
        }
        (0..4)
            .filter(|&j| used & (1 << j) == 0)
            .map(|j| m[row][j].abs() * sum_terms(m, row + 1, used | (1 << j)))
            .sum()
    }
    sum_terms(m, 0, 0)
}

/// The determinant of a 4x4 matrix of [`I512`] values, by Laplace expansion.
#[expect(clippy::many_single_char_names, reason = "clarity")]
fn det44i(mat: &[[I512; 4]; 4]) -> I512 {
    let [[a, b, c, d], [e, f, g, h], [i, j, k, l], [m, n, o, p]] = *mat;

    // Each minor uses the bottom two rows. The subscripts are its column indices.
    let minor_23 = k * p - l * o;
    let minor_13 = j * p - l * n;
    let minor_12 = j * o - k * n;
    let minor_03 = i * p - l * m;
    let minor_02 = i * o - k * m;
    let minor_01 = i * n - j * m;

    a * (f * minor_23 - g * minor_13 + h * minor_12)
        - b * (e * minor_23 - g * minor_03 + h * minor_02)
        + c * (e * minor_13 - f * minor_03 + h * minor_01)
        - d * (e * minor_12 - f * minor_02 + g * minor_01)
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
        assert_eq!(orient4d_exact(X, O, Y, Z, W), Ok(-1));
    }

    #[test]
    fn hypercoplanar_points_give_zero() {
        let in_plane = Cartesian::from([1.0, 1.0, 0.0, 0.0]);
        assert_eq!(orient4d(O, X, Y, Z, in_plane), Ok(0));
    }

    #[test]
    fn exact_exponent_span_boundary() {
        let tiny = Cartesian::from([0.0, 0.0, 0.0, 2.0_f64.powi(-72)]);
        assert_eq!(orient4d_exact(O, X, Y, Z, tiny), Ok(1));

        let tiny = Cartesian::from([0.0, 0.0, 0.0, 2.0_f64.powi(-73)]);
        assert_eq!(
            orient4d_exact(O, X, Y, Z, tiny),
            Err(Error::NumericallyAmbiguousPolytope)
        );
    }

    #[test]
    fn nonfinite_coordinates_error() {
        let nan = Cartesian::from([f64::NAN, 0.0, 0.0, 0.0]);
        assert_eq!(
            orient4d(O, X, Y, Z, nan),
            Err(Error::NumericallyAmbiguousPolytope)
        );
    }
}

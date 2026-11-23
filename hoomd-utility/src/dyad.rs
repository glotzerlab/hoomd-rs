// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! An exact dyadic decomposition of floating point numbers.

use thiserror::Error;

/// Enumerate possible sources of error in the dyadic decomposition.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// A finite value is required.
    #[error("{0} is not finite")]
    NotFinite(f64),
}

/// An exact dyadic rational `mantissa · 2^exponent`.
///
/// Every finite f64 is exactly representable in this form, and [`Dyad::try_from`]
/// finds such a decomposition with `|mantissa| ≤ 2^53 - 1` (and `mantissa == 0` only
/// for zero), matching the 53-bit significand of the input.
///
/// # Example
///
/// ```
/// use hoomd_utility::dyad::Dyad;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let dyad = Dyad::try_from(16.0)?;
/// // 16 = 2^52 * 2^-48
/// assert_eq!(dyad.mantissa(), 1_i64 << 52);
/// assert_eq!(dyad.exponent(), -48);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Dyad {
    /// The integer significand.
    mantissa: i64,
    /// The power of two.
    exponent: i32,
}

/// Decompose a finite f64 into `mantissa · 2^exponent` exactly.
///
/// This dyad does not store signed zeros: `0.0` and `-0.0` both become [`Dyad::ZERO`].
///
/// # Errors
///
/// Returns [`Error::NotFinite`] when `value` is not finite.
impl TryFrom<f64> for Dyad {
    type Error = Error;

    #[inline]
    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if !value.is_finite() {
            return Err(Error::NotFinite(value));
        }

        let bits = value.to_bits();
        let biased = (bits >> 52) & 0x7ff;
        let fraction = bits & ((1_u64 << 52) - 1);

        // A normal value is (1 + fraction / 2^52) * 2^(biased - 1023), i.e.
        // (2^52 + fraction) * 2^(biased - 1075). A subnormal value has no
        // hidden bit and the minimum exponent; zero decodes to a zero mantissa
        let (mantissa, exponent) = if biased == 0 {
            (i64::try_from(fraction).expect("52 bits"), -1074)
        } else {
            (
                i64::try_from(fraction | (1_u64 << 52)).expect("53 bits"),
                i32::try_from(biased).expect("11 bits") - 1075,
            )
        };

        Ok(Self {
            mantissa: if bits >> 63 == 1 { -mantissa } else { mantissa },
            exponent,
        })
    }
}

impl Dyad {
    /// The dyadic zero, which has `0` mantissa.
    pub const ZERO: Self = Self {
        mantissa: 0,
        exponent: -1074,
    };

    /// The integer significand.
    ///
    /// `|mantissa| ≤ 2^53 - 1` for values from [`Dyad::try_from`], and zero for 0.
    #[must_use]
    #[inline]
    pub fn mantissa(self) -> i64 {
        self.mantissa
    }

    /// The power of two.
    #[must_use]
    #[inline]
    pub fn exponent(self) -> i32 {
        self.exponent
    }

    /// Whether the value is zero.
    #[must_use]
    #[inline]
    pub fn is_zero(self) -> bool {
        self.mantissa == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::check;

    /// Decompose selected values, including every regime of the encoding.
    #[test]
    fn test_try_from_f64() {
        let cases = [
            (0.0, 0_i64, -1074_i32),
            (-0.0, 0, -1074),
            (1.0, 1_i64 << 52, -52),
            (-1.0, -(1_i64 << 52), -52),
            (-6.0, -3 * (1_i64 << 51), -50),
            (f64::from_bits(1), 1, -1074),
            (f64::MIN_POSITIVE, 1_i64 << 52, -1074),
            (f64::MAX, (1_i64 << 53) - 1, 971),
            (f64::MIN, -((1_i64 << 53) - 1), 971),
        ];
        for (value, mantissa, exponent) in cases {
            let dyad = Dyad::try_from(value).expect("finite");
            check!(dyad.mantissa() == mantissa, "{value}");
            check!(dyad.exponent() == exponent, "{value}");
            check!(dyad.is_zero() == (value == 0.0), "{value}");
        }

        check!(Dyad::try_from(f64::INFINITY) == Err(Error::NotFinite(f64::INFINITY)));
        check!(Dyad::try_from(f64::NEG_INFINITY) == Err(Error::NotFinite(f64::NEG_INFINITY)));
        check!(matches!(Dyad::try_from(f64::NAN), Err(Error::NotFinite(_))));
        check!(Dyad::ZERO.is_zero());
        check!(Dyad::ZERO == Dyad::try_from(0.0).expect("finite"));
    }

    /// Every power of two decodes to the significand one, at its exponent.
    #[test]
    #[allow(clippy::cast_possible_truncation, reason = "part of the test")]
    fn test_powers_of_two() {
        for biased in 0..=2046_u64 {
            let value = f64::from_bits(biased << 52);
            let dyad = Dyad::try_from(value).expect("finite");
            if biased == 0 {
                check!(dyad.is_zero());
            } else {
                check!(dyad.mantissa() == 1_i64 << 52, "biased {biased}");
                check!(dyad.exponent() == biased as i32 - 1075, "biased {biased}");
            }
        }
    }
}

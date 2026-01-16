// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `PositiveReal`

use serde::{Deserialize, Serialize};
use std::{
    fmt,
    ops::{Div, DivAssign, Mul, MulAssign},
};

use crate::Error;

/// A f64 value that is not +/- inf, nan, or a value <= 0.
///
/// # Example
///
/// ```
/// use hoomd_utility::valid::PositiveReal;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let positive = PositiveReal::try_from(1.0)?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PositiveReal(f64);

impl PositiveReal {
    /// Access the value.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_utility::valid::PositiveReal;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let positive = PositiveReal::try_from(1.0)?;
    ///
    /// assert_eq!(positive.get(), 1.0);
    /// # Ok(())
    /// # }
    #[must_use]
    #[inline]
    pub fn get(&self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for PositiveReal {
    type Error = Error;

    /// Convert [`f64`] to [`PositiveReal`].
    ///
    /// # Example
    ///
    /// Valid conversion:
    /// ```
    /// use hoomd_utility::valid::PositiveReal;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let positive = PositiveReal::try_from(1.0)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Invalid conversion
    /// ```
    /// use hoomd_utility::valid::PositiveReal;
    ///
    /// let result = PositiveReal::try_from(-1.0);
    /// assert!(matches!(result, Err(hoomd_utility::Error::NotPositive(_))));
    /// ```
    ///
    /// # Errors
    ///
    /// `[Error::NotFinite]` when `v` is not finite.
    /// `[Error::NotPositive]` when `v` is not a positive value
    #[inline]
    fn try_from(v: f64) -> Result<PositiveReal, Error> {
        if !v.is_finite() {
            Err(Error::NotFinite(v))
        } else if v <= 0.0 {
            Err(Error::NotPositive(v))
        } else {
            Ok(PositiveReal(v))
        }
    }
}

impl Default for PositiveReal {
    /// The default value is 1.0.
    #[inline]
    fn default() -> Self {
        PositiveReal(1.0)
    }
}

impl fmt::Display for PositiveReal {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Mul for PositiveReal {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl MulAssign<PositiveReal> for PositiveReal {
    #[inline]
    fn mul_assign(&mut self, rhs: PositiveReal) {
        self.0 *= rhs.0;
    }
}

impl Div for PositiveReal {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self(self.0 / rhs.0)
    }
}

impl DivAssign<PositiveReal> for PositiveReal {
    #[inline]
    fn div_assign(&mut self, rhs: PositiveReal) {
        self.0 /= rhs.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::check;

    #[test]
    fn positive_real_validation() {
        let result = PositiveReal::try_from(f64::INFINITY);
        check!(result == Err(Error::NotFinite(f64::INFINITY)));

        let result = PositiveReal::try_from(-f64::INFINITY);
        check!(result == Err(Error::NotFinite(-f64::INFINITY)));

        let result = PositiveReal::try_from(f64::NAN);
        check!(matches!(result, Err(Error::NotFinite(_))));

        let result = PositiveReal::try_from(0.0);
        check!(result == Err(Error::NotPositive(0.0)));

        let result = PositiveReal::try_from(-1.0);
        check!(result == Err(Error::NotPositive(-1.0)));
    }
}

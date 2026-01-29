// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `OpenUnitIntervalNumber`

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::Error;

/// A f64 value in the interval (0,1)
///
/// # Example
///
/// ```
/// use hoomd_utility::valid::OpenUnitIntervalNumber;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let v = OpenUnitIntervalNumber::try_from(0.5)?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpenUnitIntervalNumber(f64);

impl OpenUnitIntervalNumber {
    /// Access the value.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_utility::valid::OpenUnitIntervalNumber;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v = OpenUnitIntervalNumber::try_from(0.5)?;
    ///
    /// assert_eq!(v.get(), 0.5);
    /// # Ok(())
    /// # }
    #[must_use]
    #[inline]
    pub fn get(&self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for OpenUnitIntervalNumber {
    type Error = Error;

    /// Convert [`f64`] to [`OpenUnitIntervalNumber`].
    ///
    /// # Example
    ///
    /// Valid conversion:
    /// ```
    /// use hoomd_utility::valid::OpenUnitIntervalNumber;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v = OpenUnitIntervalNumber::try_from(0.5)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Invalid conversion
    /// ```
    /// use hoomd_utility::valid::OpenUnitIntervalNumber;
    ///
    /// let result = OpenUnitIntervalNumber::try_from(2.0);
    /// assert!(matches!(
    ///     result,
    ///     Err(hoomd_utility::Error::NotInOpenUnitInterval(_))
    /// ));
    /// ```
    ///
    /// # Errors
    ///
    /// `[Error::NotFinite]` when `v` is not finite.
    /// `[Error::NotInOpenUnitInterval]` when `v` is not in (0,1).
    #[inline]
    fn try_from(v: f64) -> Result<OpenUnitIntervalNumber, Error> {
        if !v.is_finite() {
            Err(Error::NotFinite(v))
        } else if v <= 0.0 || v >= 1.0 {
            Err(Error::NotInOpenUnitInterval(v))
        } else {
            Ok(OpenUnitIntervalNumber(v))
        }
    }
}

impl fmt::Display for OpenUnitIntervalNumber {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::check;

    #[test]
    fn open_unit_validation() {
        let result = OpenUnitIntervalNumber::try_from(f64::INFINITY);
        check!(result == Err(Error::NotFinite(f64::INFINITY)));

        let result = OpenUnitIntervalNumber::try_from(-f64::INFINITY);
        check!(result == Err(Error::NotFinite(-f64::INFINITY)));

        let result = OpenUnitIntervalNumber::try_from(f64::NAN);
        check!(matches!(result, Err(Error::NotFinite(_))));

        let result = OpenUnitIntervalNumber::try_from(0.0);
        check!(matches!(result, Err(Error::NotInOpenUnitInterval(_))));

        let result = OpenUnitIntervalNumber::try_from(-1.0);
        check!(matches!(result, Err(Error::NotInOpenUnitInterval(_))));

        let result = OpenUnitIntervalNumber::try_from(1.0_f64.next_down());
        check!(result == Ok(OpenUnitIntervalNumber(1.0_f64.next_down())));
    }
}

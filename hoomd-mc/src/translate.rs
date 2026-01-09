// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Translate

use std::{fmt, marker::PhantomData};

use hoomd_utility::valid::PositiveReal;

use crate::Adjust;

mod cartesian;
mod hyperboloid;
mod sphere;

/// Move the position of a body by a small distance.
///
/// `Translate` proposes local trial moves that translate the position of a body
/// in space by up to a maximum distance, given by a [`PositiveReal`].
///
/// [`PositiveReal`]: hoomd_utility::valid::PositiveReal
///
/// The generic type names are:
/// * `P`: The type of the point to translate.
///
/// # Example
///
/// ```
/// use hoomd_mc::Translate;
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let d = 0.1;
/// let translate =
///     Translate::<Cartesian<2>>::with_maximum_distance(d.try_into()?);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Translate<P> {
    /// The maximum distance a body can be translated in one trial move.
    maximum_distance: PositiveReal,
    /// Mark the type of the point to be translated.
    marker: PhantomData<P>,
}

impl<P> Translate<P> {
    /// Construct a [`Translate`] move with the given maximum distance.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_mc::Translate;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let d = 0.1;
    /// let translate =
    ///     Translate::<Cartesian<2>>::with_maximum_distance(d.try_into()?);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn with_maximum_distance(maximum_distance: PositiveReal) -> Self {
        Self {
            maximum_distance,
            marker: PhantomData,
        }
    }

    /// Get the maximum distance.
    #[must_use]
    #[inline]
    pub fn maximum_distance(&self) -> &PositiveReal {
        &self.maximum_distance
    }

    /// Get the maximum distance.
    #[inline]
    pub fn maximum_distance_mut(&mut self) -> &mut PositiveReal {
        &mut self.maximum_distance
    }
}

impl<P> Adjust for Translate<P> {
    /// Change the maximum trial move size by the given scale factor.
    #[inline]
    fn adjust(&mut self, factor: PositiveReal) {
        self.maximum_distance *= factor;
    }
}

impl<P> fmt::Display for Translate<P> {
    /// Format a [`Translate`] as `{maximum_distance}`.
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.maximum_distance.fmt(f)
    }
}

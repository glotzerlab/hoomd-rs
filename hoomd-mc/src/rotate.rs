// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Rotate

use std::marker::PhantomData;

use hoomd_utility::valid::PositiveReal;

mod angle;

/// Change the orientation of a body by a small amount.
///
/// [`Rotate`] proposes local trial moves that rotate the orientation of a body
/// by a small amount. The [`maximum_rotation`] parameter sets the largest possible
/// rotation.
///
/// For the 2D [`Angle`], [`maximum_rotation`] is measured in radians and the
/// rotation is uniformly chosen between `-maximum_rotation` and `maximum_rotation`.
///
/// TODO: document 3d maximum.
///
/// [`Angle`]: hoomd_vector::Angle
/// [`maximum_rotation`]: Self::maximum_rotation
///
/// The generic type names are:
/// * `O`: The type of the orientation to rotate.
///
/// # Example
///
/// ```
/// use hoomd_mc::Rotate;
/// use hoomd_vector::Angle;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let a = 0.1;
/// let rotate = Rotate::<Angle>::with_maximum_rotation(a.try_into()?);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Rotate<O> {
    /// Limit the maximum rotation applied during a single trial move.
    maximum_rotation: PositiveReal,
    /// Mark the type of the orientation to be rotated.
    marker: PhantomData<O>,
}

impl<P> Rotate<P> {
    /// Construct a [`Rotate`] move with the given maximum rotation.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_mc::Rotate;
    /// use hoomd_vector::Angle;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let a = 0.1;
    /// let rotate = Rotate::<Angle>::with_maximum_rotation(a.try_into()?);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn with_maximum_rotation(maximum_rotation: PositiveReal) -> Self {
        Self {
            maximum_rotation,
            marker: PhantomData,
        }
    }

    /// Get the maximum rotation.
    #[must_use]
    #[inline]
    pub fn maximum_rotation(&self) -> &PositiveReal {
        &self.maximum_rotation
    }

    /// Get the maximum rotation.
    #[inline]
    pub fn maximum_rotation_mut(&mut self) -> &mut PositiveReal {
        &mut self.maximum_rotation
    }
}

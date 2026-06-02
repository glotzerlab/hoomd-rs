use crate::{Quaternion, Versor};

/// A pair of [`Versor`]s that represent a 4D rotation.
///
/// Each [`Versor`] represents an independent rotation about a plane in R^4. We assume
/// that the left part of the rotation is about the `XY` plane and the right part is
/// about the `ZW plane`.
pub struct DoubleVersor {
    /// The left-isoclinic part of the rotation.
    l: Versor,
    /// The right-isoclinic part of the rotation.
    r: Versor,
}

impl Default for DoubleVersor {
    /// Create an identity rotation.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::DoubleVersor;
    ///
    /// let v = DoubleVersor::default();
    /// ```
    #[inline]
    fn default() -> Self {
        Self {
            l: Versor::default(),
            r: Versor::default(),
        }
    }
}

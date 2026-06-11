use crate::{Cartesian, Quaternion, Rotate, RotationMatrix, Versor};

/// A pair of [`Versor`]s that represent a 4D rotation.
///
/// Each [`Versor`] represents an independent rotation about a plane in R^4.
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

impl From<(Versor, Versor)> for DoubleVersor {
    #[inline]
    fn from(value: (Versor, Versor)) -> Self {
        Self {
            l: value.0,
            r: value.1,
        }
    }
}

impl DoubleVersor {
    #[inline]
    #[must_use]
    /// Get the left-isoclinic part of the rotation.
    pub fn left_isoclinic(&self) -> Versor {
        self.l
    }
    #[inline]
    #[must_use]
    /// Get the right-isoclinic part of the rotation.
    pub fn right_isoclinic(&self) -> Versor {
        self.r
    }
    #[inline]
    #[must_use]
    /// Create a purely left-isoclinic double versor.
    pub fn from_left_isoclinic(&self, l: Versor) -> Self {
        Self {
            l,
            r: Versor::default(),
        }
    }
    #[inline]
    #[must_use]
    /// Create a purely right-isoclinic double versor.
    pub fn from_right_isoclinic(&self, r: Versor) -> Self {
        Self {
            l: Versor::default(),
            r,
        }
    }
}

impl Rotate<Cartesian<4>> for DoubleVersor {
    type Matrix = RotationMatrix<4>;

    /// Rotate a [`Cartesian<4>`] by a [`DoubleVersor`]
    ///
    /// ```math
    /// \mathbf{q} \vec{a} \mathbf{q}^*
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_vector::{Cartesian, Rotate, Rotation, DoubleVersor, Versor};
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let a = Cartesian::from([-1.0, 0.0, 0.0, 0.0]);
    /// let v = DoubleVersor::from_left_isoclinic(
    ///     Versor::from_axis_angle([0.0, 0.0, 1.0].try_into()?, PI / 2.0)
    /// );
    ///
    /// // Initializing from left isoclinic implies the right isoclinic is unit
    /// assert_eq!(v.right_isoclinic(), Versor::default());
    ///
    /// let b = v.rotate(&a);
    /// assert_relative_eq!(b, [0.0, -1.0, 0.0, 0.0].into());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn rotate(&self, vector: &Cartesian<4>) -> Cartesian<4> {
        todo!()
    }
}

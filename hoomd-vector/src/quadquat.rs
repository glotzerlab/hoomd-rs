//! Implement [`QuadQuaternion`], a representation of rotations in five dimensions.
//! Similar to [`Versor`] in 3D and [`DoubleVersor`] in 4D, this approach is more
//! numerically stable and space efficient than a 5x5 matrix representation.

use serde_with::{DeserializeAs, SerializeAs, serde_as};

use crate::{Cartesian, Quaternion, Rotate, RotationMatrix, Vector};

/// A unitary quaternion-valued matrix representing a rotation in SO(5).
///
/// Note that, in contrast to [`DoubleVersor`], the elements of this rotation may not
/// be unit quaternions. That said, the matrix is represented in such a way that all
/// valid [`QuadQuaternion`] structs are valid rotations in SO(5).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct QuadQuaternion {
    /// Rows of the quad-quaternion matrix.
    rows: [[Quaternion; 2]; 2],
}

impl Rotate<Cartesian<5>> for QuadQuaternion {
    type Matrix = RotationMatrix<5>;
    /// Rotate a [`Cartesian<5>`] by a [`QuadQuaternion`]
    ///
    /// ```math
    /// \mathbf{Q} \vec{a} \mathbf{Q}^\dagger
    /// ```
    fn rotate(&self, vector: &Cartesian<5>) -> Cartesian<5> {
        todo!()
    }
}

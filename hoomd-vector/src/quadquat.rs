//! Implement [`QuadQuaternion`], a representation of rotations in five dimensions.
//! Similar to [`Versor`] in 3D and [`DoubleVersor`] in 4D, this approach is more
//! numerically stable and space efficient than a 5x5 matrix representation.

use crate::{Cartesian, Quaternion, Rotate, Vector};

/// A Hermitian, traceless quaternion-valued matrix representing a rotation in five dimensions.
///
/// Note that, in contrast to [`DoubleVersor`], the elements of this rotation may not
/// be unit quaternions. That said, the matrix is represented in such a way that all
/// valid [`QuadQuaternion`] structs are valid rotations in SO(5).
pub(crate) struct QuadQuaternion {
    /// The real components of the constrained quaternions making up the matrix diagonal.
    diagonal: [f64; 2],
    /// The (fully unconstrained) quaternions making up the upper right and lower left entries.
    antidiagonal: [Quaternion; 2],
}

impl Rotate<Cartesian<5>> for QuadQuaternion {
    type Matrix = RotationMatrix<5>;
    /// Rotate a [`Cartesian<5>`] by a [`QuadQuaternion`]
    ///
    /// ```math
    /// \mathbf{M} \vec{a} \mathbf{M}^\dagger
    /// ```
    fn rotate(&self, vector: &Cartesian<5>) -> Cartesian<5> {
        todo!()
    }
}

//! Implement [`QuadQuaternion`], a representation of rotations in five dimensions.
//! Similar to [`Versor`] in 3D and [`DoubleVersor`] in 4D, this approach is more
//! numerically stable and space efficient than a 5x5 matrix representation.

use serde_with::{DeserializeAs, SerializeAs, serde_as};

use crate::{Cartesian, Quaternion, Rotate, RotationMatrix, Vector};

/// A unitary quaternion-valued matrix representing a rotation in SO(5).
///
///
/// All quaternions composing this matrix are subject to the following constraints:
/// ```math
/// \begin{aligned}
/// \|q_{00}\|^2 + \|q_{10}\|^2 &= 1 \\
/// \|q_{01}\|^2 + \|q_{11}\|^2 &= 1 \\
/// q_{00}^* q_{01} + q_{10}^* q_{11} &= 0
/// \end{aligned}
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct QuadQuaternion {
    /// Rows of the quad-quaternion matrix.
    rows: [[Quaternion; 2]; 2],
}

impl QuadQuaternion {
    /// Promote a [`Cartesian<5>`] to a hermitian traceless
    fn promote_vec5(v: Cartesian<5>) -> Self {
        let [p, np] = std::array::from_fn(|i| Quaternion {
            scalar: v[0] * if i == 0 { 1.0 } else { -1.0 },
            vector: [0.0; 3].into(),
        });
        Self {
            rows: [[p, q], [q.conjugate(), np]],
        }
    }
}

impl Rotate<Cartesian<5>> for QuadQuaternion {
    type Matrix = RotationMatrix<5>;
    /// Rotate a [`Cartesian<5>`] by a [`QuadQuaternion`]
    ///
    /// ```math
    /// \mathbf{Q} \vec{a} \mathbf{Q}^\dagger
    /// ```
    fn rotate(&self, vector: &Cartesian<5>) -> Cartesian<5> {
        // Promote a Cartesian<5> to the components of a Hermitian, traceless QuadQuat
        // [
        //  [x, q],
        //  [q^*, -x]
        // ]
        let (x, q) = (
            vector[0],
            Quaternion::from([vector[1], vector[2], vector[3], vector[4]]),
        );
        let q_conj = q.conjugate();
        todo!()
    }
}

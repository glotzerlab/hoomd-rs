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

    #[inline]
    pub(crate) fn a(&self) -> Quaternion {
        self.rows[0][0]
    }
    #[inline]
    pub(crate) fn b(&self) -> Quaternion {
        self.rows[0][1]
    }
    #[inline]
    pub(crate) fn c(&self) -> Quaternion {
        self.rows[1][0]
    }
    #[inline]
    pub(crate) fn d(&self) -> Quaternion {
        self.rows[1][1]
    }
}

impl Rotate<Cartesian<5>> for QuadQuaternion {
    type Matrix = RotationMatrix<5>;
    /// Rotate a [`Cartesian<5>`] by a [`QuadQuaternion`]
    ///
    /// ```math
    /// \mathbf{M} \vec{a} \mathbf{M}^\dagger
    /// ```
    fn rotate(&self, vector: &Cartesian<5>) -> Cartesian<5> {
        // Promote a Cartesian<5> to the components of a Hermitian, traceless QuadQuat
        // [
        //  [x, q],
        //  [q^*, -x]
        // ]
        let (m_00, m_01) = (
            vector[0],
            Quaternion::from([vector[1], vector[2], vector[3], vector[4]]),
        );
        let m_10 = m_01.conjugate();

        // Build the first row of the intermediate product Y = M @ V
        let [y_00, y_01] = [
            self.a() * m_00 + self.b() * m_10,
            self.a() * m_01 - self.b() * m_00,
        ];

        // Apply the right multiplication by M†
        let scalar_part = y_00 * self.a().conjugate() + y_01 * self.b().conjugate();
        let quaternion_part = y_00 * self.c().conjugate() + y_01 * self.d().conjugate();

        // Non-real components of the scalar part should be ~ 0
        (0..3).for_each(|i| debug_assert!(scalar_part.vector[i].abs() <= 1e-12));

        let (w, [x, y, z]) = (quaternion_part.scalar, quaternion_part.vector.coordinates);
        Cartesian::from([scalar_part.scalar, w, x, y, z])
    }
}

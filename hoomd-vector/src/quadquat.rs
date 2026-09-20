// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`QuadQuaternion`], a representation of rotations in five dimensions.
//! Similar to [`Versor`] in 3D and [`DoubleVersor`] in 4D, this approach is more
//! numerically stable and space efficient than the equivalent matrix representation,
//! but slower when applying rotations.

use std::fmt;

use approxim::{AbsDiffEq, RelativeEq};
use rand::{
    Rng, RngExt,
    distr::{Distribution, StandardUniform},
};
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

use crate::{Cartesian, Error, Metric, Quaternion, Rotate, Rotation, RotationMatrix, Versor};

/// The four components of the quaternion algebra as [`Quaternion`] values.
const QUATERNION_BASIS: [Quaternion; 4] = [
    crate::quaternion!(1.0, [0.0, 0.0, 0.0]),
    crate::quaternion!(0.0, [1.0, 0.0, 0.0]),
    crate::quaternion!(0.0, [0.0, 1.0, 0.0]),
    crate::quaternion!(0.0, [0.0, 0.0, 1.0]),
];

/// A unitary quaternion-valued matrix representing a rotation in SO(5).
///
/// All quaternions composing this matrix are subject to the following constraints:
/// ```math
/// \begin{aligned}
/// \|q_{00}\|^2 + \|q_{10}\|^2 &= 1 \\
/// \|q_{01}\|^2 + \|q_{11}\|^2 &= 1 \\
/// q_{00}^* q_{01} + q_{10}^* q_{11} &= 0
/// \end{aligned}
/// ```
// Reference: D. Haydys, "Holonomy groups in Riemannian geometry", Lecture 8, slide 10
// https://www.math.uni-bielefeld.de/~haydys/teaching/tcc11holonomy_files/lect08.pdf
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuadQuaternion {
    /// Rows of the quaternionic matrix.
    rows: [[Quaternion; 2]; 2],
}

impl AbsDiffEq for QuadQuaternion {
    type Epsilon = <Quaternion as AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        Quaternion::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.rows
            .iter()
            .flatten()
            .zip(other.rows.iter().flatten())
            .all(|(a, b)| a.abs_diff_eq(b, epsilon))
    }
}

impl RelativeEq for QuadQuaternion {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        Quaternion::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.rows
            .iter()
            .flatten()
            .zip(other.rows.iter().flatten())
            .all(|(a, b)| a.relative_eq(b, epsilon, max_relative))
    }
}

impl Default for QuadQuaternion {
    /// Create an identity rotation.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::QuadQuaternion;
    ///
    /// let q = QuadQuaternion::default();
    /// ```
    #[inline]
    fn default() -> Self {
        let one = *Versor::default().get();
        let zero = Quaternion::from([0.0; 4]);
        Self {
            rows: [[one, zero], [zero, one]],
        }
    }
}

impl TryFrom<[[Quaternion; 2]; 2]> for QuadQuaternion {
    type Error = Error;

    /// Create a [`QuadQuaternion`] by projecting the rows of a quaternionic matrix onto
    /// the manifold of unitary matrices.
    ///
    /// The projection is a quaternionic Gram-Schmidt orthonormalization of the
    /// matrix columns, following `Quaternion::to_versor`: like normalizing a
    /// single quaternion, it corrects nearly unitary input and only fails when
    /// the blocks do not determine a unitary matrix at all (a zero or linearly
    /// dependent column). Note that the projection is the Gram-Schmidt
    /// factorization, not the nearest unitary matrix (the polar factorization).
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::{Error, QuadQuaternion, Quaternion};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // The identity blocks are projected to the identity rotation.
    /// let one = Quaternion::from([1.0, 0.0, 0.0, 0.0]);
    /// let zero = Quaternion::from([0.0; 4]);
    /// let q = QuadQuaternion::try_from([[one, zero], [zero, one]])?;
    ///
    /// // Linearly dependent blocks do not span H^2 and cannot be projected.
    /// assert_eq!(
    ///     QuadQuaternion::try_from([[one, one], [zero, zero]]),
    ///     Err(Error::InvalidQuadQuaternionSpan)
    /// );
    /// # Ok(())
    /// # }
    /// ```
    // Reference: F. Mezzadri, "How to generate random matrices from the classical
    // compact groups", arxiv.org/abs/math-ph/0609050. Section 4
    #[inline]
    fn try_from(rows: [[Quaternion; 2]; 2]) -> Result<Self, Self::Error> {
        Self::project(rows).ok_or(Error::InvalidQuadQuaternionSpan)
    }
}

impl QuadQuaternion {
    /// Project the columns of a quaternionic matrix onto the unit sphere in `H^2`.
    ///
    /// Returns `None` when the columns are linearly dependent (the first column is zero
    /// or the second column lies in the span of the first), because then the blocks do
    /// not determine a unitary matrix.
    #[inline]
    fn project(rows: [[Quaternion; 2]; 2]) -> Option<Self> {
        let [[a, b], [c, d]] = rows;

        // Normalize the first column.
        let column_0_norm = (a.norm_squared() + c.norm_squared()).sqrt();
        if !(column_0_norm > 0.0 && column_0_norm.is_finite()) {
            return None;
        }
        let column_scale = 1.0 / column_0_norm;
        let (a, c) = (a * column_scale, c * column_scale);

        // Remove the component of the second column along the first, then notmalize:
        // <u, v - u<u, v>> = <u, v> - <u, u><u, v> = 0.
        let overlap = a.conjugate() * b + c.conjugate() * d;
        let b_orthogonal = b - a * overlap;
        let d_orthogonal = d - c * overlap;
        let column_1_norm = (b_orthogonal.norm_squared() + d_orthogonal.norm_squared()).sqrt();
        if !(column_1_norm > 0.0 && column_1_norm.is_finite()) {
            return None;
        }
        let new_scale = 1.0 / column_1_norm;

        Some(Self {
            rows: [[a, b_orthogonal * new_scale], [c, d_orthogonal * new_scale]],
        })
    }

    /// Promote a [`Cartesian<5>`] to a Hermitian traceless quaternionic matrix.
    #[inline]
    fn promote_vec5(v: Cartesian<5>) -> Self {
        let p = Quaternion::from([v[0], 0.0, 0.0, 0.0]);
        let np = Quaternion::from([-v[0], 0.0, 0.0, 0.0]);
        let q = Quaternion::from([v[1], v[2], v[3], v[4]]);
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

impl Distribution<QuadQuaternion> for StandardUniform {
    /// Sample a uniformly random 5D rotation (Haar measure on SO(5)).
    ///
    /// A [`QuadQuaternion`] is a 2×2 quaternionic unitary matrix, i.e. an element
    /// of Sp(2) ≅ Spin(5), the double cover of SO(5). Since `Q` and `-Q` act
    /// identically on a 5-vector, a Haar-distributed element of Sp(2) is exactly a
    /// uniformly random SO(5) rotation. This implements the subgroup algorithm
    /// specialized to Sp(2) ≅ S⁷ × S³ (Mezzadri 2007, "How to generate random
    /// matrices from the classical compact groups", §6–8; Diaconis & Shahshahani
    /// 1987), the direct generalization of the `Versor` sampler. No 5×5 matrix is
    /// constructed.
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> QuadQuaternion {
        // First column of the 2x2 quaternionic matrix, uniform on S^7 (the unit
        // sphere in H^2 ~ R^8): normalize 8 i.i.d. standard normals into the pair
        // (x1, x2).
        let v: [f64; 8] = std::array::from_fn(|_| rng.sample::<f64, _>(StandardNormal));
        let v_norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let x1 = Quaternion::from([v[0] / v_norm, v[1] / v_norm, v[2] / v_norm, v[3] / v_norm]);
        let x2 = Quaternion::from([v[4] / v_norm, v[5] / v_norm, v[6] / v_norm, v[7] / v_norm]);

        // Residual Sp(1) ~ S^3 freedom: a uniform unit quaternion (as for Versor).
        let q: [f64; 4] = std::array::from_fn(|_| rng.sample::<f64, _>(StandardNormal));
        let q_norm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        let q = Quaternion::from([q[0] / q_norm, q[1] / q_norm, q[2] / q_norm, q[3] / q_norm]);

        // Phase q1 of the first component x1 (Mezzadri eq 7.26). The -conj(q1)
        // prefactor on the Householder reflection below is mandatory for Haar
        // measure; without it the distribution is wrong.
        let x1_norm = x1.norm();
        let q1 = if x1_norm > 1e-12 {
            x1 * (1.0 / x1_norm)
        } else {
            // Measure-zero degenerate case: x1 ~ 0, so the direction is e1.
            Quaternion::from([1.0, 0.0, 0.0, 0.0])
        };

        // Householder vector u = normalize(v + q1*e1), with e1 = (1, 0) in H^2.
        let u0 = x1 + q1;
        let u1 = x2;
        let u_norm = (u0.norm_squared() + u1.norm_squared()).sqrt();
        let u0 = u0 * (1.0 / u_norm);
        let u1 = u1 * (1.0 / u_norm);

        // 2x2 quaternionic Householder reflection H2 = -conj(q1) * (I - 2 u u^*).
        let one = Quaternion::from([1.0, 0.0, 0.0, 0.0]);
        let mq1 = q1.conjugate() * -1.0;
        let h00 = mq1 * (one - (u0 * u0.conjugate()) * 2.0);
        let h01 = mq1 * ((u0 * u1.conjugate()) * -2.0);
        let h10 = mq1 * ((u1 * u0.conjugate()) * -2.0);
        let h11 = mq1 * (one - (u1 * u1.conjugate()) * 2.0);

        // S = H2^dagger * diag(1, q): conjugate-transpose, then right-multiply the
        // second column by q. The four blocks of S are the QuadQuaternion.
        let a = h00.conjugate();
        let b = h10.conjugate() * q;
        let c = h01.conjugate();
        let d = h11.conjugate() * q;

        QuadQuaternion {
            rows: [[a, b], [c, d]],
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::{assert_abs_diff_eq, assert_relative_eq};
    use hoomd_linear_algebra::matrix::Matrix;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    /// Build the 5x5 SO(5) matrix representing a [`QuadQuaternion`] by applying it
    /// to each Cartesian<5> basis vector.
    fn so5_matrix(q: &QuadQuaternion) -> [[f64; 5]; 5] {
        let mut m = [[0.0; 5]; 5];
        for col in 0..5 {
            let mut basis = [0.0; 5];
            basis[col] = 1.0;
            let rotated = q.rotate(&Cartesian::from(basis));
            for row in 0..5 {
                m[row][col] = rotated[row];
            }
        }
        m
    }

    #[test]
    fn identity_rotation_is_noop() {
        let identity = QuadQuaternion::default();
        let vector = Cartesian::from([1.0, 2.0, -3.0, 4.0, -5.0]);
        assert_relative_eq!(identity.rotate(&vector), vector);
    }

    #[test]
    fn random_rotation() {
        // Every random QuadQuaternion must act as a proper SO(5) rotation
        // (orthogonal, determinant +1), and the ensemble must be Haar-uniform.
        // Correctness is a per-sample property, so it is checked on a small prefix;
        // the Haar moments are statistical and use the full sample. Exact targets
        // are derived oracle-free from the Weyl eigenangle density and Weingarten
        // calculus (see /tmp/claude/haar_higher_moments.py for the cross-check).
        const SAMPLES: u32 = 40_000;
        const CORRECTNESS_CHECKS: u32 = 100;

        let mut rng = StdRng::seed_from_u64(1);

        let mut entry_sq = 0.0; // E[Mij^2]
        let mut entry_4th = 0.0; // E[Mij^4]
        let mut trace = 0.0; // E[Tr M]
        let mut trace_sq = 0.0; // E[Tr M^2]

        for i in 0..SAMPLES {
            let m = so5_matrix(&rng.random::<QuadQuaternion>());

            if i < CORRECTNESS_CHECKS {
                // Orthogonality: M^T M = I.
                for row in 0..5 {
                    for col in 0..5 {
                        let dot = (0..5).map(|k| m[k][row] * m[k][col]).sum::<f64>();
                        let target = if row == col { 1.0 } else { 0.0 };
                        assert_abs_diff_eq!(dot, target, epsilon = 1e-10);
                    }
                }
                // det M = +1.
                assert_abs_diff_eq!(Matrix { rows: m }.determinant(), 1.0, epsilon = 1e-9);
            }

            entry_sq += m[0][0] * m[0][0];
            entry_4th += m[0][0].powi(4);
            trace += (0..5).map(|d| m[d][d]).sum::<f64>();
            let mut tr2 = 0.0;
            for row in 0..5 {
                for col in 0..5 {
                    tr2 += m[row][col] * m[col][row];
                }
            }
            trace_sq += tr2;
        }

        let n = f64::from(SAMPLES);
        // E[Mij^2] = 1/N = 0.2.
        assert_abs_diff_eq!(entry_sq / n, 0.2, epsilon = 0.01);
        // E[Mij^4] = 3 / (N(N+2)) = 3/35.
        assert_abs_diff_eq!(entry_4th / n, 3.0 / 35.0, epsilon = 0.005);
        // E[Tr M] = 0 (odd spectral moment of Haar-SO(5)).
        assert_abs_diff_eq!(trace / n, 0.0, epsilon = 0.02);
        // E[Tr M^2] ~ 1.001 (exact from the SO(5) Weyl density).
        assert_abs_diff_eq!(trace_sq / n, 1.0, epsilon = 0.03);
    }
}

    #[test]
    fn serde_and_display() {
        let mut rng = StdRng::seed_from_u64(1);
        let q: QuadQuaternion = rng.random();

        let serialized = postcard::to_allocvec(&q).expect("serialization should succeed");
        let deserialized: QuadQuaternion =
            postcard::from_bytes(&serialized).expect("deserialization should succeed");
        assert_relative_eq!(deserialized, q);

        let formatted = format!("{q}");
        assert!(formatted.starts_with("[["));
        assert!(formatted.contains('\n'));
    }

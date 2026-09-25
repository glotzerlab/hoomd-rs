// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`QuadQuaternion`], a representation of rotations in five dimensions.
//! Similar to [`Versor`] in 3D and [`DoubleVersor`] in 4D, this approach is more
//! numerically stable and space efficient than the equivalent matrix representation,
//! but slower when applying rotations.

use std::{fmt, ops::Mul};

use approxim::{AbsDiffEq, RelativeEq};
use rand::{
    Rng, RngExt,
    distr::{Distribution, StandardUniform},
};
use rand_distr::StandardNormal;
use serde::{Deserialize, Serialize};

use crate::{Cartesian, Error, Quaternion, Rotate, Rotation, RotationMatrix, Versor};

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

    /// Get the element of the matrix in position `self.rows[0][0]`.
    #[inline]
    pub(crate) fn a(&self) -> Quaternion {
        self.rows[0][0]
    }
    /// Get the element of the matrix in position `self.rows[0][1]`.
    #[inline]
    pub(crate) fn b(&self) -> Quaternion {
        self.rows[0][1]
    }
    /// Get the element of the matrix in position `self.rows[1][0]`.
    #[inline]
    pub(crate) fn c(&self) -> Quaternion {
        self.rows[1][0]
    }
    /// Get the element of the matrix in position `self.rows[1][1]`.
    #[inline]
    pub(crate) fn d(&self) -> Quaternion {
        self.rows[1][1]
    }
}

impl Distribution<QuadQuaternion> for StandardUniform {
    /// Sample a uniformly random 5D rotation (Haar measure on SO(5)).
    ///
    /// A [`QuadQuaternion`] is an element of Sp(2) ≅ Spin(5), the double
    /// cover of SO(5), so uniformly sampling Sp(2) is uniform on SO(5) as well.
    /// This implements the Mezzadri subgroup algorithm specialized to
    /// Sp(2) ≅ S⁷ × Sp(1), generalizing how we sample `Versor`s.
    ///
    /// Reference: F. Mezzadri, "How to generate random matrices from the
    /// classical compact groups", Notices AMS 54, 592 (2007),
    /// arXiv:math-ph/0609050, sections 5-8. <https://arxiv.org/pdf/math-ph/0609050>
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> QuadQuaternion {
        // First column (a, b), uniform on S^7 in H^2: (Mezzadri eq. (8.10))
        let a_unnormalized = Quaternion::from(std::array::from_fn(|_| {
            rng.sample::<f64, _>(StandardNormal)
        }));
        let b_unnormalized = Quaternion::from(std::array::from_fn(|_| {
            rng.sample::<f64, _>(StandardNormal)
        }));
        let norm = (a_unnormalized.norm_squared() + b_unnormalized.norm_squared()).sqrt();
        let a = a_unnormalized * (1.0 / norm);
        let c = b_unnormalized * (1.0 / norm);

        // Uniform element of O(N) / O(N-1) ~ S^{N-1} (Mezzadri theorem 5).
        // "Intuitively", I think this is similar to sampling an axis (a point on a
        // 3-1=2-sphere) in the SO(3) case, but here our "axis" is a fiber (?) on S^7
        let phase = *rng.random::<Versor>().get();

        // Phase of x1 (Mezzadri eq. (7.26): x1 = q1 |x1|).
        let a_norm = a.norm();
        let q1 = if a_norm != 0.0 && a_norm.is_finite() {
            a / a_norm
        } else {
            QUATERNION_BASIS[0]
        };

        // Householder vector u = normalize(v + q1 e1) (Mezzadri eq. (7.26)),
        // with the closed form ||v + q1 e1||^2 = 2(1 + |x1|) (eq. (7.16)).
        let u_norm = (2.0 * (1.0 + a_norm)).sqrt();
        let u0 = (a + q1) * (1.0 / u_norm);
        let u1 = c * (1.0 / u_norm);

        // Householder reflection H = -conj(q1)(I - 2 u u^dagger) : (eq. (7.25-7.27)),
        let neg_q1 = q1.conjugate() * -1.0;
        let h00 = neg_q1 * (QUATERNION_BASIS[0] - u0 * u0.conjugate() * 2.0);
        let h01 = neg_q1 * (u0 * u1.conjugate() * -2.0);
        let h10 = neg_q1 * (u1 * u0.conjugate() * -2.0);
        let h11 = neg_q1 * (QUATERNION_BASIS[0] - u1 * u1.conjugate() * 2.0);

        // Reconstruct the full group element from the blocks we've calculated
        // S = H^dagger * diag(1, phase) (eq. 8.15).
        QuadQuaternion {
            rows: [
                [h00.conjugate(), h10.conjugate() * phase],
                [h01.conjugate(), h11.conjugate() * phase],
            ],
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
    #[inline]
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

impl From<QuadQuaternion> for RotationMatrix<5> {
    /// Construct a rotation matrix equivalent to this rotation.
    ///
    /// When rotating many vectors by the same [`QuadQuaternion`], improve performance
    /// by converting to a matrix first and applying that matrix to the vectors.
    ///
    /// The entries are the inner products
    /// ```math
    /// R_{ij} = \langle E_i, \mathbf{M} E_j \mathbf{M}^\dagger \rangle
    /// ```
    /// in the following basis:
    /// ```math
    /// \{E_0, \ldots, E_4\} = \left\{\frac{\mathrm{diag}(1, -1)}{\sqrt 2},
    /// \frac{B(1)}{\sqrt 2}, \frac{B(i)}{\sqrt 2}, \frac{B(j)}{\sqrt 2},
    /// \frac{B(k)}{\sqrt 2}\right\}
    /// ```
    /// where $`B(u)`$ is the matrix with the quaternion $`u`$ in both off-diagonal
    /// entries. Index 0 selects the $`\mathrm{diag}(1, -1)`$ component and indices 1-4
    /// select the quaternion components $`1, i, j, k`$
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_vector::{Cartesian, QuadQuaternion, Quaternion, Rotate, RotationMatrix};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let i_quat = Quaternion::from([0.0, 1.0, 0.0, 0.0]);
    /// let zero = Quaternion::from([0.0; 4]);
    /// let rotation = QuadQuaternion::try_from([[i_quat, zero], [zero, i_quat]])?;
    ///
    /// // The rotation by PI in the (j, k) plane fixes the first three
    /// // components and negates the last two.
    /// let matrix = RotationMatrix::from(rotation);
    /// assert_relative_eq!(
    ///     matrix.rotate(&Cartesian::from([0.0, 0.0, 0.0, 1.0, 0.0])),
    ///     Cartesian::from([0.0, 0.0, 0.0, -1.0, 0.0])
    /// );
    /// assert_relative_eq!(
    ///     matrix.rotate(&Cartesian::from([1.0, 0.0, 0.0, 0.0, 0.0])),
    ///     Cartesian::from([1.0, 0.0, 0.0, 0.0, 0.0])
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn from(value: QuadQuaternion) -> RotationMatrix<5> {
        let (block_00, block_01, block_10, block_11) = (value.a(), value.b(), value.c(), value.d());

        let mut result = [[0.0; 5]; 5];

        // Corner: <diag(1, -1), M diag(1, -1) M^dagger>, in a form symmetrized
        // over the two column norms so rounding errors in the blocks cancel.
        result[0][0] = 0.5
            * (block_00.norm_squared() + block_11.norm_squared()
                - block_01.norm_squared()
                - block_10.norm_squared());

        // First column: the components of a c^dagger - b d^dagger.
        let first_column = block_00 * block_10.conjugate() - block_01 * block_11.conjugate();
        result[1][0] = first_column.scalar;
        result[2][0] = first_column.vector[0];
        result[3][0] = first_column.vector[1];
        result[4][0] = first_column.vector[2];

        // First row: 2 Re[(b^dagger a) v] for the basis quaternions v.
        let first_row = block_01.conjugate() * block_00;
        result[0][1] = 2.0 * first_row.scalar;
        result[0][2] = -2.0 * first_row.vector[0];
        result[0][3] = -2.0 * first_row.vector[1];
        result[0][4] = -2.0 * first_row.vector[2];

        // Block: for each basis quaternion v, the components of
        // (b v^dagger) c^dagger + (a v) d^dagger.
        for (column_index, &basis_quaternion) in QUATERNION_BASIS.iter().enumerate() {
            let column_quaternion = (block_01 * basis_quaternion.conjugate())
                * block_10.conjugate()
                + (block_00 * basis_quaternion) * block_11.conjugate();
            result[1][column_index + 1] = column_quaternion.scalar;
            result[2][column_index + 1] = column_quaternion.vector[0];
            result[3][column_index + 1] = column_quaternion.vector[1];
            result[4][column_index + 1] = column_quaternion.vector[2];
        }

        RotationMatrix {
            rows: result.map(Cartesian::from),
        }
    }
}

impl Mul for QuadQuaternion {
    type Output = Self;

    /// Multiply two `QuadQuaternion` matrices.
    ///
    /// This operation is a matrix multiplication, and is not commutative, so the order
    /// of the operands matters: in the product `a * b`, the columns of `b` are
    /// transformed by `a`.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::{QuadQuaternion, Quaternion};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let zero = Quaternion::from([0.0; 4]);
    /// let i = Quaternion::from([0.0, 1.0, 0.0, 0.0]);
    /// let rotation = QuadQuaternion::try_from([[i, zero], [zero, i]])?;
    ///
    /// // The identity is the neutral element of the product.
    /// assert_eq!(rotation * QuadQuaternion::default(), rotation);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let [[q_a, q_b], [q_c, q_d]] = self.rows;
        let [[r_a, r_b], [r_c, r_d]] = rhs.rows;
        Self {
            rows: [
                [q_a * r_a + q_b * r_c, q_a * r_b + q_b * r_d],
                [q_c * r_a + q_d * r_c, q_c * r_b + q_d * r_d],
            ],
        }
    }
}

impl Rotation for QuadQuaternion {
    /// Combine two rotations.
    ///
    /// The resulting rotation rotates by **first** `other` _followed by_
    /// `self`, which is the quaternionic matrix product `self * other`.
    #[inline]
    fn combine(&self, other: &Self) -> Self {
        *self * *other
    }

    /// The identity rotation.
    #[inline]
    fn identity() -> Self {
        Self::default()
    }

    /// Inverse the rotation.
    ///
    /// The inverse of a unitary matrix is its conjugate transpose.
    #[inline]
    fn inverted(self) -> Self {
        Self {
            rows: [
                [self.a().conjugate(), self.c().conjugate()],
                [self.b().conjugate(), self.d().conjugate()],
            ],
        }
    }
}

impl fmt::Display for QuadQuaternion {
    /// Format a [`QuadQuaternion`] as `[[a, b],\n [c, d]]`.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::QuadQuaternion;
    ///
    /// let q = QuadQuaternion::default();
    /// assert_eq!(
    ///     format!("{q}"),
    ///     "[[[1, [0, 0, 0]], [0, [0, 0, 0]]],\n [[0, [0, 0, 0]], [1, [0, 0, 0]]]]"
    /// );
    /// ```
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[[{}, {}],\n [{}, {}]]",
            self.rows[0][0], self.rows[0][1], self.rows[1][0], self.rows[1][1]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::{assert_abs_diff_eq, assert_relative_eq, assert_relative_ne};
    use hoomd_linear_algebra::{MatMul, SquareMatrix, matrix::Matrix};
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
        // Check that samples are proper rotations (orthogonal, det +1) and
        // the ensemble matches the exact Haar-SO(5) moments from the Weyl
        // eigenangle density: E[M_ij^2] = 1/5, E[M_ij^4] = 3/35,
        // E[Tr M] = 0, E[Tr M^2] = 1.
        const N: usize = 40_000;
        let mut rng = StdRng::seed_from_u64(1);

        let (mut e2, mut e4, mut tr, mut tr2) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..N {
            let m = Matrix {
                rows: so5_matrix(&rng.random::<QuadQuaternion>()),
            };

            if i < 100 {
                // Orthogonality: M^T M = I, and det = +1.
                let gram = m.transpose().matmul(&m);
                assert!(
                    gram.rows
                        .iter()
                        .flatten()
                        .zip(Matrix::<5, 5>::identity().rows.iter().flatten())
                        .all(|(a, b)| (a - b).abs() <= 1e-10)
                );
                assert_abs_diff_eq!(m.determinant(), 1.0, epsilon = 1e-9);
            }

            e2 += m.rows[0][0] * m.rows[0][0];
            e4 += m.rows[0][0].powi(4);
            tr += m.trace();
            tr2 += m.matmul(&m).trace();
        }

        let n = N as f64;
        assert_abs_diff_eq!(e2 / n, 0.2, epsilon = 0.01);
        assert_abs_diff_eq!(e4 / n, 3.0 / 35.0, epsilon = 0.005);
        assert_abs_diff_eq!(tr / n, 0.0, epsilon = 0.02);
        assert_abs_diff_eq!(tr2 / n, 1.0, epsilon = 0.03);
    }

    #[test]
    fn group_operations() {
        let mut rng = StdRng::seed_from_u64(1);
        let identity = QuadQuaternion::identity();
        for _ in 0..64 {
            let p: QuadQuaternion = rng.random();
            let q: QuadQuaternion = rng.random();
            let r: QuadQuaternion = rng.random();

            // Inverse and associativity.
            assert_relative_eq!(p.combine(&p.inverted()), identity, epsilon = 1e-12);
            assert_relative_eq!(
                p.combine(&q).combine(&r),
                p.combine(&q.combine(&r)),
                epsilon = 1e-12
            );

            // Composition matches the product of the matrix representations.
            let product = Matrix {
                rows: so5_matrix(&p),
            }
            .matmul(&Matrix {
                rows: so5_matrix(&q),
            });
            assert!(
                so5_matrix(&p.combine(&q))
                    .iter()
                    .flatten()
                    .zip(product.rows.iter().flatten())
                    .all(|(a, b)| (a - b).abs() <= 1e-12)
            );
        }
    }

    #[test]
    fn mul() {
        let mut rng = StdRng::seed_from_u64(1);
        let identity = QuadQuaternion::default();

        for _ in 0..64 {
            let p: QuadQuaternion = rng.random();
            let q: QuadQuaternion = rng.random();
            let r: QuadQuaternion = rng.random();

            // The identity is the neutral element on both sides.
            assert_relative_eq!(p * identity, p);
            assert_relative_eq!(identity * p, p);

            // The product is associative.
            assert_relative_eq!((p * q) * r, p * (q * r), epsilon = 1e-12);

            // The product is not commutative for generic rotations.
            assert_relative_ne!(p * q, q * p);
        }
    }

    #[test]
    fn rotation_matrix_conversion() {
        // The identity rotation converts to the identity matrix exactly.
        let identity_matrix: [[f64; 5]; 5] = RotationMatrix::from(QuadQuaternion::identity())
            .rows()
            .map(|row| row.coordinates);
        assert_eq!(identity_matrix, Matrix::<5, 5>::identity().rows);

        let mut rng = StdRng::seed_from_u64(1);

        for _ in 0..4096 {
            let rotation: QuadQuaternion = rng.random();
            let converted = RotationMatrix::from(rotation);
            let converted_rows = converted.rows().map(|row| row.coordinates);

            assert_abs_diff_eq!(converted_rows, so5_matrix(&rotation), epsilon = 1e-14);

            // Double cover -> -M and M should rotate V the same.
            let negated = QuadQuaternion {
                rows: rotation.rows.map(|row| row.map(|block| block * -1.0)),
            };
            assert_eq!(
                RotationMatrix::from(negated)
                    .rows()
                    .map(|row| row.coordinates),
                converted_rows
            );
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
}

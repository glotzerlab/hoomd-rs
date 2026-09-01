// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`DoubleVersor`], a representation of rotations in four dimensions.
//! Similar to [`Versor`] in 3D, this approach is more numerically stable and space
//! efficient than the equivalent matrix representation, but slower when applying
//! rotations.

use std::f64::consts::TAU;

use approxim::RelativeEq;
use hoomd_linear_algebra::{MatMul, matrix::Matrix44};
use rand::{Rng, RngExt};
use rand_distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};

use crate::{Cartesian, Metric, Quaternion, Rotate, Rotation, RotationMatrix, Versor};

/// A pair of [`Versor`]s that represent a 4D rotation.
///
/// Every rotation in SO(4) factors into a *left-isoclinic* and a *right-isoclinic*
/// rotation via the Van Elfrinkhof decomposition, each of which can be described as an
/// ordinary 3D [`Versor`] with fully independent degrees of freedom. [`DoubleVersor`]
/// stores one unit quaternion for each factor, $`\mathbf{q_l}`$ and $`\mathbf{q_r}`$.
/// The two rotate a [`Cartesian<4>`] vector $`\vec{a}`$ (viewed as a quaternion) with a
/// double-sided product that mirrors the three-dimensional [`Versor::rotate`]:
///
/// ```math
/// \vec{a}' = \mathbf{q}_l \, \vec{a} \, \mathbf{q}_r
/// ```
/// As normal [`Versor`]s do with SO(3), the pair $`(\mathbf{q}_l, \mathbf{q}_r)`$ forms
/// a double cover of SO(4).
#[derive(Clone, Copy, Debug, PartialEq, RelativeEq, Serialize, Deserialize)]
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
    pub fn from_left_isoclinic(l: Versor) -> Self {
        Self {
            l,
            r: Versor::default(),
        }
    }
    #[inline]
    #[must_use]
    /// Create a purely right-isoclinic double versor.
    pub fn from_right_isoclinic(r: Versor) -> Self {
        Self {
            l: Versor::default(),
            r,
        }
    }

    #[inline]
    #[must_use]
    /// Normalize the double versor.
    ///
    /// Nominally, all [`DoubleVersor`] instances have unit components. Due to limited
    /// floating point precision, this assumption may not hold after repeated
    /// operations. Normalize double versors when needed to correct this issue.
    pub fn normalized(self) -> Self {
        // Normalization is a projection onto the manifold as normal, and projection
        // onto a product manifold can be separated into the components in the product.
        Self {
            l: self.l.normalized(),
            r: self.r.normalized(),
        }
    }

    #[inline]
    #[must_use]
    /// The distance between two [`DoubleVersor`]s measured along a chord.
    ///
    /// Explicitly, this is the straight-line (chordal) distance between two
    /// rotations $`\vec{u}`$ and $`\vec{v}`$. It is NOT the intrinsic distance
    /// along the manifold, which is defined by the [`Metric`] implementation.
    ///
    /// ```math
    /// d_{SO(4)}(\vec{u}, \vec{v}) = \sqrt{8 \left(1 - (\vec{u}_l \cdot \vec{v}_l)\,(\vec{u}_r \cdot \vec{v}_r)\right)}
    /// ```
    ///
    /// This is equivalent to the matrix form $`\lVert R_u - R_v \rVert_F`$, but does not require
    /// forming the matrix representation of each rotation.
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_vector::{DoubleVersor, Metric, Versor};
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let identity = DoubleVersor::default();
    /// // A rotation by PI/2 in both invariant planes.
    /// let v = DoubleVersor::from_left_isoclinic(Versor::from_axis_angle(
    ///     [1.0, 0.0, 0.0].try_into()?,
    ///     PI,
    /// ));
    ///
    /// assert_relative_eq!(v.chordal_distance(&identity), 8.0_f64.sqrt());
    /// // The intrinsic distance along the manifold is longer than the chord.
    /// assert!(v.distance(&identity) > 8.0_f64.sqrt());
    /// # Ok(())
    /// # }
    /// ```
    pub fn chordal_distance(&self, other: &Self) -> f64 {
        // We choose a form based on the dot product of pairs of left and right
        // quaternions -- as expected, this is exactly the same as the standard
        // Frobenius metric on SO(N), $`||A-B||_F^2`$. The following Mathematica code
        // symbolically proves the expressions are equivalent
        // (* See RotationMatrix::from<DoubleVersor> *)
        // LMat[{a_, b_, c_, d_}] := {{a, -b, -c, -d}, {b, a, -d, c}, {c, d, a, -b}, {d, -c, b, a}}
        // RMat[{p_, q_, r_, s_}] := {{p, -q, -r, -s}, {q, p, s, -r}, {r, -s, p, q}, {s, r, -q, p}}

        // qL1 = {a1, b1, c1, d1}; qL2 = {a2, b2, c2, d2};
        // qR1 = {p1, q1, r1, s1}; qR2 = {p2, q2, r2, s2};

        // (* Construct the explicit SO(4) matrices *)
        // A = LMat[qL1] . RMat[qR1]; B = LMat[qL2] . RMat[qR2];

        // (* Frobenius Norm-based metric *)
        // (* Note we use ComplexExpand, as otherwise the symbolic algebra gets stuck resolving sqrts *)
        // explicitFrobeniusSquared = ComplexExpand[Norm[A - B, "Frobenius"]^2];

        // (* Quaternion form *)
        // algebraicForm = 8 - 8 * (qL1 . qL2) * (qR1 . qR2);

        // (* Contrain our solutions to the manifold *)
        // constraints = {
        //   a1^2 + b1^2 + c1^2 + d1^2 == 1, a2^2 + b2^2 + c2^2 + d2^2 == 1,
        //   p1^2 + q1^2 + r1^2 + s1^2 == 1, p2^2 + q2^2 + r2^2 + s2^2 == 1
        // };

        // (* Validate *)
        // Print["Norm[A-B, \"Frobenius\"]^2 == 8 - 8*(qL1.qL2)*(qR1.qR2)"];
        // isEquivalent = FullSimplify[explicitFrobeniusSquared == algebraicForm, constraints];
        // Print["Result: ", isEquivalent];
        let left_dot = self.l.dot_as_cartesian(&other.l);
        let right_dot = self.r.dot_as_cartesian(&other.r);
        (8.0 * (1.0 - left_dot * right_dot)).max(0.0).sqrt()
    }
}

impl From<DoubleVersor> for RotationMatrix<4> {
    /// Construct a rotation matrix equivalent to this double versor's rotation.
    ///
    /// When rotating many vectors by the same [`DoubleVersor`], converting to a matrix
    /// and applying that matrix to the vectors may be faster than direct rotation.
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_vector::{
    ///     Cartesian, DoubleVersor, Rotate, RotationMatrix, Versor,
    /// };
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let a = Cartesian::from([1.0, 2.0, 0.0, 0.0]);
    /// let v = DoubleVersor::from_left_isoclinic(Versor::from_axis_angle(
    ///     [1.0, 0.0, 0.0].try_into()?,
    ///     PI,
    /// ));
    ///
    /// let matrix = RotationMatrix::from(v);
    /// assert_relative_eq!(matrix.rotate(&a), v.rotate(&a));
    /// assert_relative_eq!(matrix.rotate(&a), [-2.0, 1.0, 0.0, 0.0].into());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[expect(clippy::many_single_char_names, reason = "Clarity.")]
    fn from(versor: DoubleVersor) -> RotationMatrix<4> {
        let (&q_l, &q_r) = (
            versor.left_isoclinic().get(),
            versor.right_isoclinic().get(),
        );
        let (a, [b, c, d]) = (q_l.scalar, q_l.vector.coordinates);
        let (p, [q, r, s]) = (q_r.scalar, q_r.vector.coordinates);

        // Construct the left-isoclinic matrix L(Q_L)
        let l_mat = [[a, -b, -c, -d], [b, a, -d, c], [c, d, a, -b], [d, -c, b, a]];

        // Construct the right-isoclinic matrix R(Q_R)
        let r_mat = [[p, -q, -r, -s], [q, p, s, -r], [r, -s, p, q], [s, r, -q, p]];

        // Combine the left and right isoclinic parts as L@R
        Matrix44 { rows: l_mat }
            .matmul(&Matrix44 { rows: r_mat })
            .into()
    }
}

impl Rotate<Cartesian<4>> for DoubleVersor {
    type Matrix = RotationMatrix<4>;

    /// Rotate a [`Cartesian<4>`] by a [`DoubleVersor`].
    ///
    /// ```math
    /// \mathbf{q_l} \vec{a} \mathbf{q_r}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_vector::{Cartesian, DoubleVersor, Rotate, Rotation, Versor};
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let a = Cartesian::from([1.0, 2.0, 0.0, 0.0]);
    ///
    /// // A left-isoclinic rotation that rotates by PI/2 radians in both the
    /// // `xy` and `zw` planes. Note that each plane rotates by half the given angle
    /// let v = DoubleVersor::from_left_isoclinic(Versor::from_axis_angle(
    ///     [1.0, 0.0, 0.0].try_into()?,
    ///     PI,
    /// ));
    ///
    /// // Initializing from left isoclinic implies the right isoclinic is [1 0 0 0]
    /// assert_eq!(v.right_isoclinic(), Versor::default());
    ///
    /// let b = v.rotate(&a);
    /// assert_relative_eq!(b, [-2.0, 1.0, 0.0, 0.0].into());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn rotate(&self, vector: &Cartesian<4>) -> Cartesian<4> {
        let q = *self.l.get() * Quaternion::from(vector.coordinates) * *self.r.get();
        q.embed_in_cartesian_4()
    }
}

impl Distribution<DoubleVersor> for StandardUniform {
    /// Sample a random [`DoubleVersor`] from the uniform distribution over all rotations in SO(4).
    ///
    /// This is implemented as a random sampling of *pairs* of [`Versor`]'s, which is
    /// equivalent to a uniform sampling of the full manifold.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::DoubleVersor;
    /// use rand::{RngExt, SeedableRng, rngs::StdRng};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut rng = StdRng::seed_from_u64(1);
    /// let v: DoubleVersor = rng.random();
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> DoubleVersor {
        (rng.random(), rng.random()).into()
    }
}

impl Rotation for DoubleVersor {
    /// Combine two rotations.
    ///
    /// The resulting versor is obtained by left and right quaternion
    /// multiplications. As with [`Versor`], `self.combine(other)` first applies
    /// `other`, then `self`. Note that the right components multiply in the
    /// reverse order!
    /// ```math
    /// \mathbf{q}_{l_{ab}} = \mathbf{q}_{l_a} \mathbf{q}_{l_b}
    /// \mathbf{q}_{r_{ab}} = \mathbf{q}_{r_b} \mathbf{q}_{r_a}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_vector::{Cartesian, DoubleVersor, Rotate, Rotation, Versor};
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let a = Cartesian::from([1.0, 2.0, -3.0, 0.5]);
    /// let left = DoubleVersor::from_left_isoclinic(Versor::from_axis_angle(
    ///     [1.0, 0.0, 0.0].try_into()?,
    ///     PI / 2.0,
    /// ));
    /// let right = DoubleVersor::from_right_isoclinic(Versor::from_axis_angle(
    ///     [0.0, 1.0, 0.0].try_into()?,
    ///     PI / 3.0,
    /// ));
    ///
    /// // Combining applies `right` first, then `left`.
    /// let combined = left.combine(&right);
    /// assert_relative_eq!(combined.rotate(&a), left.rotate(&right.rotate(&a)));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn combine(&self, other: &Self) -> Self {
        Self {
            l: self.l.combine(&other.l),
            r: other.r.combine(&self.r),
        }
    }

    /// Create the identity [`DoubleVersor`]: ([1, [0, 0, 0]], [1, [0, 0, 0]])
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::{DoubleVersor, Rotation};
    ///
    /// let identity = DoubleVersor::identity();
    /// ```
    #[inline]
    fn identity() -> Self {
        Self::default()
    }

    /// Create a [`DoubleVersor`] that performs the inverse rotation of the given double versor.
    ///
    /// Both components are conjugated:
    /// ```math
    /// (\mathbf{q}_l, \mathbf{q}_r)^{-1} = (\mathbf{q}_l^*, \mathbf{q}_r^*)
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::{DoubleVersor, Rotation, Versor};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v = DoubleVersor::from_left_isoclinic(Versor::from_axis_angle(
    ///     [0.0, 1.0, 0.0].try_into()?,
    ///     1.5,
    /// ));
    /// let v_star = v.inverted();
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn inverted(self) -> Self {
        Self {
            l: self.l.inverted(),
            r: self.r.inverted(),
        }
    }
}

impl Metric for DoubleVersor {
    #[inline]
    fn distance_squared(&self, other: &Self) -> f64 {
        let left_angle = self.l.dot_as_cartesian(&other.l).clamp(-1.0, 1.0).acos();
        let right_angle = self.r.dot_as_cartesian(&other.r).clamp(-1.0, 1.0).acos();

        // The principal angles of the relative rotation are theta_1 = left_angle
        // + right_angle and theta_2 = |left_angle - right_angle|. Fold theta_1
        // into [0, pi] to account for the double cover of SO(4) by the versor pair.
        let theta_1 = f64::min(left_angle + right_angle, TAU - (left_angle + right_angle));
        let theta_2 = f64::abs(left_angle - right_angle);

        // Multiply by 2.0 to match the standard ||log(R_A^T R_B)||_F^2 scaling.
        2.0 * (theta_1.powi(2) + theta_2.powi(2))
    }

    #[inline]
    /// The dimension of the manifold of SO(N) is $`\frac{N(N-1)}{2}`$, or 6 for SO(4).
    fn n_dimensions() -> usize {
        6
    }
    #[inline]
    /// The intrinsic distance between two [`DoubleVersor`]s.
    ///
    /// Explicitly, the metric for two points $`\vec{u}`$ and $`\vec{v}`$ on the
    /// manifold SO(4):
    ///
    /// ```math
    /// d_{SO(4)}(\vec{u}, \vec{v}) = \sqrt{2\theta_1^2 + 2\theta_2^2}
    /// ```
    ///
    /// Where the principal planar angles $`\theta_1`$ and $`\theta_2`$ of the
    /// relative rotation are derived from the left and right quaternion arc
    /// lengths, folding the sum into $`[0, \pi]`$ to account for the double
    /// cover of the manifold:
    /// * $`\alpha_l = \arccos(\vec{u}_L \cdot \vec{v}_L)`$
    /// * $`\alpha_r = \arccos(\vec{u}_R \cdot \vec{v}_R)`$
    /// * $`\theta_1 = \min(\alpha_l + \alpha_r, 2\pi - (\alpha_l + \alpha_r))`$
    /// * $`\theta_2 = |\alpha_l - \alpha_r|`$
    ///
    /// This is equivalent to the scaled matrix logarithm form $`||\log(R_u^T R_v)||_F`$,
    /// measuring the shortest path along the curved manifold.
    fn distance(&self, other: &Self) -> f64 {
        self.distance_squared(other).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use approxim::assert_relative_eq;
    use hoomd_linear_algebra::{MatMul, matrix::Matrix44};
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use rstest::rstest;
    use std::f64::consts::{PI, TAU};

    use crate::{
        Cartesian, DoubleVersor, InnerProduct, Metric, Rotate, Rotation, RotationMatrix, Unit,
        Versor,
    };

    #[rstest]
    fn random_rotations_match_matrix(
        #[values([0.0, 0.0, 0.0, 0.0], [1.0,2.0,3.0,4.0], [-3.0, 9.12, -0.1, 1.25])] v: [f64; 4],
    ) {
        let mut rng = StdRng::seed_from_u64(42);
        let v = Cartesian::from(v);

        for _ in 0..10_000 {
            let dv: DoubleVersor = rng.random();
            assert_relative_eq!(
                dv.rotate(&v),
                RotationMatrix::from(dv).rotate(&v),
                epsilon = 1e-14,
            );
        }
    }

    #[test]
    fn normalized_restores_unit_components() {
        let mut rng = StdRng::seed_from_u64(99);
        let mut q: DoubleVersor = rng.random();
        for _ in 0..100_000 {
            q = q.combine(&rng.random());
        }

        let norm_l = q.left_isoclinic().get().norm();
        let norm_r = q.right_isoclinic().get().norm();

        let n = q.normalized();
        assert_relative_eq!(n.left_isoclinic().get().norm(), 1.0, epsilon = 1e-15);
        assert_relative_eq!(n.right_isoclinic().get().norm(), 1.0, epsilon = 1e-15);

        // The drifted rotation is the normalized one scaled by norm_l * norm_r.
        let a = Cartesian::from([1.0, 2.0, -3.0, 0.5]);
        let scale = norm_l * norm_r;
        assert_relative_eq!(n.rotate(&a), q.rotate(&a) / scale, max_relative = 1e-12);
    }

    #[rstest]
    fn random_combine_invert_roundtrip(
        #[values([0.0, 0.0, 0.0, 0.0], [1.0,2.0,3.0,4.0], [-3.0, 9.12, -0.1, 1.25])] v: [f64; 4],
    ) {
        let mut rng = StdRng::seed_from_u64(42);
        let v = Cartesian::from(v);

        for _ in 0..10_000 {
            let a: DoubleVersor = rng.random();
            let b: DoubleVersor = rng.random();
            let c: DoubleVersor = rng.random();

            // Inverting undoes the rotation
            assert_relative_eq!(a.inverted().rotate(&a.rotate(&v)), v, epsilon = 1e-14);

            // Combine then invert gives identity
            let combined = a.combine(&b);
            assert_relative_eq!(
                combined.inverted().combine(&combined).rotate(&v),
                v,
                epsilon = 1e-12,
            );

            // Combine is associative
            assert_relative_eq!(
                a.combine(&b).combine(&c).rotate(&v),
                a.combine(&b.combine(&c)).rotate(&v),
                epsilon = 1e-12,
            );

            // Double inverse is the original rotation
            assert_relative_eq!(a.inverted().inverted(), a);

            // Combine then rotate matches applying sequentially
            assert_relative_eq!(
                a.combine(&b).rotate(&v),
                a.rotate(&b.rotate(&v)),
                epsilon = 1e-12,
            );
        }
    }

    #[test]
    fn chordal_distance_matches_frobenius() {
        let mut rng = StdRng::seed_from_u64(11);

        for _ in 0..10_000 {
            let u: DoubleVersor = rng.random();
            let v: DoubleVersor = rng.random();

            let mu = RotationMatrix::from(u);
            let mv = RotationMatrix::from(v);

            let frobenius = (0..4)
                .flat_map(|i| (0..4).map(move |j| mu.rows()[i][j] - mv.rows()[i][j]))
                .map(|difference| difference * difference)
                .sum::<f64>()
                .sqrt();

            assert_relative_eq!(u.chordal_distance(&v), frobenius, epsilon = 1e-12);
        }
    }

    #[test]
    fn metric_properties_hold_up() {
        let mut rng = StdRng::seed_from_u64(31);

        for _ in 0..1_000 {
            let u: DoubleVersor = rng.random();
            let v: DoubleVersor = rng.random();
            let w: DoubleVersor = rng.random();

            // Symmetry
            assert_relative_eq!(u.distance(&v), v.distance(&u), epsilon = 1e-14);

            // Zero self-distance
            assert_relative_eq!(u.distance(&u), 0.0, epsilon = 1e-7);

            // Triangle inequality
            assert!(
                u.distance(&w) <= u.distance(&v) + v.distance(&w) + 1e-14,
                "triangle inequality violated: {} > {} + {}",
                u.distance(&w),
                u.distance(&v),
                v.distance(&w)
            );
        }

        let x: Unit<Cartesian<3>> = [1.0, 0.0, 0.0]
            .try_into()
            .expect("hard-coded axis should have non-zero length");
        let y: Unit<Cartesian<3>> = [0.0, 1.0, 0.0]
            .try_into()
            .expect("hard-coded axis should have non-zero length");
        for (theta_l, theta_r) in [(0.3, 1.7), (2.0, 0.9), (2.9, 3.0), (0.0, PI)] {
            let u = DoubleVersor::from((
                Versor::from_axis_angle(x, theta_l),
                Versor::from_axis_angle(y, theta_r),
            ));
            let negated_u = DoubleVersor::from((
                Versor::from_axis_angle(x, theta_l + TAU),
                Versor::from_axis_angle(y, theta_r + TAU),
            ));

            assert_relative_eq!(u.distance(&negated_u), 0.0, epsilon = 1e-7);
        }
    }

    #[test]
    fn rotation_matrix_rows_orthonormal() {
        let mut rng = StdRng::seed_from_u64(41);

        for _ in 0..10_000 {
            let dv: DoubleVersor = rng.random();
            let rows = RotationMatrix::from(dv).rows();

            for (i, row_i) in rows.iter().enumerate() {
                assert_relative_eq!(row_i.dot(row_i), 1.0, epsilon = 1e-14);
                for row_j in rows.iter().skip(i + 1) {
                    assert_relative_eq!(row_i.dot(row_j), 0.0, epsilon = 1e-14);
                }
            }
        }
    }
}

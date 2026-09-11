// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Angle`]

use serde::{Deserialize, Serialize};
use std::{f64::consts::PI, fmt};

use approxim::approx_derive::RelativeEq;
use rand::{
    Rng,
    distr::{Distribution, StandardUniform, Uniform},
};

use crate::{Cartesian, Rotate, Rotation, RotationMatrix};

/// Rotation in the plane.
///
/// The rotation is represented by an angle `theta` in radians. Positive values rotate
/// counter-clockwise.
///
/// ## Constructing [`Angle`]
///
/// The default Angle rotates by 0 radians:
///
/// ```
/// use hoomd_vector::Angle;
///
/// let a = Angle::default();
/// assert_eq!(a.theta, 0.0)
/// ```
///
/// Create an [`Angle`] with a given value:
/// ```
/// use hoomd_vector::Angle;
/// use std::f64::consts::PI;
///
/// let a = Angle::from(PI / 2.0);
/// assert_eq!(a.theta, PI / 2.0);
/// ```
///
/// Create a random [`Angle`] from the uniform distribution over all rotations:
/// ```
/// use hoomd_vector::Angle;
/// use rand::{RngExt, SeedableRng, rngs::StdRng};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut rng = StdRng::seed_from_u64(1);
/// let a: Angle = rng.random();
/// # Ok(())
/// # }
/// ```
///
/// ## Operations using [`Angle`]
///
/// Rotate a [`Cartesian<2>`] vector by an [`Angle`]:
/// ```
/// use approxim::assert_relative_eq;
/// use hoomd_vector::{Angle, Cartesian, Rotate, Rotation};
/// use std::f64::consts::PI;
///
/// let v = Cartesian::from([-1.0, 0.0]);
/// let a = Angle::from(PI / 2.0);
/// let rotated = a.rotate(&v);
/// assert_relative_eq!(rotated, [0.0, -1.0].into())
/// ```
///
/// Combine two rotations together:
/// ```
/// use hoomd_vector::{Angle, Rotation};
/// use std::f64::consts::PI;
///
/// let a = Angle::from(PI / 2.0);
/// let b = Angle::from(-PI / 4.0);
/// let c = a.combine(&b);
/// assert_eq!(c.theta, PI / 4.0);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, RelativeEq, Serialize, Deserialize)]
pub struct Angle {
    /// Rotation angle (radians).
    pub theta: f64,
}

impl Angle {
    /// Reduce the rotation.
    ///
    /// [`Angle`] rotations are well-defined for any value of `theta`. However, combining small
    /// rotations with large ones will introduce floating point round-off error. Reducing an [`Angle`]
    /// creates an equivalent rotation with `theta` in the range from 0 to 2 pi.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::Angle;
    /// use std::f64::consts::PI;
    ///
    /// let a = Angle::from(20.0 * PI);
    /// let b = a.to_reduced();
    /// assert_eq!(b.theta, 0.0)
    /// ```
    #[inline]
    #[must_use]
    pub fn to_reduced(self) -> Self {
        Self {
            theta: self.theta.rem_euclid(2.0 * PI),
        }
    }
}

impl From<Angle> for RotationMatrix<2> {
    /// Construct a rotation matrix equivalent to this angle's rotation.
    ///
    /// When rotating many vectors by the same [`Angle`], improve performance
    /// by converting to a matrix first and applying that matrix to the vectors.
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_vector::{Angle, Cartesian, Rotate, RotationMatrix};
    /// use std::f64::consts::PI;
    ///
    /// let v = Cartesian::from([-1.0, 0.0]);
    /// let a = Angle::from(PI / 2.0);
    ///
    /// let matrix = RotationMatrix::from(a);
    /// let rotated = matrix.rotate(&v);
    /// assert_relative_eq!(rotated, [0.0, -1.0].into());
    /// ```
    #[inline]
    fn from(angle: Angle) -> RotationMatrix<2> {
        let sin_theta = angle.theta.sin();
        let cos_theta = angle.theta.cos();
        RotationMatrix {
            rows: [
                [cos_theta, -sin_theta].into(),
                [sin_theta, cos_theta].into(),
            ],
        }
    }
}

impl From<f64> for Angle {
    /// Create a rotation by `theta` radians.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::Angle;
    ///
    /// let a = Angle::from(1.5);
    /// assert_eq!(a.theta, 1.5);
    /// ```
    #[inline]
    fn from(theta: f64) -> Self {
        Self { theta }
    }
}

impl Rotate<Cartesian<2>> for Angle {
    type Matrix = RotationMatrix<2>;

    #[inline]
    /// Rotate a [`Cartesian<2>`] in the plane by an [`Angle`]
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_vector::{Angle, Cartesian, Rotate, Rotation};
    /// use std::f64::consts::PI;
    ///
    /// let v = Cartesian::from([-1.0, 0.0]);
    /// let a = Angle::from(PI / 2.0);
    /// let rotated = a.rotate(&v);
    /// assert_relative_eq!(rotated, [0.0, -1.0].into());
    /// ```
    fn rotate(&self, vector: &Cartesian<2>) -> Cartesian<2> {
        let sin_theta = self.theta.sin();
        let cos_theta = self.theta.cos();
        Cartesian::from([
            vector.coordinates[0] * cos_theta - vector.coordinates[1] * sin_theta,
            vector.coordinates[0] * sin_theta + vector.coordinates[1] * cos_theta,
        ])
    }
}

impl Rotation for Angle {
    #[inline]
    /// Create an [`Angle`] that rotates by 0 radians.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::{Angle, Rotation};
    ///
    /// let a = Angle::default();
    /// assert_eq!(a.theta, 0.0);
    /// ```
    fn identity() -> Self {
        Self::default()
    }

    #[inline]
    /// Create an [`Angle`] that rotates by the same amount in the opposite direction.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::{Angle, Rotation};
    /// use std::f64::consts::PI;
    ///
    /// let a = Angle::from(PI / 3.0);
    /// let b = a.inverted();
    /// assert_eq!(b.theta, -PI / 3.0);
    /// ```
    fn inverted(self) -> Self {
        Self::from(-self.theta)
    }

    #[inline]
    /// Create an [`Angle`] that rotates by the sum of the two angles.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::{Angle, Rotation};
    /// use std::f64::consts::PI;
    ///
    /// let a = Angle::from(PI / 2.0);
    /// let b = Angle::from(-PI / 4.0);
    /// let c = a.combine(&b);
    /// assert_eq!(c.theta, PI / 4.0);
    /// ```
    fn combine(&self, other: &Self) -> Self {
        Self::from(self.theta + other.theta)
    }
}

impl fmt::Display for Angle {
    /// Format an Angle as `<{theta}>`.
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<{}>", self.theta)
    }
}

impl Distribution<Angle> for StandardUniform {
    /// Sample a random angle from the uniform distribution over all rotations.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::Angle;
    /// use rand::{RngExt, SeedableRng, rngs::StdRng};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut rng = StdRng::seed_from_u64(1);
    /// let v: Angle = rng.random();
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Angle {
        let uniform = Uniform::new(0.0, 2.0 * PI).expect("hard-coded distribution should be valid");
        Angle::from(uniform.sample(rng))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use rstest::*;
    use std::f64::consts::PI;

    // Test named cases of the three input values (angle, vector input, and answer)
    #[rstest]
    #[case::pi_halves(PI / 2.0, (1.0, -0.5), (0.5, 1.0))]
    #[case::negative_pi_thirds(-PI / 3.0, (1.0, 0.0), (0.5, -f64::sqrt(3.0) / 2.0))]
    #[case::negative_pi(-PI, (3.1, -0.2), (-3.1, 0.2))]
    #[case::two_pi(PI*2.0, (3.1, -0.2), (3.1, -0.2))]
    #[case::zero(0.0, (3.1, -0.2), (3.1, -0.2))]
    #[case::negative_zero(-0.0, (3.1, -0.2), (3.1, -0.2))]
    fn rotate_2d(#[case] angle: f64, #[case] vec: (f64, f64), #[case] ans: (f64, f64)) {
        let angle = Angle::from(angle);
        let vec = Cartesian::from(vec);
        let ans = Cartesian::from(ans);

        assert_relative_eq!(angle.rotate(&vec), ans, epsilon = 4.0 * f64::EPSILON);
        assert_relative_eq!(
            RotationMatrix::from(angle).rotate(&vec),
            ans,
            epsilon = 4.0 * f64::EPSILON
        );
    }

    // Test with Cartesian product of the input arrays
    #[rstest(
        ang1 => [0.0, PI / 2.0, 1e-12 * PI, -3.0, 12345.6],
        ang2 => [-0.0, -PI / 3.0, PI, 2.0 * PI]
    )]
    fn combine_2d(ang1: f64, ang2: f64) {
        let (angle1, angle2) = (Angle::from(ang1), Angle::from(ang2));
        assert_relative_eq!(angle1.combine(&angle2).theta, ang1 + ang2);
    }

    #[test]
    fn default() {
        let a = Angle::default();
        assert_eq!(a.theta, 0.0);
    }

    #[test]
    fn identity() {
        let a = Angle::identity();
        assert_eq!(a.theta, 0.0);
    }

    #[rstest(theta => [0.0, 1.0, 2.125, 14.875, -4.5])]
    fn inverted(theta: f64) {
        let angle1 = Angle::from(theta);
        let angle2 = angle1.inverted();
        assert_eq!(angle2.theta, -theta);
        assert_relative_eq!(angle1.combine(&angle2), Angle::identity());
    }

    #[test]
    fn display() {
        let a = Angle::from(1.5);
        let s = format!("{a}");
        assert_eq!(s, "<1.5>");
    }

    #[test]
    fn reduced() {
        let two_pi = 2.0 * PI;

        assert_relative_eq!(Angle::from(0.125).to_reduced(), (0.125).into());
        assert_relative_eq!(Angle::from(2.0 * PI + 0.125).to_reduced(), (0.125).into());
        assert_relative_eq!(Angle::from(2.0 * 2.0 * PI + 0.5).to_reduced(), (0.5).into());
        assert_relative_eq!(Angle::from(3.0 * 2.0 * PI + 3.0).to_reduced(), (3.0).into());
        assert_relative_eq!(
            Angle::from(2.0 * PI - 0.125).to_reduced(),
            (2.0 * PI - 0.125).into()
        );

        assert_relative_eq!(Angle::from(two_pi).to_reduced(), (0.0).into());
        assert_relative_eq!(Angle::from(-0.125).to_reduced(), (2.0 * PI - 0.125).into());
        assert_relative_eq!(Angle::from(-3.0).to_reduced(), (2.0 * PI - 3.0).into());
        assert_relative_eq!(
            Angle::from(-2.0 * PI - 0.125).to_reduced(),
            (2.0 * PI - 0.125).into()
        );
        assert_relative_eq!(
            Angle::from(10.0 * -2.0 * PI - 0.125).to_reduced(),
            (2.0 * PI - 0.125).into()
        );
    }

    #[test]
    fn random() {
        let mut rng = StdRng::seed_from_u64(1);

        for _ in 0..10000 {
            let a: Angle = rng.random();
            assert!(a.theta >= 0.0 && a.theta < 2.0 * PI);
        }
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Quaternion`] and related types.
use std::{
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use approxim::approx_derive::RelativeEq;
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
use rand_distr::StandardNormal;

use crate::{Cartesian, Cross, Error, InnerProduct, Rotate, Rotation, RotationMatrix, Unit};

/// Extended complex number.
///
/// A quaternion has a real value and three complex values, represented by scalar and 3-vector
/// respectively:
/// ```math
/// \mathbf{q} = (s, \vec{v})
/// ```
///
/// Looking for the quaternion representation of 3D rotations? See [`Versor`].
///
/// ## Constructing quaternions
///
/// Create a quaternion with an array of coordinates (`[scalar, vector_0, vector_1, vector_2]`).
/// ```
/// use hoomd_vector::Quaternion;
///
/// let q = Quaternion::from([1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(q.scalar, 1.0);
/// assert_eq!(q.vector, [2.0, 3.0, 4.0].into());
/// ```
///
/// ## Quaternion properties
///
/// Compute a quaternion's norm:
/// ```
/// use hoomd_vector::Quaternion;
///
/// let q = Quaternion::from([3.0, 0.0, 4.0, 0.0]);
/// let norm = q.norm();
/// assert_eq!(norm, 5.0);
/// ```
///
/// Form the conjugate:
/// ```
/// use hoomd_vector::Quaternion;
///
/// let q = Quaternion::from([1.0, 2.0, 3.0, 4.0]);
/// let q_star = q.conjugate();
/// assert_eq!(q_star, [1.0, -2.0, -3.0, -4.0].into());
/// ```
///
/// ## Operating on quaternions
///
/// All operation examples use the following two quaternions:
/// ```
/// use hoomd_vector::Quaternion;
///
/// let a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
/// let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
/// ```
///
/// Addition:
///
/// ```
/// # use hoomd_vector::Quaternion;
/// # let a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
/// # let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
/// let c = a + b;
/// assert_eq!(c, [-1.0, 4.0, 10.0, -3.0].into());
/// ```
///
/// ```
/// # use hoomd_vector::Quaternion;
/// # let mut a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
/// # let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
/// a += b;
/// assert_eq!(a, [-1.0, 4.0, 10.0, -3.0].into());
/// ```
///
/// Subtraction:
///
/// ```
/// # use hoomd_vector::Quaternion;
/// # let a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
/// # let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
/// let c = a - b;
/// assert_eq!(c, [3.0, -8.0, 2.0, -5.0].into());
/// ```
///
/// ```
/// # use hoomd_vector::Quaternion;
/// # let mut a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
/// # let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
/// a -= b;
/// assert_eq!(a, [3.0, -8.0, 2.0, -5.0].into());
/// ```
///
/// Multiplication by a scalar:
///
/// ```
/// # use hoomd_vector::Quaternion;
/// # let a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
/// let c = a * 2.0;
/// assert_eq!(c, [2.0, -4.0, 12.0, -8.0].into());
/// ```
///
/// ```
/// # use hoomd_vector::Quaternion;
/// # let mut a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
/// # let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
/// a *= 2.0;
/// assert_eq!(a, [2.0, -4.0, 12.0, -8.0].into());
/// ```
///
/// Division by a scalar:
///
/// ```
/// # use hoomd_vector::Quaternion;
/// # let a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
/// let c = a / 2.0;
/// assert_eq!(c, [0.5, -1.0, 3.0, -2.0].into());
/// ```
///
/// ```
/// # use hoomd_vector::Quaternion;
/// # let mut a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
/// # let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
/// a /= 2.0;
/// assert_eq!(a, [0.5, -1.0, 3.0, -2.0].into());
/// ```
///
/// Quaternion multiplication:
///
/// ```
/// # use hoomd_vector::Quaternion;
/// # let a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
/// # let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
/// let c = a * b;
/// assert_eq!(c, [-10.0, 32.0, -30.0, -35.0].into());
/// ```
///
/// ```
/// # use hoomd_vector::Quaternion;
/// # let mut a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
/// # let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
/// a *= b;
/// assert_eq!(a, [-10.0, 32.0, -30.0, -35.0].into());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, RelativeEq)]
pub struct Quaternion {
    /// Scalar component
    pub scalar: f64,

    /// Vector component
    pub vector: Cartesian<3>,
}

impl Quaternion {
    /// The norm of the quaternion, squared.
    /// ```math
    /// |\mathbf{q}|^2
    /// ```
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::Quaternion;
    ///
    /// let q = Quaternion::from([3.0, 0.0, 4.0, 0.0]);
    /// let norm_squared = q.norm_squared();
    /// assert_eq!(norm_squared, 25.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        self.scalar * self.scalar + self.vector.dot(&self.vector)
    }

    /// The norm of the quaternion.
    /// ```math
    /// |\mathbf{q}|
    /// ```
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::Quaternion;
    ///
    /// let q = Quaternion::from([3.0, 0.0, 4.0, 0.0]);
    /// let norm = q.norm();
    /// assert_eq!(norm, 5.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Construct the conjugate of this quaternion.
    /// ```math
    /// \mathbf{q}^* = (s, -\vec{v})
    /// ```
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::Quaternion;
    ///
    /// let q = Quaternion::from([1.0, 2.0, 3.0, 4.0]);
    /// let q_star = q.conjugate();
    /// assert_eq!(q_star, [1.0, -2.0, -3.0, -4.0].into());
    /// ```
    #[inline]
    #[must_use]
    pub fn conjugate(self) -> Self {
        Self {
            scalar: self.scalar,
            vector: -self.vector,
        }
    }

    /// Create a [`Versor`] by normalizing the given quaternion.
    ///
    /// ```math
    /// \mathbf{v} = \frac{\mathbf{q}}{|\mathbf{q}|}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::{Quaternion, Versor};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let q = Quaternion::from([3.0, 0.0, 0.0, 4.0]);
    /// let v = q.to_versor()?;
    /// assert_eq!(*v.get(), [3.0 / 5.0, 0.0, 0.0, 4.0 / 5.0].into());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::InvalidQuaternionMagnitude`] when `self` is the 0 quaternion.
    #[inline]
    pub fn to_versor(self) -> Result<Versor, Error> {
        let mag = self.norm();
        if mag == 0.0 {
            Err(Error::InvalidQuaternionMagnitude)
        } else {
            Ok(Versor(self / mag))
        }
    }

    /// Create a [`Versor`] by normalizing the given quaternion.
    ///
    /// ```math
    /// \mathbf{v} = \frac{\mathbf{q}}{|\mathbf{q}|}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::{Quaternion, Versor};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let q = Quaternion::from([3.0, 0.0, 0.0, 4.0]);
    /// let v = q.to_versor_unchecked();
    /// assert_eq!(*v.get(), [3.0 / 5.0, 0.0, 0.0, 4.0 / 5.0].into());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// Divide by 0 when `self` is the 0 quaternion.
    #[inline]
    #[must_use]
    pub fn to_versor_unchecked(self) -> Versor {
        Versor(self / self.norm())
    }
}

impl From<[f64; 4]> for Quaternion {
    /// Construct a [`Quaternion`] from 4 values.
    ///
    /// The first value is the real part. The 2nd through 4th are the complex vector part:
    /// `[scalar, vector_0, vector_1, vector_2]`.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::Quaternion;
    ///
    /// let q = Quaternion::from([1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(q.scalar, 1.0);
    /// assert_eq!(q.vector, [2.0, 3.0, 4.0].into());
    /// ```
    #[inline]
    fn from(value: [f64; 4]) -> Self {
        Self {
            scalar: value[0],
            vector: [value[1], value[2], value[3]].into(),
        }
    }
}

impl fmt::Display for Quaternion {
    /// Format a [`Quaternion`] as `[{s}, [{v[0]}, {v[1]}, {v[2]}]]`.
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {}]", self.scalar, self.vector)
    }
}

impl Add for Quaternion {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            scalar: self.scalar + rhs.scalar,
            vector: self.vector + rhs.vector,
        }
    }
}

impl AddAssign for Quaternion {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.scalar += rhs.scalar;
        self.vector += rhs.vector;
    }
}

impl Div<f64> for Quaternion {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self {
        Self {
            scalar: self.scalar / rhs,
            vector: self.vector / rhs,
        }
    }
}

impl DivAssign<f64> for Quaternion {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.scalar /= rhs;
        self.vector /= rhs;
    }
}

impl Mul<f64> for Quaternion {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            scalar: self.scalar * rhs,
            vector: self.vector * rhs,
        }
    }
}

impl MulAssign<f64> for Quaternion {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.scalar *= rhs;
        self.vector *= rhs;
    }
}

impl Mul<Quaternion> for Quaternion {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Quaternion) -> Self {
        Self {
            scalar: (self.scalar * rhs.scalar - self.vector.dot(&rhs.vector)),
            vector: (rhs.vector * self.scalar
                + self.vector * rhs.scalar
                + self.vector.cross(&rhs.vector)),
        }
    }
}

impl MulAssign<Quaternion> for Quaternion {
    #[inline]
    fn mul_assign(&mut self, rhs: Quaternion) {
        let result = *self * rhs;
        self.scalar = result.scalar;
        self.vector = result.vector;
    }
}

impl Sub for Quaternion {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            scalar: self.scalar - rhs.scalar,
            vector: self.vector - rhs.vector,
        }
    }
}

impl SubAssign for Quaternion {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.scalar -= rhs.scalar;
        self.vector -= rhs.vector;
    }
}

/// A unit [`Quaternion`] that represents a 3D rotation.
///
/// [`Versor`] represents a 3D rotation with a **unit quaternion**. Rotation follows the Hamilton
/// convention.
///
/// ## Constructing a [`Versor`]:
///
/// The default [`Versor`] is the identity:
///
/// ```
/// use hoomd_vector::Versor;
///
/// let v = Versor::default();
/// assert_eq!(*v.get(), [1.0, 0.0, 0.0, 0.0].into());
/// ```
///
/// Create a [`Versor`] that rotates by an angle about an axis:
/// ```
/// use hoomd_vector::Versor;
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let v = Versor::from_axis_angle([0.0, 1.0, 0.0].try_into()?, PI / 2.0);
/// assert_eq!(
///     *v.get(),
///     [(PI / 4.0).cos(), 0.0, (PI / 4.0).sin(), 0.0].into()
/// );
/// # Ok(())
/// # }
/// ```
///
/// Create a [`Versor`] by normalizing a [`Quaternion`]:
/// ```
/// use hoomd_vector::{Quaternion, Versor};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let q = Quaternion::from([3.0, 0.0, 0.0, 4.0]);
/// let v = q.to_versor()?;
/// assert_eq!(*v.get(), [3.0 / 5.0, 0.0, 0.0, 4.0 / 5.0].into());
/// # Ok(())
/// # }
/// ```
///
/// Create a random [`Versor`]:
/// ```
/// use hoomd_vector::Versor;
/// use rand::{Rng, SeedableRng, rngs::StdRng};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut rng = StdRng::seed_from_u64(1);
/// let v: Versor = rng.random();
/// # Ok(())
/// # }
/// ```
///
/// ## Operations using [`Versor`]
///
/// Rotate a [`Cartesian<3>`] by a [`Versor`]:
/// ```
/// use approxim::assert_relative_eq;
/// use hoomd_vector::{Cartesian, Rotate, Rotation, Versor};
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let a = Cartesian::from([-1.0, 0.0, 0.0]);
/// let v = Versor::from_axis_angle([0.0, 0.0, 1.0].try_into()?, PI / 2.0);
/// let b = v.rotate(&a);
/// assert_relative_eq!(b, [0.0, -1.0, 0.0].into());
/// # Ok(())
/// # }
/// ```
///
/// Combine two rotations together:
/// ```
/// use hoomd_vector::{Rotation, Versor};
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let a = Versor::from_axis_angle([1.0, 0.0, 1.0].try_into()?, PI / 2.0);
/// let b = Versor::from_axis_angle([0.0, 0.0, 1.0].try_into()?, PI / 4.0);
/// let c = a.combine(&b);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, RelativeEq)]
pub struct Versor(Quaternion);

impl Versor {
    /// Take the dot product of the Versor as an element of $`\mathbb{R}^4`$.
    #[inline]
    fn dot_as_cartesian(&self, other: &Self) -> f64 {
        self.get().scalar * other.get().scalar + self.get().vector.dot(&other.get().vector)
    }
    /// Create a [`Versor`] that rotates by an angle (in radians)
    /// counterclockwise about an axis.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::Versor;
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v = Versor::from_axis_angle([0.0, 1.0, 0.0].try_into()?, PI / 2.0);
    /// assert_eq!(
    ///     *v.get(),
    ///     [(PI / 4.0).cos(), 0.0, (PI / 4.0).sin(), 0.0].into()
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: Unit<Cartesian<3>>, angle: f64) -> Self {
        let Unit(axis_vector) = axis;

        Versor(Quaternion {
            scalar: (angle / 2.0).cos(),
            vector: axis_vector * (angle / 2.0).sin(),
        })
    }

    /// Normalize the versor.
    ///
    /// Nominally, all [`Versor`] instances retain a unit norm. Due to limited
    /// floating point precision, this assumption may not hold after repeated
    /// operations. Normalize versors when needed to correct this issue.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::Versor;
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let a = Versor::from_axis_angle([0.0, 1.0, 0.0].try_into()?, PI / 2.0);
    /// let b = a.normalized();
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn normalized(self) -> Self {
        let Versor(q) = self;
        let f = 1.0 / q.norm();
        Self(Quaternion {
            scalar: q.scalar * f,
            vector: q.vector * f,
        })
    }

    /// Get the unit quaternion.
    #[inline]
    #[must_use]
    pub fn get(&self) -> &Quaternion {
        &self.0
    }

    /// A metric quantifying the angle (in radians) of the spherical arc separating two Versors.
    ///
    /// $`d : \mathbb{H} \times \mathbb{H} \to \mathbb{R}^+, \quad d(q_0, q_1) = \arccos(|q_0 \cdot q_1|)`$
    ///
    /// This value always lies in the range $`[0, \pi]`$, and is symmetric: while there
    /// are multiple arcs separating a pair of quaternions, this metric always chooses
    /// the shortest.
    #[inline]
    #[must_use]
    pub fn arc_distance(&self, other: &Self) -> f64 {
        self.dot_as_cartesian(other).acos()
    }
    /// A fast metric on Versors representing elements of SO(3).
    ///
    /// $`d : \mathbb{H} \times \mathbb{H} \to \mathbb{R}^+, \quad d(q_0, q_1) = 1 - |q_0 \cdot q_1 |`$
    ///
    /// This has less geometric meaning than the [`arc_distance`](Versor::arc_distance) metric. However, it
    /// is much faster while still obeying the triangle inequality and the axiom
    /// $`d(q_0, q_1) = d(q_1, q_0)`$. This metric always lies in the range
    /// $`[0, 1]`$, and is symmetric such that $`d(q, q)`$ = $`d(q, -q)`$.
    #[inline]
    #[must_use]
    pub fn half_euclidean_norm_squared(&self, other: &Self) -> f64 {
        1.0 - self.dot_as_cartesian(other)
    }
}

impl From<Versor> for RotationMatrix<3> {
    /// Construct a rotation matrix equivalent to this versor's rotation.
    ///
    /// When rotating many vectors by the same [`Versor`], improve performance
    /// by converting to a matrix first and applying that matrix to the vectors.
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_vector::{Cartesian, Rotate, RotationMatrix, Versor};
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let a = Cartesian::from([-1.0, 0.0, 0.0]);
    /// let v = Versor::from_axis_angle([0.0, 0.0, 1.0].try_into()?, PI / 2.0);
    ///
    /// let matrix = RotationMatrix::from(v);
    /// let b = matrix.rotate(&a);
    /// assert_relative_eq!(b, [0.0, -1.0, 0.0].into());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn from(versor: Versor) -> RotationMatrix<3> {
        let Versor(quaternion) = versor;
        let a = quaternion.scalar;
        let b = quaternion.vector[0];
        let c = quaternion.vector[1];
        let d = quaternion.vector[2];

        RotationMatrix {
            rows: [
                [
                    a * a + b * b - c * c - d * d,
                    2.0 * b * c - 2.0 * a * d,
                    2.0 * b * d + 2.0 * a * c,
                ]
                .into(),
                [
                    2.0 * b * c + 2.0 * a * d,
                    a * a - b * b + c * c - d * d,
                    2.0 * c * d - 2.0 * a * b,
                ]
                .into(),
                [
                    2.0 * b * d - 2.0 * a * c,
                    2.0 * c * d + 2.0 * a * b,
                    a * a - b * b - c * c + d * d,
                ]
                .into(),
            ],
        }
    }
}

impl Default for Versor {
    /// Create an identity rotation.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::Versor;
    ///
    /// let v = Versor::default();
    /// ```
    #[inline]
    fn default() -> Self {
        Self(Quaternion {
            scalar: 1.0,
            vector: [0.0, 0.0, 0.0].into(),
        })
    }
}

impl Rotate<Cartesian<3>> for Versor {
    type Matrix = RotationMatrix<3>;

    /// Rotate a [`Cartesian<3>`] by a [`Versor`]
    ///
    /// ```math
    /// \mathbf{q} \vec{a} \mathbf{q}^*
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_vector::{Cartesian, Rotate, Rotation, Versor};
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let a = Cartesian::from([-1.0, 0.0, 0.0]);
    /// let v = Versor::from_axis_angle([0.0, 0.0, 1.0].try_into()?, PI / 2.0);
    /// let b = v.rotate(&a);
    /// assert_relative_eq!(b, [0.0, -1.0, 0.0].into());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn rotate(&self, vector: &Cartesian<3>) -> Cartesian<3> {
        let Versor(quaternion) = self;

        *vector
            * (quaternion.scalar * quaternion.scalar - quaternion.vector.dot(&quaternion.vector))
            + quaternion.vector.cross(vector) * (2.0 * quaternion.scalar)
            + quaternion.vector * (2.0 * quaternion.vector.dot(vector))
    }
}

impl Rotation for Versor {
    /// Combine two rotations.
    ///
    /// The resulting versor is obtained by quaternion multiplication.
    /// ```math
    /// \mathbf{q}_{ab} = \mathbf{q}_a \mathbf{q}_b
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::{Rotation, Versor};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let q_a = Versor::from_axis_angle([0.0, 1.0, 0.0].try_into()?, 1.5);
    /// let q_b = Versor::from_axis_angle([1.0, 0.0, 0.0].try_into()?, 0.125);
    /// let q_ab = q_a.combine(&q_b);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn combine(&self, other: &Self) -> Self {
        let Versor(a) = self;
        let Versor(b) = other;

        Versor(a.mul(*b))
    }

    /// Create the identity [`Versor`]: [1, [0, 0, 0]]
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::{Rotation, Versor};
    ///
    /// let identity = Versor::identity();
    /// ```
    #[inline]
    fn identity() -> Self {
        Self::default()
    }

    /// Create a [`Versor`] that performs the inverse rotation of the given versor.
    ///
    /// ```math
    /// \mathbf{q}^*
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::{Rotation, Versor};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v = Versor::from_axis_angle([0.0, 1.0, 0.0].try_into()?, 1.5);
    /// let v_star = v.inverted();
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn inverted(self) -> Self {
        let Versor(quaternion) = self;

        Versor(quaternion.conjugate())
    }
}

impl fmt::Display for Versor {
    /// Format a [`Versor`] as `[{s}, [{v[0]}, {v[1]}, {v[2]}]]`.
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Distribution<Versor> for StandardUniform {
    /// Sample a random [`Versor`] from the uniform distribution over all rotations.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::Versor;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut rng = StdRng::seed_from_u64(1);
    /// let v: Versor = rng.random();
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Versor {
        // See method 19 from: https://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        let scalar = rng.sample::<f64, _>(StandardNormal);
        let vector = Cartesian::<3>::from(std::array::from_fn(|_| rng.sample(StandardNormal)));

        let norm = (vector.norm_squared() + (scalar * scalar)).sqrt();

        Versor(Quaternion {
            scalar: scalar / norm,
            vector: vector / norm,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::{assert_abs_diff_eq, assert_relative_eq};
    use rand::{SeedableRng, rngs::StdRng};
    use rstest::*;
    use std::f64::consts::PI;

    mod quaternion {
        use super::*;

        #[test]
        fn from_array() {
            let q = Quaternion::from([2.0, -3.0, 4.0, 7.0]);
            assert!(q.scalar == 2.0);
            assert!(q.vector == [-3.0, 4.0, 7.0].into());
        }

        #[test]
        fn norm() {
            let q = Quaternion::from([1.0, 4.0, -3.0, -2.0]);
            assert_eq!(q.norm_squared(), 30.0);
            assert_eq!(q.norm(), 30.0_f64.sqrt());
        }

        #[test]
        fn conjugate() {
            let q1 = Quaternion::from([1.0, -2.0, 4.0, -0.5]);
            let q2 = q1.conjugate();
            assert_eq!(q2, [1.0, 2.0, -4.0, 0.5].into());
            assert_relative_eq!(q2 * q1, [q2.norm() * q1.norm(), 0.0, 0.0, 0.0].into());
        }

        #[test]
        fn to_versor() {
            let q = Quaternion::from([5.0, 3.0, -1.0, 1.0]);

            assert_relative_eq!(
                q.to_versor()
                    .expect("hard-coded quatnernion should be non zero"),
                Versor(Quaternion {
                    scalar: 5.0 / 6.0,
                    vector: [3.0 / 6.0, -1.0 / 6.0, 1.0 / 6.0].into()
                })
            );

            assert_relative_eq!(
                q.to_versor_unchecked(),
                Versor(Quaternion {
                    scalar: 5.0 / 6.0,
                    vector: [3.0 / 6.0, -1.0 / 6.0, 1.0 / 6.0].into()
                })
            );

            let zero = Quaternion::from([0.0, 0.0, 0.0, 0.0]);
            assert!(matches!(
                zero.to_versor(),
                Err(Error::InvalidQuaternionMagnitude)
            ));
        }

        #[test]
        fn ops() {
            let a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
            let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);

            // +, +=
            assert_eq!(a + b, [-1.0, 4.0, 10.0, -3.0].into());
            let mut c = a;
            c += b;
            assert_eq!(c, [-1.0, 4.0, 10.0, -3.0].into());

            // -, -=
            assert_eq!(a - b, [3.0, -8.0, 2.0, -5.0].into());
            let mut c = a;
            c -= b;
            assert_eq!(c, [3.0, -8.0, 2.0, -5.0].into());

            // Scalar * and /
            assert_eq!(a * 2.0, [2.0, -4.0, 12.0, -8.0].into());
            let mut c = a;
            c *= 2.0;
            assert_eq!(c, [2.0, -4.0, 12.0, -8.0].into());

            assert_eq!(a / 2.0, [0.5, -1.0, 3.0, -2.0].into());
            let mut c = a;
            c /= 2.0;
            assert_eq!(c, [0.5, -1.0, 3.0, -2.0].into());

            // Quaternion multiplication
            assert_eq!(a * b, [-10.0, 32.0, -30.0, -35.0].into());
            let mut c = a;
            c *= b;
            assert_eq!(c, [-10.0, 32.0, -30.0, -35.0].into());
        }

        #[test]
        fn display() {
            let q = Quaternion {
                scalar: 0.5,
                vector: [0.125, -0.875, 2.125].into(),
            };
            let s = format!("{q}");
            assert_eq!(s, "[0.5, [0.125, -0.875, 2.125]]");
        }
    }

    mod versor {
        use super::*;
        #[test]
        fn default() {
            let a = Versor::default();
            assert!(a.get() == &[1.0, 0.0, 0.0, 0.0].into());
        }

        #[test]
        fn identity() {
            let a = Versor::identity();
            assert!(a.get() == &[1.0, 0.0, 0.0, 0.0].into());
        }

        #[rstest(
        theta => [0.0, PI / 2.0, 1e-12 * PI, -3.0, 12345.6],
        axis => [[1.0, 0.0, 0.0].try_into().expect("hard-coded vector should have non-zero length"), [1.0, -1.0, 1.0].try_into().expect("hard-coded vector should have non-zero length")],
    )]
        fn from_axis_angle(theta: f64, axis: Unit<Cartesian<3>>) {
            let Unit(axis_vector) = axis;

            let Versor(q) = Versor::from_axis_angle(axis, theta);
            assert_relative_eq!(q.scalar, (theta / 2.0).cos());
            assert_relative_eq!(q.vector, axis_vector * (theta / 2.0).sin());
        }

        #[rstest(
        theta_1 => [0.0, PI / 2.0, -3.0],
        theta_2 => [-0.0, -PI / 3.0, PI, 2.0 * PI]
    )]
        fn combine_same_axis(theta_1: f64, theta_2: f64) {
            let axis = [1.0, 0.0, 0.0]
                .try_into()
                .expect("hard-coded vector should have non-zero length");
            let Unit(axis_vector) = axis;

            let a = Versor::from_axis_angle(axis, theta_1);
            let b = Versor::from_axis_angle(axis, theta_2);
            let c = a.combine(&b);

            let theta = theta_1 + theta_2;
            let Versor(q) = c;
            assert_relative_eq!(q.scalar, (theta / 2.0).cos());
            assert_relative_eq!(q.vector, axis_vector * (theta / 2.0).sin());
        }

        fn validate_rotations<R: Rotate<Cartesian<3>>>(z_pi_2: &R, y_pi_4: &R) {
            assert_relative_eq!(
                z_pi_2.rotate(&[0.0, 0.0, 1.0].into()),
                [0.0, 0.0, 1.0].into()
            );
            assert_relative_eq!(
                z_pi_2.rotate(&[1.0, 0.0, 4.25].into()),
                [0.0, 1.0, 4.25].into()
            );
            assert_relative_eq!(
                z_pi_2.rotate(&[0.0, 1.0, -8.75].into()),
                [-1.0, 0.0, -8.75].into()
            );

            let sqrt_2_2 = 2.0_f64.sqrt() / 2.0;
            assert_relative_eq!(
                y_pi_4.rotate(&[0.0, -10.0, 0.0].into()),
                [0.0, -10.0, 0.0].into()
            );
            assert_relative_eq!(
                y_pi_4.rotate(&[1.0, -15.0, 0.0].into()),
                [sqrt_2_2, -15.0, -sqrt_2_2].into()
            );
            assert_relative_eq!(
                y_pi_4.rotate(&[sqrt_2_2, -15.0, -sqrt_2_2].into()),
                [0.0, -15.0, -1.0].into()
            );
        }

        #[test]
        fn rotate() {
            let z_pi_2 = Versor::from_axis_angle(
                [0.0, 0.0, 1.0]
                    .try_into()
                    .expect("hard-coded vector should have non-zero length"),
                PI / 2.0,
            );
            let y_pi_4 = Versor::from_axis_angle(
                [0.0, 1.0, 0.0]
                    .try_into()
                    .expect("hard-coded vector should have non-zero length"),
                PI / 4.0,
            );

            validate_rotations(&z_pi_2, &y_pi_4);
        }

        #[test]
        fn precompute() {
            let z_pi_2 = RotationMatrix::from(Versor::from_axis_angle(
                [0.0, 0.0, 1.0]
                    .try_into()
                    .expect("hard-coded vector should have non-zero length"),
                PI / 2.0,
            ));
            let y_pi_4 = RotationMatrix::from(Versor::from_axis_angle(
                [0.0, 1.0, 0.0]
                    .try_into()
                    .expect("hard-coded vector should have non-zero length"),
                PI / 4.0,
            ));

            validate_rotations(&z_pi_2, &y_pi_4);
        }

        #[test]
        fn combine_different_axis() {
            let a = Versor::from_axis_angle(
                [1.0, 0.0, 0.0]
                    .try_into()
                    .expect("hard-coded vector should have non-zero length"),
                PI / 4.0,
            );
            let b = Versor::from_axis_angle(
                [0.0, 0.0, 1.0]
                    .try_into()
                    .expect("hard-coded vector should have non-zero length"),
                PI / 2.0,
            );

            let q = a.combine(&b);
            let v = q.rotate(&[1.0, 0.0, 0.0].into());
            assert_relative_eq!(v, [0.0, 2.0_f64.sqrt() / 2.0, 2.0_f64.sqrt() / 2.0].into());
        }

        #[rstest(theta => [0.0, 1.0, 2.125])]
        fn inverted(theta: f64) {
            let q1 = Versor::from_axis_angle(
                [1.0, 0.5, -2.0]
                    .try_into()
                    .expect("hard-coded vector should have non-zero length"),
                theta,
            );
            let q2 = q1.inverted();
            assert_relative_eq!(q1.combine(&q2), Versor::identity());
        }

        #[test]
        fn display() {
            let v = Versor(Quaternion {
                scalar: 0.5,
                vector: [0.125, -0.875, 2.125].into(),
            });
            let s = format!("{v}");
            assert_eq!(s, "[0.5, [0.125, -0.875, 2.125]]");
        }

        #[test]
        fn normalized() {
            let v = Versor(Quaternion {
                scalar: 5.0,
                vector: [3.0, -1.0, 1.0].into(),
            });
            assert_relative_eq!(
                v.normalized(),
                Versor(Quaternion {
                    scalar: 5.0 / 6.0,
                    vector: [3.0 / 6.0, -1.0 / 6.0, 1.0 / 6.0].into()
                })
            );
        }

        #[test]
        fn random() {
            const CHECK_VECTORS: [Cartesian<3>; 3] = [
                Cartesian {
                    coordinates: [1.0, 0.0, 0.0],
                },
                Cartesian {
                    coordinates: [0.0, 1.0, 0.0],
                },
                Cartesian {
                    coordinates: [1.0, 0.0, 1.0],
                },
            ];

            // Perform basic checks on random versors.
            // 1) Ensure that each randomly generated versor is unit.
            // 2) Check that the result of rotating a reference vector by random versors does not
            // point in any special direction. The average dot product should be close to 0.
            let samples: u32 = 20_000;

            let reference = Cartesian::from([1.0, 0.0, 0.0]);
            let mut dot_sums = [0.0; CHECK_VECTORS.len()];

            let mut rng = StdRng::seed_from_u64(1);

            for _ in 0..samples {
                let q: Versor = rng.random();
                assert_relative_eq!(q.get().norm_squared(), 1.0, max_relative = 1e-15);

                let v = q.rotate(&reference);
                for i in 0..CHECK_VECTORS.len() {
                    dot_sums[i] += v.dot(&CHECK_VECTORS[i]);
                }
            }

            for dot_sum in dot_sums {
                assert_abs_diff_eq!(dot_sum / f64::from(samples), 0.0, epsilon = 0.01);
            }
        }
    }
}

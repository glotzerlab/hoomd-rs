// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Biquaternion`] and a four-dimensional matrix representation
//! of SO(3,1).

use num::complex::Complex;
use rand::{
    Rng,
    distr::{Distribution, StandardUniform, Uniform},
};
use std::{
    array, fmt,
    iter::zip,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use crate::{Error, HyperbolicRotate, HyperbolicRotationMatrix, Minkowski};
#[expect(unused_imports, reason = "Needed for doc link")]
use hoomd_vector::Quaternion;

/// ## A quaternion with complex coefficients.
///
/// Biquaternions are the set of numbers $`a + b\mathbf{i} + c\mathbf{j} + d\mathbf{k}`$
/// where $`a,b,c,d`$ are complex numbers and $`\{1,\mathbf{i},\mathbf{j},\mathbf{k}\}`$
/// are the quaternion algebra. Biquaternions can be thought of as a
/// generalization of quaternions which allow for complex coefficients.
/// Analogous to quaternions and SO(3), biquaternions furnish a representation
/// of SO(3,1)
///
/// ## Construction of Biquaternions
///
/// Create a biquaternion from an array of four complex numbers. Note that
/// components are in the order $`[\mathbf{i},\mathbf{j},\mathbf{k},1]`$
/// (i.e., the scalar component is at the end)
/// ```
/// use hoomd_manifold::Biquaternion;
/// use num::complex::Complex;
///
/// let q = Biquaternion::from([
///     Complex::new(1.0, 4.0),
///     Complex::new(2.0, 3.0),
///     Complex::new(3.0, 2.0),
///     Complex::new(4.0, 1.0),
/// ]);
/// assert_eq!(4.0, q.components[0].im);
/// ```
///
/// ## Operations with Biquaternions.
///
/// Similar to [`Quaternion`], biquaternions support vector operations
/// (addition, multiplication by a scalar, etc.):
/// ```
/// use hoomd_manifold::Biquaternion;
/// use num::complex::Complex;
///
/// let mut a = Biquaternion::from([
///     Complex::new(1.0, 0.0),
///     Complex::new(2.0, 0.0),
///     Complex::new(3.0, 0.0),
///     Complex::new(0.0, 1.0),
/// ]);
/// let mut b = Biquaternion::from([
///     Complex::new(0.0, 4.0),
///     Complex::new(0.0, 3.0),
///     Complex::new(0.0, 2.0),
///     Complex::new(1.0, 0.0),
/// ]);
/// b /= 2.0;
/// let mut c = a + b;
/// assert_eq!(Complex::new(1.0, 2.0), c.components[0]);
/// ```
///
/// Biquaternions also support the following operations:
///
/// Hamiltonian conjugate/ biconjugate:
/// Denoted by the method "bar", the Hamiltonian conjugate multiplies the
/// vector part of the biquaternion by -1.0.
/// ```
/// use hoomd_manifold::Biquaternion;
/// use num::complex::Complex;
///
/// let q = Biquaternion::from([
///     Complex::new(-1.0, 0.0),
///     Complex::new(-1.0, 2.0),
///     Complex::new(1.0, 0.0),
///     Complex::new(1.0, 0.0),
/// ]);
/// let p = Biquaternion::from([
///     Complex::new(1.0, 0.0),
///     Complex::new(1.0, -2.0),
///     Complex::new(-1.0, 0.0),
///     Complex::new(1.0, 0.0),
/// ]);
///
/// assert_eq!(p, q.bar());
/// ```
///
/// Complex conjugation:
/// Denoted by method "conj", takes the complex conjugate of all components of
/// the biquaternion
/// ```
/// use hoomd_manifold::Biquaternion;
/// use num::complex::Complex;
///
/// let q = Biquaternion::from([
///     Complex::new(1.0, 8.0),
///     Complex::new(2.0, 7.0),
///     Complex::new(3.0, 6.0),
///     Complex::new(4.0, 5.0),
/// ]);
/// let p = Biquaternion::from([
///     Complex::new(1.0, -8.0),
///     Complex::new(2.0, -7.0),
///     Complex::new(3.0, -6.0),
///     Complex::new(4.0, -5.0),
/// ]);
///
/// assert_eq!(p, q.conj());
/// ```
///
/// Biquaternion Product:
/// The biquaternion product takes two biquaternions and outputs another
/// biquaternion.
/// ```
/// use hoomd_manifold::Biquaternion;
/// use num::complex::Complex;
///
/// let q = Biquaternion::from([
///     Complex::new(2.0, 0.0),
///     Complex::new(0.0, 1.0),
///     Complex::new(1.0, 0.0),
///     Complex::new(1.0, 0.0),
/// ]);
/// let p = Biquaternion::from([
///     Complex::new(3.0, 0.0),
///     Complex::new(2.0, 0.0),
///     Complex::new(1.0, 0.0),
///     Complex::new(0.0, 1.0),
/// ]);
/// let c = Biquaternion::from([
///     Complex::new(1.0, 3.0),
///     Complex::new(2.0, 0.0),
///     Complex::new(5.0, -2.0),
///     Complex::new(-7.0, -1.0),
/// ]);
/// assert_eq!(c, q.dot(&p));
/// ```
///
/// Scalar Product:
/// The scalar product takes two biquaternions and outputs a complex number
/// according to
/// ```math
/// \frac{1}{2}(a\overline{b} + b\overline{a})
/// ```
/// ```
/// use hoomd_manifold::Biquaternion;
/// use num::complex::Complex;
///
/// let q = Biquaternion::from([
///     Complex::new(2.0, 0.0),
///     Complex::new(0.0, 1.0),
///     Complex::new(1.0, 0.0),
///     Complex::new(1.0, 0.0),
/// ]);
/// let p = Biquaternion::from([
///     Complex::new(3.0, 0.0),
///     Complex::new(2.0, 0.0),
///     Complex::new(1.0, 0.0),
///     Complex::new(0.0, 1.0),
/// ]);
/// assert_eq!(Complex::new(7.0, 3.0), q.scalar_product(&p));
/// ```
///
/// Biquaternion Norm:
/// The scalar product furnishes a "norm" for the biquaternion.
/// ```
/// use hoomd_manifold::Biquaternion;
/// use num::complex::Complex;
///
/// let q = Biquaternion::from([
///     Complex::new(3.0, 0.0),
///     Complex::new(0.0, 1.0),
///     Complex::new(4.0, 0.0),
///     Complex::new(0.0, 2.0),
/// ]);
/// assert_eq!(Complex::new(20.0_f64, 0.0).sqrt(), q.norm());
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Biquaternion {
    /// Components of the biquaternion, in the order [i,j,k,1].
    pub components: [Complex<f64>; 4],
}

impl Biquaternion {
    /// Compute the Hamiltonian conjugate or biconjugate of a biquaternion.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Biquaternion;
    /// use num::complex::Complex;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let q = Biquaternion::from([
    ///     Complex::new(-1.0, 0.0),
    ///     Complex::new(0.0, 1.0),
    ///     Complex::new(1.0, 0.0),
    ///     Complex::new(1.0, 0.0),
    /// ]);
    /// let p = Biquaternion::from([
    ///     Complex::new(1.0, 0.0),
    ///     Complex::new(0.0, -1.0),
    ///     Complex::new(-1.0, 0.0),
    ///     Complex::new(1.0, 0.0),
    /// ]);
    /// assert_eq!(p, q.bar());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn bar(&self) -> Self {
        Biquaternion::from([
            (self.components[0]).scale(-1.0),
            (self.components[1]).scale(-1.0),
            (self.components[2]).scale(-1.0),
            (self.components[3]),
        ])
    }
    /// Compute the complex conjugate of a biquaternion.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Biquaternion;
    /// use num::complex::Complex;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let q = Biquaternion::from([
    ///     Complex::new(1.0, 0.0),
    ///     Complex::new(0.0, 1.0),
    ///     Complex::new(1.0, 0.0),
    ///     Complex::new(1.0, 2.0),
    /// ]);
    /// let p = Biquaternion::from([
    ///     Complex::new(1.0, 0.0),
    ///     Complex::new(0.0, -1.0),
    ///     Complex::new(1.0, 0.0),
    ///     Complex::new(1.0, -2.0),
    /// ]);
    /// assert_eq!(p, q.conj());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn conj(&self) -> Self {
        Biquaternion::from([
            (self.components[0]).conj(),
            (self.components[1]).conj(),
            (self.components[2]).conj(),
            (self.components[3]).conj(),
        ])
    }
    /// Compute the squared norm of a biquaternion.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Biquaternion;
    /// use num::complex::Complex;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let q = Biquaternion::from([
    ///     Complex::new(1.0, 0.0),
    ///     Complex::new(0.0, 1.0),
    ///     Complex::new(1.0, 0.0),
    ///     Complex::new(1.0, 0.0),
    /// ]);
    /// assert_eq!(2.0, q.norm_squared().re);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn norm_squared(&self) -> Complex<f64> {
        self.scalar_product(self)
    }
    /// the norm of a biquaternion
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Biquaternion;
    /// use num::complex::Complex;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let q = Biquaternion::from([
    ///     Complex::new(3.0, 0.0),
    ///     Complex::new(0.0, 1.0),
    ///     Complex::new(4.0, 0.0),
    ///     Complex::new(1.0, 0.0),
    /// ]);
    /// assert_eq!(Complex::new(5.0, 0.0), q.norm());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn norm(&self) -> Complex<f64> {
        self.norm_squared().sqrt()
    }
    /// Compute the quaternion product of two biquaternions.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Biquaternion;
    /// use num::complex::Complex;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let q = Biquaternion::from([
    ///     Complex::new(2.0, 0.0),
    ///     Complex::new(0.0, 1.0),
    ///     Complex::new(1.0, 0.0),
    ///     Complex::new(1.0, 0.0),
    /// ]);
    /// let p = Biquaternion::from([
    ///     Complex::new(3.0, 0.0),
    ///     Complex::new(2.0, 0.0),
    ///     Complex::new(1.0, 0.0),
    ///     Complex::new(0.0, 1.0),
    /// ]);
    /// let c = Biquaternion::from([
    ///     Complex::new(1.0, 3.0),
    ///     Complex::new(2.0, 0.0),
    ///     Complex::new(5.0, -2.0),
    ///     Complex::new(-7.0, -1.0),
    /// ]);
    /// assert_eq!(c, q.dot(&p));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn dot(&self, other: &Self) -> Self {
        Biquaternion::from([
            self.components[3] * other.components[0]
                + other.components[3] * self.components[0]
                + self.components[1] * other.components[2]
                - other.components[1] * self.components[2],
            self.components[3] * other.components[1]
                + other.components[3] * self.components[1]
                + self.components[2] * other.components[0]
                - other.components[2] * self.components[0],
            self.components[3] * other.components[2]
                + other.components[3] * self.components[2]
                + self.components[0] * other.components[1]
                - other.components[0] * self.components[1],
            self.components[3] * other.components[3]
                - self.components[0] * other.components[0]
                - self.components[1] * other.components[1]
                - self.components[2] * other.components[2],
        ])
    }
    /// Compute the scalar product of two biquaternions.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Biquaternion;
    /// use num::complex::Complex;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let q = Biquaternion::from([
    ///     Complex::new(2.0, 0.0),
    ///     Complex::new(0.0, 1.0),
    ///     Complex::new(1.0, 0.0),
    ///     Complex::new(1.0, 0.0),
    /// ]);
    /// let p = Biquaternion::from([
    ///     Complex::new(3.0, 0.0),
    ///     Complex::new(2.0, 0.0),
    ///     Complex::new(1.0, 0.0),
    ///     Complex::new(0.0, 1.0),
    /// ]);
    /// assert_eq!(Complex::new(7.0, 3.0), q.scalar_product(&p));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn scalar_product(&self, other: &Self) -> Complex<f64> {
        zip(self.components.iter(), other.components.iter())
            .fold(Complex::new(0.0, 0.0), |product, x| product + x.0 * x.1)
    }
    /// Create a [`UnitBiquaternion`] by normalizing the given biquaternion.
    #[inline]
    #[expect(clippy::missing_errors_doc, reason = "maps to error message")]
    pub fn to_unit(self) -> Result<UnitBiquaternion, Error> {
        let mag = self.norm();
        if mag == Complex::new(0.0, 0.0) {
            return Err(Error::InvalidBiquaternionMagnitude);
        }
        Ok(UnitBiquaternion(self / mag))
    }
    /// Create a [`UnitBiquaternion`] by normalizing the given biquaternion
    /// without returning an Option type.
    #[inline]
    #[must_use]
    pub fn to_unit_unchecked(self) -> UnitBiquaternion {
        UnitBiquaternion(self)
    }
}

impl Default for Biquaternion {
    /// Create a biquaternion with all zeros.
    #[inline]
    fn default() -> Self {
        Self {
            components: array::from_fn(|_| Complex::new(0.0, 0.0)),
        }
    }
}

impl From<[Complex<f64>; 4]> for Biquaternion {
    /// Construct a [`Biquaternion`] from 4 complex values.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Biquaternion;
    /// use num::complex::Complex;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let q = Biquaternion::from([
    ///     Complex::new(1.0, 0.0),
    ///     Complex::new(0.0, 0.1),
    ///     Complex::new(1.0, 0.0),
    ///     Complex::new(1.0, 1.0),
    /// ]);
    /// assert_eq!(
    ///     q.components,
    ///     [
    ///         Complex::new(1.0, 0.0),
    ///         Complex::new(0.0, 0.1),
    ///         Complex::new(1.0, 0.0),
    ///         Complex::new(1.0, 1.0)
    ///     ]
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn from(value: [Complex<f64>; 4]) -> Self {
        Self { components: value }
    }
}

impl fmt::Display for Biquaternion {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[{}, {}, {}, {}]",
            self.components[0], self.components[1], self.components[2], self.components[3]
        )
    }
}

impl Add for Biquaternion {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            components: array::from_fn(|i| self.components[i] + rhs.components[i]),
        }
    }
}

impl AddAssign for Biquaternion {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for n in 0..4 {
            self.components[n] += rhs.components[n];
        }
    }
}

impl Sub for Biquaternion {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            components: array::from_fn(|i| self.components[i] - rhs.components[i]),
        }
    }
}

impl SubAssign for Biquaternion {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for n in 0..4 {
            self.components[n] -= rhs.components[n];
        }
    }
}

impl Mul<f64> for Biquaternion {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            components: array::from_fn(|i| (self.components[i]).scale(rhs)),
        }
    }
}

impl Mul<Complex<f64>> for Biquaternion {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Complex<f64>) -> Self {
        Self {
            components: array::from_fn(|i| self.components[i] * rhs),
        }
    }
}

impl MulAssign<f64> for Biquaternion {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        for n in 0..4 {
            self.components[n] *= Complex::new(rhs, 0.0);
        }
    }
}
impl Div<f64> for Biquaternion {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self {
        Self {
            components: array::from_fn(|i| (self.components[i]).scale(1.0 / rhs)),
        }
    }
}

impl Div<Complex<f64>> for Biquaternion {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Complex<f64>) -> Self {
        Self {
            components: array::from_fn(|i| self.components[i] / rhs),
        }
    }
}

impl DivAssign<f64> for Biquaternion {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        for n in 0..4 {
            self.components[n] /= Complex::new(rhs, 0.0);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// ## Represent SO(3,1) with a normalized biquaternion.
///
/// Unit-norm Biquaternions furnish a representation of SO(3,1), analogous to
/// quaternions and SO(3). If $`\vec{x} = (x_1, x_2, x_3, x_4)`$ is a vector
/// in Minkowski space, then $\vec{x}$ can be mapped to a biquaternion
/// ```math
/// \vec{x} \mapsto X = [x_1, x_2, x_3,h x_4]
/// ```
/// (where h is the imaginary number) whose squared norm is
/// ```math
/// |X|^2 = x_1^2 + x_2^2 + x_3^2 - x_4^2
/// ```
/// It can be shown that, for
/// a unit biquaternion $`q`$, the transformation
/// ```math
/// q^* X \overline{q} = X'
/// ```
/// preserves the norm, i.e.,
/// ```math
/// |X|^2 = |X'|^2
/// ```
/// We therefore have that this action by unit biquaternions
/// produces a representation of SO(3,1). The biquaternion algebra can be used
/// directly to transform Minkowski 4-vectors, or unit biquaternions can be
/// represented as matrices using [`HyperbolicRotationMatrix<4>`].
///
/// Like quaternions, the unit biquaternion
/// ```math
/// q = \cos(\theta/2) + \bf{i}\sin(\theta/2)
/// ```
/// generates a rotation about the $` \mathbf{i} `$ axis by angle $`\theta`$:
/// ```
/// use approx::assert_relative_eq;
/// use hoomd_manifold::{
///     Biquaternion, HyperbolicRotate, HyperbolicRotationMatrix, Minkowski,
///     UnitBiquaternion,
/// };
/// use num::complex::Complex;
/// use std::f64::consts::PI;
///
/// let q = Biquaternion::from([
///     Complex::new((PI / 4.0).sin(), 0.0),
///     Complex::new(0.0, 0.0),
///     Complex::new(0.0, 0.0),
///     Complex::new((PI / 4.0).cos(), 0.0),
/// ]);
/// let v = q.to_unit();
/// let x = Minkowski::from([1.0, 1.0, 1.0, 1.0]);
/// let rotation =
///     HyperbolicRotationMatrix::from(v.expect("non-zero biquaternion"));
/// let rotated = rotation.hyperbolic_rotate(&x);
/// assert_relative_eq!(rotated.coordinates[0], 1.0, epsilon = 1e-12);
/// assert_relative_eq!(rotated.coordinates[1], -1.0, epsilon = 1e-12);
/// assert_relative_eq!(rotated.coordinates[2], 1.0, epsilon = 1e-12);
/// assert_relative_eq!(rotated.coordinates[3], 1.0, epsilon = 1e-12);
/// ```
///
/// However, biquaternions also generate boosts via
/// ```math
/// q = \cosh(v) + \mathbf{i}h\sinh(v)
/// ```
/// which represents a boost of rapidity $`v`$ in the $`\mathbf{i}`$ direction:
/// ```
/// use approx::assert_relative_eq;
/// use hoomd_manifold::{
///     Biquaternion, HyperbolicRotate, HyperbolicRotationMatrix, Minkowski,
///     UnitBiquaternion,
/// };
/// use num::complex::Complex;
/// use std::f64::consts::PI;
///
/// let q = Biquaternion::from([
///     Complex::new(0.0, (0.2_f64).sinh()),
///     Complex::new(0.0, 0.0),
///     Complex::new(0.0, 0.0),
///     Complex::new((0.2_f64).cosh(), 0.0),
/// ]);
/// let v = q.to_unit();
/// let x = Minkowski::from([0.0, 0.0, 0.0, 1.0]);
/// let boost = HyperbolicRotationMatrix::from(
///     v.expect("hard-coded unit biquaternion"),
/// );
/// let boosted = boost.hyperbolic_rotate(&x);
/// assert_relative_eq!(
///     boosted.coordinates[0],
///     (0.4_f64).sinh(),
///     epsilon = 1e-12
/// );
/// assert_relative_eq!(boosted.coordinates[1], 0.0, epsilon = 1e-12);
/// assert_relative_eq!(boosted.coordinates[2], 0.0, epsilon = 1e-12);
/// assert_relative_eq!(
///     boosted.coordinates[3],
///     (0.4_f64).cosh(),
///     epsilon = 1e-12
/// );
/// ```
pub struct UnitBiquaternion(Biquaternion);

impl UnitBiquaternion {
    /// Normalize a biquaternion.
    #[inline]
    #[must_use]
    pub fn normalized(self) -> Self {
        let UnitBiquaternion(q) = self;
        let f = 1.0 / q.norm();
        Self(q * f)
    }
    /// Compute the square of the norm of a biquaternion.
    #[inline]
    #[must_use]
    pub fn norm_squared(self) -> Complex<f64> {
        let UnitBiquaternion(q) = self;
        q.norm_squared()
    }
}

impl Distribution<UnitBiquaternion> for StandardUniform {
    /// Sample a random [`UnitBiquaternion`]
    ///
    /// # Example
    ///
    /// ```
    /// use approx::assert_relative_eq;
    /// use hoomd_manifold::{Biquaternion, UnitBiquaternion};
    /// use num::complex::Complex;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut rng = StdRng::seed_from_u64(1);
    /// let v: UnitBiquaternion = rng.random();
    /// assert_relative_eq!(v.norm_squared().re, 1.0, epsilon = 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> UnitBiquaternion {
        #[expect(
            clippy::expect_used,
            reason = "This constants chosen for this distribution are valid"
        )]
        let uniform = Uniform::new(-1.0, 1.0).expect("hard-coded distribution should be valid");

        let array_re: [f64; 4] = array::from_fn(|_| uniform.sample(rng));
        let array_im: [f64; 4] = array::from_fn(|_| uniform.sample(rng));
        let mut scale =
            zip(array_re.iter(), array_im.iter()).fold(Complex::new(0.0, 0.0), |product, x| {
                product + Complex::new((x.0).powi(2) - (x.1).powi(2), 2.0_f64 * (x.1) * (x.0))
            });
        scale = scale.sqrt();
        UnitBiquaternion(Biquaternion {
            components: array::from_fn(|i| Complex::new(array_re[i], array_im[i]) / scale),
        })
    }
}

impl From<UnitBiquaternion> for HyperbolicRotationMatrix<4> {
    #[inline]
    #[expect(clippy::many_single_char_names, reason = "dummy variables")]
    fn from(q: UnitBiquaternion) -> HyperbolicRotationMatrix<4> {
        let UnitBiquaternion(biquaternion) = q;
        let [a, b, c, d]: [Complex<f64>; 4] = array::from_fn(|i| biquaternion.components[i]);

        HyperbolicRotationMatrix {
            rows: [
                [
                    (d * d.conj() + a * a.conj() - b * b.conj() - c * c.conj()).re,
                    (a * b.conj() + b * a.conj() - c * d.conj() - d * c.conj()).re,
                    (a * c.conj() + c * a.conj() + b * d.conj() + d * b.conj()).re,
                    -(d * a.conj() - a * d.conj() + b * c.conj() - c * b.conj()).im,
                ]
                .into(),
                [
                    (b * a.conj() + a * b.conj() + c * d.conj() + d * c.conj()).re,
                    (d * d.conj() - a * a.conj() + b * b.conj() - c * c.conj()).re,
                    (b * c.conj() + c * b.conj() - a * d.conj() - d * a.conj()).re,
                    -(d * b.conj() - b * d.conj() + c * a.conj() - a * c.conj()).im,
                ]
                .into(),
                [
                    (c * a.conj() + a * c.conj() - b * d.conj() - d * b.conj()).re,
                    (c * b.conj() + b * c.conj() + a * d.conj() + d * a.conj()).re,
                    (d * d.conj() - a * a.conj() - b * b.conj() + c * c.conj()).re,
                    -(d * c.conj() - c * d.conj() + a * b.conj() - b * a.conj()).im,
                ]
                .into(),
                [
                    (a * d.conj() - d * a.conj() + b * c.conj() - c * b.conj()).im,
                    (b * d.conj() - d * b.conj() + c * a.conj() - a * c.conj()).im,
                    (c * d.conj() - d * c.conj() + a * b.conj() - b * a.conj()).im,
                    (a * a.conj() + b * b.conj() + c * c.conj() + d * d.conj()).re,
                ]
                .into(),
            ],
        }
    }
}

impl HyperbolicRotate<Minkowski<4>> for UnitBiquaternion {
    type Matrix = HyperbolicRotationMatrix<4>;

    /// Transform a [`Minkowski<4>`] by a [`UnitBiquaternion`].
    ///
    /// ```math
    /// \overline{\mathbf{q}} \vec{a} \mathbf{q}^*
    /// ```
    ///
    /// # Examples
    /// Rotation about z axis:
    /// ```
    /// use approx::assert_relative_eq;
    /// use hoomd_manifold::{
    ///     Biquaternion, HyperbolicRotate, Minkowski, UnitBiquaternion,
    /// };
    /// use num::complex::Complex;
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let x = Minkowski::from([1.0, 0.0, 0.0, 1.0]);
    /// let q = Biquaternion::from([
    ///     Complex::new(0.0, 0.0),
    ///     Complex::new(0.0, 0.0),
    ///     Complex::new((PI / 4.0).sin(), 0.0),
    ///     Complex::new((PI / 4.0).cos(), 0.0),
    /// ]);
    /// let v = q.to_unit_unchecked();
    /// let rotated = v.hyperbolic_rotate(&x);
    /// assert_relative_eq!(rotated.coordinates[0], 0.0, epsilon = 1e-12);
    /// assert_relative_eq!(rotated.coordinates[1], 1.0, epsilon = 1e-12);
    /// assert_relative_eq!(rotated.coordinates[2], 0.0, epsilon = 1e-12);
    /// assert_relative_eq!(rotated.coordinates[3], 1.0, epsilon = 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Boost in x direction:
    /// ```
    /// use approx::assert_relative_eq;
    /// use hoomd_manifold::{
    ///     Biquaternion, HyperbolicRotate, Minkowski, UnitBiquaternion,
    /// };
    /// use num::complex::Complex;
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let x = Minkowski::from([0.0, 0.0, 0.0, 1.0]);
    /// let q = Biquaternion::from([
    ///     Complex::new(0.0, PI / 4.0).sin(),
    ///     Complex::new(0.0, 0.0),
    ///     Complex::new(0.0, 0.0),
    ///     Complex::new(0.0, PI / 4.0).cos(),
    /// ]);
    /// let v = q.to_unit_unchecked();
    /// let boosted = v.hyperbolic_rotate(&x);
    /// assert_relative_eq!(
    ///     boosted.coordinates[0],
    ///     (PI / 2.0).sinh(),
    ///     epsilon = 1e-12
    /// );
    /// assert_relative_eq!(boosted.coordinates[1], 0.0, epsilon = 1e-12);
    /// assert_relative_eq!(boosted.coordinates[2], 0.0, epsilon = 1e-12);
    /// assert_relative_eq!(
    ///     boosted.coordinates[3],
    ///     (PI / 2.0).cosh(),
    ///     epsilon = 1e-12
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn hyperbolic_rotate(&self, vector: &Minkowski<4>) -> Minkowski<4> {
        let UnitBiquaternion(biquaternion) = self;
        let x = Biquaternion::from([
            Complex::new(vector[0], 0.0),
            Complex::new(vector[1], 0.0),
            Complex::new(vector[2], 0.0),
            Complex::new(0.0, vector[3]),
        ]);
        let x_transformed = (biquaternion.conj()).dot(&x.dot(&(biquaternion.bar())));
        Minkowski::from([
            x_transformed.components[0].re,
            x_transformed.components[1].re,
            x_transformed.components[2].re,
            x_transformed.components[3].im,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num::complex::Complex;
    use rstest::*;
    use std::f64::consts::PI;

    #[test]
    fn from_array() {
        let q = Biquaternion::from([
            Complex::new(-1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
        ]);
        assert!(q.components[0] == Complex::new(-1.0, 0.0));
        assert!(q.components[1] == Complex::new(0.0, 1.0));
        assert!(q.components[2] == Complex::new(1.0, 0.0));
        assert!(q.components[3] == Complex::new(1.0, 0.0));
    }

    #[test]
    fn bar() {
        let q = Biquaternion::from([
            Complex::new(-2.0, 0.0),
            Complex::new(-1.0, 1.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
        ]);
        let p = Biquaternion::from([
            Complex::new(2.0, 0.0),
            Complex::new(1.0, -1.0),
            Complex::new(-1.0, 0.0),
            Complex::new(1.0, 0.0),
        ]);
        assert_eq!(p, q.bar());
    }

    #[test]
    fn conjugate() {
        let q = Biquaternion::from([
            Complex::new(1.0, 8.0),
            Complex::new(2.0, 7.0),
            Complex::new(3.0, 6.0),
            Complex::new(4.0, 5.0),
        ]);
        let p = Biquaternion::from([
            Complex::new(1.0, -8.0),
            Complex::new(2.0, -7.0),
            Complex::new(3.0, -6.0),
            Complex::new(4.0, -5.0),
        ]);
        assert_eq!(p, q.conj());
    }

    #[test]
    fn biquat_product() {
        let q = Biquaternion::from([
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
        ]);
        let p = Biquaternion::from([
            Complex::new(3.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
        ]);
        let c = Biquaternion::from([
            Complex::new(1.0, 3.0),
            Complex::new(2.0, 0.0),
            Complex::new(5.0, -2.0),
            Complex::new(-7.0, -1.0),
        ]);
        assert_eq!(c, q.dot(&p));
    }

    #[test]
    fn scalar_product() {
        let q = Biquaternion::from([
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
        ]);
        let p = Biquaternion::from([
            Complex::new(3.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
        ]);
        assert_eq!(Complex::new(7.0, 3.0), q.scalar_product(&p));
    }

    #[test]
    fn norm() {
        let q = Biquaternion::from([
            Complex::new(3.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(4.0, 0.0),
            Complex::new(0.0, 2.0),
        ]);
        assert_eq!(Complex::new(20.0_f64, 0.0).sqrt(), q.norm());
    }

    #[test]
    fn ops() {
        let a = Biquaternion::from([
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 1.0),
            Complex::new(-3.0, 0.0),
            Complex::new(4.0, 2.0),
        ]);
        let b = Biquaternion::from([
            Complex::new(4.0, 0.0),
            Complex::new(0.0, 3.0),
            Complex::new(-2.0, 0.0),
            Complex::new(1.0, 0.0),
        ]);

        // +, +=
        assert_eq!(
            a + b,
            Biquaternion::from([
                Complex::new(5.0, 0.0),
                Complex::new(2.0, 4.0),
                Complex::new(-5.0, 0.0),
                Complex::new(5.0, 2.0)
            ])
        );
        let mut c = a;
        c += b;
        assert_eq!(
            c,
            Biquaternion::from([
                Complex::new(5.0, 0.0),
                Complex::new(2.0, 4.0),
                Complex::new(-5.0, 0.0),
                Complex::new(5.0, 2.0)
            ])
        );

        // -, -=
        assert_eq!(
            a - b,
            Biquaternion::from([
                Complex::new(-3.0, 0.0),
                Complex::new(2.0, -2.0),
                Complex::new(-1.0, 0.0),
                Complex::new(3.0, 2.0)
            ])
        );
        let mut c = a;
        c -= b;
        assert_eq!(
            c,
            Biquaternion::from([
                Complex::new(-3.0, 0.0),
                Complex::new(2.0, -2.0),
                Complex::new(-1.0, 0.0),
                Complex::new(3.0, 2.0)
            ])
        );

        // Scalar * and /
        assert_eq!(
            a * 2.0,
            Biquaternion::from([
                Complex::new(2.0, 0.0),
                Complex::new(4.0, 2.0),
                Complex::new(-6.0, 0.0),
                Complex::new(8.0, 4.0)
            ])
        );
        let mut c = a;
        c *= 2.0;
        assert_eq!(
            c,
            Biquaternion::from([
                Complex::new(2.0, 0.0),
                Complex::new(4.0, 2.0),
                Complex::new(-6.0, 0.0),
                Complex::new(8.0, 4.0)
            ])
        );

        assert_eq!(
            a / 2.0,
            Biquaternion::from([
                Complex::new(0.5, 0.0),
                Complex::new(1.0, 0.5),
                Complex::new(-1.5, 0.0),
                Complex::new(2.0, 1.0)
            ])
        );
        let mut c = a;
        c /= 2.0;
        assert_eq!(
            c,
            Biquaternion::from([
                Complex::new(0.5, 0.0),
                Complex::new(1.0, 0.5),
                Complex::new(-1.5, 0.0),
                Complex::new(2.0, 1.0)
            ])
        );
    }

    #[test]
    fn display() {
        let a = Biquaternion::from([
            Complex::new(0.5, -3.1),
            Complex::new(1.2, 5.2),
            Complex::new(-1.5, 1.5),
            Complex::new(2.1, 1.5),
        ]);
        let s = format!("{a}");
        assert_eq!(s, "[0.5-3.1i, 1.2+5.2i, -1.5+1.5i, 2.1+1.5i]");
    }

    #[test]
    fn to_unit() {
        let a = Biquaternion::from([
            Complex::new(0.5, -3.0),
            Complex::new(1.0, 5.0),
            Complex::new(-1.5, 1.5),
            Complex::new(2.0, 1.0),
        ]);
        let a_unit = a.to_unit().expect("hard-coded to be nonzero");
        assert_eq!(a_unit.norm_squared().re, 1.0);
    }

    // Test named cases of three input values (rotation biquaternion, Minkowski input, and answer)
    #[rstest]
    #[case::y_rotate_pi_halves([Complex::new(0.0,0.0),
                                    Complex::new((PI/4.0).sin(),0.0),
                                    Complex::new(0.0, 0.0),
                                    Complex::new((PI/4.0).cos(), 0.0)],
                                [1.0,0.0,0.0,1.0],
                                [0.0,0.0,-1.0,1.0])]
    #[case::x_boost_half([Complex::new(0.0, 0.25).sin(),
                            Complex::new(0.0,0.0),
                            Complex::new(0.0, 0.0),
                            Complex::new(0.0, 0.25).cos()],
                           [0.0,0.0,0.0,1.0],
                           [(0.5_f64).sinh(),0.0,0.0,(0.5_f64).cosh()])]
    #[case::z_boost_one([Complex::new(0.0, 0.0),
                           Complex::new(0.0,0.0),
                           Complex::new(0.0, 0.5).sin(),
                           Complex::new(0.0, 0.5).cos()],
                          [0.0,0.0,0.0,1.0],
                          [0.0,0.0,(1.0_f64).sinh(),(1.0_f64).cosh()])]
    #[case::z_rotate_pi([Complex::new(0.0, 0.0),
                          Complex::new(0.0,0.0),
                          Complex::new((PI/2.0).sin(),0.0),
                          Complex::new((PI/2.0).cos(),0.0)],
                         [1.0,0.0,0.0,1.0],
                         [-1.0,0.0,0.0,1.0])]
    fn rotate(#[case] biquat: [Complex<f64>; 4], #[case] vec: [f64; 4], #[case] ans: [f64; 4]) {
        let q = Biquaternion::from(biquat);
        let v = q.to_unit().expect("hard-coded to be nonzero");
        let x = Minkowski::from(vec);

        // using matrix representation
        let matrix = HyperbolicRotationMatrix::from(v);
        let from_matrix = matrix.hyperbolic_rotate(&x);
        assert_relative_eq!(from_matrix.coordinates[0], ans[0], epsilon = 1e-12);
        assert_relative_eq!(from_matrix.coordinates[1], ans[1], epsilon = 1e-12);
        assert_relative_eq!(from_matrix.coordinates[2], ans[2], epsilon = 1e-12);
        assert_relative_eq!(from_matrix.coordinates[3], ans[3], epsilon = 1e-12);

        // using biquaternion algebra
        let from_algebra = v.hyperbolic_rotate(&x);
        assert_relative_eq!(from_algebra.coordinates[0], ans[0], epsilon = 1e-12);
        assert_relative_eq!(from_algebra.coordinates[1], ans[1], epsilon = 1e-12);
        assert_relative_eq!(from_algebra.coordinates[2], ans[2], epsilon = 1e-12);
        assert_relative_eq!(from_algebra.coordinates[3], ans[3], epsilon = 1e-12);
    }
}

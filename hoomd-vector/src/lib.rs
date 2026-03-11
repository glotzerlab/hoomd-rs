// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

//! Vector and quaternion math.
//!
//! `hoomd_vector` implements vector math types and operations used in scientific
//! computations, specifically those used in the HOOMD molecular simulation software
//! suite. Its API is firmly rooted in mathematical principles. Users in
//! other fields may find `hoomd_vector` useful outside the context of `HOOMD`.
//!
//! ## Vectors
//!
//! The [`Vector`] trait describes any type that is a member of a metric vector
//! space. Write code with a [`Vector`] trait bound when you can express the
//! computation with vector arithmetic and a distance metric. Your generic code can
//! then be invoked on vector types with any dimension or representation (e.g.
//! spherical coordinates).
//!
//! ```
//! use hoomd_vector::Vector;
//!
//! fn some_function<V: Vector>(a: &V, b: &V, c: &V) -> f64 {
//!     (*a + *b).distance(&c)
//! }
//! ```
//!
//! The [`InnerProduct`] subtrait of [`Vector`] describes any type that is a member of
//! an inner product space. [`InnerProduct`] implements vector norms and dot products.
//!
//! ```
//! use hoomd_vector::InnerProduct;
//!
//! fn some_other_function<V: InnerProduct>(a: &V, b: &V) -> f64 {
//!     a.dot(b) / (a.norm_squared())
//! }
//! ```
//!
//! Require additional trait bounds to perform more specific operations, such as [`Cross`]:
//! ```
//! use hoomd_vector::{Cross, InnerProduct};
//!
//! fn triple<V: InnerProduct + Cross>(a: &V, b: &V, c: &V) -> f64 {
//!     a.dot(&b.cross(c))
//! }
//! ```
//!
//! Use the provided [`Cartesian`] type to concretely represent N-dimensional
//! vectors, or when your algorithm requires Cartesian coordinates:
//!
//! ```
//! use hoomd_vector::{Cartesian, InnerProduct};
//!
//! let a = Cartesian::from([1.0, 2.0]);
//! let b = Cartesian::from([-2.0, 1.0]);
//!
//! let product = a.dot(&b);
//! assert_eq!(product, 0.0);
//!
//! let x = a[0];
//! let y = a[1];
//! ```
//!
//! ## Quaternions
//!
//! Quaternions are generalized complex numbers and a convenient way to describe the motion
//! of rotating bodies. The [`Quaternion`] type describes a single quaternion and implements
//! the associated algebra.
//!
//! ```
//! use hoomd_vector::Quaternion;
//!
//! let a = Quaternion::from([1.0, -2.0, 6.0, -4.0]);
//! let b = Quaternion::from([-2.0, 6.0, 4.0, 1.0]);
//!
//! let norm = a.norm();
//! assert_eq!(norm, 57.0_f64.sqrt());
//!
//! let sum = a + b;
//! assert_eq!(sum, [-1.0, 4.0, 10.0, -3.0].into());
//!
//! let product = a * b;
//! assert_eq!(product, [-10.0, 32.0, -30.0, -35.0].into());
//! ```
//!
//! A **unit quaternion** (called a [`Versor`] in mathematics) can represent a 3D rotation.
//!
//! ```
//! use hoomd_vector::{Quaternion, Versor};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let q = Quaternion::from([3.0, 0.0, 0.0, 4.0]);
//! let v = q.to_versor()?;
//! assert_eq!(*v.get(), [3.0 / 5.0, 0.0, 0.0, 4.0 / 5.0].into());
//! # Ok(())
//! # }
//! ```
//!
//! ## Rotations
//!
//! A [`Rotation`] describes a transformation from one orthonormal basis to
//! another. A type that implements [`Rotation`] has an
//! [`identity`](Rotation::identity). Instances of that type have an
//! [`inverse`](Rotation::inverted) and can be [`combined`](Rotation::combine)
//! with other rotations.
//!
//! Through the [`Rotate<V>`] trait, a [`Rotation`] can rotate a vector.
//!
//! As with [`Vector`], you can implement methods that operate on generic types:
//! ```
//! use hoomd_vector::{Rotate, Vector};
//!
//! fn rotate_and_translate<R: Rotate<V>, V: Vector>(r: &R, a: &V, b: &V) -> V {
//!     r.rotate(a) + *b
//! }
//! ```
//!
//! [`Angle`] implements rotations on [`Cartesian<2>`] vectors.
//! ```
//! use approxim::assert_relative_eq;
//! use hoomd_vector::{Angle, Cartesian, Rotate, Rotation};
//! use std::f64::consts::PI;
//!
//! let v = Cartesian::from([-1.0, 0.0]);
//! let a = Angle::from(PI / 2.0);
//! let rotated = a.rotate(&v);
//! assert_relative_eq!(rotated, [0.0, -1.0].into());
//! ```
//!
//! [`Versor`] implements rotations on [`Cartesian<3>`] vectors.
//! ```
//! use approxim::assert_relative_eq;
//! use hoomd_vector::{Cartesian, Rotate, Rotation, Versor};
//! use std::f64::consts::PI;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let a = Cartesian::from([-1.0, 0.0, 0.0]);
//! let v = Versor::from_axis_angle([0.0, 0.0, 1.0].try_into()?, PI / 2.0);
//! let b = v.rotate(&a);
//! assert_relative_eq!(b, [0.0, -1.0, 0.0].into());
//! # Ok(())
//! # }
//! ```
//!
//! Convert to a [`RotationMatrix`] when you need to rotate many vectors by the same
//! rotation. [`RotationMatrix::rotate`] is typically several times faster than
//! [`Versor::rotate`].
//!
//! # Random distributions
//!
//! `hoomd_vector` interoperates with [`rand`] to generate random vectors and rotations.
//!
//! The [`StandardUniform`](rand::distr::StandardUniform) distribution randomly samples
//! rotations uniformly from the set of all vectors or rotations.
//!
//! - Vectors are uniformly sampled from the `[-1,1]` hypercube
//! - Angles are uniformly sampled from the half-open interval `[0, 2π)`
//! - Versors are uniformly sampled from the surface of the `3-Sphere`, which doubly
//!   covers `SO(3)`, the manifold of rotations in three dimensions.
//!
//! ```
//! use hoomd_vector::{Angle, Cartesian, Versor};
//! use rand::{RngExt, SeedableRng, rngs::StdRng};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut rng = StdRng::seed_from_u64(1);
//! let vector: Cartesian<3> = rng.random();
//! let angle: Angle = rng.random();
//! let versor: Versor = rng.random();
//! # Ok(())
//! # }
//! ```
//!
//! The [`Ball`](crate::distribution::Ball) distribution samples vectors from
//! the interior of an `n-Ball`, the set of all points whose distance from the origin is
//! in `[0, 1)`.
//!
//! ```
//! use hoomd_vector::{Cartesian, distribution::Ball};
//! use rand::{Rng, SeedableRng, distr::Distribution, rngs::StdRng};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut rng = StdRng::seed_from_u64(1);
//! let ball = Ball {
//!     radius: 3.0.try_into()?,
//! };
//! let v: Cartesian<3> = ball.sample(&mut rng);
//! # Ok(())
//! # }
//! ```
//!
//! # Complete documentation
//!
//! `hoomd-vector` is is a part of *hoomd-rs*. Read the [complete documentation]
//! for more information.
//!
//! [complete documentation]: https://glotzerlab-hoomd-rs.readthedocs-hosted.com

mod angle;
mod cartesian;
pub mod distribution;
mod quaternion;

pub use angle::Angle;
pub use cartesian::{Cartesian, RotationMatrix};
pub use quaternion::{Quaternion, Versor};

use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use thiserror::Error;

/// Enumerate possible sources of error in fallible vector math operations.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// Attempted converting a value to a vector with a dimension not equal to the value's length.
    #[error("source length does not match the target dimensions")]
    InvalidVectorLength,

    /// Attempted to normalize a vector with an invalid magnitude.
    #[error("cannot normalize the 0 vector")]
    InvalidVectorMagnitude,

    /// Attempted to normalize a quaternion with an invalid magnitude.
    #[error("cannot normalize the 0 quaternion")]
    InvalidQuaternionMagnitude,
}

/// Operate on elements of a metric vector space.
///
/// Specifically, [`Vector`] defines methods that can be performed on any vector in a metric vector
/// space. Note that this is not an inner product space by default, and calculations requiring an
/// inner product should use the [`InnerProduct`] subtrait.
///
/// ## Vector Operations
///
/// The following examples demonstrate vector operations applied to the following
/// vectors:
///
/// ```
/// use hoomd_vector::Cartesian;
///
/// # fn main() {
/// let mut a = Cartesian::from([1.0, 2.0]);
/// let mut b = Cartesian::from([4.0, 8.0]);
/// # }
/// ```
///
/// Vector addition:
///
/// ```
/// # use hoomd_vector::Cartesian;
/// # fn main() {
/// # let mut a = Cartesian::from([1.0, 2.0]);
/// # let mut b = Cartesian::from([4.0, 8.0]);
/// let c = a + b;
/// assert_eq!(c, [5.0, 10.0].into())
/// # }
/// ```
///
/// ```
/// # use hoomd_vector::Cartesian;
/// # fn main() {
/// # let mut a = Cartesian::from([1.0, 2.0]);
/// # let mut b = Cartesian::from([4.0, 8.0]);
/// a += b;
/// assert_eq!(a, [5.0, 10.0].into())
/// # }
/// ```
///
/// Vector subtraction:
///
/// ```
/// # use hoomd_vector::Cartesian;
/// # fn main() {
/// # let mut a = Cartesian::from([1.0, 2.0]);
/// # let mut b = Cartesian::from([4.0, 8.0]);
/// let c = b - a;
/// assert_eq!(c, [3.0, 6.0].into())
/// # }
/// ```
///
/// ```
/// # use hoomd_vector::Cartesian;
/// # fn main() {
/// # let mut a = Cartesian::from([1.0, 2.0]);
/// # let mut b = Cartesian::from([4.0, 8.0]);
/// b -= a;
/// assert_eq!(b, [3.0, 6.0].into())
/// # }
/// ```
///
/// Multiplication of a vector by a scalar:
///
/// ```
/// # use hoomd_vector::Cartesian;
/// # fn main() {
/// # let mut a = Cartesian::from([1.0, 2.0]);
/// # let mut b = Cartesian::from([4.0, 8.0]);
/// let c = a * 2.0;
/// assert_eq!(c, [2.0, 4.0].into())
/// # }
/// ```
///
/// ```
/// # use hoomd_vector::Cartesian;
/// # fn main() {
/// # let mut a = Cartesian::from([1.0, 2.0]);
/// # let mut b = Cartesian::from([4.0, 8.0]);
/// a *= 2.0;
/// assert_eq!(a, [2.0, 4.0].into())
/// # }
/// ```
///
/// Division of a vector by a scalar:
///
/// ```
/// # use hoomd_vector::Cartesian;
/// # fn main() {
/// # let mut a = Cartesian::from([1.0, 2.0]);
/// # let mut b = Cartesian::from([4.0, 8.0]);
/// let c = b / 2.0;
/// assert_eq!(c, [2.0, 4.0].into())
/// # }
/// ```
///
/// ```
/// # use hoomd_vector::Cartesian;
/// # fn main() {
/// # let mut a = Cartesian::from([1.0, 2.0]);
/// # let mut b = Cartesian::from([4.0, 8.0]);
/// b /= 2.0;
/// assert_eq!(b, [2.0, 4.0].into())
/// # }
/// ```
///
/// Negation:
///
/// ```
/// # use hoomd_vector::Cartesian;
/// # fn main() {
/// # let mut a = Cartesian::from([1.0, 2.0]);
/// # let mut b = Cartesian::from([4.0, 8.0]);
/// let mut c = -a;
/// assert_eq!(c, [-1.0, -2.0].into());
/// # }
/// ```
///
/// Equality:
///
/// ```
/// # use hoomd_vector::Cartesian;
/// # fn main() {
/// # let mut a = Cartesian::from([1.0, 2.0]);
/// # let mut b = Cartesian::from([4.0, 8.0]);
/// assert!(a != b)
/// # }
/// ```
pub trait Vector:
    Add<Self, Output = Self>
    + AddAssign
    + Copy
    + Div<f64, Output = Self>
    + DivAssign<f64>
    + PartialEq
    + Metric
    + Mul<f64, Output = Self>
    + MulAssign<f64>
    + Sub<Self, Output = Self>
    + SubAssign
    + Neg<Output = Self>
{
}

/// Operate on elements of a wedge product space.
/// 
/// The [`WedgeProduct`] subtrait defines the wedge product method, which returns the bivector
/// for two elements. In 2-D vector space, the bivector of two vectors is a scalar;
/// in 3-D it is another 3-vector. In cartesian space, torque always takes the form
/// of a bivector.
/// TODO: confirm whether the elements must be vectors
pub trait WedgeProduct: Vector {
    /// Type of the resulting [bivector](https://en.wikipedia.org/wiki/Bivector).
    type Bivector;
    /// Compute the wedge product between two elements.
    /// 
    /// ```math
    /// \textbf{A}=\textbf{a}\wedge{\textbf{b}}
    /// ```
    /// 
    /// # Example
    /// ```
    /// use hoomd_vector::{Cartesian, WedgeProduct}
    /// 
    /// # fn main() {
    /// let a = Cartesian::from([1.0, 1.0]);
    /// let b = Cartesian::from([0.0, 1.0]);
    /// let c = Cartesian::from([1.0, 1.0, 1.0]);
    /// let d = Cartesian::from([0.0, 1.0, 0.0]);
    /// assert_eq!(a.cross(b), a.wedge_product(b));
    /// assert_eq!(c.cross(d), c.wedge_product(d));
    /// # }
    /// ```
    fn wedge_product(&self, other: &Self) -> Self::Bivector;
}

/// Operate on elements of a tensor product space.
/// 
/// The [`TensorProduct`] subtrait defines the tensor product method, which returns
/// the tensor resulting from matrix-style multiplication of two elements.
/// TODO: confirm whether the elements must be vectors
pub trait TensorProduct: Vector {
    /// Type of the resulting tensor.
    type Tensor;
    /// Compute the tensor product between two elements.
    /// 
    /// ```math
    /// a \otimes b = \begin{bmatrix} a_{0}
    ///  \\ \vdots 
    ///  \\ a_{n}
    /// \end{bmatrix}
    /// \begin{bmatrix}
    /// b_{0} & \dots & b_{n}
    /// \end{bmatrix}
    /// =
    /// \begin{bmatrix}
    /// a_{0}b_{0} & \dots & a_{0}b_{n} \\
    /// \vdots & \ddots & \vdots \\
    /// a_{n}b_{0} & \dots & a_{n}b_{n}
    /// \end{bmatrix}
    /// ```
    /// 
    /// # Example
    /// ```
    /// use hoomd_vector::{Cartesian, Matrix, TensorProduct}
    /// 
    /// # fn main() {
    /// let a = Cartesian::from([1.0, 1.0]);
    /// let b = Cartesian::from([0.0, 1.0]);
    /// let m = Matrix {
    ///     rows: [[0.0, 1.0], [0.0, 1.0]]
    /// }
    /// assert_eq!(a.tensor_product(b), m)
    /// # }
    /// ```
    fn tensor_product(&self, other: &Self) -> Self::Tensor;
}

/// Operates on elements of a metric space.
///
/// [`Metric`] implements a distance metric between points.
pub trait Metric {
    /// Compute the squared distance between two vectors belonging to a metric space.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::{Cartesian, Metric};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let x = Cartesian::from([0.0, 1.0, 1.0]);
    /// let y = Cartesian::from([1.0, 0.0, 0.0]);
    /// assert_eq!(3.0, x.distance_squared(&y));
    /// # Ok(())
    /// # }
    /// ```
    fn distance_squared(&self, other: &Self) -> f64;

    /// Return the number of dimensions in this vector space.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::{Cartesian, Metric};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let vec2 = Cartesian::<2>::default();
    /// let vec3 = Cartesian::<3>::default();
    /// assert_eq!(2, vec2.n_dimensions());
    /// assert_eq!(3, vec3.n_dimensions());
    /// # Ok(())
    /// # }
    /// ```
    fn n_dimensions(&self) -> usize;

    /// Compute the distance between two vectors belonging to a metric space.
    /// # Example
    /// ```
    /// use hoomd_vector::{Cartesian, Metric};
    ///
    /// let x = Cartesian::from([0.0, 0.0]);
    /// let y = Cartesian::from([3.0, 4.0]);
    /// assert_eq!(5.0, x.distance(&y));
    /// ```
    fn distance(&self, other: &Self) -> f64;
}

/// Operate on elements of an inner product space.
///
/// The [`InnerProduct`] subtrait defines additional methods that can be performed on any vector
/// in an inner product space, specifically vector norms and inner products.
pub trait InnerProduct: Vector {
    /// Compute the vector dot product between two vectors.
    ///
    /// ```math
    /// c = \vec{a} \cdot \vec{b}
    /// ```
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::{Cartesian, InnerProduct};
    ///
    /// # fn main() {
    /// let a = Cartesian::from([1.0, 2.0]);
    /// let b = Cartesian::from([3.0, 4.0]);
    /// let c = a.dot(&b);
    /// assert_eq!(c, 11.0);
    /// # }
    /// ```
    #[must_use]
    fn dot(&self, other: &Self) -> f64;

    /// Compute the squared norm of the vector.
    ///
    /// ```math
    /// \left| \vec{v} \right|^2
    /// ```
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::{Cartesian, InnerProduct};
    ///
    /// # fn main() {
    /// let v = Cartesian::from([2.0, 4.0]);
    /// let norm_squared = v.norm_squared();
    /// assert_eq!(norm_squared, 20.0);
    /// # }
    /// ```
    #[must_use]
    #[inline]
    fn norm_squared(&self) -> f64 {
        self.dot(self)
    }

    /// Compute the norm of the vector.
    ///
    /// ```math
    /// \left| \vec{v} \right|
    /// ```
    ///
    /// <div class="warning">
    ///
    /// Computing the norm calls `sqrt`. Prefer
    /// [`norm_squared`](InnerProduct::norm_squared) when possible.
    ///
    /// </div>
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::{Cartesian, InnerProduct};
    ///
    /// # fn main() {
    /// let v = Cartesian::from([3.0, 4.0]);
    /// let norm = v.norm();
    /// assert_eq!(norm, 5.0);
    /// # }
    /// ```
    #[must_use]
    #[inline]
    fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Create a vector of unit length pointing in the same direction as the given vector.
    ///
    /// Returns a tuple containing unit vector along with the original vector's norm:
    /// ```math
    /// \frac{\vec{v}}{|\vec{v}|}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::{Cartesian, InnerProduct, Unit};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let a = Cartesian::from([3.0, 4.0]);
    /// let (unit, norm) = a.to_unit()?;
    /// assert_eq!(*unit.get(), [3.0 / 5.0, 4.0 / 5.0].into());
    /// assert_eq!(norm, 5.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::InvalidVectorMagnitude`] when `self` is the 0 vector.
    #[inline]
    fn to_unit(self) -> Result<(Unit<Self>, f64), Error> {
        let norm = self.norm();
        if norm == 0.0 {
            Err(Error::InvalidVectorMagnitude)
        } else {
            Ok((Unit(self / norm), norm))
        }
    }

    /// Create a vector of unit length pointing in the same direction as the given vector.
    ///
    /// Returns a tuple containing unit vector along with the original vector's norm:
    /// ```math
    /// \frac{\vec{v}}{|\vec{v}|}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::{Cartesian, InnerProduct, Unit};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let a = Cartesian::from([3.0, 4.0]);
    /// let (unit, norm) = a.to_unit_unchecked();
    /// assert_eq!(*unit.get(), [3.0 / 5.0, 4.0 / 5.0].into());
    /// assert_eq!(norm, 5.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// Divide by 0 when `self` is the 0 vector.
    #[inline]
    fn to_unit_unchecked(self) -> (Unit<Self>, f64) {
        let norm = self.norm();
        (Unit(self / norm), norm)
    }

    /// Project one vector onto another.
    /// ```math
    /// \left(\frac{\vec{a} \cdot \vec{b}}{|\vec{b}|^2}\right) \vec{b}
    /// ```
    /// where `self` is $`\vec{a}`$.
    /// # Example
    /// ```
    /// use hoomd_vector::{Cartesian, InnerProduct, Vector};
    /// let a = Cartesian::from([1.0, 2.0]);
    /// let b = Cartesian::from([4.0, 0.0]);
    /// let c = a.project(&b);
    /// assert_eq!(c, [1.0, 0.0].into());
    /// ```
    #[inline]
    #[must_use]
    fn project(&self, b: &Self) -> Self {
        *b * self.dot(b) / b.norm_squared()
    }

    /// Create a unit vector in the space.
    ///
    /// Each vector space defines its own default unit vector.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::{Cartesian, InnerProduct};
    ///
    /// let u = Cartesian::<2>::default_unit();
    /// assert_eq!(*u.get(), [0.0, 1.0].into());
    /// ```
    fn default_unit() -> Unit<Self>;
}

/// A [`Vector`] with magnitude 1.0.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Unit<V>(V);

impl<V> Unit<V> {
    /// Get the unit vector.
    #[inline]
    pub fn get(&self) -> &V {
        &self.0
    }
}

/// A vector space where the cross product is defined.
pub trait Cross {
    /// Perform the cross product.
    /// Compute the cross product (right-handed) of two vectors:
    ///
    /// ```math
    /// \vec{c} = \vec{a} × \vec{b}
    /// ```
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::{Cartesian, Cross, Vector};
    ///
    /// # fn main() {
    /// let a = Cartesian::from([1.0, 0.0, 0.0]);
    /// let b = Cartesian::from([0.0, 1.0, 0.0]);
    /// let c = a.cross(&b);
    /// assert_eq!(c, [0.0, 0.0, 1.0].into());
    /// # }
    /// ```
    #[must_use]
    fn cross(&self, other: &Self) -> Self;
}

/// Applies the rotation operation to vectors.
///
/// The [`Rotate`] trait describes a type that can rotate a given vector. The rotated vector has the
/// same magnitude, but possibly a different direction.
///
/// Types that implement [`Rotate`] may or _may not_ implement [`Rotation`].
pub trait Rotate<V: Vector> {
    /// Type of the related rotation matrix
    type Matrix: Rotate<V>;

    /// Rotate a vector.
    ///
    /// ```math
    /// \vec{b} = R(\vec{a})
    /// ```
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_vector::{Angle, Cartesian, Rotate, Rotation};
    ///
    /// let v = Cartesian::from([-1.0, 0.0]);
    /// let a = Angle::from(std::f64::consts::PI / 2.0);
    /// let rotated = a.rotate(&v);
    /// assert_relative_eq!(rotated, [0.0, -1.0].into());
    /// ```
    #[must_use]
    fn rotate(&self, vector: &V) -> V;
}

/// Describes the transformation from one orthonormal basis to another.
///
/// A [`Rotation`] represents a single rotation operation. Rotations change the direction of a vector
/// while keeping its magnitude constant. To maintain generality, this documentation shows rotations
/// mathematically as _functions_:
/// ```math
/// \vec{b} = R(\vec{a})
/// ```
///
/// All types that implement [`Rotation`] _should_ implement [`Rotate`] for at least one vector type.
pub trait Rotation: Copy {
    /// The identity rotation.
    /// ```math
    /// \vec{a} = I(\vec{a})
    /// ```
    #[must_use]
    fn identity() -> Self;

    /// Inverse the rotation.
    /// ```math
    /// \vec{a} = R^{-1}(R(\vec{a}))
    /// ```
    ///
    /// # Example
    /// ```
    /// # use hoomd_vector::{Rotation};
    /// # fn inverse<R: Rotation>(r: R) {
    /// let r_inverse = r.inverted();
    /// # }
    /// ```
    #[must_use]
    fn inverted(self) -> Self;

    /// Combine two rotations.
    ///
    /// The resulting rotation `R_ab` will rotate by **first** `R_b` _followed by_ a
    /// rotation of `R_a`.
    ///
    /// ```math
    /// R_{ab}(\vec{v})= R_a(R_b(\vec{v}))
    /// ```
    ///
    /// # Example
    /// ```
    /// # use hoomd_vector::{Rotation};
    /// # fn inverse<R: Rotation>(R_a: &R, R_b: &R) {
    /// let R_ab = R_a.combine(R_b);
    /// # }
    /// ```
    #[must_use]
    fn combine(&self, other: &Self) -> Self;
}

/// Get the relative position and orientation given pairs of positions and orientations.
///
/// # Example
///
/// ```
/// use approxim::assert_relative_eq;
/// use hoomd_vector::{self, Angle, Cartesian};
/// use std::f64::consts::PI;
///
/// let r_a = Cartesian::from([1.0, -2.0]);
/// let o_a = Angle::from(PI / 2.0);
///
/// let r_b = Cartesian::from([2.0, -1.0]);
/// let o_b = Angle::from(PI);
///
/// let (v_ab, o_ab) =
///     hoomd_vector::pair_system_to_local(&r_a, &o_a, &r_b, &o_b);
/// assert_relative_eq!(v_ab[0], 1.0);
/// assert_relative_eq!(v_ab[1], -1.0);
/// assert_relative_eq!(o_ab.theta, PI / 2.0);
/// ```
#[inline]
pub fn pair_system_to_local<V, R>(r_a: &V, o_a: &R, r_b: &V, o_b: &R) -> (V, R)
where
    V: Vector,
    R: Rotation + Rotate<V>,
{
    let r_ab = *r_b - *r_a;
    let o_a_inverted = o_a.inverted();
    let v_ij = o_a_inverted.rotate(&r_ab);
    let o_ij = o_a_inverted.combine(o_b);
    (v_ij, o_ij)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use assert2::check;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    fn compute_add_generic<T>(a: T, b: T) -> T
    where
        T: Vector,
    {
        a + b
    }

    #[test]
    fn add_generic() {
        let a = Cartesian::from([1.0, 2.0, 3.0]);
        let b = Cartesian::from([4.0, 5.0, 6.0]);
        let c = compute_add_generic(a, b);
        check!(c == [5.0, 7.0, 9.0].into());
    }

    #[test]
    fn test_pair_system_to_local_2d() {
        let mut rng = StdRng::seed_from_u64(1);

        for _ in 0..1_000 {
            let o_a: Angle = rng.random();
            let o_b: Angle = rng.random();

            let r_a: Cartesian<2> = rng.random();
            let r_b: Cartesian<2> = rng.random();

            let c_in_b: Cartesian<2> = rng.random();

            // Test self-consistency by locating c in both a's and b's reference frames.
            // Check that they are equivalent in the global frame.
            let (v_ij, o_ij) = pair_system_to_local(&r_a, &o_a, &r_b, &o_b);
            let c_in_a = v_ij + o_ij.rotate(&c_in_b);

            assert_relative_eq!(
                r_a + o_a.rotate(&c_in_a),
                r_b + o_b.rotate(&c_in_b),
                epsilon = 4.0 * f64::EPSILON
            );

            let (v_ji, o_ji) = pair_system_to_local(&r_b, &o_b, &r_a, &o_a);
            assert_relative_eq!(
                v_ji + o_ji.rotate(&c_in_a),
                c_in_b,
                epsilon = 4.0 * f64::EPSILON
            );
        }
    }

    #[test]
    fn test_pair_system_to_local_3d() {
        let mut rng = StdRng::seed_from_u64(1);

        for _ in 0..1_000 {
            let o_a: Versor = rng.random();
            let o_b: Versor = rng.random();

            let r_a: Cartesian<3> = rng.random();
            let r_b: Cartesian<3> = rng.random();

            let c_in_b: Cartesian<3> = rng.random();

            // Test self-consistency by locating c in both a's and b's reference frames.
            // Check that they are equivalent in the global frame.
            let (v_ij, o_ij) = pair_system_to_local(&r_a, &o_a, &r_b, &o_b);
            let c_in_a = v_ij + o_ij.rotate(&c_in_b);

            assert_relative_eq!(
                r_a + o_a.rotate(&c_in_a),
                r_b + o_b.rotate(&c_in_b),
                epsilon = 10.0 * f64::EPSILON
            );

            let (v_ji, o_ji) = pair_system_to_local(&r_b, &o_b, &r_a, &o_a);
            assert_relative_eq!(
                v_ji + o_ji.rotate(&c_in_a),
                c_in_b,
                epsilon = 10.0 * f64::EPSILON
            );
        }
    }
}

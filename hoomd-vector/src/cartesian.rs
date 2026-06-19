// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement canonical vector types.

use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::{
    array, fmt,
    iter::{Sum, zip},
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

use approxim::approx_derive::RelativeEq;
use rand::{
    Rng,
    distr::{Distribution, StandardUniform, Uniform},
};

use crate::{Cross, Error, InnerProduct, Metric, Outer, Rotate, Unit, Vector, Wedge};
use hoomd_linear_algebra::{MatMul, matrix::Matrix};

/// A [`Vector`] represented by `N` `f64` coordinates.
///
/// [`Cartesian`] is the canonical implementation of [`InnerProduct`].
///
/// ## Constructing vectors
///
/// The default is the 0 vector:
/// ```
/// use hoomd_vector::Cartesian;
///
/// let v = Cartesian::<3>::default();
/// assert_eq!(v, [0.0; 3].into())
/// ```
///
/// Create a vector with an array of coordinates:
/// ```
/// use hoomd_vector::Cartesian;
///
/// let v = Cartesian::from([1.0, 2.0, 3.0, 4.0, 5.0]);
/// ```
///
/// 2D and 3D vectors can also be initialized from tuples:
/// ```
/// use hoomd_vector::Cartesian;
///
/// let a = Cartesian::from((1.0, 2.0, 3.0));
/// let b = Cartesian::from((4.0, 5.0));
/// ```
///
/// Construct a random vector in the [-1, 1] hypercube:
///
/// ```
/// use hoomd_vector::Cartesian;
/// use rand::{RngExt, SeedableRng, rngs::StdRng};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut rng = StdRng::seed_from_u64(1);
/// let v: Cartesian<3> = rng.random();
/// # Ok(())
/// # }
/// ```
///
/// ## Operating on vectors
///
/// Use vector math operations when you can:
/// ```
/// use hoomd_vector::{Cartesian, InnerProduct};
///
/// let a = Cartesian::from([1.0, 2.0]);
/// let b = Cartesian::from([4.0, 8.0]);
/// let c = (a + b).dot(&a);
/// ```
///
/// Access the coordinates directly when needed:
/// ```
/// use hoomd_vector::Cartesian;
///
/// let a = Cartesian::from((1.0, 2.0));
/// let b = Cartesian::from((a[1], 0.0));
/// ```
///
/// Compute the sum of an iterator over vectors:
/// ```
/// use hoomd_vector::Cartesian;
///
/// let total: Cartesian<2> =
///     [Cartesian::from((1.0, 2.0)), Cartesian::from((3.0, 4.0))]
///         .into_iter()
///         .sum();
/// ```
#[serde_as]
#[derive(Clone, Copy, Debug, PartialEq, RelativeEq, Serialize, Deserialize)]
#[approx(epsilon_type = f64)]
pub struct Cartesian<const N: usize> {
    /// The vector's coordinates.
    #[serde_as(as = "[_; N]")]
    #[approx(into_iter)]
    pub coordinates: [f64; N],
}

impl<const N: usize> Default for Cartesian<N> {
    /// Create a 0 vector.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::Cartesian;
    ///
    /// let v = Cartesian::<3>::default();
    /// assert_eq!(v, [0.0; 3].into())
    /// ```
    #[inline]
    fn default() -> Self {
        Cartesian::from([0.0; N])
    }
}

impl<const N: usize> From<[f64; N]> for Cartesian<N> {
    /// Create a Cartesian vector with the given coordinates.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::Cartesian;
    ///
    /// let v = Cartesian::from([4.0, 3.0]);
    /// ```
    #[inline]
    fn from(coordinates: [f64; N]) -> Self {
        Self { coordinates }
    }
}

impl<const N: usize> IntoIterator for Cartesian<N> {
    type Item = f64;
    type IntoIter = <[f64; N] as IntoIterator>::IntoIter;

    /// Iterate over the components of the vector.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::Cartesian;
    ///
    /// let a = Cartesian::from([1.0, 2.0]);
    /// let mut iter = a.into_iter();
    ///
    /// assert_eq!(iter.next(), Some(1.0));
    /// assert_eq!(iter.next(), Some(2.0));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.coordinates.into_iter()
    }
}

impl From<(f64, f64)> for Cartesian<2> {
    #[inline]
    fn from(coordinates: (f64, f64)) -> Self {
        Self {
            coordinates: coordinates.into(),
        }
    }
}

impl From<(f64, f64, f64)> for Cartesian<3> {
    #[inline]
    fn from(coordinates: (f64, f64, f64)) -> Self {
        Self {
            coordinates: coordinates.into(),
        }
    }
}

impl<const N: usize> TryFrom<Vec<f64>> for Cartesian<N> {
    type Error = Error;

    /// Create a Cartesian vector with coordinates given by a [`Vec<f64>`]
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v = Cartesian::<3>::try_from(vec![3.0, 4.0, 5.0])?;
    /// assert_eq!(v, [3.0, 4.0, 5.0].into());
    /// # Ok(())
    /// # }
    /// ```
    /// <div class="warning">
    ///
    /// This method deallocates the Vec after copying it.
    /// Use `Cartesian::From<[f64; N]>` in performance critical code.
    ///
    /// </div>
    #[inline]
    fn try_from(value: Vec<f64>) -> Result<Self, Self::Error> {
        let coordinates = value.try_into().map_err(|_| Error::InvalidVectorLength)?;
        Ok(Self { coordinates })
    }
}

impl<const N: usize> TryFrom<std::ops::Range<usize>> for Cartesian<N> {
    type Error = Error;

    /// Create a Cartesian vector with coordinates given by a range.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v = Cartesian::<3>::try_from(3..6)?;
    /// assert_eq!(v, [3.0, 4.0, 5.0].into());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn try_from(value: std::ops::Range<usize>) -> Result<Self, Self::Error> {
        if value.len() != N {
            return Err(Error::InvalidVectorLength);
        }

        let coordinates = array::from_fn(|i| (value.start + i) as f64);
        Ok(Self { coordinates })
    }
}

impl<const N: usize> TryFrom<[f64; N]> for Unit<Cartesian<N>> {
    type Error = Error;

    /// Create a unit Cartesian vector by normalizing the given coordinates.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::{Cartesian, Unit};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let n = Unit::<Cartesian<3>>::try_from([2.0, 0.0, 0.0])?;
    /// assert_eq!(*n.get(), [1.0, 0.0, 0.0].into());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn try_from(value: [f64; N]) -> Result<Self, Self::Error> {
        Cartesian::from(value).to_unit().map(|t| t.0)
    }
}

impl<const N: usize> Vector for Cartesian<N> {}

impl<const N: usize> InnerProduct for Cartesian<N> {
    #[inline]
    fn dot(&self, other: &Self) -> f64 {
        zip(self.coordinates.iter(), other.coordinates.iter())
            .fold(0.0, |product, (&x, &y)| x.mul_add(y, product))
    }

    /// Create a unit vector in the space.
    ///
    /// The default unit vector in Cartesian space is `[0.0, 0.0, ...., 1.0]`.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::{Cartesian, InnerProduct};
    ///
    /// let u = Cartesian::<2>::default_unit();
    /// assert_eq!(*u.get(), [0.0, 1.0].into());
    ///
    /// let u = Cartesian::<3>::default_unit();
    /// assert_eq!(*u.get(), [0.0, 0.0, 1.0].into());
    /// ```
    #[inline]
    fn default_unit() -> Unit<Self> {
        assert!(N >= 1);
        let mut coordinates = [0.0; N];
        coordinates[N - 1] = 1.0;
        Unit(Self { coordinates })
    }
}

impl<const N: usize> Metric for Cartesian<N> {
    /// Computes the squared distance between two points in Euclidean space.
    /// ```math
    /// d^2(\vec{x},\vec{y}) = \sum_{i=1}^{N} (x_i - y_i)^2
    /// ```
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
    #[inline]
    fn distance_squared(&self, other: &Self) -> f64 {
        zip(self.coordinates.iter(), other.coordinates.iter())
            .fold(0.0, |product, x| product + (x.0 - x.1).powi(2))
    }
    #[inline]
    fn distance(&self, other: &Self) -> f64 {
        (self.distance_squared(other)).sqrt()
    }
    /// Return the number of dimensions in this Cartesian vector space.
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
    #[inline]
    fn n_dimensions(&self) -> usize {
        N
    }
}

impl<const N: usize> Add for Cartesian<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut coordinates = [0.0; N];

        for (result, (a, b)) in coordinates
            .iter_mut()
            .zip(self.coordinates.iter().zip(rhs.coordinates.iter()))
        {
            *result = a + b;
        }
        Self { coordinates }
    }
}

impl<const N: usize> AddAssign for Cartesian<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for (result, a) in self.coordinates.iter_mut().zip(rhs.coordinates.iter()) {
            *result += a;
        }
    }
}

impl<const N: usize> Div<f64> for Cartesian<N> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self {
        Self {
            coordinates: self.coordinates.map(|x| x / rhs),
        }
    }
}

impl<const N: usize> DivAssign<f64> for Cartesian<N> {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.coordinates = self.coordinates.map(|x| x / rhs);
    }
}

impl<const N: usize> Mul<f64> for Cartesian<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            coordinates: self.coordinates.map(|x| x * rhs),
        }
    }
}

impl<const N: usize> MulAssign<f64> for Cartesian<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.coordinates = self.coordinates.map(|x| x * rhs);
    }
}

impl<const N: usize> Sub for Cartesian<N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut coordinates = [0.0; N];
        for (result, (a, b)) in coordinates
            .iter_mut()
            .zip(self.coordinates.iter().zip(rhs.coordinates.iter()))
        {
            *result = a - b;
        }
        Self { coordinates }
    }
}

impl<const N: usize> SubAssign for Cartesian<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for (result, a) in self.coordinates.iter_mut().zip(rhs.coordinates) {
            *result -= a;
        }
    }
}

impl<const N: usize> fmt::Display for Cartesian<N> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.coordinates
                .iter()
                .map(f64::to_string)
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

impl<const N: usize> Neg for Cartesian<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            coordinates: self.coordinates.map(|x| -x),
        }
    }
}

impl Cross for Cartesian<3> {
    #[inline]
    fn cross(&self, other: &Self) -> Self {
        Cartesian::from((
            self[1] * other[2] - self[2] * other[1],
            self[2] * other[0] - self[0] * other[2],
            self[0] * other[1] - self[1] * other[0],
        ))
    }
}

impl Wedge for Cartesian<3> {
    type Bivector = Cartesian<3>;

    #[inline]
    /// Compute the wedge product of two vectors.
    ///
    /// ```math
    /// \textbf{A}=\textbf{a}\wedge{\textbf{b}}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::{Cartesian, Wedge};
    ///
    /// let a = Cartesian::from([1.0, 0.0, 0.0]);
    /// let b = Cartesian::from([0.0, 1.0, 0.0]);
    /// assert_eq!(a.wedge(&b), [0.0, 0.0, 1.0].into());
    /// ```
    fn wedge(&self, other: &Self) -> Self::Bivector {
        self.cross(other)
    }
}

impl Cartesian<4> {
    /// Compute the `N-1`-ary ([co-unary]) cross product of a four-dimensional vector.
    ///
    /// This function is a generalization of the cross product to general dimension,
    /// identifying a unique vector perpendicular to `N-1` other vectors. In two
    /// dimensions, this is [`Cartesian::<2>::perp`], and in three dimensions this is
    /// simply [`Cross`].
    ///
    /// In geometric algebra terms, this is the Hodge dual of the exterior (Wedge)
    /// product. Geometrically, the dot product of the resulting vector with all inputs
    /// is zero and the magnitude of the vector is the hypervolume of the
    /// parallelepiped spanned by the inputs.
    ///
    /// [co-unary]: https://ncatlab.org/nlab/show/cross+product#counary
    #[inline]
    #[must_use]
    pub fn counary_cross(vectors: &[Self; 3]) -> Self {
        /// Calculate a 2x2 matrix determinant while compensating for errors.
        /// <https://pharr.org/matt/blog/2019/11/03/difference-of-floats>
        #[inline(always)]
        fn diff_of_products(a: f64, b: f64, c: f64, d: f64) -> f64 {
            let cd = c * d;
            let err = (-c).mul_add(d, cd);
            let dop = a.mul_add(b, -cd);
            dop + err
        }
        let u = &vectors[0];
        let v = &vectors[1];
        let w = &vectors[2];

        // 2x2 Minors via Kahan's Exact FMA
        let m01 = diff_of_products(v[0], w[1], v[1], w[0]);
        let m02 = diff_of_products(v[0], w[2], v[2], w[0]);
        let m03 = diff_of_products(v[0], w[3], v[3], w[0]);
        let m12 = diff_of_products(v[1], w[2], v[2], w[1]);
        let m13 = diff_of_products(v[1], w[3], v[3], w[1]);
        let m23 = diff_of_products(v[2], w[3], v[3], w[2]);

        // Association is important here, as we can accumulate small sign errors if we
        // do not correctly group terms!
        [
            (u[1] * m23 - u[2] * m13 + u[3] * m12),
            -(u[0] * m23 - u[2] * m03 + u[3] * m02),
            (u[0] * m13 - u[1] * m03 + u[3] * m01),
            -(u[0] * m12 - u[1] * m02 + u[2] * m01),
        ]
        .into()
    }
}

impl<const N: usize> Distribution<Cartesian<N>> for StandardUniform {
    /// Sample a Cartesian vector from the uniform [-1, 1] hypercube.
    ///
    /// Each coordinate in the vector is in the closed range [-1, 1].
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::Cartesian;
    /// use rand::{RngExt, SeedableRng, rngs::StdRng};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut rng = StdRng::seed_from_u64(1);
    /// let v: Cartesian<3> = rng.random();
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Cartesian<N> {
        let uniform = Uniform::new_inclusive(-1.0, 1.0)
            .expect("hard-coded range should form a valid distribution");
        Cartesian {
            coordinates: array::from_fn(|_| uniform.sample(rng)),
        }
    }
}

impl<const N: usize, T> Index<T> for Cartesian<N>
where
    T: Into<usize> + std::slice::SliceIndex<[f64], Output = f64>,
{
    type Output = f64;
    /// Get the value of the vector at coordinate i.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v = Cartesian::<3>::try_from(3..6)?;
    /// assert_eq!((v[0], v[1], v[2]), (3.0, 4.0, 5.0));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn index(&self, index: T) -> &Self::Output {
        &self.coordinates[index]
    }
}

impl Cartesian<2> {
    /// Construct a 2-vector perpendicular to self.
    ///
    /// Given a vector $`(v_x, v_y)`$ `perpendicular` returns the vector
    /// rotated by $`\pi/2`$:
    /// ```math
    /// (-v_y, v_x)
    /// ```
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::Cartesian;
    ///
    /// let v = Cartesian::from([1.0, -4.5]);
    /// assert_eq!(v.perpendicular(), [4.5, 1.0].into());
    /// ```
    #[inline]
    #[must_use]
    pub fn perpendicular(self) -> Self {
        Cartesian::from([-self[1], self[0]])
    }
}

impl Wedge for Cartesian<2> {
    type Bivector = f64;

    /// Compute the wedge product of two vectors.
    ///
    /// ```math
    /// \textbf{A}=\textbf{a}\wedge{\textbf{b}}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::{Cartesian, Wedge};
    ///
    /// let a = Cartesian::from([2.0, 1.0]);
    /// let b = Cartesian::from([3.0, 1.0]);
    ///
    /// assert_eq!(a.wedge(&b), -1.0);
    /// ```
    #[inline]
    fn wedge(&self, other: &Self) -> Self::Bivector {
        self[0] * other[1] - self[1] * other[0]
    }
}

impl<const N: usize> Outer for Cartesian<N> {
    type Output = Matrix<N, N>;

    #[inline]
    fn outer(&self, other: &Self) -> Self::Output {
        self.to_column_matrix().matmul(&other.to_row_matrix())
    }
}

impl<const N: usize, T> IndexMut<T> for Cartesian<N>
where
    T: Into<usize> + std::slice::SliceIndex<[f64], Output = f64>,
{
    /// Get a mutable reference to the value of the vector at coordinate i.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut v = Cartesian::<3>::try_from(3..6)?;
    /// assert_eq!((v[0], v[1], v[2]), (3.0, 4.0, 5.0));
    /// v[0] += 1.0;
    /// assert_eq!(v[0], 4.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn index_mut(&mut self, index: T) -> &mut Self::Output {
        &mut self.coordinates[index]
    }
}

impl<const N: usize> Sum for Cartesian<N> {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Cartesian::default(), |acc, x| acc + x)
    }
}

/// Rotate vectors efficiently.
///
/// Construct a [`RotationMatrix`] to efficiently rotate many vectors by the same rotation.
///
/// See:
/// * [`RotationMatrix::from<Angle>`]
/// * [`RotationMatrix::from<Versor>`]
///
/// [`RotationMatrix`] _intentionally_ does not implement [`Rotation`](crate::Rotation).
/// [`Angle`](crate::Angle) and [`Versor`](crate::Versor) are representations of
/// rotations that are often the most effective and numerically stable to
/// manipulate.
#[serde_as]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RotationMatrix<const N: usize> {
    /// Rows of the rotation matrix.
    #[serde_as(as = "[_; N]")]
    pub(crate) rows: [Cartesian<N>; N],
}

impl<const N: usize> From<RotationMatrix<N>> for Matrix<N, N> {
    #[inline]
    fn from(value: RotationMatrix<N>) -> Self {
        Self {
            rows: value.rows().map(|arr| arr.coordinates),
        }
    }
}

impl<const N: usize> RotationMatrix<N> {
    /// Get the rows of the rotation matrix.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::{Angle, InnerProduct, RotationMatrix, Vector};
    /// use std::f64::consts::PI;
    ///
    /// let a = Angle::from(PI / 2.0);
    ///
    /// let matrix = RotationMatrix::from(a);
    /// assert!(matrix.rows()[0].dot(&[0.0, -1.0].into()) > 0.99);
    /// assert!(matrix.rows()[1].dot(&[1.0, 0.0].into()) > 0.99);
    /// ```
    #[inline]
    #[must_use]
    pub fn rows(&self) -> [Cartesian<N>; N] {
        self.rows
    }

    /// Create a matrix that performs the inverse rotation.
    ///
    /// Matrix inversion is cheaper than [`Angle`](crate::Angle) ->
    /// [`RotationMatrix`] and [`Versor`](crate::Versor) -> [`RotationMatrix`]
    /// conversions. When you need both rotations, convert once and then invert.
    ///
    /// # Example
    /// ```
    /// use hoomd_vector::{Angle, InnerProduct, RotationMatrix, Vector};
    /// use std::f64::consts::PI;
    ///
    /// let a = Angle::from(PI / 2.0);
    ///
    /// let matrix = RotationMatrix::from(a);
    /// let inverted_matrix = matrix.inverted();
    /// assert!(inverted_matrix.rows()[0].dot(&[0.0, 1.0].into()) > 0.99);
    /// assert!(inverted_matrix.rows()[1].dot(&[-1.0, 0.0].into()) > 0.99);
    /// ```
    #[inline]
    #[must_use]
    pub fn inverted(self) -> Self {
        Self {
            rows: array::from_fn(|i| array::from_fn::<_, N, _>(|j| self.rows()[j][i]).into()),
        }
    }
}

impl<const N: usize> Default for RotationMatrix<N> {
    /// Create an identity matrix.
    ///
    /// ```math
    /// \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
    /// ```
    /// ,
    /// ```math
    /// \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}
    /// ```
    /// , and so on.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_vector::RotationMatrix;
    ///
    /// let identity = RotationMatrix::<3>::default();
    /// ```
    #[inline]
    fn default() -> RotationMatrix<N> {
        RotationMatrix {
            rows: array::from_fn(|i| array::from_fn(|j| if i == j { 1.0 } else { 0.0 }).into()),
        }
    }
}

impl<const N: usize> fmt::Display for RotationMatrix<N> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.rows
                .iter()
                .map(Cartesian::<N>::to_string)
                .collect::<Vec<String>>()
                .join("\n ")
        )
    }
}

impl<const N: usize> Rotate<Cartesian<N>> for RotationMatrix<N> {
    type Matrix = RotationMatrix<N>;

    #[inline]
    /// Rotate a [`Cartesian<N>`] by a [`RotationMatrix`]
    ///
    /// # Examples
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
    ///
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
    fn rotate(&self, vector: &Cartesian<N>) -> Cartesian<N> {
        let mut coordinates = [0.0; N];

        for (result, row) in coordinates.iter_mut().zip(self.rows.iter()) {
            *result = row.dot(vector);
        }

        Cartesian { coordinates }
    }
}

impl<const N: usize, const K: usize> MatMul<Matrix<N, K>> for RotationMatrix<N> {
    type Output = Matrix<N, K>;

    #[inline]
    fn matmul(&self, rhs: &Matrix<N, K>) -> Self::Output {
        Matrix::from(*self).matmul(rhs)
    }
}

impl<const N: usize> Cartesian<N> {
    /// Convert a [`Cartesian<N>`] into a row matrix [`Matrix<1, N>`].
    ///
    /// # Example
    /// ```
    /// use hoomd_linear_algebra::matrix::Matrix;
    /// use hoomd_vector::Cartesian;
    ///
    /// let a = Cartesian::from([1.0, -2.0, 3.0]);
    ///
    /// let b = a.to_row_matrix();
    /// assert_eq!(b.rows, [[1.0, -2.0, 3.0]]);
    /// ```
    #[inline]
    #[must_use]
    pub fn to_row_matrix(self) -> Matrix<1, N> {
        Matrix {
            rows: [self.coordinates],
        }
    }
    /// Convert a [`Cartesian<N>`] into a column matrix [`Matrix<N, 1>`].
    ///
    /// # Example
    /// ```
    /// use hoomd_linear_algebra::matrix::Matrix;
    /// use hoomd_vector::Cartesian;
    ///
    /// let a = Cartesian::from([1.0, -2.0, 3.0]);
    ///
    /// let b = a.to_column_matrix();
    /// assert_eq!(b.rows, [[1.0], [-2.0], [3.0]]);
    /// ```
    #[inline]
    #[must_use]
    pub fn to_column_matrix(self) -> Matrix<N, 1> {
        Matrix {
            rows: std::array::from_fn(|i| [self[i]]),
        }
    }
}

impl<const N: usize> From<Matrix<1, N>> for Cartesian<N> {
    #[inline]
    fn from(value: Matrix<1, N>) -> Self {
        value.rows[0].into()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Angle, Rotation, Versor};

    use super::*;
    use approxim::assert_relative_eq;
    use paste::paste;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use rstest::rstest;

    // Parameterize a test function over an array of vector lengths
    macro_rules! parameterize_vector_length {
        // macro with name as above that takes an identifier (fn) and an expression
        // $(...),* matches 0 or more expressions (values) separated by commas
        ($test_body:ident, [$($dim:expr),*]) => {

            // Now, we repeat the test block 0 or more times, one for each $dim
            $(
                // paste package combines values in [< >] to form a new ident
                paste! {
                    #[test]
                    fn [< $test_body "_" $dim>]() {
                        const DIM: usize = $dim;
                        $test_body::<DIM>();
                    }
                }
            )*
        };
    }

    /// Generate a pair of length N vectors.
    /// The first vector ranges from [0, N-1] and the second ranges from [N, 2*N-1]
    fn generate_vector_pair<const N: usize>() -> (Cartesian<N>, Cartesian<N>) {
        (
            Cartesian::try_from(0..N).unwrap(),
            Cartesian::try_from(N..N * 2).unwrap(),
        )
    }

    fn index<const N: usize>() {
        let (_, b) = generate_vector_pair::<N>();
        assert!(zip(0..N, b.coordinates.iter()).all(|(i, &x)| b[i] == x));
    }
    parameterize_vector_length!(index, [2, 3, 4, 8, 16, 32]);

    fn index_mut<const N: usize>() {
        let (a, mut b) = generate_vector_pair::<N>();
        zip(0..N, b.coordinates.iter_mut()).for_each(|(i, x)| *x = a[i]);
        assert_eq!(a, b);
    }
    parameterize_vector_length!(index_mut, [2, 3, 4, 8, 16, 32]);

    fn add_explicit<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();
        let c = a.add(b);

        let addition_answer: Vec<f64> = (0..(2 * N))
            .step_by(2)
            .map(|x| (x + N) as f64)
            .collect::<Vec<_>>();

        assert_eq!(c, Cartesian::try_from(addition_answer).unwrap());
    }
    parameterize_vector_length!(add_explicit, [2, 3, 4, 8, 16, 32]);

    fn add_operator<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();
        let c = a + b;

        let addition_answer: Vec<f64> = (0..(2 * N))
            .step_by(2)
            .map(|x| (x + N) as f64)
            .collect::<Vec<_>>();

        assert_eq!(c, Cartesian::try_from(addition_answer).unwrap());
    }
    parameterize_vector_length!(add_operator, [2, 3, 4, 8, 16, 32]);

    fn add_assign<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();
        let mut c = a;
        c += b;

        let addition_answer: Vec<f64> = (0..(2 * N))
            .step_by(2)
            .map(|x| (x + N) as f64)
            .collect::<Vec<_>>();

        assert_eq!(c, Cartesian::try_from(addition_answer).unwrap());
    }
    parameterize_vector_length!(add_assign, [2, 3, 4, 8, 16, 32]);

    fn sub_operator<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();
        let c = a - b;

        let subtraction_answer = [-(N as f64); N];

        assert_eq!(c, subtraction_answer.into());
    }
    parameterize_vector_length!(sub_operator, [2, 3, 4, 8, 16, 32]);

    fn sub_assign<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();
        let mut c = a;
        c -= b;

        let subtraction_answer = [-(N as f64); N];

        assert_eq!(c, subtraction_answer.into());
    }

    parameterize_vector_length!(sub_assign, [2, 3, 4, 8, 16, 32]);

    fn mul_operator<const N: usize>() {
        let (a, _) = generate_vector_pair::<N>();
        let b = 12.0;
        let c = a * b;

        let multiplication_answer: Vec<f64> = (0..N).map(|x| (x as f64) * b).collect::<Vec<_>>();

        assert_eq!(c, Cartesian::try_from(multiplication_answer).unwrap());
    }
    parameterize_vector_length!(mul_operator, [2, 3, 4, 8, 16, 32]);

    fn mul_assign<const N: usize>() {
        let (mut a, _) = generate_vector_pair::<N>();
        let b = 12.0;
        a *= b;

        let multiplication_answer: Vec<f64> = (0..N).map(|x| (x as f64) * b).collect::<Vec<_>>();

        assert_eq!(a, Cartesian::try_from(multiplication_answer).unwrap());
    }

    parameterize_vector_length!(mul_assign, [2, 3, 4, 8, 16, 32]);

    fn div_operator<const N: usize>() {
        let (a, _) = generate_vector_pair::<N>();
        let b = 12.0;
        let c = a / b;

        let division_answer: Vec<f64> = (0..N).map(|x| (x as f64) / b).collect::<Vec<_>>();

        assert_eq!(c, Cartesian::try_from(division_answer).unwrap());
    }
    parameterize_vector_length!(div_operator, [2, 3, 4, 8, 16, 32]);

    fn div_assign<const N: usize>() {
        let (mut a, _) = generate_vector_pair::<N>();
        let b = 12.0;
        a /= b;

        let division_answer: Vec<f64> = (0..N).map(|x| (x as f64) / b).collect::<Vec<_>>();

        assert_eq!(a, Cartesian::try_from(division_answer).unwrap());
    }

    parameterize_vector_length!(div_assign, [2, 3, 4, 8, 16, 32]);

    fn compute_add_ref_ref<const N: usize>(a: &Cartesian<N>, b: &Cartesian<N>) -> Cartesian<N> {
        *a + *b
    }

    fn compute_add_ref_type<const N: usize>(a: &Cartesian<N>, b: Cartesian<N>) -> Cartesian<N> {
        *a + b
    }

    fn compute_add_type_ref<const N: usize>(a: Cartesian<N>, b: &Cartesian<N>) -> Cartesian<N> {
        a + *b
    }

    fn add_with_refs<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();

        let addition_answer = Cartesian::try_from(
            (0..(2 * N))
                .step_by(2)
                .map(|x| (x + N) as f64)
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let c = compute_add_ref_ref(&a, &b);
        assert_eq!(c, addition_answer);

        let c = compute_add_ref_type(&a, b);
        assert_eq!(c, addition_answer);

        let c = compute_add_type_ref(a, &b);
        assert_eq!(c, addition_answer);
    }
    parameterize_vector_length!(add_with_refs, [2, 3, 4, 8, 16, 32]);

    fn dot<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();
        let c = a.dot(&b);

        let n = N as f64;

        // Analytical solution to sum of k * (n+k), k = 0 to n-1
        let dot_ans = (5.0 * n.powi(3) - 6.0 * n.powi(2) + n) / 6.0;

        assert_eq!(c, dot_ans);
    }
    parameterize_vector_length!(dot, [2, 3, 4, 8, 16, 32]);

    fn neg<const N: usize>() {
        let a: Cartesian<N> = Cartesian::try_from(0..N).unwrap();
        let b = -a;
        for (i, j) in zip(a.coordinates.iter(), b.coordinates.iter()) {
            assert_eq!(*i, -j);
        }
    }
    parameterize_vector_length!(neg, [2, 3, 4, 8, 16, 32]);

    #[test]
    fn cross() {
        let (a, b) = generate_vector_pair::<3>();
        let c = a.cross(&b);

        // Analytical solution
        let cross_ans = Cartesian::from((-3.0, 6.0, -3.0));

        assert_eq!(c, cross_ans);

        let a = Cartesian::from([1.0, 0.0, 0.0]);
        let b = Cartesian::from([0.0, 1.0, 0.0]);
        assert_eq!(a.cross(&b), [0.0, 0.0, 1.0].into());

        let a = Cartesian::from([0.0, 3.0, 0.0]);
        let b = Cartesian::from([0.0, 0.0, 2.0]);
        assert_eq!(a.cross(&b), [6.0, 0.0, 0.0].into());

        let a = Cartesian::from([2.0, 0.0, 0.0]);
        let b = Cartesian::from([0.0, 0.0, 4.0]);
        assert_eq!(a.cross(&b), [0.0, -8.0, 0.0].into());
    }

    #[test]
    fn display() {
        let a = Cartesian::from([1.5, 2.125, -3.875]);
        let s = format!("{a}");
        assert_eq!(s, "[1.5, 2.125, -3.875]");

        let a = Cartesian::from([10.0, 20.0, 30.0, 40.0]);
        let s = format!("{a}");
        assert_eq!(s, "[10, 20, 30, 40]");
    }

    #[test]
    fn from_2_tuple() {
        let a = Cartesian::from((3.0, 0.125));
        assert_eq!(a.coordinates, [3.0, 0.125]);
    }

    #[test]
    fn from_3_tuple() {
        let a = Cartesian::from((-0.5, 2.0, 18.125));
        assert_eq!(a.coordinates, [-0.5, 2.0, 18.125]);
    }

    fn from_vec<const N: usize>() {
        let mut vec = Vec::with_capacity(N);

        assert_eq!(
            Cartesian::<N>::try_from(vec.clone()),
            Err(Error::InvalidVectorLength)
        );

        for i in 0..N {
            vec.push(i as f64 * 0.5);
        }
        let a = Cartesian::<N>::try_from(vec.clone()).unwrap();

        assert_eq!(vec, Vec::from(a.coordinates));

        vec.push(1.0);
        assert_eq!(
            Cartesian::<N>::try_from(vec.clone()),
            Err(Error::InvalidVectorLength)
        );
    }
    parameterize_vector_length!(from_vec, [2, 3, 4, 8, 16, 32]);

    fn random_in_range<const N: usize>() {
        // Loosely verify we are drawing from the correct distribution
        let mut rng = StdRng::seed_from_u64(1);
        let a: Cartesian<N> = rng.random();

        assert!(a.coordinates.iter().all(|&x| -1.0 < x && x < 1.0));

        // This test will fail ~1e-3008 percent of the time - it's probably fine
        if N == 10_000 {
            assert!(a.coordinates.iter().any(|&x| x < 0.0));
        }
    }

    parameterize_vector_length!(random_in_range, [2, 3, 4, 8, 16, 32, 10_000]);

    #[test]
    fn to_unit() {
        let a = Cartesian::from((2.0, 0.0, 0.0));
        let (Unit(unit_a), _) = a
            .to_unit()
            .expect("hard-coded vector should have non-zero length");
        assert_eq!(unit_a, [1.0, 0.0, 0.0].into());

        let (Unit(unit_a), _) = a.to_unit_unchecked();
        assert_eq!(unit_a, [1.0, 0.0, 0.0].into());

        let Unit(unit_a) = Unit::<Cartesian<3>>::try_from([1.0, 0.0, 0.0])
            .expect("hard-coded vector should have non-zero length");
        assert_eq!(unit_a, [1.0, 0.0, 0.0].into());

        let a = Cartesian::from((3.0, 0.0, 4.0));
        let (Unit(unit_a), _) = a
            .to_unit()
            .expect("hard-coded vector should have non-zero length");
        assert_eq!(unit_a, [3.0 / 5.0, 0.0, 4.0 / 5.0].into());

        let (Unit(unit_a), _) = a.to_unit_unchecked();
        assert_eq!(unit_a, [3.0 / 5.0, 0.0, 4.0 / 5.0].into());

        let Unit(unit_a) = Unit::<Cartesian<3>>::try_from([3.0, 0.0, 4.0])
            .expect("hard-coded vector should have non-zero length");
        assert_eq!(unit_a, [3.0 / 5.0, 0.0, 4.0 / 5.0].into());

        let a = Cartesian::from((0.0, 0.0, 0.0));
        assert!(matches!(a.to_unit(), Err(Error::InvalidVectorMagnitude)));
        assert!(matches!(
            Unit::<Cartesian<3>>::try_from([0.0, 0.0, 0.0]),
            Err(Error::InvalidVectorMagnitude)
        ));
    }

    #[test]
    fn sum() {
        let total: Cartesian<2> = [Cartesian::from((1.0, 2.0)), Cartesian::from((-2.0, -1.0))]
            .into_iter()
            .sum();

        assert_eq!(total, [-1.0, 1.0].into());
    }

    #[test]
    fn perpendicular() {
        let v = Cartesian::from([1.0, -4.5]);
        assert_eq!(v.perpendicular(), [4.5, 1.0].into());
    }

    #[test]
    fn test_rotationmatrix_display() {
        let m = RotationMatrix::<3>::default();
        let s = format!("{m}");
        assert_eq!(
            s,
            "[[1, 0, 0]
 [0, 1, 0]
 [0, 0, 1]]"
        );

        let m = RotationMatrix::<2>::default();
        let s = format!("{m}");
        assert_eq!(
            s,
            "[[1, 0]
 [0, 1]]"
        );
    }

    #[rstest(
        angle => [
            Angle::default(),
            Angle::from(std::f64::consts::FRAC_PI_3),
            Angle::from(999.9 * std::f64::consts::PI / 1.234)
        ]
    )]
    fn test_rotationmatrix_invert_2d(angle: Angle) {
        let transposed = RotationMatrix::from(angle).inverted();
        let inverted = RotationMatrix::from(angle.inverted());
        for (a, b) in transposed.rows().iter().zip(inverted.rows().iter()) {
            assert_relative_eq!(a, b);
        }
    }
    #[rstest(
        versor => [
            Versor::default(),
            Versor::from_axis_angle(
                Unit::try_from([1.0, 0.0, 0.0]).unwrap(), std::f64::consts::PI / 1.234
            )
        ]
    )]
    fn test_rotationmatrix_invert_3d(versor: Versor) {
        let transposed = RotationMatrix::from(versor).inverted();
        let inverted = RotationMatrix::from(versor.inverted());
        for (a, b) in transposed.rows().iter().zip(inverted.rows().iter()) {
            assert_relative_eq!(a, b);
        }
    }

    #[rstest]
    fn test_generalized_cross_product_combinations(
        #[values(1.0, -2.5, 4.0)] r1: f64,
        #[values(1.0, 3.0, -5.1)] r2: f64,
        #[values(2.0, -1.5, 0.0)] r3: f64,
        #[values(0.0, 33.0, -12.5, 0.04)] x: f64,
        #[values(0.0, 1.0, 3.0, -2.6)] y: f64,
        #[values(0.0, 2.0, -1.5, -0.1)] z: f64,
    ) {
        // Construct a lower-triangular matrix of vectors
        let vecs = [
            Cartesian::from([r1, 0.0, 0.0, 0.0]),
            Cartesian::from([x, r2, 0.0, 0.0]),
            Cartesian::from([y, z, r3, 0.0]),
        ];

        let cross = Cartesian::<4>::counary_cross(&vecs);

        // Validate the result is perpendicular to all inputs
        for v in &vecs {
            assert_relative_eq!(cross.dot(v), 0.0);
        }

        // Validate the magnitude is exactly the spanned volume
        // For a triangular matrix, the volume is the product of the diagonals.
        let expected_volume_sq = (r1 * r2 * r3).powi(2);
        assert_relative_eq!(cross.norm_squared(), expected_volume_sq);
    }
}

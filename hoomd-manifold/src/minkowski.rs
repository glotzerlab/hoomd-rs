// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement vector types in Minkowski space.

use std::{
    array,
    f64::consts::PI,
    fmt,
    iter::zip,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

use approxim::{approx_derive::RelativeEq, assert_relative_eq};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform, Uniform},
};

use crate::{Error, HyperbolicRotate};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Metric, Vector};

/// A vector in N-dimensional Minkowski space.
///
/// [`Minkowski<N>`] implements (N-1,1)-dimensional Minkowski space with the
/// metric signature $`(+ , \cdots , + , -)`$. [`Minkowski`] supports
/// [`Vector`] operations such as vector addition and rescaling.
///
/// ## Constructing Minkowski vectors
///
/// Similar to [`Cartesian`], N-dimensional vectors can be
/// constructed using an array of (real-valued) coordinates. Three- and
/// four-dimensional vectors can also be constructed from tuples:
/// ```
/// use hoomd_manifold::Minkowski;
///
/// fn from_array() -> Minkowski<5> {
///     Minkowski::from([1.0, 2.0, 3.0, 4.0, 5.0])
/// }
/// fn from_tuples() -> Minkowski<3> {
///     Minkowski::from((6.0, 7.0, 8.0))
/// }
/// ```
///
/// ## Operating on Minkowski vectors
///
/// [`Minkowski`] implements everything from [`Vector`], which includes vector
/// addition/subtraction and multiplication by a scalar.
///
/// ```
/// use hoomd_manifold::Minkowski;
///
/// // Vector addition
/// let mut a = Minkowski::from([1.0, 1.0, 1.0, 1.0]);
/// let mut b = Minkowski::from([0.0, 0.0, 0.0, 2.0]);
/// a += b;
///
/// // Multiplication by a scalar
/// let mut c = a * 4.0;
///
/// // Division by a scalar
/// c /= 2.0;
///
/// assert_eq!(c, [2.0, 2.0, 2.0, 6.0].into());
/// ```
///
/// The distance metric on Minkowski space is given by the "spacetime interval"
/// ```math
/// d_M^2(\vec{u},\vec{v}) = (\vec{u}-\vec{v})^T \eta (\vec{u}-\vec{v})
/// = (u_1-v_1)^2 +\cdots + (u_{N-1}-v_{N-1})^2 - (u_N - v_N)^2
/// ```
/// Note that because this metric is not positive-definite, [`Minkowski`] this
/// "spacetime interval" is not a true inner-product, and therefore
/// [`Minkowski`] does not implement the methods of [`InnerProduct`].
///
/// [`InnerProduct`]: hoomd_vector::InnerProduct
/// [`Cartesian`]: hoomd_vector::Cartesian
///
/// ```
/// use hoomd_manifold::Minkowski;
/// use hoomd_vector::Metric;
///
/// let x = Minkowski::from([1.0, 0.0, 5.0]);
/// let y = Minkowski::from([0.0, 0.0, 3.0]);
/// assert_eq!(-3.0, x.distance_squared(&y));
/// ```

#[derive(Clone, Copy, Debug, PartialEq, RelativeEq)]
#[approx(epsilon_type = f64)]
pub struct Minkowski<const N: usize> {
    /// The vector's coordinates.
    ///
    /// The final component is the one associated with a minus sign (-) in the metric.
    #[approx(into_iter)]
    pub coordinates: [f64; N],
}

impl<const N: usize> Default for Minkowski<N> {
    /// Create a 0 vector in Minkowski space.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Minkowski;
    ///
    /// let v = Minkowski::<3>::default();
    /// assert_eq!(v, [0.0; 3].into())
    /// ```
    #[inline]
    fn default() -> Self {
        Minkowski::from([0.0; N])
    }
}

impl<const N: usize> From<[f64; N]> for Minkowski<N> {
    /// Create a vector in Minkowski space with the given coordinates.
    ///
    /// The last component has a (-) signature, while the preceding coordinates
    /// have (+) signatures in the metric.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Minkowski;
    ///
    /// let v = Minkowski::from([1.0, 2.0]);
    /// ```
    #[inline]
    fn from(coordinates: [f64; N]) -> Self {
        Self { coordinates }
    }
}

impl From<(f64, f64)> for Minkowski<2> {
    #[inline]
    fn from(coordinates: (f64, f64)) -> Self {
        Self {
            coordinates: coordinates.into(),
        }
    }
}

impl From<(f64, f64, f64)> for Minkowski<3> {
    #[inline]
    fn from(coordinates: (f64, f64, f64)) -> Self {
        Self {
            coordinates: coordinates.into(),
        }
    }
}

impl From<(f64, f64, f64, f64)> for Minkowski<4> {
    #[inline]
    fn from(coordinates: (f64, f64, f64, f64)) -> Self {
        Self {
            coordinates: coordinates.into(),
        }
    }
}

impl<const N: usize> TryFrom<Vec<f64>> for Minkowski<N> {
    type Error = Error;

    /// Create a vector in Minkowski with coordinates given by a [`Vec<f64>`]
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Minkowski;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v = Minkowski::<3>::try_from(vec![5.0, 4.0, 3.0])?;
    /// assert_eq!(v, [5.0, 4.0, 3.0].into());
    /// # Ok(())
    /// # }
    /// ```
    /// <div class="warning">
    ///
    /// Use `Minkowski::From<[f64; N]>` in performance critical code.
    ///
    /// </div>
    #[inline]
    fn try_from(value: Vec<f64>) -> Result<Self, Self::Error> {
        let coordinates = value.try_into().map_err(|_| Error::InvalidVectorLength)?;
        Ok(Self { coordinates })
    }
}

impl<const N: usize> TryFrom<std::ops::Range<usize>> for Minkowski<N> {
    type Error = Error;

    /// Create a vector in Minkowski space with coordinates given by a range.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Minkowski;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v = Minkowski::<3>::try_from(1..4)?;
    /// assert_eq!(v, [1.0, 2.0, 3.0].into());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn try_from(value: std::ops::Range<usize>) -> Result<Self, Self::Error> {
        if value.len() != N {
            return Err(Error::InvalidVectorLength);
        }

        // The default value of 0 will never be used due to the above error check.
        let mut iter = value;
        let coordinates = array::from_fn(|_| iter.next().unwrap_or(0) as f64);
        Ok(Self { coordinates })
    }
}

impl<const N: usize> Metric for Minkowski<N> {
    /// Computes the squared distance between two points in Minkowski space.
    ///
    /// Employs the "mostly plusses" metric signature (+ ... + -).
    /// ```math
    /// d^2_M(\vec{x},\vec{y}) = -(x_N-y_N)^2 + \sum_{i=1}^{N-1} (x_i - y_i)^2
    /// ```
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Minkowski;
    /// use hoomd_vector::{Metric, Vector};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let x = Minkowski::from([0.0, 2.0, 3.0]);
    /// let y = Minkowski::from([1.0, 0.0, 0.0]);
    /// assert_eq!(-4.0, x.distance_squared(&y));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn distance_squared(&self, other: &Self) -> f64 {
        let last_component = -(self.coordinates[N - 1] - other.coordinates[N - 1]).powi(2);
        zip(
            self.coordinates[0..N - 1].iter(),
            other.coordinates[0..N - 1].iter(),
        )
        .fold(last_component, |product, x| product + (x.0 - x.1).powi(2))
    }
    #[inline]
    fn distance(&self, other: &Self) -> f64 {
        ((self.distance_squared(other)).abs()).sqrt()
    }
    #[inline]
    fn n_dimensions(&self) -> usize {
        N
    }
}

impl<const N: usize> Vector for Minkowski<N> {}

impl<const N: usize> Add for Minkowski<N> {
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

impl<const N: usize> AddAssign for Minkowski<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for (result, a) in self.coordinates.iter_mut().zip(rhs.coordinates.iter()) {
            *result += a;
        }
    }
}

impl<const N: usize> Div<f64> for Minkowski<N> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self {
        let mut coordinates = [0.0; N];

        for (result, a) in coordinates.iter_mut().zip(self.coordinates) {
            *result = a / rhs;
        }
        Self { coordinates }
    }
}

impl<const N: usize> DivAssign<f64> for Minkowski<N> {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        for result in &mut self.coordinates {
            *result /= rhs;
        }
    }
}

impl<const N: usize> Mul<f64> for Minkowski<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        let mut coordinates = [0.0; N];
        for (result, a) in coordinates.iter_mut().zip(self.coordinates) {
            *result = a * rhs;
        }
        Self { coordinates }
    }
}

impl<const N: usize> MulAssign<f64> for Minkowski<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        for result in &mut self.coordinates {
            *result *= rhs;
        }
    }
}

impl<const N: usize> Sub for Minkowski<N> {
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

impl<const N: usize> SubAssign for Minkowski<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for (result, a) in self.coordinates.iter_mut().zip(rhs.coordinates) {
            *result -= a;
        }
    }
}

impl<const N: usize> fmt::Display for Minkowski<N> {
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

impl<const N: usize> Neg for Minkowski<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        let mut result = self;
        result.coordinates.iter_mut().for_each(|x| *x = -*x);
        result
    }
}

impl<const N: usize, T> Index<T> for Minkowski<N>
where
    T: Into<usize> + std::slice::SliceIndex<[f64], Output = f64>,
{
    type Output = f64;
    /// Get the value of the vector at coordinate i.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Minkowski;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v = Minkowski::<3>::try_from(4..7)?;
    /// assert_eq!((v[0], v[1], v[2]), (4.0, 5.0, 6.0));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn index(&self, index: T) -> &Self::Output {
        &self.coordinates[index]
    }
}

impl<const N: usize, T> IndexMut<T> for Minkowski<N>
where
    T: Into<usize> + std::slice::SliceIndex<[f64], Output = f64>,
{
    /// Get a mutable reference to the value of the vector at coordinate `i`.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Minkowski;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut v = Minkowski::<3>::try_from(4..7)?;
    /// assert_eq!((v[0], v[1], v[2]), (4.0, 5.0, 6.0));
    /// v[0] += 1.0;
    /// assert_eq!(v[0], 5.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn index_mut(&mut self, index: T) -> &mut Self::Output {
        &mut self.coordinates[index]
    }
}

impl<const N: usize> Distribution<Minkowski<N>> for StandardUniform {
    /// Sample a Minkowski vector from the uniform $` [-1, 1] `$ hypercube.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Minkowski;
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut rng = StdRng::seed_from_u64(1);
    /// let v: Minkowski<3> = rng.random();
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Minkowski<N> {
        let uniform = Uniform::new_inclusive(-1.0, 1.0)
            .expect("hard-coded range should form a valid distribution");
        Minkowski {
            coordinates: array::from_fn(|_| uniform.sample(rng)),
        }
    }
}

/// Point on the top sheet of a Hyperboloid.
///
/// [`Hyperbolic`] implements an embedding of the top sheet of an
/// (N-1)-dimensional two-sheeted hyperboloid in N-dimensional Minkowski space.
/// This surface has constant negative curvature and therefore serves as a model
/// of (N-1)-dimensional hyperbolic space.
///
/// Explicitly, for N-dimensional Minkowski space with metric $`\eta =
/// \operatorname{diag}(+,\cdots,+,-)`$, the Hyperbolic with skirt width $`R`$ is
/// defined by the set of points with components satisfying
/// ```math
/// x_1^2 +\cdots x_{N-1}^2 - x_{N}^2 = -R^2
/// ```
/// Where the "top sheet" is defined by the $`x_N>0`$ solutions. In Minkowski
/// space, the hyperboloid has a natural interpretation as the set of points
/// with the same spacetime interval
/// ```math
/// \Delta s^2 = \vec{x}^T \eta \vec{x} = x_1^2 +\cdots x_{N-1}^2 - x_{N}^2
/// ```
///
/// [`Hyperbolic`] implements a [`Metric`] that computes the distance of the
/// geodesic passing between two points on a hyperboloid with some given skirt
/// width.
///
/// Two points on the hyperboloid with skirt width $` R = 1.0 `$:
/// ```
/// use hoomd_manifold::{Hyperbolic, Minkowski};
/// use hoomd_vector::Metric;
///
/// let x =
///     Hyperbolic::from_minkowski_coordinates([0.0, 0.0, 1.0].into(), 1.0_f64);
///
/// let y = Hyperbolic::from_minkowski_coordinates(
///     [0.0, 1.0, (2.0_f64).sqrt()].into(),
///     1.0_f64,
/// );
///
/// assert_eq!(((2.0_f64).sqrt()).acosh(), x.distance(&y));
/// ```
///
/// [`Metric`]: hoomd_vector::Metric
#[derive(Clone, Copy, Debug, PartialEq, RelativeEq)]
pub struct Hyperbolic<const N: usize> {
    /// A point on the surface of the upper sheet of a two-sheeted hyperboloid.
    point: Minkowski<N>,
    /// The skirt width of the Hyperbolic.
    skirt: f64,
}

impl<const N: usize> Hyperbolic<N> {
    /// Get the coordinates of the point on the hyperboloid.
    #[must_use]
    #[inline]
    pub fn coordinates(&self) -> &[f64; N] {
        &self.point.coordinates
    }
    /// Get the Minkowski point of the hyperboloid.
    #[must_use]
    #[inline]
    pub fn point(&self) -> &Minkowski<N> {
        &self.point
    }
    /// Get the skirt width of the hyperboloid.
    #[must_use]
    #[inline]
    pub fn skirt(&self) -> f64 {
        self.skirt
    }
    /// Create a Hyperbolic point from a Minkowski vector.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::{Hyperbolic, Minkowski};
    /// use hoomd_vector::Metric;
    ///
    /// let x =
    ///     Hyperbolic::from_minkowski_coordinates([0.0, 0.0, 1.0].into(), 1.0_f64);
    /// ```
    #[must_use]
    #[inline]
    pub fn from_minkowski_coordinates(point: Minkowski<N>, skirt: f64) -> Hyperbolic<N> {
        let skirt_squared = -point.distance_squared(&Minkowski::<N>::default());
        assert_relative_eq!(skirt_squared, skirt.powi(2), epsilon = 1e-12);
        Hyperbolic { point, skirt }
    }
}

impl Hyperbolic<3> {
    /// Create a point on the surface of a three-dimensional hyperboloid from the polar representation.
    #[must_use]
    #[inline]
    pub fn from_polar_coordinates(v: f64, theta: f64, skirt: f64) -> Hyperbolic<3> {
        let theta_mod = theta.rem_euclid(2.0 * PI);
        let point = Minkowski::from([
            skirt * (v.sinh()) * (theta_mod.cos()),
            skirt * (v.sinh()) * (theta_mod.sin()),
            skirt * (v.cosh()),
        ]);
        Hyperbolic::from_minkowski_coordinates(point, skirt)
    }
}

impl Hyperbolic<4> {
    /// Create a point on the surface of a four-dimensional hyperboloid from the spherical representation.
    #[must_use]
    #[inline]
    pub fn from_polar_coordinates(v: f64, theta: f64, phi: f64, skirt: f64) -> Hyperbolic<4> {
        let theta_mod = theta.rem_euclid(2.0 * PI);
        let phi_mod = phi.rem_euclid(PI);
        let point = Minkowski::from([
            skirt * (v.sinh()) * (theta_mod.cos()),
            skirt * (v.sinh()) * (theta_mod.sin()) * (phi_mod.cos()),
            skirt * (v.sinh()) * (theta_mod.sin()) * (phi_mod.sin()),
            skirt * (v.cosh()),
        ]);
        Hyperbolic::from_minkowski_coordinates(point, skirt)
    }
}

impl<const N: usize> Hyperbolic<N> {
    /// Compute the distance from a point to the cusp.
    ///
    /// Computes the length of the geodesic passing between the cusp
    /// $`(0,\cdots,0,\rho)`$ and a given point on the hyperboloid with a given
    /// skirt length.
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_manifold::{Hyperbolic, Minkowski};
    /// use hoomd_vector::Vector;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v: f64 = 4.2;
    /// let rho: f64 = 1.0;
    /// let x = Hyperbolic::from_minkowski_coordinates(
    ///     [rho * (v.sinh()), 0.0, rho * (v.cosh())].into(),
    ///     rho,
    /// );
    /// assert_relative_eq!(v * rho, x.distance_from_cusp(), epsilon = 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn distance_from_cusp(&self) -> f64 {
        self.skirt * ((self.point.coordinates[N - 1]) / self.skirt).acosh()
    }

    /// Projects points on the hyperboloid onto the Poincare disk/ball.
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_manifold::{Hyperbolic, Minkowski};
    /// use hoomd_vector::Vector;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v: f64 = 1.098612;
    /// let rho: f64 = 1.0;
    /// let x = Hyperbolic::from_minkowski_coordinates(
    ///     [v.sinh(), 0.0, v.cosh()].into(),
    ///     rho,
    /// );
    /// let projection = x.to_poincare();
    /// assert_relative_eq!(
    ///     v.sinh() / (v.cosh() + 1.0),
    ///     projection[0],
    ///     epsilon = 1e-12
    /// );
    /// assert_relative_eq!(0.0, projection[1], epsilon = 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn to_poincare(&self) -> Vec<f64> {
        (0..N - 1)
            .collect::<Vec<usize>>()
            .iter()
            .map(|i| {
                self.point.coordinates[*i] / (1.0 + self.point.coordinates[N - 1] / self.skirt)
            })
            .collect::<Vec<f64>>()
    }
}

impl<const N: usize> Default for Hyperbolic<N> {
    /// Construct a default point on a hyperboloid.
    ///
    /// The default `Hyperbolic<N>` point is on the cusp of a hyperboloid with
    /// skirt width of 1 (i.e., the point $`(0, \cdots, 0, 1)`$).
    #[inline]
    fn default() -> Self {
        let mut zero = Minkowski::<N>::default();
        zero.coordinates[N - 1] = 1.0;
        Hyperbolic {
            point: zero,
            skirt: 1.0_f64,
        }
    }
}

impl Metric for Hyperbolic<3> {
    /// The distance between two [`Hyperbolic<3>`] points.
    ///
    /// Explicitly, the metric for two points $`\vec{u}`$ and $`\vec{v}`$ on a
    /// Hyperbolic with skirt width $`\rho`$ is given by
    ///
    /// ```math
    /// d_{H_2}(\vec{u}, \vec{v}) = \rho \operatorname{arccosh}\left[\frac{1}{\rho^2}(u_3v_3 - u_1v_1 - u_2v_2)\right]
    /// ```
    /// This choice of metric furnishes a representation of 2-dimensional hyperbolic
    /// space with Gaussian curvature $`K = -1/\rho^2`$.
    #[inline]
    fn distance(&self, other: &Self) -> f64 {
        assert_eq!(
            self.skirt, other.skirt,
            "points must be on the same Hyperbolic"
        );
        let last_component = self.point.coordinates[2] * other.point.coordinates[2];
        let arg = zip(
            self.point.coordinates[0..2].iter(),
            other.point.coordinates[0..2].iter(),
        )
        .fold(last_component, |product, x| product - (x.0 * x.1));
        self.skirt * (arg / (self.skirt.powi(2))).acosh()
    }

    #[inline]
    fn distance_squared(&self, other: &Self) -> f64 {
        self.distance(other).powi(2)
    }

    #[inline]
    fn n_dimensions(&self) -> usize {
        2_usize
    }
}

impl Metric for Hyperbolic<4> {
    /// The distance between two [`Hyperbolic<4>`] points.
    ///
    /// Explicitly, the metric for two points $`\vec{u}`$ and $`\vec{v}`$ on a hyperboloid with
    /// skirt width $`\rho`$ is given by
    ///
    /// ```math
    /// d_{H_3}(\vec{u}, \vec{v}) = \rho \operatorname{arccosh}\left[\frac{1}{\rho^2}(u_4v_4 - u_1v_1 - u_2v_2 - u_3v_3)\right]
    /// ```
    /// This choice of metric furnishes a representation of 3-dimensional hyperboloid
    /// space with with Gaussian curvature $`K = -1/\rho^2`$.
    #[inline]
    fn distance(&self, other: &Self) -> f64 {
        assert_eq!(
            self.skirt, other.skirt,
            "points must be on the same hyperboloid"
        );
        let last_component = self.point.coordinates[3] * other.point.coordinates[3];
        let arg = zip(
            self.point.coordinates[0..3].iter(),
            other.point.coordinates[0..3].iter(),
        )
        .fold(last_component, |product, x| product - (x.0 * x.1));
        self.skirt * (arg / (self.skirt.powi(2))).acosh()
    }

    #[inline]
    fn distance_squared(&self, other: &Self) -> f64 {
        self.distance(other).powi(2)
    }

    #[inline]
    fn n_dimensions(&self) -> usize {
        3_usize
    }
}

/// Hyperbolic rotations in Minkowski Space
///
/// Construct a [`HyperbolicRotationMatrix`] to apply SO(N-1, 1)
/// transformations to N-dimensional Minkowski vectors. For Minkowski 4-vectors,
/// [`Biquaternion`] should be used instead for numerical stability. See
/// documentation in [`HyperbolicAngle`] for details on SO(2,1) transformations
/// (i.e., two-dimensional hyperbolic space), and in [`Biquaternion`] for
/// details on SO(3,1) transformations (i.e., three-dimensional hyperbolic
/// space).
///
/// [`Biquaternion`]: crate::Biquaternion
/// [`HyperbolicAngle`]: crate::HyperbolicAngle
///
/// In two dimensional hyperbolic space:
/// ```
/// use hoomd_manifold::{
///     HyperbolicAngle, HyperbolicRotate, HyperbolicRotationMatrix, Minkowski,
/// };
/// use std::f64::consts::PI;
///
/// fn rotate_about_z(minkowski_vector: &Minkowski<3>) -> Minkowski<3> {
///     let generators = HyperbolicAngle::from((PI, 0.0_f64, 0.0_f64));
///     let rotation_matrix = HyperbolicRotationMatrix::from(generators);
///     rotation_matrix.hyperbolic_rotate(&minkowski_vector)
/// }
///
/// fn boost_in_x(minkowski_vector: &Minkowski<3>) -> Minkowski<3> {
///     let generators = HyperbolicAngle::from((0.0_f64, 0.2_f64, 0.0_f64));
///     let boost_matrix = HyperbolicRotationMatrix::from(generators);
///     boost_matrix.hyperbolic_rotate(&minkowski_vector)
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HyperbolicRotationMatrix<const N: usize> {
    /// Rows of the rotation matrix.
    pub(crate) rows: [Minkowski<N>; N],
}

impl<const N: usize> HyperbolicRotate<Minkowski<N>> for HyperbolicRotationMatrix<N> {
    type Matrix = HyperbolicRotationMatrix<N>;

    #[inline]
    /// Rotate a [`Minkowski<N>`] by a [`HyperbolicRotationMatrix`]
    ///
    /// # Examples
    ///
    /// Rotate point in 2D hyperbolic space about z-axis:
    /// ```
    /// use hoomd_manifold::{
    ///     HyperbolicAngle, HyperbolicRotate, HyperbolicRotationMatrix, Minkowski,
    /// };
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v = Minkowski::from([1.0, 0.0, 1.0]);
    /// let spatial_rotation = HyperbolicAngle::from((PI / 2.0, 0.0_f64, 0.0_f64));
    /// let matrix = HyperbolicRotationMatrix::from(spatial_rotation);
    /// let rotated = matrix.hyperbolic_rotate(&v);
    /// let c = Minkowski::from([(PI / 2.0).cos(), (PI / 2.0).sin(), 1.0]);
    /// assert_eq!(c, rotated);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Boost point in 2D hyperbolic space in x direction:
    /// ```
    /// use hoomd_manifold::{
    ///     HyperbolicAngle, HyperbolicRotate, HyperbolicRotationMatrix, Minkowski,
    /// };
    /// use num::complex::Complex;
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v = Minkowski::from([1.0, 0.0, 1.0]);
    /// let small_boost = HyperbolicAngle::from((0.0_f64, 0.1_f64, 0.0_f64));
    /// let matrix = HyperbolicRotationMatrix::from(small_boost);
    /// let rotated = matrix.hyperbolic_rotate(&v);
    /// let c = Minkowski::from([
    ///     (0.1_f64).sinh() + (0.1_f64).cosh(),
    ///     0.0,
    ///     (0.1_f64).sinh() + (0.1_f64).cosh(),
    /// ]);
    /// assert_eq!(c, rotated);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Zero angles and rapidities does nothing:
    /// ```
    /// use hoomd_manifold::{
    ///     HyperbolicAngle, HyperbolicRotate, HyperbolicRotationMatrix, Minkowski,
    /// };
    /// use num::complex::Complex;
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v = Minkowski::from([1.0, 2.0, 1.0]);
    /// let identity = HyperbolicAngle::from((0.0_f64, 0.0_f64, 0.0_f64));
    /// let matrix = HyperbolicRotationMatrix::from(identity);
    /// let rotated = matrix.hyperbolic_rotate(&v);
    /// let c = Minkowski::from([1.0, 2.0, 1.0]);
    /// assert_eq!(c, rotated);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Rotate point in 3D hyperbolic space about y axis using matrix representation:
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_manifold::{
    ///     Biquaternion, HyperbolicRotate, HyperbolicRotationMatrix, Minkowski,
    ///     UnitBiquaternion,
    /// };
    /// use num::complex::Complex;
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let q = Biquaternion::from([
    ///     Complex::new(0.0, 0.0),
    ///     Complex::new((PI / 4.0).sin(), 0.0),
    ///     Complex::new(0.0, 0.0),
    ///     Complex::new((PI / 4.0).cos(), 0.0),
    /// ]);
    /// let v = q.to_unit()?;
    /// let x = Minkowski::from([1.0, 0.0, 0.0, 1.0]);
    /// let rotation = HyperbolicRotationMatrix::from(v);
    /// let rotated = rotation.hyperbolic_rotate(&x);
    /// assert_relative_eq!(rotated, [0.0, 0.0, -1.0, 1.0].into(), epsilon = 1e-12);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Boost point in 3D hyperbolic space in x direction using biquaternion algebra:
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_manifold::{
    ///     Biquaternion, HyperbolicRotate, Minkowski, UnitBiquaternion,
    /// };
    /// use num::complex::Complex;
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let x = Minkowski::from([0.0, 0.0, 0.0, 1.0]);
    /// let q = Biquaternion::from([
    ///     Complex::new(0.0, 0.25).sin(),
    ///     Complex::new(0.0, 0.0),
    ///     Complex::new(0.0, 0.0),
    ///     Complex::new(0.0, 0.25).cos(),
    /// ]);
    /// let v = q.to_unit()?;
    /// let boosted = v.hyperbolic_rotate(&x);
    /// assert_relative_eq!(
    ///     boosted,
    ///     [(0.5_f64).sinh(), 0.0, 0.0, (0.5_f64).cosh()].into(),
    ///     epsilon = 1e-12
    /// );
    /// # Ok(())
    /// # }
    /// ```
    fn hyperbolic_rotate(&self, vector: &Minkowski<N>) -> Minkowski<N> {
        let mut coordinates = [0.0; N];

        for (result, row) in coordinates.iter_mut().zip(self.rows.iter()) {
            *result = zip(row.coordinates.iter(), vector.coordinates.iter())
                .fold(0.0, |product, x| product + x.0 * x.1);
        }

        Minkowski { coordinates }
    }
}

/// Randomly distribute points locally on a hyperboloid.
///
/// [`HyperbolicDisk`] is a uniform distribution of points within distance `r`
/// of a point on the 2-dimensional hyperboloid with a given skirt width.
///
/// # Example
///
/// ```
/// use hoomd_manifold::{
///     Hyperbolic, HyperbolicAngle, HyperbolicDisk, HyperbolicRotate,
///     HyperbolicRotationMatrix, Minkowski,
/// };
/// use hoomd_vector::Metric;
/// use rand::{Rng, SeedableRng, distr::Distribution, rngs::StdRng};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut rng = StdRng::seed_from_u64(12);
///
/// let rho: f64 = 1.0;
/// let v: HyperbolicAngle = rng.random();
/// let matrix = HyperbolicRotationMatrix::from(v);
/// let origin = Minkowski::from([0.0, 0.0, rho]);
/// let random_point = Hyperbolic::from_minkowski_coordinates(
///     matrix.hyperbolic_rotate(&origin),
///     rho,
/// );
///
/// let r = 0.1;
/// let mut rng_2 = StdRng::seed_from_u64(239);
/// let disk = HyperbolicDisk {
///     disk_radius: r.try_into()?,
///     point: random_point,
/// };
/// let transformed_random_point: Hyperbolic<3> = disk.sample(&mut rng_2);
///
/// assert!(r > random_point.distance(&transformed_random_point));
///
/// # Ok(())
/// # }
/// ```
pub struct HyperbolicDisk {
    /// Max distance away from point.
    pub disk_radius: PositiveReal,
    /// The center of the disk.
    pub point: Hyperbolic<3>,
}

impl Distribution<Hyperbolic<3>> for HyperbolicDisk {
    /// Sample a random point in the hyperbolic disk.
    ///
    /// The implementation translates Minkowski 3-vector `point` along
    /// the Hyperbolic by maximum distance of `disk_radius`. Note that because SO(2,1) is
    /// non-Abelian, the point must be transformed to the cusp before a trial
    /// move is applied (and then the point is transformed back). This ensures
    /// that the max distance translated by the trial move does not exceed `disk_radius`.
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Hyperbolic<3> {
        let rho = self.point.skirt;
        let max_boost = (self.disk_radius.get()) / rho;
        let point = self.point;
        let eta = (point.point.coordinates[2] / rho).acosh();
        let phi = point.point.coordinates[1].atan2(point.point.coordinates[0]);
        let trial_boost = Uniform::new(0.0, 1.0).expect("r is positive and real");
        let trial_rotation =
            Uniform::new(-PI, PI).expect("hard-coded distribution should be valid");
        let theta = trial_rotation.sample(rng);
        let v1: f64 = trial_boost.sample(rng);
        let v = v1.sqrt() * max_boost;
        let trial_coords = [
            rho * v.sinh() * theta.cos(),
            rho * v.sinh() * theta.sin(),
            rho * v.cosh(),
        ];
        let transformed_point = Minkowski::from([
            trial_coords[0] * (eta.cosh()) * (phi.cos()) - trial_coords[1] * (phi.sin())
                + trial_coords[2] * (eta.sinh()) * (phi.cos()),
            trial_coords[0] * (eta.cosh()) * (phi.sin())
                + trial_coords[1] * (phi.cos())
                + trial_coords[2] * (eta.sinh()) * (phi.sin()),
            trial_coords[0] * (eta.sinh()) + trial_coords[2] * (eta.cosh()),
        ]);
        Hyperbolic::from_minkowski_coordinates(transformed_point, rho)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use paste::paste;
    use rand::{SeedableRng, rngs::StdRng};

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
    fn generate_vector_pair<const N: usize>() -> (Minkowski<N>, Minkowski<N>) {
        (
            Minkowski::try_from(0..N).unwrap(),
            Minkowski::try_from(N..N * 2).unwrap(),
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

        assert_eq!(c, Minkowski::try_from(addition_answer).unwrap());
    }
    parameterize_vector_length!(add_explicit, [2, 3, 4, 8, 16, 32]);

    fn add_operator<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();
        let c = a + b;

        let addition_answer: Vec<f64> = (0..(2 * N))
            .step_by(2)
            .map(|x| (x + N) as f64)
            .collect::<Vec<_>>();

        assert_eq!(c, Minkowski::try_from(addition_answer).unwrap());
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

        assert_eq!(c, Minkowski::try_from(addition_answer).unwrap());
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

        assert_eq!(c, Minkowski::try_from(multiplication_answer).unwrap());
    }
    parameterize_vector_length!(mul_operator, [2, 3, 4, 8, 16, 32]);

    fn div_operator<const N: usize>() {
        let (a, _) = generate_vector_pair::<N>();
        let b = 12.0;
        let c = a / b;

        let division_answer: Vec<f64> = (0..N).map(|x| (x as f64) / b).collect::<Vec<_>>();

        assert_eq!(c, Minkowski::try_from(division_answer).unwrap());
    }
    parameterize_vector_length!(div_operator, [2, 3, 4, 8, 16, 32]);

    fn div_assign<const N: usize>() {
        let (mut a, _) = generate_vector_pair::<N>();
        let b = 12.0;
        a /= b;

        let division_answer: Vec<f64> = (0..N).map(|x| (x as f64) / b).collect::<Vec<_>>();

        assert_eq!(a, Minkowski::try_from(division_answer).unwrap());
    }

    parameterize_vector_length!(div_assign, [2, 3, 4, 8, 16, 32]);

    fn compute_add_ref_ref<const N: usize>(a: &Minkowski<N>, b: &Minkowski<N>) -> Minkowski<N> {
        *a + *b
    }

    fn compute_add_ref_type<const N: usize>(a: &Minkowski<N>, b: Minkowski<N>) -> Minkowski<N> {
        *a + b
    }

    fn compute_add_type_ref<const N: usize>(a: Minkowski<N>, b: &Minkowski<N>) -> Minkowski<N> {
        a + *b
    }

    fn add_with_refs<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();

        let addition_answer = Minkowski::try_from(
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

    #[test]
    fn display() {
        let a = Minkowski::from([1.67, -2.125, 42.01]);
        let s = format!("{a}");
        assert_eq!(s, "[1.67, -2.125, 42.01]");

        let a = Minkowski::from([10.0, 20.0, 30.0, 40.0]);
        let s = format!("{a}");
        assert_eq!(s, "[10, 20, 30, 40]");
    }

    #[test]
    fn from_2_tuple() {
        let a = Minkowski::from((13.0, 0.125));
        assert_eq!(a.coordinates, [13.0, 0.125]);
    }

    #[test]
    fn from_3_tuple() {
        let a = Minkowski::from((-0.5, 2.0, 18.125));
        assert_eq!(a.coordinates, [-0.5, 2.0, 18.125]);
    }

    fn from_vec<const N: usize>() {
        let mut vec = Vec::with_capacity(N);

        assert_eq!(
            Minkowski::<N>::try_from(vec.clone()),
            Err(Error::InvalidVectorLength)
        );

        for i in 0..N {
            vec.push(i as f64 * 0.5);
        }
        let a = Minkowski::<N>::try_from(vec.clone()).unwrap();

        assert_eq!(vec, Vec::from(a.coordinates));

        vec.push(1.0);
        assert_eq!(
            Minkowski::<N>::try_from(vec.clone()),
            Err(Error::InvalidVectorLength)
        );
    }
    parameterize_vector_length!(from_vec, [2, 3, 4, 8, 16, 32]);

    fn random_in_range<const N: usize>() {
        // Loosely verify we are drawing from the correct distribution
        let mut rng = StdRng::seed_from_u64(1);
        let a: Minkowski<N> = rng.random();

        assert!(a.coordinates.iter().all(|&x| -1.0 < x && x < 1.0));

        // This test will fail ~1e-3008 percent of the time - it's probably fine
        if N == 10_000 {
            assert!(a.coordinates.iter().any(|&x| x < 0.0));
        }
    }

    parameterize_vector_length!(random_in_range, [2, 3, 4, 8, 16, 32, 10_000]);

    /// Generate a pair of points in 2-dimensional hyperbolic space
    fn generate_h2_pair(skirt: f64) -> (Hyperbolic<3>, Hyperbolic<3>) {
        (
            Hyperbolic::<3>::from_polar_coordinates(3.2, 0.1, skirt),
            Hyperbolic::<3>::from_polar_coordinates(4.0, 3.1, skirt),
        )
    }
    /// Generate a pair of points in 3-dimensional hyperbolic space
    fn generate_h3_pair(skirt: f64) -> (Hyperbolic<4>, Hyperbolic<4>) {
        (
            Hyperbolic::<4>::from_polar_coordinates(3.5, 0.4, 0.5, skirt),
            Hyperbolic::<4>::from_polar_coordinates(4.2, 2.7, 0.1, skirt),
        )
    }

    #[test]
    #[expect(clippy::many_single_char_names, reason = "test variables")]
    fn hyperbolic_distance() {
        let (a, b) = generate_h2_pair(1.0);
        let ab_distance = a.distance(&b);
        let ab_numeric_answer = 7.194_993_724_795_472;
        assert_relative_eq!(ab_distance, ab_numeric_answer, epsilon = 1e-12);

        let (c, d) = generate_h3_pair(1.0);
        let cd_distance = c.distance(&d);
        let cd_numeric_answer = 7.525_514_513_583_905;
        assert_relative_eq!(cd_distance, cd_numeric_answer, epsilon = 1e-12);

        let (e, f) = generate_h2_pair(0.1);
        let ef_distance = e.distance(&f);
        let ef_numeric_answer = 0.719_499_372_479_547_2;
        assert_relative_eq!(ef_distance, ef_numeric_answer, epsilon = 1e-11);

        let (g, h) = generate_h3_pair(0.1);
        let gh_distance = g.distance(&h);
        let gh_numeric_answer = 0.752_551_451_358_390_5;
        assert_relative_eq!(gh_distance, gh_numeric_answer, epsilon = 1e-11);
    }

    #[test]
    fn poincare_projection() {
        let a = Hyperbolic::<3>::from_polar_coordinates(1.5, 1.5, 1.0);
        let a_poincare = a.to_poincare();
        let a_numeric_poincare = [0.044_928_659_534_049_77, 0.633_557_895_753_136_3];
        assert_relative_eq![a_poincare[0], a_numeric_poincare[0], epsilon = 1e-12];
        assert_relative_eq![a_poincare[1], a_numeric_poincare[1], epsilon = 1e-12];

        let b = Hyperbolic::<3>::from_polar_coordinates(0.5, 4.2, 1.0);
        let b_poincare = b.to_poincare();
        let b_numeric_poincare = [-0.120_074_024_591_707_93, -0.213_465_172_363_015_63];
        assert_relative_eq![b_poincare[0], b_numeric_poincare[0], epsilon = 1e-12];
        assert_relative_eq![b_poincare[1], b_numeric_poincare[1], epsilon = 1e-12];

        let c = Hyperbolic::<3>::from_polar_coordinates(1.5, 1.5, 10.0);
        let c_poincare = c.to_poincare();
        let c_numeric_poincare = [0.449_286_595_340_497_7, 6.335_578_957_531_363];
        assert_relative_eq![c_poincare[0], c_numeric_poincare[0], epsilon = 1e-12];
        assert_relative_eq![c_poincare[1], c_numeric_poincare[1], epsilon = 1e-12];

        let d = Hyperbolic::<3>::from_polar_coordinates(0.5, 4.2, 10.0);
        let d_poincare = d.to_poincare();
        let d_numeric_poincare = [-1.200_740_245_917_079_3, -2.134_651_723_630_156];
        assert_relative_eq![d_poincare[0], d_numeric_poincare[0], epsilon = 1e-12];
        assert_relative_eq![d_poincare[1], d_numeric_poincare[1], epsilon = 1e-12];
    }

    #[test]
    fn specific_distances() {
        // Distance to the cusp
        let a = Hyperbolic::<3>::from_polar_coordinates(1.2, 3.2, 1.0);
        let a_cusp_distance = a.distance_from_cusp();
        let a_cusp_distance_numeric = 1.2;
        assert_relative_eq!(a_cusp_distance, a_cusp_distance_numeric, epsilon = 1e-12);

        let b = Hyperbolic::<3>::from_polar_coordinates(2.0, 1.6, 5.0);
        let b_cusp_distance = b.distance_from_cusp();
        let b_cusp_distance_numeric = 10.0;
        assert_relative_eq!(b_cusp_distance, b_cusp_distance_numeric, epsilon = 1e-12);

        let c = Hyperbolic::<4>::from_polar_coordinates(1.2, 3.2, 1.2, 1.0);
        let c_cusp_distance = c.distance_from_cusp();
        let c_cusp_distance_numeric = 1.2;
        assert_relative_eq!(c_cusp_distance, c_cusp_distance_numeric, epsilon = 1e-12);

        let d = Hyperbolic::<4>::from_polar_coordinates(2.0, 1.6, 0.8, 5.0);
        let d_cusp_distance = d.distance_from_cusp();
        let d_cusp_distance_numeric = 10.0;
        assert_relative_eq!(d_cusp_distance, d_cusp_distance_numeric, epsilon = 1e-12);
    }

    #[test]
    fn random_hyperbolic() {
        // Generate ten random points on the Hyperbolic
        let mut rng = StdRng::seed_from_u64(42);
        let d = 0.1;
        let origin = Minkowski::from([0.0, 0.0, 1.0]);
        for _n in 0..10 {
            let disk = HyperbolicDisk {
                disk_radius: d.try_into().expect("hard-coded positive number"),
                point: Hyperbolic::<3>::from_minkowski_coordinates(origin, 1.0),
            };
            let random_point: Hyperbolic<3> = disk.sample(&mut rng);

            // check that points remain on Hyperbolic
            let rho = -random_point
                .point
                .distance_squared(&Minkowski::<3>::default());
            assert_relative_eq!(rho, 1.0, epsilon = 1e-12);

            // check that points are within distance d of cusp
            let distance = random_point.distance_from_cusp();
            assert!(d > distance);
        }
    }
}

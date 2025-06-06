// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement vector types in Minkowski space.
 */

use std::array;
use std::fmt;
use std::iter::{Sum, zip};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use rand::Rng;
use rand::distr::{Distribution, StandardUniform, Uniform};
use hoomd_vector::{Rotate, Vector};

use crate::Error;

/** Description of Minkowski space, examples of usage
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Minkowski<const N: usize> {
    // The vector's coordinates
    pub coordinates: [f64; N],
}

impl<const N: usize> Default for Minkowski<N> {
    /** Create a 0 vector in Minkowski space.

    # Example
    ```
    use hoomd_manifold::Minkowski;

    let v = Minkowski::<3>::default();
    assert_eq!(v, [0.0; 3].into())
    ```
    */
    #[inline]
    fn default() -> Self {
        Minkowski::from([0.0; N])
    }
}

impl<const N: usize> From<[f64; N]> for Minkowski<N> {
    /** Create a vector in Minkowski space with the given coordinates. Note that
    the first component has a (-) signature, while the following coordinates have
     (+) signatures in the metric.

    # Example
    ```
    use hoomd_manifold::Minkowski;

    let v = Minkowski::from([1.0, 2.0]);
    ```
    */
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

impl<const N: usize> TryFrom<Vec<f64>> for Minkowski<N> {
    type Error = Error;

    /** Create a vector in Minkowski with coordinates given by a [`Vec<f64>`]

    # Example
    ```
    use hoomd_manifold::Minkowski;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v = Minkowski::<3>::try_from(vec![5.0, 4.0, 3.0])?;
    assert_eq!(v, [5.0, 4.0, 3.0].into());
    # Ok(())
    # }
    ```
    <div class="warning">

    Use `Minkowski::From<[f64; N]>` in performance critical code.

    </div>
    */
    #[inline]
    fn try_from(value: Vec<f64>) -> Result<Self, Self::Error> {
        let coordinates = value.try_into().map_err(|_| Error::InvalidVectorLength)?;
        Ok(Self { coordinates })
    }
}

impl<const N: usize> TryFrom<std::ops::Range<usize>> for Minkowski<N> {
    type Error = Error;

    /** Create a vector in Minkowski space with coordinates given by a range.

    # Example
    ```
    use hoomd_manifold::Minkowski;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v = Minkowski::<3>::try_from(1..4)?;
    assert_eq!(v, [1.0, 2.0, 3.0].into());
    # Ok(())
    # }
    ```

    <div class="warning">

    Use `Minkowski::From<[f64; N]>` in performance critical code.

    </div>
    */
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

impl<const N: usize> Vector for Minkowski<N> {
    /** Computes the squared distance between two points in Minkowski space with 
    "mostly pluses" metric signature. 
    ```math
    d^2_M(\vec{x},\vec{y}) = -(x_0-y_0)^2 + \sum_{i=1}^{N-1} (x_i - y_i)^2
    ```

    # Example
    ```
    use hoomd_manifold::Minkowski;
    use hoomd_vector::Vector;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Minkowski::from([0.0, 1.0, 2.0]);
    let y = Minkowski::from([1.0, 0.0, 0.0]);
    assert_eq!(4.0, x.distance_squared(&y));
    # Ok(())
    # }
    */
    #[inline]
    fn distance_squared(&self, other: &Self) -> f64 {
        let zeroth_component = -1.0 * (self.coordinates[0] - other.coordinates[0]).powi(2);
        zip(self.coordinates[1..N].iter(), other.coordinates[1..N].iter())
            .fold(zeroth_component, |product, x| product + (x.0 - x.1).powi(2))
    }
}

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
    /** Get the value of the vector at coordinate i.

    # Example
    ```
    use hoomd_manifold::Minkowski;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v = Minkowski::<3>::try_from(4..7)?;
    assert_eq!((v[0], v[1], v[2]), (4.0, 5.0, 6.0));
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn index(&self, index: T) -> &Self::Output {
        &self.coordinates[index]
    }
}

impl<const N: usize, T> IndexMut<T> for Minkowski<N>
where
    T: Into<usize> + std::slice::SliceIndex<[f64], Output = f64>,
{
    /** Get a mutable reference to the value of the vector at coordinate i.

    # Example
    ```
    use hoomd_manifold::Minkowski;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut v = Minkowski::<3>::try_from(4..7)?;
    assert_eq!((v[0], v[1], v[2]), (4.0, 5.0, 6.0));
    v[0] += 1.0;
    assert_eq!(v[0], 5.0);
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn index_mut(&mut self, index: T) -> &mut Self::Output {
        &mut self.coordinates[index]
    }
}

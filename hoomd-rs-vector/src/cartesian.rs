// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::array;
use std::fmt;
use std::iter::zip;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{Dimension, Error, Vector, Cross};

/** A Cartesian vector represented by an array of `N` `f64` coordinates.

`CartesianVector` is the canonical implementation of [`Vector`].

Create a vector with an array of coordinates:
```
# use hoomd_rs_vector::CartesianVector;
let v = CartesianVector::from([1.0, 2.0, 3.0, 4.0, 5.0]);
```

2D and 3D vectors can also be initialized from tuples:
```
# use hoomd_rs_vector::CartesianVector;
let a = CartesianVector::from((1.0, 2.0, 3.0));
let b = CartesianVector::from((4.0, 5.0));
```

Use vector math operations when you can:
```
# use hoomd_rs_vector::{CartesianVector, Vector};
let a = CartesianVector::from([1.0, 2.0]);
let b = CartesianVector::from([4.0, 8.0]);
let c = (a + b).dot(&a);
```

Access the coordinates directly when needed:
```
# use hoomd_rs_vector::CartesianVector;
# let a = CartesianVector::from((1.0, 2.0));
let b = CartesianVector::from((a.coordinates[1], 0.0));
```
*/
#[allow(clippy::module_name_repetitions)] // cartesian.rs is a private implementation module.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CartesianVector<const N: usize> {
    /// The vector's coordinates.
    pub coordinates: [f64; N],
}

impl<const N: usize> Default for CartesianVector<N> {
    /** Create a 0 vector.

    ```
    # use hoomd_rs_vector::CartesianVector;
    let v = CartesianVector::<3>::default();
    assert_eq!(v, [0.0; 3].into())
    ```
    */
    #[inline]
    fn default() -> Self {
        CartesianVector::from([0.0; N])
    }
}

impl<const N: usize> From<[f64; N]> for CartesianVector<N> {
    /** Create a Cartesian vector with the given coordinates.

    ```
    # use hoomd_rs_vector::CartesianVector;
    let v = CartesianVector::from([4.0, 3.0]);
    ```
    */
    #[inline]
    fn from(coordinates: [f64; N]) -> Self {
        Self { coordinates }
    }
}

impl From<(f64, f64)> for CartesianVector<2> {
    #[inline]
    fn from(coordinates: (f64, f64)) -> Self {
        Self {
            coordinates: coordinates.into(),
        }
    }
}

impl From<(f64, f64, f64)> for CartesianVector<3> {
    #[inline]
    fn from(coordinates: (f64, f64, f64)) -> Self {
        Self {
            coordinates: coordinates.into(),
        }
    }
}

impl<const N: usize> TryFrom<Vec<f64>> for CartesianVector<N> {
    type Error = Error;

    /** Create a Cartesian vector with coordinates given by a [`Vec<f64>`]

    ```
    # use hoomd_rs_vector::CartesianVector;
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v = CartesianVector::<3>::try_from(vec![3.0, 4.0, 5.0])?;
    assert_eq!(v, [3.0, 4.0, 5.0].into());
    # Ok(())
    # }
    ```

    > Note: Use `CartesianVector::From<[f64; N]>` in performance critical code.
    */
    #[inline]
    fn try_from(value: Vec<f64>) -> Result<Self, Self::Error> {
        let coordinates = value.try_into().map_err(|_| Error::InvalidVectorLength)?;
        Ok(Self { coordinates })
    }
}

impl<const N: usize> TryFrom<std::ops::Range<usize>> for CartesianVector<N> {
    type Error = Error;

    /** Create a Cartesian vector with coordinates given by a range.

    ```
    # use hoomd_rs_vector::CartesianVector;
    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v = CartesianVector::<3>::try_from(3..6)?;
    assert_eq!(v, [3.0, 4.0, 5.0].into());
    # Ok(())
    # }
    ```

    > Note: Use `CartesianVector::From<[f64; N]>` in performance critical code.
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

impl<const N: usize> Vector for CartesianVector<N> {
    type Dimension = Dimension<N>;

    #[inline]
    fn dot(&self, rhs: &Self) -> f64 {
        zip(self.coordinates.iter(), rhs.coordinates.iter())
            .fold(0.0, |product, x| product + x.0 * x.1)
    }
}

impl<const N: usize> Add for CartesianVector<N> {
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

impl<const N: usize> AddAssign for CartesianVector<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for (result, a) in self.coordinates.iter_mut().zip(rhs.coordinates.iter()) {
            *result += a;
        }
    }
}

impl<const N: usize> Div<f64> for CartesianVector<N> {
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

impl<const N: usize> DivAssign<f64> for CartesianVector<N> {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        for result in &mut self.coordinates {
            *result /= rhs;
        }
    }
}

impl<const N: usize> Mul<f64> for CartesianVector<N> {
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

impl<const N: usize> MulAssign<f64> for CartesianVector<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        for result in &mut self.coordinates {
            *result *= rhs;
        }
    }
}

impl<const N: usize> Sub for CartesianVector<N> {
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

impl<const N: usize> SubAssign for CartesianVector<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for (result, a) in self.coordinates.iter_mut().zip(rhs.coordinates) {
            *result -= a;
        }
    }
}

impl<const N: usize> fmt::Display for CartesianVector<N> {
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

impl<const N: usize> Neg for CartesianVector<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        let mut result = self;
        result.coordinates.iter_mut().for_each(|x| *x = -*x);
        result
    }
}

impl Cross for CartesianVector<3> {
    fn cross(&self, rhs: &Self) -> Self {
        CartesianVector::from((
            self.coordinates[1] * rhs.coordinates[2] - self.coordinates[2] * rhs.coordinates[1],
            self.coordinates[2] * rhs.coordinates[0] - self.coordinates[0] * rhs.coordinates[2],
            self.coordinates[0] * rhs.coordinates[1] - self.coordinates[1] * rhs.coordinates[0]
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paste::paste;

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
    fn generate_vector_pair<const N: usize>() -> (CartesianVector<N>, CartesianVector<N>) {
        (
            CartesianVector::try_from(0..N).unwrap(),
            CartesianVector::try_from(N..N * 2).unwrap(),
        )
    }

    fn add_explicit<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();
        let c = a.add(b);

        let addition_answer: Vec<f64> = (0..(2 * N))
            .step_by(2)
            .map(|x| (x + N) as f64)
            .collect::<Vec<_>>();

        assert_eq!(c, CartesianVector::try_from(addition_answer).unwrap());
    }
    parameterize_vector_length!(add_explicit, [2, 3, 4, 8, 16, 32]);

    fn add_operator<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();
        let c = a + b;

        let addition_answer: Vec<f64> = (0..(2 * N))
            .step_by(2)
            .map(|x| (x + N) as f64)
            .collect::<Vec<_>>();

        assert_eq!(c, CartesianVector::try_from(addition_answer).unwrap());
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

        assert_eq!(c, CartesianVector::try_from(addition_answer).unwrap());
    }

    parameterize_vector_length!(add_assign, [2, 3, 4, 8, 16, 32]);

    fn compute_add_ref_ref<const N: usize>(
        a: &CartesianVector<N>,
        b: &CartesianVector<N>,
    ) -> CartesianVector<N> {
        *a + *b
    }

    fn compute_add_ref_type<const N: usize>(
        a: &CartesianVector<N>,
        b: CartesianVector<N>,
    ) -> CartesianVector<N> {
        *a + b
    }

    fn compute_add_type_ref<const N: usize>(
        a: CartesianVector<N>,
        b: &CartesianVector<N>,
    ) -> CartesianVector<N> {
        a + *b
    }

    fn add_with_refs<const N: usize>() {
        let (a, b) = generate_vector_pair::<N>();

        let addition_answer = CartesianVector::try_from(
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
        let a: CartesianVector<N> = CartesianVector::try_from(0..N).unwrap();
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
        let cross_ans = CartesianVector::from((-3.0, 6.0, -3.0));
        
        assert_eq!(c, cross_ans);
    }

    #[test]
    fn display() {
        let a = CartesianVector::from([1.5, 2.125, -3.875]);
        let s = format!("{a}");
        assert_eq!(s, "[1.5, 2.125, -3.875]");

        let a = CartesianVector::from([10.0, 20.0, 30.0, 40.0]);
        let s = format!("{a}");
        assert_eq!(s, "[10, 20, 30, 40]");
    }
}

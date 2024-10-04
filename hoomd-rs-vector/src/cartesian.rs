// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd_rs, released under the BSD 3-Clause License.

use std::fmt;
use std::iter::zip;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::Vector;

/// A Cartesian vector with dimension `N`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CartesianVector<const N: usize> {
    coordinates: [f64; N],
}

impl<const N: usize> Default for CartesianVector<N> {
    fn default() -> Self {
        CartesianVector::from([0.0; N])
    }
}

impl<const N: usize> From<[f64; N]> for CartesianVector<N> {
    #[inline]
    fn from(coordinates: [f64; N]) -> Self {
        Self { coordinates }
    }
}

impl<const N: usize> From<std::ops::Range<usize>> for CartesianVector<N> {
    #[inline]
    fn from(coordinates: std::ops::Range<usize>) -> Self {
        let arr: [f64; N] = coordinates
            .map(|x| x as f64)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        CartesianVector::from(arr)
    }
}

impl<const N: usize> Vector for CartesianVector<N> {
    const DIMENSION: usize = N;

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
            *result = a + b
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
        for result in self.coordinates.iter_mut() {
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
        for result in self.coordinates.iter_mut() {
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
            *result = a - b
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.coordinates
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use rstest::*;

    #[test]
    fn add_explicit() {
        let a = CartesianVector::from([1.0, 2.0, 3.0]);
        let b = CartesianVector::from([4.0, 5.0, 6.0]);
        let c = a.add(b);
        assert_eq!(c, [5.0, 7.0, 9.0].into());
    }

    #[test]
    fn add_operator() {
        let a = CartesianVector::from([1.0, 2.0, 3.0]);
        let b = CartesianVector::from([4.0, 5.0, 6.0]);
        let c = a + b;
        assert_eq!(c, [5.0, 7.0, 9.0].into());
    }

    #[test]
    fn add_assign() {
        let a = CartesianVector::from([1.0, 2.0, 3.0]);
        let b = CartesianVector::from([4.0, 5.0, 6.0]);
        let mut c = a;
        c += b;
        assert_eq!(c, [5.0, 7.0, 9.0].into());
    }

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

    #[test]
    fn add_with_refs() {
        let a = CartesianVector::from([1.0, 2.0, 3.0]);
        let b = CartesianVector::from([4.0, 5.0, 6.0]);
        let c = compute_add_ref_ref(&a, &b);
        assert_eq!(c, [5.0, 7.0, 9.0].into());

        let c = compute_add_ref_type(&a, b);
        assert_eq!(c, [5.0, 7.0, 9.0].into());

        let c = compute_add_type_ref(a, &b);
        assert_eq!(c, [5.0, 7.0, 9.0].into());
    }

    #[test]
    fn dot() {
        let a = CartesianVector::from([1.0, 2.0, 3.0]);
        let b = CartesianVector::from([4.0, 5.0, 6.0]);
        let c = a.dot(&b);
        assert_eq!(c, 32.0);
    }

    #[test]
    fn display() {
        let a = CartesianVector::from([1.15, 2.0, 3.999999999]);
        println!("Test array: {a}, printed.");
        assert_eq!(a.to_string(), "[1.15, 2, 3.999999999]");
    }

    use paste::paste;

    macro_rules! parameterized_tests {
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

    fn this_is_my_test_name<const N: usize>() {
        println!("Worked! {}", N);
        println!("{}", CartesianVector::<N>::from(0..N));
    }
    fn this_is_another_test_name<const N: usize>() {
        println!("Worked! {}", N);
        println!("{}", CartesianVector::<N>::from(0..N));
    }

    parameterized_tests!(this_is_my_test_name, [2, 3]);
    parameterized_tests!(this_is_another_test_name, [2, 3]);
}

// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd_rs, released under the BSD 3-Clause License.

use std::fmt::Display;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::Vector;

/// A Cartesian vector with dimension `N` and `Real`-valued coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CartesianVector<const N: usize> {
    coordinates: [f64; N],
}

impl<const N: usize> From<[f64; N]> for CartesianVector<N> {
    #[inline]
    fn from(coordinates: [f64; N]) -> Self {
        CartesianVector { coordinates }
    }
}

impl<const N: usize> Vector for CartesianVector<N> {
    const DIMENSION: usize = N;

    #[inline]
    fn dot(&self, rhs: &Self) -> f64 {
        let mut product = 0.0;
        for i in 0..N {
            product += self.coordinates[i] * rhs.coordinates[i];
        }
        product
    }
}

impl<const N: usize> Add for CartesianVector<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = self.coordinates[i] + rhs.coordinates[i];
        }
        CartesianVector {
            coordinates: result,
        }
    }
}

impl<const N: usize> AddAssign for CartesianVector<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coordinates[i] += rhs.coordinates[i];
        }
    }
}

impl<const N: usize> Div<f64> for CartesianVector<N> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self {
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = self.coordinates[i] / rhs;
        }
        CartesianVector {
            coordinates: result,
        }
    }
}

impl<const N: usize> DivAssign<f64> for CartesianVector<N> {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        for i in 0..N {
            self.coordinates[i] /= rhs;
        }
    }
}

impl<const N: usize> Mul<f64> for CartesianVector<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = self.coordinates[i] * rhs;
        }
        CartesianVector {
            coordinates: result,
        }
    }
}

impl<const N: usize> MulAssign<f64> for CartesianVector<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        for i in 0..N {
            self.coordinates[i] *= rhs;
        }
    }
}

impl<const N: usize> Sub for CartesianVector<N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut result = [0.0; N];
        for i in 0..N {
            result[i] = self.coordinates[i] - rhs.coordinates[i];
        }
        CartesianVector {
            coordinates: result,
        }
    }
}

impl<const N: usize> SubAssign for CartesianVector<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coordinates[i] -= rhs.coordinates[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn compute_add_ref_ref<const N: usize>(
        a: &CartesianVector<N>,
        b: &CartesianVector<N>,
    ) -> CartesianVector<N> {
        a.clone() + b.clone()
    }

    fn compute_add_ref_type<const N: usize>(
        a: &CartesianVector<N>,
        b: CartesianVector<N>,
    ) -> CartesianVector<N> {
        a.clone() + b
    }

    fn compute_add_type_ref<const N: usize>(
        a: CartesianVector<N>,
        b: &CartesianVector<N>,
    ) -> CartesianVector<N> {
        a + b.clone()
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
}

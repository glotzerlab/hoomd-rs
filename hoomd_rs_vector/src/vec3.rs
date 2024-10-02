// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd_rs, released under the BSD 3-Clause License.

use std::fmt::Display;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::Vector;

/// A Cartesian vector with dimension `N` and `Real`-valued coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CartesianVector3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Default for CartesianVector3 {
    fn default() -> Self {
        CartesianVector3::from([0.0; 3])
    }
}

impl From<[f64; 3]> for CartesianVector3 {
    #[inline]
    fn from(coordinates: [f64; 3]) -> Self {
        Self { x: coordinates[0],
                          y: coordinates[1],
                          z: coordinates[2], }
    }
}

impl Vector for CartesianVector3 {
    const DIMENSION: usize = 3;

    #[inline]
    fn dot(&self, rhs: &Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

impl Add for CartesianVector3 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        CartesianVector3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl AddAssign for CartesianVector3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Div<f64> for CartesianVector3 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self {
        CartesianVector3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl DivAssign<f64> for CartesianVector3 {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

impl Mul<f64> for CartesianVector3 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl MulAssign<f64> for CartesianVector3 {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl Sub for CartesianVector3 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl SubAssign for CartesianVector3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_explicit() {
        let a = CartesianVector3::from([1.0, 2.0, 3.0]);
        let b = CartesianVector3::from([4.0, 5.0, 6.0]);
        let c = a.add(b);
        assert_eq!(c, [5.0, 7.0, 9.0].into());
    }

    #[test]
    fn add_operator() {
        let a = CartesianVector3::from([1.0, 2.0, 3.0]);
        let b = CartesianVector3::from([4.0, 5.0, 6.0]);
        let c = a + b;
        assert_eq!(c, [5.0, 7.0, 9.0].into());
    }

    fn compute_add_ref_ref(
        a: &CartesianVector3,
        b: &CartesianVector3,
    ) -> CartesianVector3 {
        a.clone() + b.clone()
    }

    fn compute_add_ref_type(
        a: &CartesianVector3,
        b: CartesianVector3,
    ) -> CartesianVector3 {
        a.clone() + b
    }

    fn compute_add_type_ref(
        a: CartesianVector3,
        b: &CartesianVector3,
    ) -> CartesianVector3 {
        a + b.clone()
    }

    #[test]
    fn add_with_refs() {
        let a = CartesianVector3::from([1.0, 2.0, 3.0]);
        let b = CartesianVector3::from([4.0, 5.0, 6.0]);
        let c = compute_add_ref_ref(&a, &b);
        assert_eq!(c, [5.0, 7.0, 9.0].into());

        let c = compute_add_ref_type(&a, b);
        assert_eq!(c, [5.0, 7.0, 9.0].into());

        let c = compute_add_type_ref(a, &b);
        assert_eq!(c, [5.0, 7.0, 9.0].into());
    }
}

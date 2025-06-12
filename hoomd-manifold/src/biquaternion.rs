// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Biquaternion`] and SO(3,1) representation. 
 */

use std::array;
use num::complex::Complex;
use std::fmt;
use std::iter::zip;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};
use libm::acosh;
use std::f64::consts::PI;
use hoomd_vector::{Vector, Angle, Rotate, Rotation, InnerProduct};

use crate::{Error,Minkowski};

/** Documentation for biquaternions
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Biquaternion {
    pub components: [Complex<f64>; 4]
}

impl Biquaternion {
    /** the Hamiltonian conjugate of a biquaternion

    # Example
    ```
    use hoomd_manifold::Biquaternion;
    use num::complex::Complex;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = Biquaternion::from([Complex::new(1.0,0.0),
                                Complex::new(0.0,1.0),
                                Complex::new(1.0,0.0),
                                Complex::new(1.0,0.0)]);
    let p = Biquaternion::from([Complex::new(1.0,0.0),
                                Complex::new(0.0,-1.0),
                                Complex::new(-1.0,0.0),
                                Complex::new(-1.0,0.0)]);
    assert_eq!(p, q.bar());
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn bar(&self) -> Self {
        Biquaternion::from([self.components[0], (self.components[1]).scale(-1.0),
        (self.components[2]).scale(-1.0),(self.components[3]).scale(-1.0)])
    }
    /** the complex conjugate of a biquaternion

    # Example
    ```
    use hoomd_manifold::Biquaternion;
    use num::complex::Complex;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = Biquaternion::from([Complex::new(1.0,0.0),
                                Complex::new(0.0,1.0),
                                Complex::new(1.0,0.0),
                                Complex::new(1.0,2.0)]);
    let p = Biquaternion::from([Complex::new(1.0,0.0),
                                Complex::new(0.0,-1.0),
                                Complex::new(1.0,0.0),
                                Complex::new(1.0,-2.0)]);
    assert_eq!(p, q.conj());
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn conj(&self) -> Self {
        Biquaternion::from([(self.components[0]).conj(), (self.components[1]).conj(),
        (self.components[2]).conj(),(self.components[3]).conj()])
    }
    /** the norm of a biquaternion

    # Example
    ```
    use hoomd_manifold::Biquaternion;
    use num::complex::Complex;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = Biquaternion::from([Complex::new(1.0,0.0),
                                Complex::new(0.0,1.0),
                                Complex::new(1.0,0.0),
                                Complex::new(1.0,0.0)]);
    assert_eq!(2.0, q.norm());
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn norm(&self) -> f64 {
        zip(self.components.iter(), self.components.iter())
            .fold(0.0, |product, x| product + (x.0 * x.1).re)
    }
    /** the biquaternion product of a biquaternion

    # Example
    ```
    use hoomd_manifold::Biquaternion;
    use num::complex::Complex;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = Biquaternion::from([Complex::new(1.0,0.0),
                                Complex::new(0.0,1.0),
                                Complex::new(1.0,0.0),
                                Complex::new(1.0,0.0)]);
    assert_eq!(2.0, q.norm());
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn dot(&self, other: &Self) -> Complex<f64> {
        let arg1 = self.components[4]*other.components[4]
                - zip(self.components[0..4].iter(), self.components[0..4].iter())
                .fold(0.0, |product, x| product + x.0 * x.1.re);
        let arg2 = 
    }
}

impl From<[Complex<f64>; 4]> for Biquaternion {
    /** Construct a [`Biquaternion`] from 4 complex values.

    # Example
    ```
    use hoomd_manifold::Biquaternion;
    use num::complex::Complex;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = Biquaternion::from([Complex::new(1.0,0.0),
                                Complex::new(0.0,0.1),
                                Complex::new(1.0,0.0),
                                Complex::new(1.0,1.0)]);
    assert_eq!(q.components, [Complex::new(1.0,0.0),
                                Complex::new(0.0,0.1),
                                Complex::new(1.0,0.0),
                                Complex::new(1.0,1.0)]);
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn from(value: [Complex<f64>; 4]) -> Self {
        Self {
            components: value.into(),
        }
    }
}


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


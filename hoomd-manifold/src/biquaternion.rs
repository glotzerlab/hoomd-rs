// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Biquaternion`] and SO(3,1) representation. 
 */


use num::complex::Complex;
use std::fmt;
use std::iter::zip;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use crate::{Error,Minkowski, HyperbolicRotationMatrix};

/** Documentation for biquaternions
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Biquaternion {
    pub components: [Complex<f64>; 4]
}

impl Biquaternion {
    /** the Hamiltonian conjugate or biconjugate of a biquaternion

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
    /** the squared norm of a biquaternion

    # Example
    ```
    use hoomd_manifold::Biquaternion;
    use num::complex::Complex;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = Biquaternion::from([Complex::new(1.0,0.0),
                                Complex::new(0.0,1.0),
                                Complex::new(1.0,0.0),
                                Complex::new(1.0,0.0)]);
    assert_eq!(2.0, q.norm_squared());
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        self.scalar_product(&self).re
    }
    /** the norm of a biquaternion

    # Example
    ```
    use hoomd_manifold::Biquaternion;
    use num::complex::Complex;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = Biquaternion::from([Complex::new(3.0,0.0),
                                Complex::new(0.0,1.0),
                                Complex::new(4.0,0.0),
                                Complex::new(1.0,0.0)]);
    assert_eq!(5.0, q.norm());
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }
    /** the quaternion product of two biquaternions

    # Example
    ```
    use hoomd_manifold::Biquaternion;
    use num::complex::Complex;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = Biquaternion::from([Complex::new(2.0,0.0),
                                Complex::new(0.0,1.0),
                                Complex::new(1.0,0.0),
                                Complex::new(1.0,0.0)]);
    let p = Biquaternion::from([Complex::new(3.0,0.0),
                                Complex::new(2.0,0.0),
                                Complex::new(1.0,0.0),
                                Complex::new(0.0,1.0)]);
    let c = Biquaternion::from([Complex::new(1.0,3.0),
                                Complex::new(2.0,0.0),
                                Complex::new(5.0,-2.0),
                                Complex::new(-7.0,-1.0)]);
    assert_eq!(c, q.dot(&p));
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn dot(&self, other: &Self) -> Self {
        Biquaternion::from([
            self.components[3]*other.components[0] + other.components[3]*self.components[0]
            + self.components[1]*other.components[2] - other.components[1]*self.components[2],
            self.components[3]*other.components[1] + other.components[3]*self.components[1]
            + self.components[2]*other.components[0] - other.components[2]*self.components[0],
            self.components[3]*other.components[2] + other.components[3]*self.components[2]
            + self.components[0]*other.components[1] - other.components[0]*self.components[1],
            self.components[3]*other.components[3] - self.components[0]*other.components[0]
            - self.components[1]*other.components[1] - self.components[2]*other.components[2]
            ])
    }
    /** the scalar product of two biquaternions

    # Example
    ```
    use hoomd_manifold::Biquaternion;
    use num::complex::Complex;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = Biquaternion::from([Complex::new(2.0,0.0),
                                Complex::new(0.0,1.0),
                                Complex::new(1.0,0.0),
                                Complex::new(1.0,0.0)]);
    let p = Biquaternion::from([Complex::new(3.0,0.0),
                                Complex::new(2.0,0.0),
                                Complex::new(1.0,0.0),
                                Complex::new(0.0,1.0)]);
    assert_eq!(Complex::new(7.0,3.0), q.scalar_product(&p));
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn scalar_product(&self, other: &Self) -> Complex<f64> {
        zip(self.components.iter(), other.components.iter())
            .fold(Complex::new(0.0,0.0), |product, x| product + x.0 * x.1)
    }
    /** Convert a biquaternion into a Minkowski 4-vector
    # Example
    ```
    use hoomd_manifold::{Biquaternion, Minkowski};
    use num::complex::Complex;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = Biquaternion::from([Complex::new(2.0,0.0),
                                Complex::new(1.0,0.0),
                                Complex::new(1.0,0.0),
                                Complex::new(0.0,1.0)]);
    let x = Minkowski::from([2.0,1.0,1.0,1.0]);
    assert_eq!(x, q.to_4_vector()?);
    # Ok(())
    # }
    ```
    */
    #[inline]
    pub fn to_4_vector(self) -> Result<Minkowski<4>, Error> {
        if self + (self.conj()).bar() != Biquaternion::default() {
                Err(Error::InvalidBiquaternion4Vector)
        } else {
            Ok(Minkowski::from([self.components[0].re,
                            self.components[1].re,
                            self.components[2].re,
                            self.components[3].im]))
        }
    }
}

impl Default for Biquaternion {
    /** Create a biquaternion with all zeros
    */
    #[inline]
    fn default() -> Self {
        Self{
            components:[
                Complex::new(0.0,0.0),
                Complex::new(0.0,0.0),
                Complex::new(0.0,0.0),
                Complex::new(0.0,0.0),
                ]
        }
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

impl fmt::Display for Biquaternion {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {}, {}, {}]", self.components[0], self.components[1], 
        self.components[2], self.components[3])
    }
}

impl Add for Biquaternion {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            components: [self.components[0] + rhs.components[0],
                        self.components[1] + rhs.components[1],
                        self.components[2] + rhs.components[2],
                        self.components[3] + rhs.components[3]]
        }
    }
}

impl AddAssign for Biquaternion {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for n in 0..4 {
            self.components[n] += rhs.components[n];
        }
    }
}

impl Sub for Biquaternion {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            components: [self.components[0] - rhs.components[0],
                        self.components[1] - rhs.components[1],
                        self.components[2] - rhs.components[2],
                        self.components[3] - rhs.components[3]]
        }
    }
}

impl SubAssign for Biquaternion {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for n in 0..4 {
            self.components[n] -= rhs.components[n];
        }
    }
}

impl Mul<f64> for Biquaternion {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            components: [(self.components[0]).scale(rhs),
                        (self.components[1]).scale(rhs),
                        (self.components[2]).scale(rhs),
                        (self.components[3]).scale(rhs)]
        }
    }
}

impl MulAssign<f64> for Biquaternion {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        for n in 0..4 {
            self.components[n] *= Complex::new(rhs,0.0);
        }
    }
}
impl Div<f64> for Biquaternion {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self {
        Self {
            components: [(self.components[0]).scale(1.0/rhs),
                        (self.components[1]).scale(1.0/rhs),
                        (self.components[2]).scale(1.0/rhs),
                        (self.components[3]).scale(1.0/rhs)]
        }
    }
}

impl DivAssign<f64> for Biquaternion {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        for n in 0..4 {
            self.components[n] /= Complex::new(rhs,0.0);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HyperbolicVersor(Biquaternion);

impl HyperbolicVersor {
    /** A unit-norm biquaternion that represents an SO(3,1) transformation.
    */
}

impl From<HyperbolicVersor> for HyperbolicRotationMatrix<4> {
    #[inline]
    fn from(q: HyperbolicVersor) -> HyperbolicRotationMatrix<4> {
        let HyperbolicVersor(biquaternion) = q;
        let a = biquaternion.components[0];
        let b = biquaternion.components[1];
        let c = biquaternion.components[2];
        let d = biquaternion.components[3];

        HyperbolicRotationMatrix {
            rows: [
                [(d*d.conj() + a*a.conj() - b*b.conj() - c*c.conj()).re,
                (a*b.conj() + b*a.conj() - c*d.conj() - d*c.conj()).re,
                (a*c.conj() + c*a.conj() + b*d.conj() + d*b.conj()).re,
                -1.0*(d*a.conj() - a*d.conj() + b*c.conj() - c*b.conj()).im]
                .into(),
                [(b*a.conj() + a*b.conj() + c*d.conj() + d*c.conj()).re,
                (d*d.conj() - a*a.conj() + b*b.conj() - c*c.conj()).re,
                (b*c.conj() + c*b.conj() - a*d.conj() - d*a.conj()).re,
                -1.0*(d*b.conj() - b*d.conj() + c*a.conj() - a*c.conj()).im]
                .into(),
                [(c*a.conj() + a*c.conj() - b*d.conj() - d*b.conj()).re,
                (c*b.conj() + b*c.conj() + a*d.conj() + d*a.conj()).re,
                (d*d.conj() - a*a.conj() - b*b.conj() + c*c.conj()).re,
                -1.0*(d*c.conj() - c*d.conj() + a*b.conj() - b*a.conj()).im]
                .into(),
                [(a*d.conj() - d*a.conj() + b*c.conj() - c*b.conj()).im,
                (b*d.conj() - d*b.conj() + c*a.conj() - a*c.conj()).im,
                (c*d.conj() - d*c.conj() + a*b.conj() - b*a.conj()).im,
                (a*a.conj() + b*b.conj() + c*c.conj() + d*d.conj()).re]
                .into(),
            ],
        }
    }
}
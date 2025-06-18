// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Biquaternion`] anda four-dimensional matrix representation
 of SO(3,1). 
 */


use num::complex::Complex;
use std::fmt;
use std::iter::zip;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign,
};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform, Uniform};

use crate::{Error,Minkowski, HyperbolicRotationMatrix, HyperbolicRotate};

/** 
## Biquaternions

Biquaternions are the set of numbers $a + b\mathbf{i} + c\mathbf{j} + d\mathbf{k}$ 
where $a,b,c,d$ are complex numbers and ${1,\mathbf{i},\mathbf{j},\mathbf{k}}$ are 
the quaternion algebra. Biquaternions can be thought of as a generalization of quaternions
 which allow for complex coefficients. 

## Construction of Biquaternions

Create a biquaternion from an array of four complex numbers. Note that components are 
in the order $[\mathbf{i},\mathbf{j},\mathbf{k},1]$ (i.e., the scalar component is at 
the end)
```
use hoomd_manifold::Biquaternion;
use num::complex::Complex;

// create biquaternion q = (4+h) + (1+4h)i + (2+3h)j + (3+2h)k
let q = Biquaternion::from([Complex::new(1.0,4.0),
                        Complex::new(2.0,3.0),
                        Complex::new(3.0,2.0),
                        Complex::new(4.0,1.0)]);
assert_eq!(4.0, q.components[0].im);

```

## Operations with biquaternions

Similar to [`Quaternion`], biquaternions support vector operations (addition, multiplication 
by a scalar, etc.):
```
use hoomd_manifold::Biquaternion;
use num::complex::Complex;

// create biquaternion q = (4+h) + (1+4h)i + (2+3h)j + (3+2h)k
let mut a = Biquaternion::from([Complex::new(1.0,0.0),
                        Complex::new(2.0,0.0),
                        Complex::new(3.0,0.0),
                        Complex::new(0.0,1.0)]);
let mut b = Biquaternion::from([Complex::new(0.0,4.0),
                        Complex::new(0.0,3.0),
                        Complex::new(0.0,2.0),
                        Complex::new(1.0,0.0)]);
b /= 2.0;
let mut c = a + b;
assert_eq!(Complex::new(1.0,2.0), c.components[0]);

```
Biquaternions also support the following operations:

Hamiltonian conjugate/ biconjugate: 
Denoted by the method "bar", the Hamiltonian conjugate multiplies the vector part 
of the biquaternion by -1.0.
```
use hoomd_manifold::Biquaternion;
use num::complex::Complex;

let q = Biquaternion::from([Complex::new(-1.0,0.0),
                            Complex::new(-1.0,2.0),
                            Complex::new(1.0,0.0),
                            Complex::new(1.0,0.0)]);
let p = Biquaternion::from([Complex::new(1.0,0.0),
                            Complex::new(1.0,-2.0),
                            Complex::new(-1.0,0.0),
                            Complex::new(1.0,0.0)]);

// Hamiltonian conjugate denoted by "bar"
assert_eq!(p, q.bar());
```

Complex conjugation:
Deonted by method "conj", takes the complex conjugate of all components of the 
biquaternion
```
use hoomd_manifold::Biquaternion;
use num::complex::Complex;

let q = Biquaternion::from([Complex::new(1.0,8.0),
                            Complex::new(2.0,7.0),
                            Complex::new(3.0,6.0),
                            Complex::new(4.0,5.0)]);
let p = Biquaternion::from([Complex::new(1.0,-8.0),
                            Complex::new(2.0,-7.0),
                            Complex::new(3.0,-6.0),
                            Complex::new(4.0,-5.0)]);

// Complex conjugate denoted by "conj"
assert_eq!(p, q.conj());
```

Biquaternion Product:
The biquaternion product takes two biquaternions and outputs another biquaternion.
```
use hoomd_manifold::Biquaternion;
use num::complex::Complex;

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
```

Scalar Product:
The scalar product takes two biquaternions and outputs a complex number according to 
```math
\frac{1}{2}(a\overline{b} + b\overline{a}) 
```
```
use hoomd_manifold::Biquaternion;
use num::complex::Complex;

let q = Biquaternion::from([Complex::new(2.0,0.0),
                            Complex::new(0.0,1.0),
                            Complex::new(1.0,0.0),
                            Complex::new(1.0,0.0)]);
let p = Biquaternion::from([Complex::new(3.0,0.0),
                            Complex::new(2.0,0.0),
                            Complex::new(1.0,0.0),
                            Complex::new(0.0,1.0)]);
assert_eq!(Complex::new(7.0,3.0), q.scalar_product(&p));
```

Biquaternion Norm:
The scalar product furnishes a "norm" for the biquaternion. 
```
use hoomd_manifold::Biquaternion;
use num::complex::Complex;

let q = Biquaternion::from([Complex::new(3.0,0.0),
                            Complex::new(0.0,1.0),
                            Complex::new(4.0,0.0),
                            Complex::new(0.0,2.0)]);
assert_eq!((20.0_f64).sqrt(), q.norm());
```

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
    let q = Biquaternion::from([Complex::new(-1.0,0.0),
                                Complex::new(0.0,1.0),
                                Complex::new(1.0,0.0),
                                Complex::new(1.0,0.0)]);
    let p = Biquaternion::from([Complex::new(1.0,0.0),
                                Complex::new(0.0,-1.0),
                                Complex::new(-1.0,0.0),
                                Complex::new(1.0,0.0)]);
    assert_eq!(p, q.bar());
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn bar(&self) -> Self {
        Biquaternion::from([(self.components[0]).scale(-1.0), (self.components[1]).scale(-1.0),
        (self.components[2]).scale(-1.0),(self.components[3])])
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
    /** create a [`UnitBiquaternion`] by normalizing the given biquaternion
     */
    #[inline]
    pub fn to_unit(self) -> Result<UnitBiquaternion, Error> {
        let mag = self.norm();
        if mag == 0.0 {
            Err(Error::InvalidBiquaternionMagnitude)
        } else {
            Ok(UnitBiquaternion(self / mag))
        }
    }
    /** create a [`UnitBiquaternion`] by normalizing the given biquaternion without 
    checking
     */
     #[inline]
     pub fn to_unit_unchecked(self) -> UnitBiquaternion {
        UnitBiquaternion(self)
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
/** 
## Representation of SO(3,1)
Unit-norm Biquaternions furnish a representation of SO(3,1), analogous to quaternions and SO(3). 
If $\vec{x} = (x_1,x_2,x_3,x_4)$ is a vector in Minkowski space, then $\vec{x}$ can be 
mapped to a biquaternion $\vec{x} \mapsto X = [x_1,x_2,x_3,h x_4]$ (where $h$) is the imaginary 
number) whose squared norm is $|X|^2 = x_1^2 + x_2^2 + x_3^2 - x_4^2$. It can be shown that, for 
a unit biquaternion $q$, the transformation 
```math
q^* X \overline{q} = X'
``` 
preserves the norm, i.e., $|X|^2 = |X'|^2$. We therefore have that this action by unit biquaternions 
produces a representation of SO(3,1). The biquaternion algebra can be used directly to transform Minkowski
 4-vectors, or unit biquaternions can be represented as matrices using [`HyperbolicRotationMatrix<4>`].

Like quaternions, the unit biquaternion 
```math
q = \cos(\theta/2) + \bf{i}\sin(\theta/2)
``` 
generates a rotation about the $\mathbf{i}$ axis by angle $\theta$: 
```
use hoomd_manifold::{HyperbolicRotationMatrix, Minkowski, HyperbolicRotate,
                    Biquaternion, UnitBiquaternion};
use std::f64::consts::PI;
use num::complex::Complex;

// biquaternion representing a rotation of pi/2 radians about x-axis
let q = Biquaternion::from([Complex::new((PI/4.0).sin(),0.0),
                    Complex::new(0.0,0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new((PI/4.0).cos(), 0.0)]);
let v = q.to_unit();
let x = Minkowski::from([1.0, 1.0, 1.0, 1.0]);
let rotation = HyperbolicRotationMatrix::from(v.expect("non-zero biquaternion"));
let rotated = rotation.hyperbolic_rotate(&x);
// rotated vector is approximately [1.0, -1.0, 1.0, 1.0];
```

However, biquaternions also generate boosts via 
```math
q = \cosh(v) + \mathbf{i}h\sinh(v)
```
which represents a boost of rapidity $v$ in the $\mathbf{i}$ direction:
```
use hoomd_manifold::{UnitBiquaternion, HyperbolicRotate, HyperbolicRotationMatrix, Biquaternion, Minkowski};
use std::f64::consts::PI;
use num::complex::Complex;
use libm::{sinh,cosh};

// biquaternion representing a boost of rapidity 0.5 in x direction
let q = Biquaternion::from([Complex::new(0.0,(0.2_f64).sinh()),
                    Complex::new(0.0,0.0),
                    Complex::new(0.0,0.0),
                    Complex::new((0.2_f64).cosh(),0.0)]);
let v = q.to_unit();
let x = Minkowski::from([0.0, 0.0, 0.0, 1.0]);
let boost = HyperbolicRotationMatrix::from(v.expect("hard-coded unit biquaternion"));
let boosted = boost.hyperbolic_rotate(&x);
// boosted is approximately [(0.4_f64).sinh(), 0.0, 0.0,(0.4_f64).cosh()]
```
*/
pub struct UnitBiquaternion(Biquaternion);

impl UnitBiquaternion {
    /** Normalize a biquaternion
     */
    #[inline]
    #[must_use]
    pub fn normalized(self) -> Self {
        let UnitBiquaternion(q) = self;
        let f = 1.0 / q.norm();
        Self(q*f)
    }
}

impl Distribution<UnitBiquaternion> for StandardUniform {
    /** Sample a random [`UnitBiquaternion`] 

    # Example

    ```
    use hoomd_manifold::UnitBiquaternion;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(1);
    let v: UnitBiquaternion = rng.random();
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> UnitBiquaternion {
        #[expect(
            clippy::expect_used,
            reason = "This constants chosen for this distribution are valid"
        )]
        let uniform = Uniform::new(-1.0, 1.0).expect("hard-coded distribution should be valid");

        let mut vec = [Complex::new(0.0,0.0) ; 4];
        let mut scale = Complex::new(0.0,0.0);
        for n in 0..4 {
            let mut a = uniform.sample(rng);
            let mut b = uniform.sample(rng);
            vec[n] = Complex::new(a,b);
            scale += vec[n].powi(2);
        }
        
        UnitBiquaternion(Biquaternion {
            components: [vec[0]/scale,vec[1]/scale,vec[2]/scale,vec[3]/scale]
        })
    }
}

impl From<UnitBiquaternion> for HyperbolicRotationMatrix<4> {
    #[inline]
    fn from(q: UnitBiquaternion) -> HyperbolicRotationMatrix<4> {
        let UnitBiquaternion(biquaternion) = q;
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

impl HyperbolicRotate<Minkowski<4>> for UnitBiquaternion {
    type Matrix = HyperbolicRotationMatrix<4>;

    /** Transform a [`Minkowski<4>`] by a [`UnitBiquaternion`]

    ```math
    \overline{\mathbf{q}} \vec{a} \mathbf{q}^*
    ```

    # Example
    ```
    // rotation about z axis using biquaternion algebra
    use hoomd_manifold::{UnitBiquaternion, HyperbolicRotate, Biquaternion, Minkowski};
    use std::f64::consts::PI;
    use num::complex::Complex;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Minkowski::from([1.0, 0.0, 0.0, 1.0]);
    let q = Biquaternion::from([Complex::new(0.0,0.0),
                        Complex::new(0.0,0.0),
                        Complex::new((PI/4.0).sin(), 0.0),
                        Complex::new((PI/4.0).cos(), 0.0)]);
    let v = q.to_unit_unchecked();
    let rotated = v.hyperbolic_rotate(&x);
    // rotated is approximately [0.0, 1.0, 0.0, 1.0]
    # Ok(())
    # }
    ```

    # Example
    ```
    // boost in x direction using biquaternion algebra.
    use hoomd_manifold::{UnitBiquaternion, HyperbolicRotate, Biquaternion, Minkowski};
    use std::f64::consts::PI;
    use num::complex::Complex;
    use libm::{sinh,cosh};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Minkowski::from([0.0, 0.0, 0.0, 1.0]);
    let q = Biquaternion::from([Complex::new(0.0, PI/4.0).sin(),
                        Complex::new(0.0,0.0),
                        Complex::new(0.0, 0.0),
                        Complex::new(0.0, PI/4.0).cos()]);
    let v = q.to_unit_unchecked();
    let boosted = v.hyperbolic_rotate(&x);
    // boosted is approximately [(PI/2.0).sinh(), 0.0, 0.0, (PI/2.0).cosh()]
    # Ok(())
    # }
    ```
    */
    
    #[inline]
    fn hyperbolic_rotate(&self, vector: &Minkowski<4>) -> Minkowski<4> {
        let UnitBiquaternion(biquaternion) = self;
        let x = Biquaternion::from([Complex::new(vector[0],0.0),
                                    Complex::new(vector[1],0.0),
                                    Complex::new(vector[2],0.0),
                                    Complex::new(0.0,vector[3])]);
        let x_transformed = (biquaternion.conj()).dot(&x.dot(&(biquaternion.bar())));
        Minkowski::from([x_transformed.components[0].re,
                        x_transformed.components[1].re,
                        x_transformed.components[2].re,
                        x_transformed.components[3].im])
    }
}
// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement vector types in Minkowski space.
 */

use std::array;
use std::fmt;
use std::iter::zip;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform, Uniform};
use libm::{sin, cos, acosh, sinh, cosh, sqrt, atan2};
use std::f64::consts::PI;
use hoomd_vector::Vector;
use hoomd_utility::valid::PositiveReal;
use hoomd_microstate::{boundary::Boundary, property::Point};

use crate::{CurvedManifold, Error, Hyperboloid, HyperbolicRotate, FundamentalDomain};

/** 
[`Minkowski<N>`] implements (N-1,1)-dimensional Minkowski space with the metric signature 
$(+ \;\cdots\; +\; -)$. [`Minkowski`] supports [`Vector`] operations such as vector addition and rescaling, but 
is not a true inner product space. 

## Constructing Minkowski vectors 

Similar to [`Cartesian`], N-dimensional vectors can be constructed using an array 
of (real-valued) coordinates. Three- and four-dimensional vectors can also be 
constructed from tuples: 
```
use hoomd_manifold::Minkowski;

fn from_array() -> Minkowski<5> {
    Minkowski::from([1.0, 2.0, 3.0, 4.0, 5.0])
}
fn from_tuples() -> Minkowski<3> {
    Minkowski::from((6.0, 7.0, 8.0))
}
```

## Operating on Minkowski vectors

[`Minkowski`] implements everything from [`Vector`], which includes vector addition/subtraction,
multiplication by a scalar, and a distance metric. 

```
use hoomd_manifold::Minkowski;

// Vector addition
let mut a = Minkowski::from([1.0, 1.0, 1.0, 1.0]);
let mut b = Minkowski::from([0.0, 0.0, 0.0, 2.0]);
a += b;

// Multiplication by a scalar
let mut c = a * 4.0;

// Division by a scalar
c /= 2.0;

assert_eq!(c, [2.0, 2.0, 2.0, 6.0].into());
```

The distance metric on Minkowski space is given by the "spacetime interval"
```math
d_M^2(\vec{u},\vec{v}) = (\vec{u}-\vec{v})^T \eta (\vec{u}-\vec{v}) 
= (u_1-v_1)^2 +\cdots + (u_{N-1}-v_{N-1})^2 - (u_N - v_N)^2
``` 
Note that because this metric is not positive-definite, [`Minkowski`] only implements a 
squared distance metric (i.e., it does not implement a "distance" function which 
takes the square root of "distance_squared"). 

```
use hoomd_manifold::Minkowski;
use hoomd_vector::Vector;

let x = Minkowski::from([1.0, 0.0, 5.0]);
let y = Minkowski::from([0.0, 0.0, 3.0]);
assert_eq!(-3.0, x.distance_squared(&y));
```
*/

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Minkowski<const N: usize> {
    /** The vector's coordinates. The final component is the one associated with a minus sign (-)
    in the metric
    */
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
    the last component has a (-) signature, while the preceeding coordinates have
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
    the "mostly plusses" metric signature (+ ... + -). 
    ```math
    d^2_M(\vec{x},\vec{y}) = -(x_N-y_N)^2 + \sum_{i=1}^{N-1} (x_i - y_i)^2
    ```

    # Example
    ```
    use hoomd_manifold::Minkowski;
    use hoomd_vector::Vector;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Minkowski::from([0.0, 2.0, 3.0]);
    let y = Minkowski::from([1.0, 0.0, 0.0]);
    assert_eq!(-4.0, x.distance_squared(&y));
    # Ok(())
    # }
    */
    #[inline]
    fn distance_squared(&self, other: &Self) -> f64 {
        let last_component = -1.0 * (self.coordinates[N-1] - other.coordinates[N-1]).powi(2);
        zip(self.coordinates[0..N-1].iter(), other.coordinates[0..N-1].iter())
            .fold(last_component, |product, x| product + (x.0 - x.1).powi(2))
    }
}

/** The [`CurvedManifold`] trait for Minkowski implements the negatively curved metric
*/
impl<const N: usize> CurvedManifold for Minkowski<N> {
    #[inline]
    fn geodesic_distance(&self, other: &Self, rho: f64) -> f64 {
        let last_component = self.coordinates[N-1] * other.coordinates[N-1];
        let arg = zip(self.coordinates[0..N-1].iter(), other.coordinates[0..N-1].iter())
            .fold(last_component, |product, x| product - (x.0 * x.1));
        rho * acosh(arg/(rho.powi(2)))
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

impl<const N: usize> Distribution<Minkowski<N>> for StandardUniform {
    /** Sample a Minkowski vector from the uniform [-1, 1] hypercube.

    # Example
    ```
    use hoomd_manifold::Minkowski;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(1);
    let v: Minkowski::<3> = rng.random();
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Minkowski<N> {
        #[expect(
            clippy::expect_used,
            reason = "This constants chosen for this distribution are valid"
        )]
        let uniform = Uniform::new_inclusive(-1.0, 1.0)
            .expect("hard-coded range should form a valid distribution");
        Minkowski {
            coordinates: array::from_fn(|_| uniform.sample(rng)),
        }
    }
}

/** 
## Hyperboloid Model  
The trait [`Hyperboloid`] implements an embedding of the top sheet of an (N-1)-dimensional 
two-sheeted hyperboloid in N-dimensional Minkowski space. This surface has constant negative curvature 
and therefore serves as a model of (N-1)-dimensional hyperbolic space.

Explicitly, for N-dimensional Minkowski space with metric $\eta = \operatorname{diag}(+,\cdots,+,-)$, 
the hyperboloid with skirt width $R$ is defined by the set of points with components satisfying 
```math
x_1^2 +\cdots x_{N-1}^2 - x_{N}^2 = -R^2
```
Where the "top sheet" is defined by the $x_N>0$ solutions. In Minkowski space, the hyperboloid
has a natural interpretation as the set of points with the same spacetime interval
```math
\Delta s^2 = \vec{x}^T \eta \vec{x} = x_1^2 +\cdots x_{N-1}^2 - x_{N}^2
```

[`Hyperboloid`] defines a distance metric [`hyperbolic_distance`] which computes the distance 
of the geodesic passing between two points on a hyperboloid with some given skirt width. This may be 
interpreted as the metric for the hyperboloid model of hyperbolic space. 
```
use libm::acosh;
use hoomd_manifold::{Minkowski, Hyperboloid};
use hoomd_vector::Vector;

// two points on the hyperboloid with skirt width R = 1.0:
let x = Minkowski::from([0.0, 0.0, 1.0]);
let y = Minkowski::from([0.0, 1.0, (2.0_f64).sqrt()]);

assert_eq!(acosh((2.0_f64).sqrt()), x.hyperbolic_distance(&y, 1.0));
```

*/

impl<const N: usize> Hyperboloid for Minkowski<N> {
    /** Computes the length of the geodesic passing between two points. From N-dimensional Minkowski space 
    with signature (+\cdots +-), one can obtain the corresponding Minkowski bilinear form 
    ```math
    B(\vec{u},\vec{v}) = \vec{u}^T \eta \vec{v}= u_1v_1 + \cdots + u_{N-1}v_{N-1} - u_Nv_N 
    ``` 
    Now the distance (according to the Minkowski metric) between two points \vec{u} and \vec{v} on the hyperboloid
    is given by 
    ```math
    d_{H^2}(\vec{u},\vec{v}) = \rho\cdot\cosh^{-1}(-\frac{1}{\rho^2}B(\vec{u},\vec{v}))
    ```

    # Example
    ```
    use libm::acosh;
    use hoomd_vector::Vector;
    use hoomd_manifold::{Minkowski, Hyperboloid};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Minkowski::from([0.0, 0.0, 1.0]);
    let y = Minkowski::from([1.0, 0.0, (2.0_f64).sqrt()]);
    let c = acosh((2.0_f64).sqrt());
    assert_eq!(c,x.hyperbolic_distance(&y,1.0));
    # Ok(())
    # }
    ```
     */
    #[inline]
    fn hyperbolic_distance(&self, other: &Self, skirt: f64) -> f64 {
        let last_component = self.coordinates[N-1] * other.coordinates[N-1];
        let arg = zip(self.coordinates[0..N-1].iter(), other.coordinates[0..N-1].iter())
            .fold(last_component, |product, x| product - (x.0 * x.1));
        skirt * acosh(arg/(skirt.powi(2)))
    }
    /** Computes the length of the geodesic passing between the cusp $(0,\cdots,0,\rho)$ and a given
     point on the hyperboloid with a given skirt length.

    # Example
    ```
    use libm::{sinh, cosh};
    use hoomd_vector::Vector;
    use hoomd_manifold::{Minkowski, Hyperboloid};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v : f64 = 4.2;
    let rho : f64 = 1.0;
    let x = Minkowski::from([rho*(v.sinh()),0.0,rho*(v.cosh())]);
    assert_eq!(v*rho, x.distance_from_cusp(rho));
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn distance_from_cusp(&self, skirt: f64) -> f64 {
        skirt * acosh((self.coordinates[N-1])/skirt)
    }
    /** Projects points on the hyperboloid onto the Poincare disk/ball.

    # Example
    ```
    use libm::{sinh, cosh};
    use hoomd_vector::Vector;
    use hoomd_manifold::{Minkowski, Hyperboloid};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v : f64 = 1.098612;
    let rho : f64 = 1.0;
    let x = Minkowski::from([v.sinh(),0.0,v.cosh()]);
    let projection = x.to_poincare(rho);
    assert_eq!([(v.sinh())/(v.cosh() + 1.0), 0.0], [projection[0],projection[1]]);
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn to_poincare(&self, skirt: f64) -> Vec<f64> {
        (0..N-1).collect::<Vec<usize>>()
        .iter().map(|i| self.coordinates[*i] / (1.0 + self.coordinates[N-1]/skirt)).collect::<Vec<f64>>()
    }
}

// Cusp-to-vertex distance for {8,8} tiling for Gauss curvature K = -1
const EIGHTEIGHT : f64 = 2.448452447678076;

impl FundamentalDomain for Minkowski<3> {
    /** Computes the length of the geodesic passing between the cusp $(0,0,\rho)$ and the boundary 
    of the fundamental domain of the {8,8} tiling of hyperbolic space.
    # Example
    ```
    use libm::acosh;
    use hoomd_vector::Vector;
    use hoomd_manifold::{Minkowski, Hyperboloid, FundamentalDomain};
    use std::f64::consts::PI;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v : f64 = 4.2;
    let rho : f64 = 1.0;
    let theta: f64 = PI/4.0;
    let x = Minkowski::from([rho*(v.sinh())*(theta.cos()),rho*(v.sinh())*(theta.sin()),rho*(v.cosh())]);
    // x.distance_to_boundary(rho) is approximately 0
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn distance_to_boundary(&self, skirt: f64) -> f64 {
        let theta = atan2(self.coordinates[1], self.coordinates[0]);
        let angle = theta.rem_euclid(PI/4.0);
        let tile_size = EIGHTEIGHT;
        let eta = (tile_size.tanh()/(angle.cos() - angle.sin()*(1.0-(2.0_f64).sqrt()))).atanh();
        skirt * eta - self.distance_from_cusp(skirt)
    }
    /** Outputs vector of points on the boundary of the fundamental domain
    */
    #[inline] 
    fn boundary_points(m: usize, skirt: f64) -> Vec::<(f64, f64)> {
        let mut coords = Vec::<(f64,f64)>::new();
        for n in 0..m {
            let angle = (n as f64) * 2.0 * PI / (m as f64);
            let tile_size = EIGHTEIGHT;
            let eta = (tile_size.tanh()/(angle.cos() - angle.sin()*(1.0-(2.0_f64).sqrt()))).atanh();
            let x = (skirt* sinh(eta))/(1.0 + cosh(eta));
            for k in 0..8 {
                coords.push((x*cos(angle + (k as f64)*PI/4.0), x*sin(angle+ (k as f64)*PI/4.0)));
            }
        }
        coords
    }   
}

/** {8,8} tile of hyperbolic space
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EightEight {
    /// Skirt width of the hyperboloid
    pub skirt: f64
}

impl Boundary<Minkowski<3>, Point<Minkowski<3>>, Point<Minkowski<3>>> for EightEight {
    #[inline]
    fn is_inside(&self, point: &Minkowski<3>) -> bool {
        point.distance_to_boundary(self.skirt) >= 0.0
    }
}

/** 

## Hyperbolic Rotations in Minkowski Space

Construct a [`HyperbolicRotationMatrix`] to apply SO(N-1, 1) transformations to
N-dimensional Minkowski vectors. For Minkowski 4-vectors, [`Biquaternion`] should be used 
instead for numeric stability. See documentation in [`HyperbolicAngle`] for details on
 SO(2,1) transformations (i.e., two-dimensional hyperbolic space), and see documentation 
 in [`Biquaternion`] for details on SO(3,1) transformations (i.e., three-dimensional 
 hyperbolic space).

In two dimensional hyperbolic space:
```
use hoomd_manifold::{HyperbolicRotationMatrix, Minkowski, HyperbolicRotate, HyperbolicAngle};
use std::f64::consts::PI;
use libm::{sinh, cosh};

// rotation by pi radians about z axis
fn rotate_about_z(minkowski_vector: &Minkowski<3>) -> Minkowski<3> {
    let generators = HyperbolicAngle::from((PI, 0.0_f64, 0.0_f64));
    let rotation_matrix = HyperbolicRotationMatrix::from(generators);
    rotation_matrix.hyperbolic_rotate(&minkowski_vector)
}

// boost in x direction
fn boost_in_x(minkowski_vector: &Minkowski<3>) -> Minkowski<3>{
    let generators = HyperbolicAngle::from((0.0_f64, 0.2_f64, 0.0_f64));
    let boost_matrix = HyperbolicRotationMatrix::from(generators);
    boost_matrix.hyperbolic_rotate(&minkowski_vector)
}
```

*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HyperbolicRotationMatrix<const N: usize> {
    /// Rows of the rotation matrix.
    pub(crate) rows: [Minkowski<N>; N],
}

impl<const N: usize> HyperbolicRotate<Minkowski<N>> for HyperbolicRotationMatrix<N> {
    type Matrix = HyperbolicRotationMatrix<N>;

    #[inline]
    /** Rotate a [`Minkowski<N>`] by a [`HyperbolicRotationMatrix`]
    # Example
    ```
    // Rotate point in 2D hyperbolic space about z-axis
    use hoomd_manifold::{HyperbolicRotationMatrix, Minkowski, HyperbolicRotate, HyperbolicAngle};
    use std::f64::consts::PI;
    use libm::{sin, cos};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v = Minkowski::from([1.0, 0.0, 1.0]);
    let spatial_rotation = HyperbolicAngle::from((PI/2.0, 0.0_f64, 0.0_f64));
    let matrix = HyperbolicRotationMatrix::from(spatial_rotation);
    let rotated = matrix.hyperbolic_rotate(&v);
    let c = Minkowski::from([cos(PI/2.0),sin(PI/2.0),1.0]);
    assert_eq!(c,rotated);
    # Ok(())
    # }
    ```
    # Example
    ```
    // Boost point in 2D hyperbolic space in x direction
    use hoomd_manifold::{HyperbolicRotationMatrix, Minkowski, HyperbolicRotate, HyperbolicAngle};
    use std::f64::consts::PI;
    use num::complex::Complex;
    use libm::{sinh,cosh};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v = Minkowski::from([1.0, 0.0, 1.0]);
    let small_boost = HyperbolicAngle::from((0.0_f64, 0.1_f64, 0.0_f64));
    let matrix = HyperbolicRotationMatrix::from(small_boost);
    let rotated = matrix.hyperbolic_rotate(&v);
    let c = Minkowski::from([sinh(0.1)+cosh(0.1),0.0,sinh(0.1)+cosh(0.1)]);
    assert_eq!(c,rotated);
    # Ok(())
    # }
    ```
    # Example
    ```
    // inputting zero for all angles and rapidities does nothing
    use hoomd_manifold::{HyperbolicRotationMatrix, Minkowski, HyperbolicRotate, HyperbolicAngle};
    use std::f64::consts::PI;
    use num::complex::Complex;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v = Minkowski::from([1.0, 2.0, 1.0]);
    let identity = HyperbolicAngle::from((0.0_f64, 0.0_f64, 0.0_f64));
    let matrix = HyperbolicRotationMatrix::from(identity);
    let rotated = matrix.hyperbolic_rotate(&v);
    let c = Minkowski::from([1.0, 2.0, 1.0]);
    assert_eq!(c,rotated);
    # Ok(())
    # }
    ```
    # Example

    ```
    // Rotate point in 3D hyperbolic space about y axis using matrix representation
    use hoomd_manifold::{HyperbolicRotationMatrix, Minkowski, HyperbolicRotate,
                        Biquaternion, UnitBiquaternion};
    use std::f64::consts::PI;
    use num::complex::Complex;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = Biquaternion::from([Complex::new(0.0,0.0),
                        Complex::new((PI/4.0).sin(),0.0),
                        Complex::new(0.0, 0.0),
                        Complex::new((PI/4.0).cos(), 0.0)]);
    let v = q.to_unit()?;
    let x = Minkowski::from([1.0, 0.0, 0.0, 1.0]);
    let rotation = HyperbolicRotationMatrix::from(v);
    let rotated = rotation.hyperbolic_rotate(&x);
    // rotated vector is approximately [0.0, 0.0, -1.0, 1.0];
    # Ok(())
    # }
    ```
    ```
    // Boost point in 3D hyperbolic space in x direction using biquaternion algebra.
    use hoomd_manifold::{UnitBiquaternion, HyperbolicRotate, Biquaternion, Minkowski};
    use std::f64::consts::PI;
    use num::complex::Complex;
    use libm::{sinh,cosh};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Minkowski::from([0.0, 0.0, 0.0, 1.0]);
    let q = Biquaternion::from([Complex::new(0.0, 0.5).sin(),
                        Complex::new(0.0,0.0),
                        Complex::new(0.0, 0.0),
                        Complex::new(0.0, 0.5).cos()]);
    let v = q.to_unit()?;
    let boosted = v.hyperbolic_rotate(&x);
    // boosted is approximately [(0.5).sinh(), 0.0, 0.0, (0.5).cosh()]
    # Ok(())
    # }
    ```
    */
    fn hyperbolic_rotate(&self, vector: &Minkowski<N>) -> Minkowski<N> {
        let mut coordinates = [0.0; N];

        for (result, row) in coordinates.iter_mut().zip(self.rows.iter()) {
            *result = zip(row.coordinates.iter(), vector.coordinates.iter())
                        .fold(0.0, |product, x| product + x.0 * x.1);
        }

        Minkowski { coordinates }
    }
}

/** A uniform distribution of points within distance r of a point on the 2-dimensional hyperboloid
with a given skirt width. 
# Example

```
use hoomd_manifold::{Hyperboloid, HyperbolicDisk, Minkowski, HyperbolicAngle,
                    HyperbolicRotationMatrix, HyperbolicRotate};
use hoomd_vector::Vector;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand::distr::Distribution;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut rng = StdRng::seed_from_u64(12);

// generate random point
let rho: f64 = 1.0;
let v: HyperbolicAngle = rng.random();
let matrix = HyperbolicRotationMatrix::from(v);
let origin = Minkowski::from([0.0, 0.0, rho]);
let random_point = matrix.hyperbolic_rotate(&origin);
    
// generate transformation which keeps the distance moved less than r = 0.1
let r = 0.1;
let mut rng_2 = StdRng::seed_from_u64(239);
let disk = HyperbolicDisk {r: r.try_into()?, point: random_point, skirt: rho};
let transformed_random_point: Minkowski<3> = disk.sample(&mut rng_2);

assert!(r > random_point.hyperbolic_distance(&transformed_random_point, rho));

# Ok(())
# }
```
*/
pub struct HyperbolicDisk {
    /// Max distance away from point
    pub r: PositiveReal,
    /// The center of the disk
    pub point: Minkowski<3>,
    /// The skirt width of the hyperboloid
    pub skirt: f64
}

impl Distribution<Minkowski<3>> for HyperbolicDisk {
    /** Translates Minkowski 3-vector named "point" along the hyperboloid by maximum distance of r.
    Note that because SO(2,1) is non-Abelian, the point must be transformed to the cusp before the 
    trial move is applied (and then the point is transformed back). This ensures that the max distance 
    translated by the trial move does not exceed r. 
    */
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Minkowski<3> {
        let rho = self.skirt;
        let max_boost = (self.r.get())/rho;
        let point = self.point;
        let eta = acosh(point.coordinates[2]/rho);
        let phi = atan2(point.coordinates[1], point.coordinates[0]);
        let trial_boost = Uniform::new(0.0, 1.0).expect("r is positive and real");
        let trial_rotation = Uniform::new(-PI, PI).expect("hard-coded distribution should be valid");
        let theta = trial_rotation.sample(rng);
        let v1: f64 = trial_boost.sample(rng);
        let v = sqrt(v1) * max_boost;
        let trial_coords = [rho * sinh(v) * cos(theta),
                            rho * sinh(v) * sin(theta),
                            rho * cosh(v)];
        Minkowski::from([trial_coords[0]*cosh(eta)*cos(phi) - trial_coords[1]* sin(phi) + trial_coords[2]*sinh(eta)*cos(phi),
                        trial_coords[0]*cosh(eta)*sin(phi) + trial_coords[1]* cos(phi) + trial_coords[2]*sinh(eta)* sin(phi),
                        trial_coords[0]*sinh(eta) + trial_coords[2]*cosh(eta)])
    }
}

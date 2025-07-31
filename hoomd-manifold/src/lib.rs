// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Tools for non-Euclidean geometries. Includes trait [`Sphere`] which calculates geodesic
distances on a the surface of an N-sphere, and trait [`Hyperboloid`] which calculates
geodesic distances on the hyperboloid embedded in [`Minkowski`]. 

## Sphere

[`Sphere`] describes an N-sphere of radius R embedded in [`Cartesian<N+1>`]. By definition,
the components of a point on an N-sphere satisfy 
```math
\sum_{i=1}^{N+1}x_i^2 = R^2
```
[`Sphere`] implements a distance metric which calculates the geodesic distance on the 
surface of an N-sphere. Use [`Sphere`] to describe spaces with constant postive curvature. 

## Hyperboloid
[`Hyperboloid`] describes the upper sheet of an N-dimensional two-sheeted hyperboloid with 
skirt R. The components of a point on the hyperboloid satisfy 
```math
x_1^2 + \cdots + x_{N-1}^2 - x_{N}^2 = -R^2
```
[`Hyperboloid`] implements a distance metric which calculates the geodesic distance on 
the surface of a hyperboloid. Use [`Hyperboloid`] embdedded in [`Minkowski`] to describe
hyperbolic space. 

## Minkowski

[`Minkowski<N>`] implements (N-1,1)-dimensional Minkowski space with the metric signature 
$(+ \;\cdots\; +\; -)$. [`Minkowski`] supports [`Vector`] operations such as vector addition and rescaling, but 
is not a true inner product space. The distance metric on Minkowski space is given by the 
"spacetime interval"
```math
d_M^2(\vec{u},\vec{v}) = (\vec{u}-\vec{v})^T \eta (\vec{u}-\vec{v}) 
= (u_1-v_1)^2 +\cdots + (u_{N-1}-v_{N-1})^2 - (u_N - v_N)^2
``` 

```
use hoomd_manifold::Minkowski;
use hoomd_vector::Vector;

let u = Minkowski::from([1.0,0.0,0.0,-1.0]);
let v = Minkowski::from([2.0,0.0,1.0,1.0]);
let w = Minkowski::from([0.0,0.0,0.0,3.0]);
let del = (u+w).distance_squared(&v);
assert_eq!(1.0, del);

```
## Hyperbolic Rotations
"Hyperbolic rotations" describe elements of the group SO(N,1), which preserve hyperboloids 
embedded in [`Minkowski<N+1>`]. Transformations include pure spatial rotations as well as 
"boosts". 

For two-dimensional hyperbolic surfaces, use [`HyperbolicAngle`] to implement 
elements of SO(2,1) which rotate points about the z-axis or boost points along the x- and y-axes.
Use [`HyperbolicRotationMatrix`] to generate the matrix from the values defined by [`HyperbolicAngle`]. 
```
// Rotation about z axis
use hoomd_manifold::{HyperbolicRotationMatrix, Minkowski, HyperbolicRotate, HyperbolicAngle};
use std::f64::consts::PI;

let v = Minkowski::from([1.0, 0.0, 1.0]);
let rotation_about_z = HyperbolicAngle::from((PI/2.0, 0.0_f64, 0.0_f64));
let matrix = HyperbolicRotationMatrix::from(rotation_about_z);
let rotated = matrix.hyperbolic_rotate(&v);
// rotated is approximately [0.0,1.0,1.0]);
```
```
// Boost in the y direction
use hoomd_manifold::{HyperbolicRotationMatrix, Minkowski, HyperbolicRotate, HyperbolicAngle};
use std::f64::consts::PI;

let v = Minkowski::from([1.0, 0.0, 1.0]);
let boost_in_y = HyperbolicAngle::from((0.0_f64, 0.0_f64, 0.5_f64));
let matrix = HyperbolicRotationMatrix::from(boost_in_y);
let boosted = matrix.hyperbolic_rotate(&v);
// rotated is approximately [1.0,sinh(0.5),cosh(0.5)]);
```

For three-dimensional hyperbolic surfaces, use [`Biquaternion`]. Biquaternions are a 
generalization of quaternions which allow for complex coefficients. Unit biquaternions give
 a representation of SO(3,1); this can either be done by converting the biquaternions
 to a [`HyperbolicRotationMatrix`] or by using the ['UnitBiquaternion'] algebra directly. 

 ```math
// Rotate point in 3D hyperbolic space about z axis using matrix representation
use hoomd_manifold::{HyperbolicRotationMatrix, Minkowski, HyperbolicRotate,
                    Biquaternion, UnitBiquaternion};
use std::f64::consts::PI;
use num::complex::Complex;

let q = Biquaternion::from([Complex::new((PI/4.0).sin(),0.0),
                    Complex::new(0.0,0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new((PI/4.0).cos(), 0.0)]);
let v = q.to_unit();
let x = Minkowski::from([0.0, 1.0, 0.0, 1.0]);
let rotation_about_x = HyperbolicRotationMatrix::from(v);
let rotated = rotation_about_x.hyperbolic_rotate(&x);
// rotated vector is approximately [0.0, 0.0, 1.0, 1.0];
``` 
```
// Boost point in 3D hyperbolic space in x direction using biquaternion algebra
use hoomd_manifold::{UnitBiquaternion, HyperbolicRotate, Biquaternion, Minkowski};
use std::f64::consts::PI;
use num::complex::Complex;
use libm::{sinh,cosh};

let x = Minkowski::from([0.0, 0.0, 0.0, 1.0]);
let q = Biquaternion::from([Complex::new(0.0, PI/4.0).sinh(),
                    Complex::new(0.0,0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, PI/4.0).cosh()]);
let v = q.to_unit();
let boosted = v.expect("non-zero biquaternion").hyperbolic_rotate(&x);
// boosted is approximately [(PI/2.0).sinh(), 0.0, 0.0, (PI/2.0).cosh()]
```
*/

mod curved_interaction;
mod sphere;
mod minkowski;
mod hyperbolic_angle;
mod biquaternion;
mod manifold_translate;

pub use {
    minkowski::{Minkowski, Hyperboloid, HyperbolicRotationMatrix, HyperbolicDisk, EightEight},
    hyperbolic_angle::HyperbolicAngle,
    biquaternion::{Biquaternion, UnitBiquaternion},
    manifold_translate::{HyperbolicTranslate, SphericalTranslate},
    curved_interaction::CurvedIsotropic,
    sphere::{Sphere, SphericalDisk},
};

use thiserror::Error;
use hoomd_vector::Vector;

// / Enumerate possible sources of error in fallible vector math operations.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)] 
pub enum Error {
    /// Attempted converting a biquaternion not belonging to the hyperboloid to a 4-vector
    #[error("Biquaternion does not fit required format of [re,re,re,im] to describe a 4-vector")]
    InvalidBiquaternion4Vector,
    
    /// Attempted to normalize a norm zero biquaternion
    #[error("Biquaternion with norm zero cannot be normalized")]
    InvalidBiquaternionMagnitude,

    /// Attempted converting a value to a vector with a dimension not equal to the value's length.
    #[error("source length does not match the target dimensions")]
    InvalidVectorLength,

}

/** Implement methods on non-Euclidean spaces
*/
pub trait CurvedManifold {
    /** Distance of the geodesic path passing through two points on a curved manifold.
    */
    fn geodesic_distance(&self, other: &Self) -> f64;
    /** Cast points in a vector space (i.e., the embedding space) as curved manifold points
    */
    fn to_manifold(point: Vec<f64>) -> Self;
}

/** Operations for the fundamental domain on an arbitrary manifold
*/
pub trait FundamentalDomain {
    /** Distance of the geodesic path passing through a given point on the hyperbola and the
    boundary of the fundamental domain.
    */
    #[inline]
    fn distance_to_boundary(&self) -> f64;
    /** List of points on the boundary of the fundamental domain
    */
    #[inline]
    fn boundary_points(m: usize, skirt: f64) -> Vec::<(f64, f64)>;
}

/** Linear transformations preserving hyperboloids.
 */
pub trait HyperbolicRotate<V: Vector> {
    /// Type of the related rotation matrix
    type Matrix: HyperbolicRotate<V>;
    /** Apply a SO(N-1,1) transformation to an N-dimensional Minkowski vector
    */
    #[must_use]
    fn hyperbolic_rotate(&self, vector: &V) -> V;
}

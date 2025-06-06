// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement vector types on a sphere. 
 */

use std::array;
use libm::acos;
use std::fmt;
use std::iter::{Sum, zip};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use rand::Rng;
use rand::distr::{Distribution, StandardUniform, Uniform};
use crate::{Geodesic,Error};

/** Description of sphere, examples of usage
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere {
    /** Coordinates of a point on a sphere. By assumption, coordinates are listed as 
    (r, theta, phi), where r is the radial component, theta is the zenith angle, and 
    phi is the azimuth.
    */
    pub coordinates: [f64; 3],
    pub radius: f64
}

impl Geodesic for Sphere {
    /** Computes the arc length bewtween two points on a sphere of radius R. The arc length 
    is generally given by R\Delta\psi, where \Delta\psi is the angle between the two points along
    the great circle which intersects both points. For points with coordinates 
    (r_1,theta_1,phi_1) and (r_2,theta_2,phi_2), \Delta\psi between the two points is given by 
    ```math 
    \Delta \psi = \arccos\left(\sin(\theta_1)\sin(\theta_2) + \cos(\theta_1)\cos(\theta_2)\cos(\phi_1-\phi_2)\right)
    ```
    */
    #[inline]
    fn geodesic_distance(&self, other: &Self) -> f64 {
        let arg = self.coordinates[1].sin()*other.coordinates[1].sin() 
                + self.coordinates[1].cos()*other.coordinates[1].cos()*(self.coordinates[2]-other.coordinates[2]);
        let delta_psi = arg.acos();
        self.radius*delta_psi
        // Panic when points do not have the same radii
    }
}
// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement vector types on a sphere. 
 */

use libm::acos;
use hoomd_vector::{Cartesian, InnerProduct};
use crate::{Error, Sphere};

/** Description of sphere, examples of usage
*/

impl<const N: usize> Sphere for Cartesian<N> {
    /** Computes the arc length bewtween two points on an N-sphere of radius R. The arc length 
    is generally given by R\Delta\psi, where \Delta\psi is the angle between the two points along
    the great circle which intersects both points. For two points \vec{u} and \vec{v} on an N-sphere
    embedded in cartesian space, we have 
    ```math 
    \cos(\Delta\psi) = \frac{\vec{u}\cdot\vec{v}}{R}
    ```
    Therefore the arclength between \vec{u} and \vec{v} is given by 
    ```math
    d_S(\vec{u},\vec{v}) = R\delta\psi = R\arccos\left(\frac{\vec{u}\cdot\vec{v}}{R}\right)
    ```
    */
    #[inline]
    fn sphere_distance(&self, other: &Self, radius: f64) -> f64 {
        let arg = Cartesian::dot(self, other) / radius;
        radius * acos(arg)
    }
}
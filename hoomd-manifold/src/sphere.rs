// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement vector types on a sphere. 
 */

use libm::acos;
use hoomd_vector::{Cartesian, InnerProduct};
use crate::{CurvedManifold, Sphere};

/** TODO: Description of sphere, examples of usage
*/

/** [`CurvedManifold`] for Cartesian implements the positively curved metric
*/
impl<const N: usize> CurvedManifold for Cartesian<N> {
    #[inline]
    fn geodesic_distance(&self, other: &Self, rho: f64) -> f64 {
        let arg = Cartesian::dot(self, other) / rho;
        rho * acos(arg)
    }
}

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

    # Example
    ```
    use libm::acos;
    use std::f64::consts::PI;
    use hoomd_vector::{Cartesian, InnerProduct};
    use hoomd_manifold::Sphere;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Cartesian::from([1.0, 0.0, 0.0]);
    let y = Cartesian::from([0.0, 1.0, 1.0]);
    let c = PI/2.0;
    assert_eq!(c,x.sphere_distance(&y,1.0));
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn sphere_distance(&self, other: &Self, radius: f64) -> f64 {
        let arg = Cartesian::dot(self, other) / radius;
        radius * acos(arg)
    }
}
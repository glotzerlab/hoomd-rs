// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement vector types on a sphere. 
 */

use libm::{acos, atan2, sin, cos, sqrt};
use std::f64::consts::PI;
use hoomd_vector::{Cartesian, InnerProduct};
use crate::{CurvedManifold, Sphere};
use hoomd_utility::valid::PositiveReal;
use rand::Rng;
use rand::distr::{Distribution, Uniform};

/** TODO: Description of sphere, examples of usage
*/

/** [`CurvedManifold`] for Cartesian implements the positively curved metric
*/
impl<const N: usize> CurvedManifold for Cartesian<N> {
    #[inline]
    fn geodesic_distance(&self, other: &Self, rho: f64) -> f64 {
        let arg = Cartesian::dot(self, other) / rho.powi(2);
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
        let arg = Cartesian::dot(self, other) / radius.powi(2);
        radius * acos(arg)
    }
    /** Implements a stereographic projection from the N-sphere to an N-dimensional plane. 

    # Example
    ```
    use hoomd_manifold::Sphere;
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let radius = 1.0;
    let x = Cartesian::from([0.5_f64.sqrt(), 0.0, -(0.5_f64.sqrt())]);
    let projection = x.stereographic_projection(radius);
    assert_eq!([1.0/(2.0_f64.sqrt()+ 1.0), 0.0] ,[projection[0],projection[1]]);
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn stereographic_projection(&self, radius: f64) -> Vec<f64> {
        (0..N-1).collect::<Vec<usize>>()
        .iter().map(|i| self.coordinates[*i] / (1.0 - self.coordinates[N-1]/radius)).collect::<Vec<f64>>()
    }
}

/** A uniform distribution of points within distance r of a point on the 2-sphere
with a given radius. 
# Example

```
use hoomd_manifold::Sphere;
use hoomd_vector::{Vector, Cartesian};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand::distr::Distribution;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut rng = StdRng::seed_from_u64(12);

# Ok(())
# }
```
*/
pub struct SphericalDisk {
    /// Max distance away from point
    pub r: PositiveReal,
    /// The center of the disk
    pub point: Cartesian<3>,
    /// The radius of the sphere
    pub radius: f64
}

impl Distribution<Cartesian<3>> for SphericalDisk {
    /** Translates 3-dimensional cartesian vector named "point" along the surface of a sphere by maximum distance of r.
    Note that because SO(3) is non-Abelian, the point must be transformed to the "north pole" before the 
    trial move is applied (and then the point is transformed back). This ensures that the max distance 
    translated by the trial move does not exceed r. 
    */
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Cartesian<3> {
        let radius = self.radius;
        let max_trans = (self.r.get())/radius;
        let point = self.point;
        let phi = atan2(point.coordinates[1], point.coordinates[0]);
        let theta = acos(point.coordinates[2]/radius);
        let trial_zenith = Uniform::new(0.0, 1.0).expect("r is positive and real");
        let trial_azimuth = Uniform::new(-PI, PI).expect("hard-coded distribution should be valid");
        let azi = trial_azimuth.sample(rng);
        let zeni_sample: f64 = trial_zenith.sample(rng);
        let zeni = sqrt(zeni_sample) * max_trans;
        let trial_coords = [radius * sin(zeni) * cos(azi),
                            radius * sin(zeni) * sin(azi),
                            radius * cos(zeni)];
        Cartesian::from([trial_coords[0]*cos(theta)*cos(phi) - trial_coords[1]* sin(phi) + trial_coords[2]*sin(theta)*cos(phi),
                        trial_coords[0]*cos(theta)*sin(phi) + trial_coords[1]* cos(phi) + trial_coords[2]*sin(theta)* sin(phi),
                        -trial_coords[0]*sin(theta) + trial_coords[2]*cos(theta)])
    }
}
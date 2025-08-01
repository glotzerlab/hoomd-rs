// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement vector and curved manifold types on a sphere.
 */

use crate::CurvedManifold;
use approx::assert_relative_eq;
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, InnerProduct, Vector};
use libm::{acos, atan2, cos, sin, sqrt};
use rand::Rng;
use rand::distr::{Distribution, Uniform};
use std::f64::consts::PI;

/** The trait [`Sphere`] for ['Cartesian'] implements types on the embedding of an N-sphere in Euclidean space.
Explicitly, the N-sphere is defined by the set of (N+1)-dimesnional points whose components satisfy
```math
x_1^2 + x_2^2 + \cdots + x_{N+1}^1 = R^2
```
for some radius $R$.
*/

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sphere<const N: usize> {
    /** a cartesian point living on the surface of an N-sphere
     */
    pub point: Cartesian<N>,
    /** the radius of the sphere
     */
    pub radius: f64,
}
impl<const N: usize> Sphere<N> {
    /** Get the coordinates of the point
     */
    pub fn coordinates(&self) -> &[f64; N] {
        &self.point.coordinates
    }
    /** Get the radius of the sphere
     */
    pub fn radius(&self) -> f64 {
        self.radius
    }
    /** Create a sphere point from a cartesian vector
     */
    pub fn from(point: &Cartesian<N>) -> Sphere<N> {
        let radius = point.norm();
        Sphere {
            point: *point,
            radius: radius,
        }
    }
    /** Create a 2-sphere from spherical coordinates
     */
    pub fn from_2_angles(r: f64, theta: f64, phi: f64) -> Sphere<3> {
        let theta_mod = theta.rem_euclid(PI);
        let phi_mod = phi.rem_euclid(2.0 * PI);
        let point = Cartesian::from([
            r * (theta_mod.sin()) * (phi_mod.cos()),
            r * (theta_mod.sin()) * (phi_mod.sin()),
            r * (theta_mod.cos()),
        ]);
        Sphere::from(&point)
    }
    /** Create a 3-sphere from spherical coordinates
     */
    pub fn from_3_angles(r: f64, theta: f64, phi_1: f64, phi_2: f64) -> Sphere<4> {
        let theta_mod = theta.rem_euclid(PI);
        let phi_1_mod = phi_1.rem_euclid(PI);
        let phi_2_mod = phi_2.rem_euclid(2.0 * PI);
        let point = Cartesian::from([
            r * (theta_mod.sin()) * (phi_1_mod.cos()),
            r * (theta_mod.sin()) * (phi_1_mod.sin()) * (phi_2_mod.cos()),
            r * (theta_mod.sin()) * (phi_1_mod.sin()) * (phi_2_mod.sin()),
            r * (theta_mod.cos()),
        ]);
        Sphere::from(&point)
    }
    /** Implements a stereographic projection from the N-sphere to an N-dimensional plane.

    # Example
    ```
    use hoomd_manifold::Sphere;
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let radius = 1.0;
    let x = Cartesian::from([0.5_f64.sqrt(), 0.0, -(0.5_f64.sqrt())]);
    let projection = Sphere::from(&x).stereographic_projection();
    assert_eq!([1.0/(2.0_f64.sqrt()+ 1.0), 0.0] ,[projection[0],projection[1]]);
    # Ok(())
    # }
    ```
    */
    #[inline]
    pub fn stereographic_projection(&self) -> Vec<f64> {
        (0..N - 1)
            .collect::<Vec<usize>>()
            .iter()
            .map(|i| {
                self.point.coordinates[*i] / (1.0 - self.point.coordinates[N - 1] / self.radius)
            })
            .collect::<Vec<f64>>()
    }
}

/** [`CurvedManifold`] for [`Sphere`] computes the geodesic distance between two points on the surface of the sphere.
*/
impl<const N: usize> CurvedManifold for Sphere<N> {
    /** Computes the arc length bewtween two points on an N-sphere of radius R.
    For two points $\vec{u}$ and $\vec{v}$ on an N-sphere
    embedded in cartesian space, the arclength between \vec{u} and \vec{v} is given by
    ```math
    d_S(\vec{u},\vec{v}) = R\delta\psi = R\arccos\left(\frac{\vec{u}\cdot\vec{v}}{R^2}\right)
    ```

    # Example
    ```
    use libm::acos;
    use std::f64::consts::PI;
    use hoomd_vector::{Cartesian, InnerProduct};
    use hoomd_manifold::{CurvedManifold, Sphere};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let radius : f64 = 5.0;
    let x = Sphere::from(&Cartesian::from([radius, 0.0, 0.0]));
    let y = Sphere::from(&Cartesian::from([0.0, radius, 0.0]));
    assert_eq!(radius* PI/2.0, x.geodesic_distance(&y));
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn geodesic_distance(&self, other: &Self) -> f64 {
        assert_relative_eq!(self.radius, other.radius, epsilon = 1e-12);
        let arg = Cartesian::dot(&self.point, &other.point) / self.radius.powi(2);
        self.radius * acos(arg)
    }
    /** Casts a point in (N+1)-dimensional cartesian space to N-dimensional positively-curved space
     */
    #[inline]
    fn to_manifold(point: Vec<f64>) -> Sphere<N> {
        let cartesian_point = Cartesian::<N>::try_from(point);
        match cartesian_point {
            Ok(pt) => Sphere::from(&pt),
            Err(_e) => panic!("point cannot be embedded onto sphere"),
        }
    }
}

/** A uniform distribution of points within distance r of a point on the 2-sphere
with a given radius.

# Example

```
use hoomd_manifold::{CurvedManifold, Sphere, SphericalDisk};
use hoomd_vector::{Vector, Cartesian};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand::distr::Distribution;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let radius : f64 = 1.5;
let mut rng = StdRng::seed_from_u64(12);

// generate a random point
let sample_disk = SphericalDisk{
        r: 0.5_f64.try_into()?,
        point: Cartesian::from([0.01,0.01,-1.0*(radius.powi(2)-2.0*(0.01_f64).powi(2)).sqrt()]),
        radius: radius,};
let random_point: Sphere<3> = sample_disk.sample(&mut rng);

// generate transformation which keeps the distance moved less than 0.1
let disk = SphericalDisk {
    r: 0.1_f64.try_into()?,
    point: random_point.point,
    radius: radius,
};
let transformed_random_point: Sphere<3> = disk.sample(&mut rng);

assert!(0.1 > random_point.geodesic_distance(&transformed_random_point));

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
    pub radius: f64,
}

impl Distribution<Sphere<3>> for SphericalDisk {
    /** Translates 3-dimensional cartesian vector named "point" along the surface of a sphere by maximum distance of r.
    Note that because SO(3) is non-Abelian, the point must be transformed to the "north pole" before the
    trial move is applied (and then the point is transformed back). This ensures that the max distance
    translated by the trial move does not exceed r.
    */
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Sphere<3> {
        let radius = self.radius;
        let max_trans = (self.r.get()) / radius;
        let point = self.point;
        let phi = atan2(point.coordinates[1], point.coordinates[0]);
        let theta = acos(point.coordinates[2] / radius);
        let trial_zenith = Uniform::new(0.0, 1.0).expect("r is positive and real");
        let trial_azimuth = Uniform::new(-PI, PI).expect("hard-coded distribution should be valid");
        let azi = trial_azimuth.sample(rng);
        let zeni_sample: f64 = trial_zenith.sample(rng);
        let zeni = sqrt(zeni_sample) * max_trans;
        let trial_coords = [
            radius * sin(zeni) * cos(azi),
            radius * sin(zeni) * sin(azi),
            radius * cos(zeni),
        ];
        let transformed_point = Cartesian::from([
            trial_coords[0] * cos(theta) * cos(phi) - trial_coords[1] * sin(phi)
                + trial_coords[2] * sin(theta) * cos(phi),
            trial_coords[0] * cos(theta) * sin(phi)
                + trial_coords[1] * cos(phi)
                + trial_coords[2] * sin(theta) * sin(phi),
            -trial_coords[0] * sin(theta) + trial_coords[2] * cos(theta),
        ]);
        let new_sphere = Sphere::from(&transformed_point);
        assert_relative_eq!(radius, new_sphere.radius, epsilon = 1e-12);
        new_sphere
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use paste::paste;
    use rand::{SeedableRng, rngs::StdRng};
    use rstest::rstest;

    /// Generate a pair of points on the surface of a 2-sphere
    fn generate_S2_pair(radius: f64) -> (Sphere<3>, Sphere<3>) {
        (
            Sphere::<3>::from_2_angles(radius, 0.1, 0.3),
            Sphere::<3>::from_2_angles(radius, 1.1, 0.5),
        )
    }
    /// Generate a pair of points on the surface of a 3-sphere
    fn generate_S3_pair(radius: f64) -> (Sphere<4>, Sphere<4>) {
        (
            Sphere::<4>::from_3_angles(radius, 0.2, 0.3, 0.5),
            Sphere::<4>::from_3_angles(radius, 2.3, 1.1, 0.4),
        )
    }

    #[test]
    fn spherical_distance() {
        let (a, b) = generate_S2_pair(1.0);
        let ab_distance = a.geodesic_distance(&b);
        let ab_distance_numeric = 1.002106222125083;
        assert_relative_eq!(ab_distance, ab_distance_numeric, epsilon = 1e-12);

        let (c, d) = generate_S3_pair(1.0);
        let cd_distance = c.geodesic_distance(&d);
        let cd_distance_numeric = 2.1531289007720287;
        assert_relative_eq!(cd_distance, cd_distance_numeric, epsilon = 1e-12);

        let (a, b) = generate_S2_pair(10.0);
        let ab_distance = a.geodesic_distance(&b);
        let ab_distance_numeric = 10.02106222125083;
        assert_relative_eq!(ab_distance, ab_distance_numeric, epsilon = 1e-12);

        let (c, d) = generate_S3_pair(10.0);
        let cd_distance = c.geodesic_distance(&d);
        let cd_distance_numeric = 21.531289007720287;
        assert_relative_eq!(cd_distance, cd_distance_numeric, epsilon = 1e-12);
    }

    #[test]
    fn stereographic() {
        let a = Sphere::<3>::from_2_angles(1.0, 2.1, 1.5);
        let a_projection = a.stereographic_projection();
        let a_projection_numeric = vec![0.04057625219179988, 0.5721827720389171];
        assert_relative_eq![a_projection[0], a_projection_numeric[0], epsilon = 1e-12];
        assert_relative_eq![a_projection[1], a_projection_numeric[1], epsilon = 1e-12];

        let b = Sphere::<4>::from_3_angles(1.0, 2.1, 1.5, 0.5);
        let b_projection = b.stereographic_projection();
        let b_projection_numeric =
            vec![0.04057625219179988, 0.5021376229554481, 0.27431903366480376];
        assert_relative_eq![b_projection[0], b_projection_numeric[0], epsilon = 1e-12];
        assert_relative_eq![b_projection[1], b_projection_numeric[1], epsilon = 1e-12];
        assert_relative_eq![b_projection[2], b_projection_numeric[2], epsilon = 1e-12];
    }

    #[test]
    fn random_sphere() {
        // Generate ten random points on the hyperboloid
        let mut rng = StdRng::seed_from_u64(42);
        let d = 0.1;
        let n_pole = Cartesian::from([0.0, 0.0, 1.0]);
        for _n in 0..10 {
            let disk = SphericalDisk {
                r: d.try_into().expect("hard-coded positive number"),
                point: n_pole,
                radius: 1.0,
            };
            let random_point: Sphere<3> = disk.sample(&mut rng);

            //check that points remain on Sphere
            let rho = random_point.point.norm_squared();
            assert_relative_eq!(rho, 1.0, epsilon = 1e-12);

            //check that points are within distance d of north pole
            let dist = (random_point.point[2].acos()) * (rho.sqrt());
            assert!(d > dist);
        }
    }
}

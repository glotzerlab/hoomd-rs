// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement vector and curved manifold types on a sphere.

use std::f64::consts::PI;

use approxim::{approx_derive::RelativeEq, assert_relative_eq};
use rand::{
    Rng,
    distr::{Distribution, Uniform},
};

use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, InnerProduct, Metric};

/// Point on the surface of a sphere.
///
/// [`Spherical`] is a point on an N-sphere embedded in (N+1)-dimensional
/// euclidean space. Explicitly, the N-sphere is defined by the set of
/// (N+1)-dimensional points whose components satisfy
/// ```math
/// x_1^2 + x_2^2 + \cdots + x_{N+1}^1 = R^2
/// ```
/// for some radius $`R`$.
#[derive(Clone, Copy, Debug, PartialEq, RelativeEq)]
pub struct Spherical<const N: usize> {
    /// a cartesian point living on the surface of an N-sphere
    point: Cartesian<N>,
    /// the radius of the sphere
    radius: f64,
}
impl<const N: usize> Spherical<N> {
    /// Get the coordinates of the point
    #[inline]
    #[must_use]
    pub fn coordinates(&self) -> &[f64; N] {
        &self.point.coordinates
    }
    /// Get the point of the sphere
    #[inline]
    #[must_use]
    pub fn point(&self) -> &Cartesian<N> {
        &self.point
    }
    /// Get the radius of the sphere
    #[inline]
    #[must_use]
    pub fn radius(&self) -> f64 {
        self.radius
    }
    /// Construct a Sphere given a Cartesian vector and a radius.
    ///
    /// # Panics
    ///
    /// Panics when the point is not sufficiently close to the sphere's surface.
    #[inline]
    #[must_use]
    pub fn from_cartesian_coordinates(point: Cartesian<N>, radius: f64) -> Spherical<N> {
        let rad = point.norm();
        assert_relative_eq!(rad, radius, epsilon=1e-12);
        Sphere {
            point,
            radius,
        }
    }

    /// Implements a stereographic projection from the N-sphere to an N-dimensional plane.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Spherical;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let radius = 1.0;
    /// let x = Cartesian::from([0.5_f64.sqrt(), 0.0, -(0.5_f64.sqrt())]);
    /// let projection = Spherical::from_cartesian_coordinates(x, radius)
    ///     .stereographic_projection();
    /// assert_eq!(
    ///     [1.0 / (2.0_f64.sqrt() + 1.0), 0.0],
    ///     [projection[0], projection[1]]
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
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

impl Spherical<3> {
    /// Create a 2-sphere from spherical coordinates
    #[inline]
    #[must_use]
    pub fn from_polar_coordinates(r: f64, theta: f64, phi: f64) -> Spherical<3> {
        let theta_mod = theta.rem_euclid(PI);
        let phi_mod = phi.rem_euclid(2.0 * PI);
        let point = Cartesian::from([
            r * (theta_mod.sin()) * (phi_mod.cos()),
            r * (theta_mod.sin()) * (phi_mod.sin()),
            r * (theta_mod.cos()),
        ]);
        Spherical::from_cartesian_coordinates(point, r)
    }
}

impl Spherical<4> {
    /// Create a 3-sphere from spherical coordinates
    #[inline]
    #[must_use]
    pub fn from_polar_coordinates(r: f64, theta: f64, phi_1: f64, phi_2: f64) -> Spherical<4> {
        let theta_mod = theta.rem_euclid(PI);
        let phi_1_mod = phi_1.rem_euclid(PI);
        let phi_2_mod = phi_2.rem_euclid(2.0 * PI);
        let point = Cartesian::from([
            r * (theta_mod.sin()) * (phi_1_mod.cos()),
            r * (theta_mod.sin()) * (phi_1_mod.sin()) * (phi_2_mod.cos()),
            r * (theta_mod.sin()) * (phi_1_mod.sin()) * (phi_2_mod.sin()),
            r * (theta_mod.cos()),
        ]);
        Spherical::from_cartesian_coordinates(point, r)
    }
}

impl Metric for Spherical<3> {
    /// The distance between two [`Spherical<3>`] points.
    ///
    /// Explicitly, the metric for two points $`\vec{u}`$ and $`\vec{v}`$ on a
    /// 2-sphere with radius $`R`$ is given by
    ///
    /// ```math
    /// d_{S_2}(\vec{u}, \vec{v}) = R \arccos\left[\frac{1}{R^2}(u_1v_1 + u_2v_2 + u_3v_3)\right]
    /// ```
    /// This choice of metric furnishes a representation of 2-dimensional spherical
    /// space with Gaussian curvature $`K = 1/R^2`$.
    #[inline]
    fn distance(&self, other: &Self) -> f64 {
        assert_eq!(
            self.radius, other.radius,
            "points must be on the same sphere"
        );
        let arg = Cartesian::dot(&self.point, &other.point) / self.radius.powi(2);
        self.radius * (arg.acos())
    }
    #[inline]
    fn distance_squared(&self, other: &Self) -> f64 {
        (self.distance(other)).powi(2)
    }
    #[inline]
    fn n_dimensions(&self) -> usize {
        2_usize
    }
}

impl Metric for Spherical<4> {
    /// The distance between two [`Spherical<4>`] points.
    ///
    /// Explicitly, the
    /// metric for two points $`\vec{u}`$ and $`\vec{v}`$ on a 3-sphere with
    /// radius  $`R`$ is given by
    ///
    /// ```math
    /// d_{S_3}(\vec{u}, \vec{v}) = R \arccos\left[\frac{1}{R^2}(u_1v_1 + u_2v_2 + u_3v_3 + u_4v_4)\right]
    /// ```
    /// This choice of metric furnishes a representation of 3-dimensional spherical
    /// space with Gaussian curvature $`K = 1/R^2`$.
    #[inline]
    fn distance(&self, other: &Self) -> f64 {
        assert_eq!(
            self.radius, other.radius,
            "points must be on the same sphere"
        );
        let arg = Cartesian::dot(&self.point, &other.point) / self.radius.powi(2);
        self.radius * (arg.acos())
    }
    #[inline]
    fn distance_squared(&self, other: &Self) -> f64 {
        (self.distance(other)).powi(2)
    }
    #[inline]
    fn n_dimensions(&self) -> usize {
        3_usize
    }
}

/// Randomly distribute points locally on a sphere.
///
/// [`SphericalDisk`] is a uniform distribution of points within distance `r` of
/// a point on the 2-sphere with a given radius.
///
/// # Example
///
/// ```
/// use hoomd_manifold::{Spherical, SphericalDisk};
/// use hoomd_vector::{Cartesian, Metric};
/// use rand::{Rng, SeedableRng, distr::Distribution, rngs::StdRng};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let radius: f64 = 1.5;
/// let mut rng = StdRng::seed_from_u64(12);
///
/// let sample_disk = SphericalDisk {
///     disk_radius: 0.5_f64.try_into()?,
///     point: Spherical::<3>::from_cartesian_coordinates(
///         Cartesian::from([
///             0.01,
///             0.01,
///             -(radius.powi(2) - 2.0 * (0.01_f64).powi(2)).sqrt(),
///         ]),
///         radius,
///     ),
/// };
/// let random_point: Spherical<3> = sample_disk.sample(&mut rng);
///
/// let disk = SphericalDisk {
///     disk_radius: 0.1_f64.try_into()?,
///     point: random_point,
/// };
/// let transformed_random_point: Spherical<3> = disk.sample(&mut rng);
///
/// assert!(0.1 > random_point.distance(&transformed_random_point));
///
/// # Ok(())
/// # }
/// ```
pub struct SphericalDisk {
    /// Max distance away from point.
    pub disk_radius: PositiveReal,
    /// The center of the disk.
    pub point: Spherical<3>,
}

impl<const N: usize> Default for Spherical<N> {
    #[inline]
    fn default() -> Self {
        let mut zero = Cartesian::<N>::default();
        zero.coordinates[N - 1] = 1.0;
        Spherical {
            point: zero,
            radius: 1.0_f64,
        }
    }
}

impl Distribution<Spherical<3>> for SphericalDisk {
    /// Translates 3-dimensional cartesian vector named "point" along the
    /// surface of a sphere by maximum distance of r.
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Spherical<3> {
        let radius = self.point.radius;
        let max_trans = (self.disk_radius.get()) / radius;
        let point = self.point;
        let phi = point.point.coordinates[1].atan2(point.point.coordinates[0]);
        let theta = (point.point.coordinates[2] / radius).acos();
        let trial_zenith = Uniform::new(0.0, 1.0).expect("r is positive and real");
        let trial_azimuth = Uniform::new(-PI, PI).expect("hard-coded distribution should be valid");
        let azi = trial_azimuth.sample(rng);
        let zeni_sample: f64 = trial_zenith.sample(rng);
        let zeni = (zeni_sample).sqrt() * max_trans;
        let trial_coords = [
            radius * (zeni.sin()) * (azi.cos()),
            radius * (zeni.sin()) * (azi.sin()),
            radius * (zeni.cos()),
        ];
        let transformed_point = Cartesian::from([
            trial_coords[0] * (theta.cos()) * (phi.cos()) - trial_coords[1] * (phi.sin())
                + trial_coords[2] * (theta.sin()) * (phi.cos()),
            trial_coords[0] * (theta.cos()) * (phi.sin())
                + trial_coords[1] * (phi.cos())
                + trial_coords[2] * (theta.sin()) * (phi.sin()),
            -trial_coords[0] * (theta.sin()) + trial_coords[2] * (theta.cos()),
        ]);
        Spherical::from_cartesian_coordinates(transformed_point, radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use rand::{SeedableRng, rngs::StdRng};

    /// Generate a pair of points on the surface of a 2-sphere
    fn generate_s2_pair(radius: f64) -> (Spherical<3>, Spherical<3>) {
        (
            Spherical::<3>::from_polar_coordinates(radius, 0.1, 0.3),
            Spherical::<3>::from_polar_coordinates(radius, 1.1, 0.5),
        )
    }
    /// Generate a pair of points on the surface of a 3-sphere
    fn generate_s3_pair(radius: f64) -> (Spherical<4>, Spherical<4>) {
        (
            Spherical::<4>::from_polar_coordinates(radius, 0.2, 0.3, 0.5),
            Spherical::<4>::from_polar_coordinates(radius, 2.3, 1.1, 0.4),
        )
    }

    #[test]
    fn spherical_distance() {
        let (a, b) = generate_s2_pair(1.0);
        let ab_distance = a.distance(&b);
        let ab_distance_numeric = 1.002_106_222_125_083;
        assert_relative_eq!(ab_distance, ab_distance_numeric, epsilon = 1e-12);

        let (c, d) = generate_s3_pair(1.0);
        let cd_distance = c.distance(&d);
        let cd_distance_numeric = 2.153_128_900_772_028;
        assert_relative_eq!(cd_distance, cd_distance_numeric, epsilon = 1e-12);

        let (a, b) = generate_s2_pair(10.0);
        let ab_distance = a.distance(&b);
        let ab_distance_numeric = 10.021_062_221_250_83;
        assert_relative_eq!(ab_distance, ab_distance_numeric, epsilon = 1e-12);

        let (c, d) = generate_s3_pair(10.0);
        let cd_distance = c.distance(&d);
        let cd_distance_numeric = 21.531_289_007_720_287;
        assert_relative_eq!(cd_distance, cd_distance_numeric, epsilon = 1e-12);
    }

    #[test]
    fn stereographic() {
        let a = Spherical::<3>::from_polar_coordinates(1.0, 2.1, 1.5);
        let a_projection = a.stereographic_projection();
        let a_projection_numeric = [0.040_576_252_191_799_88, 0.572_182_772_038_917_1];
        assert_relative_eq![a_projection[0], a_projection_numeric[0], epsilon = 1e-12];
        assert_relative_eq![a_projection[1], a_projection_numeric[1], epsilon = 1e-12];

        let b = Spherical::<4>::from_polar_coordinates(1.0, 2.1, 1.5, 0.5);
        let b_projection = b.stereographic_projection();
        let b_projection_numeric = [
            0.040_576_252_191_799_88,
            0.502_137_622_955_448_1,
            0.274_319_033_664_803_76,
        ];
        assert_relative_eq![b_projection[0], b_projection_numeric[0], epsilon = 1e-12];
        assert_relative_eq![b_projection[1], b_projection_numeric[1], epsilon = 1e-12];
        assert_relative_eq![b_projection[2], b_projection_numeric[2], epsilon = 1e-12];
    }

    #[test]
    fn random_sphere() {
        // Generate ten random points on the Hyperbolic
        let mut rng = StdRng::seed_from_u64(42);
        let d = 0.1;
        let n_pole = Cartesian::from([0.0, 0.0, 1.0]);
        for _n in 0..10 {
            let disk = SphericalDisk {
                disk_radius: d.try_into().expect("hard-coded positive number"),
                point: Spherical::<3>::from_cartesian_coordinates(n_pole, 1.0),
            };
            let random_point: Spherical<3> = disk.sample(&mut rng);

            // check that points remain on Sphere
            let rho = random_point.point.norm_squared();
            assert_relative_eq!(rho, 1.0, epsilon = 1e-12);

            // check that points are within distance d of north pole
            let distance = (random_point.point[2].acos()) * (rho.sqrt());
            assert!(d > distance);
        }
    }
}

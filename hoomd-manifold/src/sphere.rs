// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement vector and curved manifold types on a sphere.

use approxim::{approx_derive::RelativeEq, assert_relative_eq};
use rand::{
    Rng, RngExt,
    distr::{Distribution, Uniform},
};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, InnerProduct, Metric, Quaternion, Rotate, Versor};

/// Point on the surface of a sphere.
///
/// [`Spherical`] is a point on a unit N-sphere embedded in (N+1)-dimensional
/// euclidean space. Explicitly, the N-sphere is defined by the set of
/// (N+1)-dimensional points whose components satisfy
/// ```math
/// x_1^2 + x_2^2 + \cdots + x_{N+1}^1 = 1.0
/// ```
/// Note that the radius is fixed to be 1.0.
#[derive(Clone, Copy, Debug, PartialEq, RelativeEq, Serialize, Deserialize)]
pub struct Spherical<const N: usize> {
    /// a cartesian point living on the surface of an N-sphere
    point: Cartesian<N>,
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
    /// Construct a Sphere given a Cartesian vector and a radius.
    ///
    /// # Panics
    ///
    /// Panics when the point is not sufficiently close to the sphere's surface.
    #[inline]
    #[must_use]
    pub fn from_cartesian_coordinates(point: Cartesian<N>) -> Spherical<N> {
        let rad = point.norm();
        assert_relative_eq!(rad, 1.0_f64, epsilon = 1e-6);
        Spherical { point }
    }
    /// Implements a stereographic projection from the N-sphere to an n-dimensional
    /// plane by projecting through the $`(0,\cdots, 0,1)`$ axis.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Spherical;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let x = Cartesian::from([0.5_f64.sqrt(), 0.0, -(0.5_f64.sqrt())]);
    /// let projection =
    ///     Spherical::from_cartesian_coordinates(x).stereographic_projection();
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
            .map(|i| self.point.coordinates[*i] / (1.0 - self.point.coordinates[N - 1]))
            .collect::<Vec<f64>>()
    }
}

impl Spherical<3> {
    /// Create a 2-sphere from spherical coordinates
    #[inline]
    #[must_use]
    pub fn from_polar_coordinates(theta: f64, phi: f64) -> Spherical<3> {
        let theta_mod = theta.rem_euclid(PI);
        let phi_mod = phi.rem_euclid(2.0 * PI);
        let point = Cartesian::from([
            (theta_mod.sin()) * (phi_mod.cos()),
            (theta_mod.sin()) * (phi_mod.sin()),
            (theta_mod.cos()),
        ]);
        Spherical::from_cartesian_coordinates(point)
    }
}

impl Spherical<4> {
    /// Create a point on a 3-sphere from spherical coordinates. Note that this uses
    /// the convention
    /// ```math
    /// \begin{pmatrix}
    /// \sin\theta \cos\phi_1
    /// \\ \sin\theta \sin\phi_1 \cos\phi_2
    /// \\ \sin\theta \sin\phi_1 \sin\phi_2
    /// \\ \cos\theta
    /// \end{pmatrix}
    /// ```
    /// where $`\theta`$ and $`phi_1`$ both run over the range $`0`$ to $`\pi`$ and $`\phi_2`$
    /// runs from $`0`$ to $`2\pi`$.
    #[inline]
    #[must_use]
    pub fn from_polar_coordinates(theta: f64, phi_1: f64, phi_2: f64) -> Spherical<4> {
        let theta_mod = theta.rem_euclid(PI);
        let phi_1_mod = phi_1.rem_euclid(PI);
        let phi_2_mod = phi_2.rem_euclid(2.0 * PI);
        let point = Cartesian::from([
            (theta_mod.sin()) * (phi_1_mod.cos()),
            (theta_mod.sin()) * (phi_1_mod.sin()) * (phi_2_mod.cos()),
            (theta_mod.sin()) * (phi_1_mod.sin()) * (phi_2_mod.sin()),
            theta_mod.cos(),
        ]);
        Spherical::from_cartesian_coordinates(point)
    }
    /// Create a point on a unit-radius 3-sphere from a unit quaternion.
    #[inline]
    #[must_use]
    pub fn from_versor(versor: Versor) -> Spherical<4> {
        let q = versor.get();
        let (a,b,c,d) = (q.scalar, q.vector[0], q.vector[1], q.vector[2]);
        Spherical::<4>::from_cartesian_coordinates(Cartesian::from([a, b, c, d]))
    }
    /// Create a versor which maps $`(1,0,0,0)`$ to the target `Spherical<4>` point.
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_manifold::Spherical;
    /// use hoomd_vector::{Cartesian, Quaternion, Versor};
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let radius = 1.0;
    /// let x = Spherical::<4>::from_polar_coordinates(
    ///     PI / 4.0,
    ///     PI / 8.0,
    ///     5.0 * PI / 4.0,
    /// );
    /// let x_versor = x.to_versor();
    /// let pole_versor = Quaternion::from([1.0, 0.0, 0.0, 0.0])
    ///     .to_versor()
    ///     .expect("not a null vector");
    /// let transformation =
    ///     (*x_versor.get() * *pole_versor.get() * *x_versor.get())
    ///         .to_versor()
    ///         .expect("Hard-coded example is valid");
    /// let mapped_pole = Spherical::<4>::from_versor(transformation);
    /// assert_relative_eq!(
    ///     mapped_pole.coordinates()[0],
    ///     x.coordinates()[0],
    ///     epsilon = 1e-12
    /// );
    /// assert_relative_eq!(
    ///     mapped_pole.coordinates()[1],
    ///     x.coordinates()[1],
    ///     epsilon = 1e-12
    /// );
    /// assert_relative_eq!(
    ///     mapped_pole.coordinates()[2],
    ///     x.coordinates()[2],
    ///     epsilon = 1e-12
    /// );
    /// assert_relative_eq!(
    ///     mapped_pole.coordinates()[3],
    ///     x.coordinates()[3],
    ///     epsilon = 1e-12
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn to_versor(&self) -> Versor {
        let phi = self.coordinates()[3].atan2(self.coordinates()[2]);
        let theta = ((self.coordinates()[3].powi(2) + self.coordinates()[2].powi(2)).sqrt())
            .atan2(self.coordinates()[1]);
        let psi = ((self.coordinates()[3].powi(2)
            + self.coordinates()[2].powi(2)
            + self.coordinates()[1].powi(2))
        .sqrt())
        .atan2(self.coordinates()[0]);
        let n_hat = Cartesian::from([
            theta.cos(),
            (theta.sin()) * (phi.cos()),
            (theta.sin()) * (phi.sin()),
        ])
        .to_unit_unchecked();
        Versor::from_axis_angle(n_hat.0, psi)
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
        let arg = Cartesian::dot(&self.point, &other.point);
        let arg_clipped = arg.clamp(-1.0, 1.0);
        arg_clipped.acos()
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
        let arg = Cartesian::dot(&self.point, &other.point);
        let arg_clipped = arg.clamp(-1.0, 1.0);
        arg_clipped.acos()
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
/// let mut rng = StdRng::seed_from_u64(12);
///
/// let sample_disk =
///     SphericalDisk {
///         disk_radius: 0.5_f64.try_into()?,
///         point: Spherical::<3>::from_cartesian_coordinates(Cartesian::from(
///             [0.01, 0.01, -(1.0 - 2.0 * (0.01_f64).powi(2)).sqrt()],
///         )),
///     };
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SphericalDisk<const N: usize> {
    /// Max distance away from point.
    pub disk_radius: PositiveReal,
    /// The center of the disk.
    pub point: Spherical<N>,
}

impl<const N: usize> Default for Spherical<N> {
    #[inline]
    fn default() -> Self {
        let mut zero = Cartesian::<N>::default();
        zero.coordinates[N - 1] = 1.0;
        Spherical { point: zero }
    }
}

impl Distribution<Spherical<3>> for SphericalDisk<3> {
    /// Translates 3-dimensional cartesian vector named "point" along the
    /// surface of a sphere by maximum distance of r.
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Spherical<3> {
        let max_angle = self.disk_radius.get();
        let p = self.point.point.coordinates;
        let p_norm = self.point.point.norm();
        let p_hat = [p[0] / p_norm, p[1] / p_norm, p[2] / p_norm];
        // Build an ON tangent frame (e1, e2) not parallel to p_hat
        let tmp = if p_hat[2].abs() < 0.9 {
            [0.0, 0.0, 1.0]
        } else {
            [1.0, 0.0, 0.0]
        };

        // e1 = normalized cross product of tmp and p_hat
        let e1 = [
            tmp[1] * p_hat[2] - tmp[2] * p_hat[1],
            tmp[2] * p_hat[0] - tmp[0] * p_hat[2],
            tmp[0] * p_hat[1] - tmp[1] * p_hat[0],
        ];
        let e1_norm = (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]).sqrt();
        let e1 = [e1[0] / e1_norm, e1[1] / e1_norm, e1[2] / e1_norm];
        // e2 = cross product of p_hat and e1
        let e2 = [
            p_hat[1] * e1[2] - p_hat[2] * e1[1],
            p_hat[2] * e1[0] - p_hat[0] * e1[2],
            p_hat[0] * e1[1] - p_hat[1] * e1[0],
        ];
        // sample direction in tangent plane: dir = cos(phi) * e1 + sin(phi) * e2
        let phi = Uniform::new(0.0, 2.0 * PI)
            .expect("u is positive")
            .sample(rng);
        let dir = [
            phi.cos() * e1[0] + phi.sin() * e2[0],
            phi.cos() * e1[1] + phi.sin() * e2[1],
            phi.cos() * e1[2] + phi.sin() * e2[2],
        ];
        // sample geodesic distance with area weighting
        let u: f64 = Uniform::new(0.0, 1.0).expect("u is positive").sample(rng);
        let cos_theta = 1.0 - u * (1.0 - max_angle.cos());
        let sin_theta = (cos_theta.acos()).sin();
        // exponential map exp_p(v) = R\cos(v)*p_hat + R\sin(v)*v_hat
        let new_point = [
            cos_theta * p_hat[0] + sin_theta * dir[0],
            cos_theta * p_hat[1] + sin_theta * dir[1],
            cos_theta * p_hat[2] + sin_theta * dir[2],
        ];

        Spherical::from_cartesian_coordinates(Cartesian::from(new_point))
    }
}

impl Distribution<Spherical<4>> for SphericalDisk<4> {
    /// Translates 3-dimensional cartesian vector named "point" along the
    /// surface of a sphere by maximum distance of r.
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Spherical<4> {
        let max_trans = self.disk_radius.get();
        let point = self.point;
        // generate random unit cartesian vector
        let v: Versor = rng.random();
        let b_hat = v
            .rotate(&Cartesian::from([1.0, 0.0, 0.0]))
            .to_unit()
            .expect("hard coded non-null vector");
        let eta = Uniform::new(0.0, max_trans).expect("hard coded non-negative");
        let translation_versor = Versor::from_axis_angle(b_hat.0, eta.sample(rng));

        let position_versor = Quaternion::from(*point.coordinates())
            .to_versor()
            .expect("spherical points cannot be null");
        let transformation =
            ((*translation_versor.get()) * (*position_versor.get()) * (*translation_versor.get()))
                .to_versor()
                .expect("spherical points cannot be null");
        let sphere_point = Spherical::<4>::from_versor(transformation);
        Spherical::<4>::from_cartesian_coordinates(*sphere_point.point())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use rand::{SeedableRng, rngs::StdRng};

    /// Generate a pair of points on the surface of a 2-sphere
    fn generate_s2_pair() -> (Spherical<3>, Spherical<3>) {
        (
            Spherical::<3>::from_polar_coordinates(0.1, 0.3),
            Spherical::<3>::from_polar_coordinates(1.1, 0.5),
        )
    }
    /// Generate a pair of points on the surface of a 3-sphere
    fn generate_s3_pair() -> (Spherical<4>, Spherical<4>) {
        (
            Spherical::<4>::from_polar_coordinates(0.2, 0.3, 0.5),
            Spherical::<4>::from_polar_coordinates(2.3, 1.1, 0.4),
        )
    }

    #[test]
    fn spherical_distance() {
        let (a, b) = generate_s2_pair();
        let ab_distance = a.distance(&b);
        let ab_distance_numeric = 1.002_106_222_125_083;
        assert_relative_eq!(ab_distance, ab_distance_numeric, epsilon = 1e-12);

        let (c, d) = generate_s3_pair();
        let cd_distance = c.distance(&d);
        let cd_distance_numeric = 2.153_128_900_772_028;
        assert_relative_eq!(cd_distance, cd_distance_numeric, epsilon = 1e-12);
    }

    #[test]
    fn stereographic() {
        let a = Spherical::<3>::from_polar_coordinates(2.1, 1.5);
        let a_projection = a.stereographic_projection();
        let a_projection_numeric = [0.040_576_252_191_799_88, 0.572_182_772_038_917_1];
        assert_relative_eq![a_projection[0], a_projection_numeric[0], epsilon = 1e-12];
        assert_relative_eq![a_projection[1], a_projection_numeric[1], epsilon = 1e-12];

        let b = Spherical::<4>::from_polar_coordinates(2.1, 1.5, 0.5);
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
    fn random_two_sphere() {
        // Generate ten random points on the Hyperbolic
        let mut rng = StdRng::seed_from_u64(42);
        let d = 0.1;
        let n_pole = Cartesian::from([0.0, 0.0, 1.0]);
        for _n in 0..10 {
            let disk = SphericalDisk {
                disk_radius: d.try_into().expect("hard-coded positive number"),
                point: Spherical::<3>::from_cartesian_coordinates(n_pole),
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
    #[test]
    fn random_three_sphere() {
        // Generate ten random points on the Hyperbolic
        let mut rng = StdRng::seed_from_u64(42);
        let d = 0.1;
        let n_pole = Spherical::from_cartesian_coordinates(Cartesian::from([1.0, 0.0, 0.0, 0.0]));
        for _n in 0..10 {
            let disk = SphericalDisk {
                disk_radius: d.try_into().expect("hard-coded positive number"),
                point: n_pole,
            };
            let random_point: Spherical<4> = disk.sample(&mut rng);

            // check that points remain on Sphere
            let rho = random_point.point.norm_squared();
            assert_relative_eq!(rho, 1.0, epsilon = 1e-12);

            // check that points are within distance d of north pole
            let distance = random_point.point().distance(n_pole.point());
            assert!(d > distance);
        }
    }
    #[test]
    fn from_versor() {
        // generate a 3-sphere point from a versor
        let mut rng = StdRng::seed_from_u64(358);
        let v: Versor = rng.random();
        let sphere_pt = Spherical::<4>::from_versor(v);
        assert_eq!(sphere_pt.coordinates()[0], v.get().scalar);
        assert_eq!(sphere_pt.coordinates()[1], v.get().vector.coordinates[0]);
        assert_eq!(sphere_pt.coordinates()[2], v.get().vector.coordinates[1]);
        assert_eq!(sphere_pt.coordinates()[3], v.get().vector.coordinates[2]);
    }
    #[test]
    fn to_versor() {
        // generate a 3-sphere point from a versor
        let mut rng = StdRng::seed_from_u64(5121);
        let v: Versor = rng.random();
        let sphere_pt = Spherical::<4>::from_versor(v);
        // get versor which maps identity versor to 3-sphere point
        let versor_from_spherical = sphere_pt.to_versor();
        let pole_versor = Quaternion::from([1.0, 0.0, 0.0, 0.0])
            .to_versor()
            .expect("not a null vector");
        let transformation = (*versor_from_spherical.get() * *pole_versor.get() * *versor_from_spherical.get())
            .to_versor()
            .expect("Hard-coded example is valid");
        let q = transformation.get();
        assert_relative_eq!(q.scalar, v.get().scalar,epsilon=1e-12);
        assert_relative_eq!(q.vector.coordinates[0], v.get().vector.coordinates[0],epsilon=1e-12);
        assert_relative_eq!(q.vector.coordinates[1], v.get().vector.coordinates[1],epsilon=1e-12);
        assert_relative_eq!(q.vector.coordinates[2], v.get().vector.coordinates[2],epsilon=1e-12);
    }
}

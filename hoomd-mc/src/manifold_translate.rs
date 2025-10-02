// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Translation moves on curved surfaces

use crate::{LocalTrial, Translate};
use hoomd_manifold::{HyperbolicDisk, Hyperboloid, Minkowski, Sphere, SphericalDisk};
use hoomd_microstate::property::Position;
use hoomd_vector::{Cartesian, InnerProduct};

use rand::{Rng, distr::Distribution};

impl<B> LocalTrial<Hyperboloid<3>, B> for Translate
where
    B: Position<Position=Hyperboloid<3>>,
{
    /// Propose local trial moves for a body on a hyperbolic surface.
    ///
    /// # Example
    /// ```
    /// use approx::assert_relative_eq;
    /// use hoomd_manifold::{Hyperboloid, Minkowski};
    /// use hoomd_mc::{Translate, LocalTrial};
    /// use hoomd_microstate::property::{Point, Position};
    /// use hoomd_vector::{Metric, Vector};
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut rng = StdRng::seed_from_u64(13);
    /// let rho: f64 = 0.8;
    /// let body_properties = Point::new(Hyperboloid::from(Minkowski::from([
    ///     1.0,
    ///     -1.0,
    ///     (2.0 + rho.powi(2)).sqrt(),
    /// ]), rho));
    /// let d = 0.1 * rho;
    /// let hyperbolic_translate = Translate {
    ///     maximum_distance: d.try_into()?,
    /// };
    ///
    /// let new_body_properties =
    ///     hyperbolic_translate.propose(&mut rng, body_properties);
    ///
    /// // Translation move keeps the point on the hyperboloid
    /// assert_relative_eq!(
    ///     new_body_properties
    ///         .position()
    ///         .point()
    ///         .distance_squared(&Minkowski::from([0.0, 0.0, 0.0])),
    ///     -(rho.powi(2)),
    ///     epsilon = 1e-12
    /// );
    ///
    /// // Translation move does not move the point more than a distance d
    /// assert!(
    ///     d > new_body_properties.position().distance(&Hyperboloid::from(
    ///         Minkowski::from([1.0, -1.0, (2.0 + rho.powi(2)).sqrt()]), rho)
    ///     )
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn propose<R: Rng>(&self, rng: &mut R, body_properties: B) -> B {
        let mut trial = body_properties;
        let rho = trial.position().skirt();
        let disk = HyperbolicDisk {
            r: self.maximum_distance,
            point: *trial.position_mut().point(),
            skirt: rho,
        };
        let trial_sample = disk.sample(rng);
        // push point back onto hyperboloid
        *trial.position_mut() = Hyperboloid::from(
            Minkowski::from([
                trial_sample.coordinates()[0],
                trial_sample.coordinates()[1],
                (trial_sample.point()[0].powi(2)
                    + trial_sample.point()[1].powi(2)
                    + trial.position().skirt().powi(2))
                .sqrt()]),
            rho
        );
        trial
    }
}

impl<B> LocalTrial<Sphere<3>, B> for Translate
where
    B: Position<Position = Sphere<3>>,
{
    /// Propose local trial moves for a body on the surface of a sphere
    ///
    /// # Example
    /// ```
    /// use approx::assert_relative_eq;
    /// use hoomd_manifold::{Sphere, SphericalDisk};
    /// use hoomd_mc::{LocalTrial, Translate};
    /// use hoomd_microstate::property::{Point, Position};
    /// use hoomd_vector::{Cartesian, Metric, Vector};
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut rng = StdRng::seed_from_u64(14);
    /// let radius: f64 = 2.0;
    /// let initial_point =
    ///     Point::new(
    ///         Sphere::from_cartesian_coordinates(
    ///             Cartesian::from(
    ///                 [2.0_f64.sqrt(),
    ///                 2.0_f64.sqrt(),
    ///                 0.0]
    ///             ),
    ///             2.0_f64
    ///         )
    ///     );
    /// let d = 0.1;
    /// let spherical_translate = Translate {
    ///     maximum_distance: d.try_into()?,
    /// };
    ///
    /// let new_body_properties =
    ///     spherical_translate.propose(&mut rng, initial_point);
    ///
    /// // Translation move keeps point on the surface of the sphere
    /// assert_relative_eq!(
    ///     new_body_properties.position().radius(),
    ///     radius,
    ///     epsilon = 1e-12
    /// );
    ///
    /// // Translation move does not translate the point more than a distance d away
    /// assert!(
    ///     d > new_body_properties.position().distance(&initial_point.position())
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn propose<R: Rng>(&self, rng: &mut R, body_properties: B) -> B {
        let mut trial = body_properties;
        let disk = SphericalDisk {
            r: self.maximum_distance,
            point: *trial.position_mut().point(),
            radius: trial.position().radius(),
        };
        *trial.position_mut() = disk.sample(rng);
        let rescale = trial.position().radius() / trial.position().point().norm();
        *trial.position_mut() = Sphere::from_cartesian_coordinates(
            Cartesian::from([
                rescale*trial.position().coordinates()[0],
                rescale*trial.position().coordinates()[1],
                rescale*trial.position().coordinates()[2],
            ]),
            trial.position().radius());
        trial
    }
}

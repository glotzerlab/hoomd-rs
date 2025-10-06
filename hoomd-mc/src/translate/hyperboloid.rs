// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Translation moves on curved surfaces

use rand::{Rng, distr::Distribution};

use crate::{LocalTrial, Translate};
use hoomd_manifold::{HyperbolicDisk, Hyperboloid, Minkowski};
use hoomd_microstate::property::Position;

impl<B> LocalTrial<B> for Translate<Hyperboloid<3>>
where
    B: Position<Position = Hyperboloid<3>>,
{
    /// Propose local trial moves for a body on a hyperbolic surface.
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_manifold::{Hyperboloid, Minkowski};
    /// use hoomd_mc::{LocalTrial, Translate};
    /// use hoomd_microstate::property::{Point, Position};
    /// use hoomd_vector::{Metric, Vector};
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut rng = StdRng::seed_from_u64(13);
    /// let rho: f64 = 0.8;
    /// let body_properties = Point::new(Hyperboloid::from_minkowski_coordinates(
    ///     [1.0, -1.0, (2.0 + rho.powi(2)).sqrt()].into(),
    ///     rho,
    /// ));
    /// let d = 0.1 * rho;
    /// let translate = Translate::with_maximum_distance(d.try_into()?);
    ///
    /// let new_body_properties = translate.propose(&mut rng, body_properties);
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
    ///     d > new_body_properties.position().distance(
    ///         &Hyperboloid::from_minkowski_coordinates(
    ///             Minkowski::from([1.0, -1.0, (2.0 + rho.powi(2)).sqrt()]),
    ///             rho
    ///         )
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
            disk_radius: *self.maximum_distance(),
            point: *trial.position_mut()
        };
        let trial_sample = disk.sample(rng);
        // push point back onto hyperboloid
        *trial.position_mut() = Hyperboloid::from_minkowski_coordinates(
            Minkowski::from([
                trial_sample.coordinates()[0],
                trial_sample.coordinates()[1],
                (trial_sample.point()[0].powi(2)
                    + trial_sample.point()[1].powi(2)
                    + trial.position().skirt().powi(2))
                .sqrt(),
            ]),
            rho,
        );
        trial
    }
}

// TODO: test

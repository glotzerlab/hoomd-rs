// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Translation moves on curved surfaces

use rand::{Rng, distr::Distribution};

use crate::{LocalTrial, Translate};
use hoomd_manifold::{Hyperbolic, HyperbolicDisk, Minkowski};
use hoomd_microstate::property::{Orientation, OrientedHyperbolicPoint, Point, Position};
use hoomd_vector::Angle;

impl LocalTrial<Point<Hyperbolic<3>>> for Translate<Point<Hyperbolic<3>>> {
    /// Propose local trial moves for a body on a hyperbolic surface.
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_manifold::{Hyperbolic, Minkowski};
    /// use hoomd_mc::{LocalTrial, Translate};
    /// use hoomd_microstate::property::{Point, Position};
    /// use hoomd_vector::Metric;
    /// use rand::{SeedableRng, rngs::StdRng};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut rng = StdRng::seed_from_u64(13);
    /// let body_properties = Point::new(Hyperbolic::from_minkowski_coordinates(
    ///     [1.0, -1.0, (3.0_f64).sqrt()].into(),
    /// ));
    /// let d = 0.1;
    /// let translate = Translate::with_maximum_distance(d.try_into()?);
    ///
    /// let new_body_properties = translate.propose(&mut rng, body_properties);
    ///
    /// // Translation move keeps the point on the Hyperboloid
    /// assert_relative_eq!(
    ///     new_body_properties
    ///         .position()
    ///         .point()
    ///         .distance_squared(&Minkowski::from([0.0, 0.0, 0.0])),
    ///     -1.0_f64,
    ///     epsilon = 1e-12
    /// );
    ///
    /// // Translation move does not move the point more than a distance d
    /// assert!(
    ///     d > new_body_properties.position().distance(
    ///         &Hyperbolic::from_minkowski_coordinates(Minkowski::from([
    ///             1.0,
    ///             -1.0,
    ///             (3.0_f64).sqrt()
    ///         ]))
    ///     )
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn propose<R: Rng>(
        &self,
        rng: &mut R,
        body_properties: Point<Hyperbolic<3>>,
    ) -> Point<Hyperbolic<3>> {
        let mut trial = body_properties;
        let disk = HyperbolicDisk {
            disk_radius: *self.maximum_distance(),
            point: *trial.position_mut(),
        };
        let trial_sample: Hyperbolic<3> = disk.sample(rng);
        // push point back onto Hyperboloid
        *trial.position_mut() = Hyperbolic::from_minkowski_coordinates(Minkowski::from([
            trial_sample.coordinates()[0],
            trial_sample.coordinates()[1],
            (trial_sample.point()[0].powi(2) + trial_sample.point()[1].powi(2) + 1.0_f64).sqrt(),
        ]));
        trial
    }
}

impl LocalTrial<OrientedHyperbolicPoint<3, Angle>>
    for Translate<OrientedHyperbolicPoint<3, Angle>>
{
    /// Propose local trial moves for an oriented body on a hyperbolic surface.
    #[inline]
    fn propose<R: Rng>(
        &self,
        rng: &mut R,
        body_properties: OrientedHyperbolicPoint<3, Angle>,
    ) -> OrientedHyperbolicPoint<3, Angle> {
        let mut trial = body_properties;
        let original_orientation = body_properties.orientation.theta;
        let disk = HyperbolicDisk {
            disk_radius: *self.maximum_distance(),
            point: *trial.position_mut(),
        };
        let (trial_sample, boost, rotation) =
            OrientedHyperbolicPoint::<3, Angle>::sample(&disk, rng);
        // compute change in orientation
        let del_phi = OrientedHyperbolicPoint::<3, Angle>::deck_transform(
            boost,
            rotation,
            &body_properties.position,
        );
        *trial.orientation_mut() = Angle::from(original_orientation + del_phi);
        // push point back onto Hyperboloid
        *trial.position_mut() = Hyperbolic::from_minkowski_coordinates(Minkowski::from([
            trial_sample.coordinates()[0],
            trial_sample.coordinates()[1],
            (trial_sample.point()[0].powi(2) + trial_sample.point()[1].powi(2) + 1.0_f64).sqrt(),
        ]));
        trial
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use hoomd_manifold::{Hyperbolic, Minkowski};
    use hoomd_microstate::property::{OrientedHyperbolicPoint, Point, Position};
    use hoomd_vector::{Angle, Metric};
    use rand::{SeedableRng, rngs::StdRng};
    use rstest::*;

    /// Number of trial moves to test
    const N: usize = 256;

    #[rstest]
    fn translate_hyperbolic_point(#[values(0.01, 0.1, 1.0)] d: f64) {
        let mut rng = StdRng::seed_from_u64(42);
        let rho: f64 = 1.0;
        let body_properties = Point::new(Hyperbolic::from_minkowski_coordinates(
            [1.0, 0.0, (2.0_f64).sqrt()].into(),
        ));
        let translate =
            Translate::with_maximum_distance(d.try_into().expect("hard-coded positive real"));

        for _ in 0..N {
            let new_body_properties = translate.propose(&mut rng, body_properties);

            // Translation move keeps the point on the Hyperboloid
            assert_relative_eq!(
                new_body_properties
                    .position()
                    .point()
                    .distance_squared(&Minkowski::from([0.0, 0.0, 0.0])),
                -(rho.powi(2)),
                epsilon = 1e-12
            );

            // Translation move does not move the point more than a distance d
            assert!(
                d > new_body_properties.position().distance(
                    &Hyperbolic::from_minkowski_coordinates(Minkowski::from([
                        1.0,
                        0.0,
                        (2.0_f64).sqrt()
                    ]),)
                )
            );
        }
    }

    #[rstest]
    fn translate_oriented_hyperbolic_point(#[values(0.01, 0.1, 1.0)] d: f64) {
        let mut rng = StdRng::seed_from_u64(42);
        let body_properties = OrientedHyperbolicPoint {
            position: Hyperbolic::from_minkowski_coordinates([1.0, 0.0, (2.0_f64).sqrt()].into()),
            orientation: Angle::from(0.0),
        };
        let translate =
            Translate::with_maximum_distance(d.try_into().expect("hard-coded positive real"));

        for _ in 0..N {
            let new_body_properties = translate.propose(&mut rng, body_properties);

            // Translation move keeps the point on the Hyperboloid
            assert_relative_eq!(
                new_body_properties
                    .position()
                    .point()
                    .distance_squared(&Minkowski::from([0.0, 0.0, 0.0])),
                -1.0,
                epsilon = 1e-12
            );

            // Translation move does not move the point more than a distance d
            assert!(
                d > new_body_properties.position().distance(
                    &Hyperbolic::from_minkowski_coordinates(Minkowski::from([
                        1.0,
                        0.0,
                        (2.0_f64).sqrt()
                    ]))
                )
            );
        }
    }
}

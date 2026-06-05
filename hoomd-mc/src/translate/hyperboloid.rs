// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Translation moves on curved surfaces

use rand::{
    Rng,
    distr::Distribution,
};

use crate::{LocalTrial, Translate};
use hoomd_manifold::{Hyperbolic, HyperbolicDisk};
use hoomd_microstate::property::{Orientation, OrientedHyperbolicPoint, Point, Position};
use hoomd_utility::valid::PositiveReal;
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
        let max_distance: PositiveReal = self.maximum_distance
            * PositiveReal::try_from(0.9).expect("hard-coded positive number");
        let disk = HyperbolicDisk {
            disk_radius: max_distance,
            point: *trial.position_mut(),
        };
        let trial_sample: Hyperbolic<3> = disk.sample(rng);
        let new = trial_sample.coordinates();
        // store in polar coordinates to ensure point is on disk
        let theta = new[1].atan2(new[0]);
        let boost = (new[2]).acosh();
        *trial.position_mut() = Hyperbolic::<3>::from_polar_coordinates(boost, theta);
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
        let max_distance: PositiveReal = self.maximum_distance
            * PositiveReal::try_from(0.9).expect("hard-coded positive number");
        let disk = HyperbolicDisk {
            disk_radius: max_distance,
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
        let new = trial_sample.coordinates();
        let theta = new[1].atan2(new[0]);
        let boost = (new[2]).acosh();
        *trial.position_mut() = Hyperbolic::<3>::from_polar_coordinates(boost, theta);
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
    const NSTEPS: usize = 1000;

    #[rstest]
    fn translate_hyperbolic_point(#[values(0.01, 0.1, 1.0)] d: f64) {
        let mut rng = StdRng::seed_from_u64(42);
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
                -1.0,
                epsilon = 1e-8
            );

            // Translation move does not move the point more than a distance d
            let dist =
                new_body_properties
                    .position()
                    .distance(&Hyperbolic::from_minkowski_coordinates(Minkowski::from([
                        1.0,
                        0.0,
                        (2.0_f64).sqrt(),
                    ])));
            assert!(d > dist);
        }
    }

    #[rstest]
    fn translate_hyperbolic_point_chain(#[values(0.001, 0.01, 0.1, 0.25)] d: f64) {
        let mut rng = StdRng::seed_from_u64(42);
        let mut body_properties = Point::new(Hyperbolic::from_minkowski_coordinates(
            [-1.0, 1.0, (3.0_f64).sqrt()].into(),
        ));
        let translate =
            Translate::with_maximum_distance(d.try_into().expect("hard-coded positive real"));

        for _ in 0..NSTEPS {
            let new_body_properties = translate.propose(&mut rng, body_properties);

            // Translation move keeps the point on the Hyperboloid
            #[allow(
                clippy::cast_possible_truncation,
                reason = "float is truncated before converting to i32"
            )]
            let tolerance = 8_i32
                - (new_body_properties.position.coordinates()[2]
                    .log10()
                    .trunc() as i32);
            assert_relative_eq!(
                new_body_properties
                    .position()
                    .point()
                    .distance_squared(&Minkowski::from([0.0, 0.0, 0.0])),
                -1.0,
                epsilon = 10.0_f64.powi(-tolerance)
            );

            // Translation move does not move the point more than a distance d
            let dist = new_body_properties
                .position()
                .distance(&body_properties.position);
            assert!(d > dist);

            body_properties.position = new_body_properties.position;
        }
    }

    #[rstest]
    fn translate_oriented_hyperbolic_point(#[values(0.001, 0.01, 0.1, 1.0)] d: f64) {
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
            #[allow(
                clippy::cast_possible_truncation,
                reason = "float is truncated before converting to i32"
            )]
            let tolerance = 8_i32
                - (new_body_properties.position.coordinates()[2]
                    .log10()
                    .trunc() as i32);
            assert_relative_eq!(
                new_body_properties
                    .position()
                    .point()
                    .distance_squared(&Minkowski::from([0.0, 0.0, 0.0])),
                -1.0,
                epsilon = 10.0_f64.powi(-tolerance)
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

    #[rstest]
    fn translate_oriented_hyperbolic_point_chain(#[values(0.001, 0.01, 0.1, 0.25)] d: f64) {
        let mut rng = StdRng::seed_from_u64(42);
        let mut body_properties = OrientedHyperbolicPoint {
            position: Hyperbolic::from_minkowski_coordinates([1.0, 0.0, (2.0_f64).sqrt()].into()),
            orientation: Angle::from(0.0),
        };
        let translate =
            Translate::with_maximum_distance(d.try_into().expect("hard-coded positive real"));

        for _ in 0..NSTEPS {
            let new_body_properties = translate.propose(&mut rng, body_properties);

            // Translation move keeps the point on the Hyperboloid
            #[allow(
                clippy::cast_possible_truncation,
                reason = "float is truncated before converting to i32"
            )]
            let tolerance = 8_i32
                - (new_body_properties.position.coordinates()[2]
                    .log10()
                    .trunc() as i32);
            assert_relative_eq!(
                new_body_properties
                    .position()
                    .point()
                    .distance_squared(&Minkowski::from([0.0, 0.0, 0.0])),
                -1.0,
                epsilon = 10.0_f64.powi(-tolerance)
            );

            // Translation move does not move the point more than a distance d
            let dist = new_body_properties
                .position()
                .distance(&body_properties.position);
            assert!(d > dist);
            body_properties.position = new_body_properties.position;
        }
    }

    #[rstest]
    fn translate_hyperbolic_point_far_from_cusp(#[values(0.001)] d: f64) {
        //, 0.01, 0.1, 0.25)] d: f64) {
        let mut rng = StdRng::seed_from_u64(42);
        let mut body_properties = Point::new(Hyperbolic::from_minkowski_coordinates(
            [10_000.0, 10_000.0, (200_000_001.0_f64).sqrt()].into(),
        ));
        let translate =
            Translate::with_maximum_distance(d.try_into().expect("hard-coded positive real"));

        for _ in 0..NSTEPS {
            let new_body_properties = translate.propose(&mut rng, body_properties);

            // Translation move keeps the point on the Hyperboloid
            #[allow(
                clippy::cast_possible_truncation,
                reason = "float is truncated before converting to i32"
            )]
            let tolerance = 8_i32
                - (new_body_properties.position.coordinates()[2]
                    .log10()
                    .trunc() as i32);
            assert_relative_eq!(
                new_body_properties
                    .position()
                    .point()
                    .distance_squared(&Minkowski::from([0.0, 0.0, 0.0])),
                -1.0,
                epsilon = 10.0_f64.powi(-tolerance)
            );

            // Translation move does not move the point more than a distance d
            let dist = new_body_properties
                .position()
                .distance(&body_properties.position);
            // when far away from cusp, small displacements are unstable
            assert!(d > dist);
            body_properties.position = new_body_properties.position;
        }
    }
    #[rstest]
    fn translate_oriented_hyperbolic_point_far_from_cusp(#[values(0.001, 0.01, 0.1)] d: f64) {
        let mut rng = StdRng::seed_from_u64(42);
        let mut body_properties = OrientedHyperbolicPoint {
            position: Hyperbolic::from_minkowski_coordinates(
                [10_000.0, 10_000.0, (200_000_001.0_f64).sqrt()].into(),
            ),
            orientation: Angle::default(),
        };
        let translate =
            Translate::with_maximum_distance(d.try_into().expect("hard-coded positive real"));

        for _ in 0..NSTEPS {
            let new_body_properties = translate.propose(&mut rng, body_properties);

            // Translation move keeps the point on the Hyperboloid
            #[allow(
                clippy::cast_possible_truncation,
                reason = "float is truncated before converting to i32"
            )]
            let tolerance = 8_i32
                - (new_body_properties.position.coordinates()[2]
                    .log10()
                    .trunc() as i32);
            assert_relative_eq!(
                new_body_properties
                    .position()
                    .point()
                    .distance_squared(&Minkowski::from([0.0, 0.0, 0.0])),
                -1.0,
                epsilon = 10.0_f64.powi(-tolerance)
            );

            // Translation move does not move the point more than a distance d
            let dist = new_body_properties
                .position()
                .distance(&body_properties.position);
            // when far away from cusp, small displacements are unstable
            assert!(d > dist);
            body_properties.position = new_body_properties.position;
        }
    }
}

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Translation moves on curved surfaces

use rand::{Rng, distr::Distribution};

use crate::{LocalTrial, Translate};
use hoomd_manifold::{Spherical, SphericalDisk};
use hoomd_microstate::property::{Point, Position};
use hoomd_vector::InnerProduct;

impl LocalTrial<Point<Spherical<3>>> for Translate<Point<Spherical<3>>> {
    /// Propose local trial moves for a body on the surface of a sphere
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_manifold::{Spherical, SphericalDisk};
    /// use hoomd_mc::{LocalTrial, Translate};
    /// use hoomd_microstate::property::{Point, Position};
    /// use hoomd_vector::{Cartesian, InnerProduct, Metric, Vector};
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut rng = StdRng::seed_from_u64(14);
    /// let initial_point = Point::new(Spherical::from_cartesian_coordinates(
    ///     [0.5_f64.sqrt(), 0.5_f64.sqrt(), 0.0].into(),
    /// ));
    /// let d = 0.1;
    /// let translate = Translate::with_maximum_distance(d.try_into()?);
    ///
    /// let new_body_properties = translate.propose(&mut rng, initial_point);
    ///
    /// // Translation move keeps point on the surface of the sphere
    /// let new_body_radius = new_body_properties.position.point().norm();
    /// assert_eq!(new_body_radius, 1.0);
    ///
    /// // Translation move does not translate the point more than a distance d away
    /// assert!(
    ///     d > new_body_properties
    ///         .position()
    ///         .distance(&initial_point.position())
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn propose<R: Rng>(
        &self,
        rng: &mut R,
        body_properties: Point<Spherical<3>>,
    ) -> Point<Spherical<3>> {
        let mut trial = body_properties;
        let disk = SphericalDisk {
            disk_radius: *self.maximum_distance(),
            point: *trial.position_mut(),
        };
        *trial.position_mut() = disk.sample(rng);
        let rescale = 1.0 / trial.position().point().norm();
        *trial.position_mut() = Spherical::from_cartesian_coordinates(
            *trial.position().point() * rescale
        );
        trial
    }
}

// TODO: test

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Translation moves on curved surfaces

use rand::{
    Rng, RngExt,
    distr::{Distribution, Uniform},
};
use rand_distr::StandardNormal;

use crate::{LocalTrial, Translate};
use hoomd_manifold::Spherical;
use hoomd_microstate::property::{Orientation, OrientedSphericalPoint, Point, Position};
use hoomd_vector::{Angle, Cartesian, InnerProduct};
use std::f64::consts::PI;

impl LocalTrial<Point<Spherical<3>>> for Translate<Point<Spherical<3>>> {
    /// Propose local trial moves for a body on the surface of a sphere.
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
    /// assert_relative_eq!(new_body_radius, 1.0, epsilon = 1e-8);
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
        let dist = Uniform::new(0.0, self.maximum_distance().get())
            .expect("max distance must be positive real");
        let displacement = dist.sample(rng);
        let (sn, cs) = (displacement.sin(), displacement.cos());
        let vec = Cartesian::<3>::from(std::array::from_fn(|_| rng.sample(StandardNormal)));
        let proj = vec.dot(trial.position.point());
        let tangent = Cartesian::from([
            vec[0] - proj * trial.position.coordinates()[0],
            vec[1] - proj * trial.position.coordinates()[1],
            vec[2] - proj * trial.position.coordinates()[2],
        ]);
        let (unit, _norm) = tangent.to_unit().expect("cannot be null");
        let new = Cartesian::from([
            trial.position.coordinates()[0] * cs + unit.get().coordinates[0] * sn,
            trial.position.coordinates()[1] * cs + unit.get().coordinates[1] * sn,
            trial.position.coordinates()[2] * cs + unit.get().coordinates[2] * sn,
        ]);
        *trial.position_mut() = Spherical::from_cartesian_coordinates(new);
        trial
    }
}

impl LocalTrial<OrientedSphericalPoint<3, Angle>> for Translate<OrientedSphericalPoint<3, Angle>> {
    /// Propose local trial moves for an oriented body on the surface of a sphere.
    ///
    /// TODO: example
    #[inline]
    fn propose<R: Rng>(
        &self,
        rng: &mut R,
        body_properties: OrientedSphericalPoint<3, Angle>,
    ) -> OrientedSphericalPoint<3, Angle> {
        let mut trial = body_properties;
        let dist = Uniform::new(0.0, 1.0)
            .expect("max distance must be positive real");
        let displacement = dist.sample(rng) * self.maximum_distance().get();
        let beta = dist.sample(rng) * 2.0 * PI; 
        let (cs, sn) = (beta.cos(), beta.sin());

        let stereo = trial.position.stereographic_projection_from_south_pole();
        //  calculate orientation change
        let tau = (displacement/2.0).tan();
        let (al, be, ga, de) = (stereo[0] + tau*cs, stereo[1]+tau*sn, 1.0-tau*(stereo[0]*cs + stereo[1]*sn), tau*(stereo[1]*cs - stereo[0]*sn));
        let del_psi = -2.0 * de.atan2(ga);
        *trial.orientation_mut() = Angle::from(body_properties.orientation.theta + del_psi);

        // transform the position using the same Möbius transform
        let pref = 1.0/(ga*ga + de*de);
        let transformed_stereo = [pref*(al*ga + be*de), pref*(be*ga - al*de)];
        let phi = (transformed_stereo[1]).atan2(transformed_stereo[0]);
        let theta = 2.0 * ((transformed_stereo[0]*transformed_stereo[0] + transformed_stereo[1]*transformed_stereo[1]).sqrt()).atan();
        *trial.position_mut() = Spherical::<3>::from_polar_coordinates(theta, phi);
        trial
    }
}

impl LocalTrial<Point<Spherical<4>>> for Translate<Point<Spherical<4>>> {
    /// Propose local trial moves for a body on the surface of a 3-sphere
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_manifold::{Spherical, SphericalDisk};
    /// use hoomd_mc::{LocalTrial, Translate};
    /// use hoomd_microstate::property::{Point, Position};
    /// use hoomd_vector::{Cartesian, InnerProduct, Metric, Vector};
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut rng = StdRng::seed_from_u64(14);
    /// let initial_point = Point::new(Spherical::<4>::from_polar_coordinates(
    ///     PI / 4.0,
    ///     PI / 10.0,
    ///     5.0 * PI / 4.0,
    /// ));
    /// let d = 0.1;
    /// let translate = Translate::with_maximum_distance(d.try_into()?);
    ///
    /// let new_body_properties = translate.propose(&mut rng, initial_point);
    ///
    /// // Translation move keeps point on the surface of the sphere
    /// assert_relative_eq!(
    ///     new_body_properties.position().point().norm(),
    ///     1.0_f64,
    ///     epsilon = 1e-8
    /// );
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
        body_properties: Point<Spherical<4>>,
    ) -> Point<Spherical<4>> {
        let mut trial = body_properties;
        let dist = Uniform::new(0.0, self.maximum_distance().get())
            .expect("max distance must be positive real");
        let displacement = dist.sample(rng);
        let (sn, cs) = ((displacement.abs()).sin(), (displacement.abs()).cos());
        let vec = Cartesian::<4>::from(std::array::from_fn(|_| rng.sample(StandardNormal)));
        let proj = vec.dot(trial.position.point());
        let tangent = Cartesian::from([
            vec[0] - proj * trial.position.coordinates()[0],
            vec[1] - proj * trial.position.coordinates()[1],
            vec[2] - proj * trial.position.coordinates()[2],
            vec[3] - proj * trial.position.coordinates()[3],
        ]);
        let (unit, _norm) = tangent.to_unit().expect("cannot be null");
        let new = Cartesian::from([
            trial.position.coordinates()[0] * cs + unit.get().coordinates[0] * sn,
            trial.position.coordinates()[1] * cs + unit.get().coordinates[1] * sn,
            trial.position.coordinates()[2] * cs + unit.get().coordinates[2] * sn,
            trial.position.coordinates()[3] * cs + unit.get().coordinates[3] * sn,
        ]);
        *trial.position_mut() = Spherical::from_cartesian_coordinates(new);
        trial
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use hoomd_vector::Metric;
    use rand::{SeedableRng, rngs::StdRng};
    use rstest::*;
    use std::f64::consts::PI;

    const N: usize = 256;

    #[rstest]
    fn translate_2spherical_point(#[values(0.01, 0.1, 1.0)] d: f64) {
        let mut rng = StdRng::seed_from_u64(1459);
        let initial_point = Point::new(Spherical::<3>::from_cartesian_coordinates(
            [0.5_f64.sqrt(), 0.0, 0.5_f64.sqrt()].into(),
        ));
        let translate =
            Translate::with_maximum_distance(d.try_into().expect("hard-coded positive value"));

        for _ in 0..N {
            let new_body_properties = translate.propose(&mut rng, initial_point);
            // Translation move keeps point on the surface of the sphere
            assert_relative_eq!(
                new_body_properties.position().point().norm(),
                1.0_f64,
                epsilon = 1e-8
            );
            // Translation move does not translate the point more than a distance d away
            assert!(
                d > new_body_properties
                    .position()
                    .distance(initial_point.position())
            );
        }
    }

    #[rstest]
    fn translate_3spherical_point(#[values(0.01, 0.1, 1.0)] d: f64) {
        let mut rng = StdRng::seed_from_u64(1459);
        let initial_point = Point::new(Spherical::<4>::from_polar_coordinates(
            PI / 4.0,
            PI / 10.0,
            5.0 * PI / 4.0,
        ));
        let translate =
            Translate::with_maximum_distance(d.try_into().expect("hard-coded positive value"));

        for _ in 0..N {
            let new_body_properties = translate.propose(&mut rng, initial_point);
            // Translation move keeps point on the surface of the sphere
            assert_relative_eq!(
                new_body_properties.position().point().norm(),
                1.0_f64,
                epsilon = 1e-8
            );
            // Translation move does not translate the point more than a distance d away
            assert!(
                d > new_body_properties
                    .position()
                    .distance(initial_point.position())
            );
        }
    }
}

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
use hoomd_microstate::property::{Point, Position};
use hoomd_vector::{Cartesian, InnerProduct};

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
        // let displacement = (self.maximum_distance().get()) * rng.sample::<f64, _>(StandardNormal);
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
        /* let v = base_point.coordinates();            // v[0] = real part
    let d = max_dist * rng.random::<f64>();      // rng.gen::<f64>() on rand 0.8
    let (sn, cs) = d.sin_cos();                  // full angle: r^2, not r

    // Random direction; normalisation folded into the sin factor
    let g: [f64; 3] = std::array::from_fn(|_| rng.sample(StandardNormal));
    let sg = sn / (g[0]*g[0] + g[1]*g[1] + g[2]*g[2]).sqrt();

    // t = s û s, expanded (singular only at v = -1)
    let k = v[1]*g[0] + v[2]*g[1] + v[3]*g[2];
    let a = 1.0 + v[0];
    let m = if a > 1e-12 { k / a } else { 0.0 }; // at v = -1, any pure û is tangent

    let new = [
        cs*v[0] - sg*k,
        cs*v[1] + sg*(g[0] - m*v[1]),
        cs*v[2] + sg*(g[1] - m*v[2]),
        cs*v[3] + sg*(g[2] - m*v[3]),
    ];
    let new_pos = Spherical::<4>::from_cartesian_coordinates(new.into());     // use an unchecked constructor if you have one
    let ultimate_displacement = new_pos.distance(&base_point);
    //debug_assert!((new_pos.distance(&base_point) - d).abs() < 1e-9);
     */
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

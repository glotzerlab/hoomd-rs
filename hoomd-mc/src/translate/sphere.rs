// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Translation moves on curved surfaces

use rand::{Rng, RngExt, distr::Uniform};
use rand_distr::StandardNormal;

use crate::{LocalTrial, Translate};
use hoomd_manifold::Spherical;
use hoomd_microstate::property::{Point, Position};

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
        let coord = trial.position.coordinates();
        let d = rng.sample(
            Uniform::new(0.0, self.maximum_distance.get()).expect("max distance must be positive"),
        );

        // create a tangent vector at v
        let mut tangent: [f64; 3] = std::array::from_fn(|_| rng.sample(StandardNormal));
        let tang_v: f64 = (0..3).map(|i| tangent[i] * coord[i]).sum();
        for i in 0..3 {
            tangent[i] -= tang_v * coord[i];
        }
        let tang_norm = tangent.iter().map(|x| x * x).sum::<f64>().sqrt();

        // Riemann exponential map
        let (sn, cs) = d.sin_cos();
        let new: [f64; 3] = std::array::from_fn(|i| cs * coord[i] + sn * tangent[i] / tang_norm);
        *trial.position_mut() = Spherical::<3>::from_cartesian_coordinates(new.into());
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
        let coord = trial.position.coordinates();
        let displacement = self.maximum_distance.get() * rng.random::<f64>();
        let (sn, cs) = displacement.sin_cos();

        let vec: [f64; 3] = std::array::from_fn(|_| rng.sample(StandardNormal));
        // based on algebraic simplifiation of quaternion "square root"
        let sg = sn / (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]).sqrt();
        let k = coord[1] * vec[0] + coord[2] * vec[1] + coord[3] * vec[2];
        let a = 1.0 + coord[0];
        // clip for instability near south pole
        let k_rescale = if a > 1e-12 { k / a } else { 0.0 };
        let new = [
            cs * coord[0] - sg * k,
            cs * coord[1] + sg * (vec[0] - k_rescale * coord[1]),
            cs * coord[2] + sg * (vec[1] - k_rescale * coord[2]),
            cs * coord[3] + sg * (vec[2] - k_rescale * coord[3]),
        ];
        *trial.position_mut() = Spherical::<4>::from_cartesian_coordinates(new.into());
        trial
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use hoomd_vector::{InnerProduct, Metric};
    use rand::{SeedableRng, rngs::StdRng};
    use rstest::*;
    use std::f64::consts::PI;

    const N: usize = 256;

    #[rstest]
    fn translate_2spherical_point(#[values(0.001, 0.01, 0.1, 1.0)] d: f64) {
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
    fn translate_3spherical_point(#[values(0.001, 0.01, 0.1, 1.0)] d: f64) {
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

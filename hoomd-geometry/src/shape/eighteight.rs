// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`EightEight`]

use serde::{Deserialize, Serialize};

use crate::IsPointInside;
use hoomd_manifold::Hyperbolic;
use std::f64::consts::PI;

/// A regular octagon in two-dimensional hyperbolic space.
///
/// [`EightEight`] implements a single regular octagon in the {8,8} tiling of
/// two-dimensional hyperbolic space. The scaling of the octagon is set such
/// that each of the angles is $` \frac{2\pi}{8} `$ so that eight equivalent
/// octagons meet at each vertex.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EightEight {
    /// Skirt width of the Hyperbolic.
    pub skirt: f64,
}

impl IsPointInside<Hyperbolic<3>> for EightEight {
    /// Checks if a given Hyperbolic point is inside [`EightEight`].
    ///
    /// # Example
    /// ```
    /// use hoomd_geometry::{IsPointInside, shape::EightEight};
    /// use hoomd_manifold::Hyperbolic;
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let eight_eight = EightEight { skirt: 1.0 };
    ///
    /// let point = Hyperbolic::<3>::from_polar_coordinates(1.0, PI / 8.0, 1.0);
    /// assert!(eight_eight.is_point_inside(&point));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn is_point_inside(&self, point: &Hyperbolic<3>) -> bool {
        EightEight::distance_to_boundary(point) >= 0.0
    }
}

impl EightEight {
    /// Computes the shortest distance between a given point and the boundary
    /// of `EightEight`.
    ///
    /// The shortest distance is along the radial path &mdash; the geodesic
    /// passing between the Hyperbolic cusp and the query point.
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::shape::EightEight;
    /// use hoomd_manifold::{Hyperbolic, Minkowski};
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v: f64 = 2.448_452_447_678_076;
    /// let rho: f64 = 1.0;
    /// let theta: f64 = PI / 4.0;
    /// let x = Hyperbolic::from_minkowski_coordinates(
    ///     [
    ///         rho * (v.sinh()) * (theta.cos()),
    ///         rho * (v.sinh()) * (theta.sin()),
    ///         rho * (v.cosh()),
    ///     ]
    ///     .into(),
    ///     1.0,
    /// );
    /// assert_relative_eq!(
    ///     EightEight::distance_to_boundary(&x),
    ///     0.0,
    ///     epsilon = 1e-12
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn distance_to_boundary(point: &Hyperbolic<3>) -> f64 {
        let theta = point.coordinates()[1].atan2(point.coordinates()[0]);
        let angle = theta.rem_euclid(PI / 4.0);
        let boost = (point.coordinates()[2] / point.skirt()).acosh();
        let tile_size = EightEight::EIGHTEIGHT;
        let eta =
            (tile_size.tanh() / (angle.cos() - angle.sin() * (1.0 - (2.0_f64).sqrt()))).atanh();
        point.skirt() * (eta - boost)
    }
    /// Points on the boundary of the fundamental domain
    #[inline]
    #[must_use]
    pub fn boundary_points(number_of_points: usize, skirt: f64) -> Vec<(f64, f64)> {
        let mut coords = Vec::<(f64, f64)>::new();
        for n in 0..number_of_points {
            let angle = (n as f64) * 2.0 * PI / (number_of_points as f64);
            let tile_size = EightEight::EIGHTEIGHT;
            let eta =
                (tile_size.tanh() / (angle.cos() - angle.sin() * (1.0 - (2.0_f64).sqrt()))).atanh();
            let x = (skirt * eta.sinh()) / (1.0 + eta.cosh());
            for k in 0..8 {
                coords.push((
                    x * (angle + f64::from(k) * PI / 4.0).cos(),
                    x * (angle + f64::from(k) * PI / 4.0).sin(),
                ));
            }
        }
        coords
    }
    /// Cusp-to-vertex distance for {8,8} tiling for Gauss curvature K = -1
    pub const EIGHTEIGHT: f64 = 2.448_452_447_678_076;
    /// Length of one of the sides of the {8,8} tiling for Gauss curvature K = -1
    pub const EDGE_LENGTH: f64 = 3.057_141_838_961_997;
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use hoomd_manifold::{Hyperbolic, HyperbolicDisk, Minkowski};
    use rand::{SeedableRng, distr::Distribution, rngs::StdRng};
    use std::ops::Not;

    #[test]
    fn boundary_distance() {
        // Distance to the edge of the {8,8} fundamental domain
        let e = Hyperbolic::<3>::from_polar_coordinates(1.0, 0.1, 1.0);
        let e_edge_distance = EightEight::distance_to_boundary(&e);
        let e_edge_distance_numeric = 0.838_080_324_331_728;
        assert_relative_eq!(e_edge_distance, e_edge_distance_numeric, epsilon = 1e-12);

        let f = Hyperbolic::<3>::from_polar_coordinates(1.0, 1.1, 1.0);
        let f_edge_distance = EightEight::distance_to_boundary(&f);
        let f_edge_distance_numeric = 0.545_034_457_278_499_5;
        assert_relative_eq!(f_edge_distance, f_edge_distance_numeric, epsilon = 1e-12);
    }

    #[test]
    fn inside_is_inside() {
        let eight_eight = EightEight { skirt: 1.0 };
        let r = 1.528_570_919_480_998;
        let mut rng = StdRng::seed_from_u64(239);
        let disk = HyperbolicDisk {
            disk_radius: r.try_into().expect("hard-coded positive number"),
            point: Hyperbolic::<3>::from_minkowski_coordinates(
                Minkowski::from([0.0, 0.0, 1.0]),
                1.0,
            ),
        };
        let random_point: Hyperbolic<3> = disk.sample(&mut rng);
        assert!(eight_eight.is_point_inside(&random_point));

        let point_1 = Hyperbolic::<3>::from_polar_coordinates(1.52, PI / 8.0, 1.0);
        assert!(eight_eight.is_point_inside(&point_1));

        let point_2 = Hyperbolic::<3>::from_polar_coordinates(2.44, PI / 4.0, 1.0);
        assert!(eight_eight.is_point_inside(&point_2));
    }

    #[test]
    fn outside_is_outside() {
        let eight_eight = EightEight { skirt: 1.0 };
        let point_1 = Hyperbolic::<3>::from_polar_coordinates(1.53, PI / 8.0, 1.0);
        assert!((eight_eight.is_point_inside(&point_1)).not());

        let point_2 = Hyperbolic::<3>::from_polar_coordinates(2.45, PI / 4.0, 1.0);
        assert!((eight_eight.is_point_inside(&point_2)).not());
    }
}

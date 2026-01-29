// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`EightEight`]

use serde::{Deserialize, Serialize};

use crate::IsPointInside;
use hoomd_manifold::{Hyperbolic, Minkowski};
use hoomd_vector::Metric;
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
    /// The shortest distance is computed by finding the arclength of the geodesic
    /// which passes through the query point and intersects the boundary at a
    /// right angle.
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::shape::EightEight;
    /// use hoomd_manifold::{Hyperbolic, Minkowski};
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v: f64 = EightEight::CUSP_TO_EDGE - 0.4;
    /// let rho: f64 = 1.0;
    /// let theta: f64 = PI / 8.0;
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
    ///     0.4,
    ///     epsilon = 1e-12
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn distance_to_boundary(point: &Hyperbolic<3>) -> f64 {
        let theta =
            (point.coordinates()[1].atan2(point.coordinates()[0])).rem_euclid(PI / 4.0) - PI / 8.0;
        let boost = (point.coordinates()[2] / point.skirt()).acosh();
        let (b_sinh, b_cosh) = (boost.sinh(), boost.cosh());
        let rho = point.skirt();
        let xi = Self::CUSP_TO_EDGE;
        let (xi_sinh, xi_cosh) = (xi.sinh(), xi.cosh());
        // boost into frame where edge is the vertical diameter
        let edge_as_diameter: Hyperbolic<3> = Hyperbolic::<3>::from_minkowski_coordinates(
            Minkowski::from([
                rho * xi_cosh * b_sinh * (theta.cos()) - rho * xi_sinh * b_cosh,
                rho * b_sinh * (theta.sin()),
                -rho * xi_sinh * b_sinh * (theta.cos()) + rho * xi_cosh * b_cosh,
            ]),
            rho,
        );
        let flipped = Hyperbolic::<3>::from_minkowski_coordinates(
            Minkowski::from([
                -edge_as_diameter.coordinates()[0],
                edge_as_diameter.coordinates()[1],
                edge_as_diameter.coordinates()[2],
            ]),
            rho,
        );
        let sign = -(edge_as_diameter.coordinates()[0]).signum();
        sign * (edge_as_diameter.distance(&flipped)) / 2.0
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
    /// Apply a lattice transformation to a point.
    #[inline]
    #[must_use]
    pub fn gamma(eta: f64, theta: f64, point: &[f64; 3]) -> [f64; 3] {
        let (eta_sinh_squared, two_eta_sinh, theta_sin, theta_cos) = (
            (eta.sinh()).powi(2),
            (2.0 * eta).sinh(),
            theta.sin(),
            theta.cos(),
        );
        [
            (2.0 * (eta_sinh_squared) * ((theta_cos).powi(2)) + 1.0) * point[0]
                + ((2.0 * theta).sin()) * (eta_sinh_squared) * point[1]
                + (two_eta_sinh) * (theta_cos) * point[2],
            ((2.0 * theta).sin()) * (eta_sinh_squared) * point[0]
                + (2.0 * (eta_sinh_squared) * ((theta_sin).powi(2)) + 1.0) * point[1]
                + (two_eta_sinh) * (theta_sin) * point[2],
            (two_eta_sinh) * (theta_cos) * point[0]
                + (two_eta_sinh) * (theta_sin) * point[1]
                + ((2.0 * eta).cosh()) * point[2],
        ]
    }
    /// Calculate the change in angle in the tangent bundle associated with a lattice
    /// transformation.
    #[inline]
    #[must_use]
    pub fn reorient(theta: f64, point: &[f64; 3]) -> f64 {
        let (q_u, q_v) = (point[0] / (1.0 + point[2]), point[1] / (1.0 + point[2]));
        let alpha = (1.0 + (PI / 4.0).cos()).sqrt();
        let beta = (theta.cos()) * ((2.0 * ((PI / 4.0).cos())).sqrt());
        let gamma = (theta.sin()) * ((2.0 * ((PI / 4.0).cos())).sqrt());
        // let prefactor = 1.0/((b*v-c*u).powi(2)+(a+b*u+c*v).powi(2)).powi(2);
        let p_x = alpha + beta * q_u + gamma * q_v;
        let p_y = beta * q_v - gamma * q_u;
        -2.0 * (p_y.atan2(p_x))
    }
    /// Cusp-to-vertex distance for the {8,8} tiling for Gauss curvature K = -1.
    pub const EIGHTEIGHT: f64 = 2.448_452_447_678_076;
    /// Cusp-to-middle-of-edge distance for the {8,8} tiling for Gauss curvature K = -1.
    pub const CUSP_TO_EDGE: f64 = 1.528_570_919_480_998;
    /// Length of one of the sides of the {8,8} tiling for Gauss curvature K = -1.
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
        let e_edge_distance_numeric = 0.631_401_734_734_821;
        assert_relative_eq!(e_edge_distance, e_edge_distance_numeric, epsilon = 1e-12);

        let f = Hyperbolic::<3>::from_polar_coordinates(0.6, 0.2 + PI / 4.0, 1.0);
        let f_edge_distance = EightEight::distance_to_boundary(&f);
        let f_edge_distance_numeric = 0.947_879_122_461_848;
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
        let point_1 = Hyperbolic::<3>::from_polar_coordinates(1.54, PI / 8.0, 1.0);
        assert!((eight_eight.is_point_inside(&point_1)).not());

        let point_2 = Hyperbolic::<3>::from_polar_coordinates(2.45, PI / 4.0, 1.0);
        assert!((eight_eight.is_point_inside(&point_2)).not());
    }
}

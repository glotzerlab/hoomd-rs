// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`TwelveTwelve`]

use serde::{Deserialize, Serialize};

use crate::IsPointInside;
use hoomd_manifold::{Hyperbolic, Minkowski};
use hoomd_vector::Metric;
use std::f64::consts::PI;

/// A regular dodecagon in two-dimensional hyperbolic space.
///
/// [`TwelveTwelve`] implements a single regular dodecagon in the {12,12} tiling
/// of two-dimensional hyperbolic space. The scaling of the octagon is set such
/// that each of the angles is $` \frac{2\pi}{12} `$ so that twelve equivalent
/// dodecagons meet at each vertex.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwelveTwelve {}

impl IsPointInside<Hyperbolic<3>> for TwelveTwelve {
    /// Checks if a given Hyperbolic point is inside [`TwelveTwelve`].
    ///
    /// # Example
    /// ```
    /// use hoomd_geometry::{IsPointInside, shape::TwelveTwelve};
    /// use hoomd_manifold::Hyperbolic;
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let twelve_twelve = TwelveTwelve {};
    ///
    /// let point = Hyperbolic::<3>::from_polar_coordinates(1.0, PI / 8.0);
    /// assert!(twelve_twelve.is_point_inside(&point));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn is_point_inside(&self, point: &Hyperbolic<3>) -> bool {
        TwelveTwelve::distance_to_boundary(point) >= 0.0
    }
}

impl TwelveTwelve {
    /// Computes the shortest distance between a given point and the boundary
    /// of `TwelveTwelve`.
    ///
    /// The shortest distance is computed by finding the arclength of the geodesic
    /// which passes through the query point and intersects the boundary at a
    /// right angle.
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::shape::TwelveTwelve;
    /// use hoomd_manifold::{Hyperbolic, Minkowski};
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let v: f64 = TwelveTwelve::CUSP_TO_EDGE - 0.4;
    /// let theta: f64 = PI / 12.0;
    /// let x = Hyperbolic::from_minkowski_coordinates(
    ///     [
    ///         (v.sinh()) * (theta.cos()),
    ///         (v.sinh()) * (theta.sin()),
    ///         (v.cosh()),
    ///     ]
    ///     .into(),
    /// );
    /// assert_relative_eq!(
    ///     TwelveTwelve::distance_to_boundary(&x),
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
            (point.coordinates()[1].atan2(point.coordinates()[0])).rem_euclid(PI / 6.0) - PI / 12.0;
        let boost = (point.coordinates()[2]).acosh();
        let (b_sinh, b_cosh) = (boost.sinh(), point.coordinates()[2]);
        let xi = Self::CUSP_TO_EDGE;
        let (xi_sinh, xi_cosh) = (xi.sinh(), xi.cosh());
        // boost into frame where edge is the vertical diameter
        let edge_as_diameter: Hyperbolic<3> =
            Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
                xi_cosh * b_sinh * (theta.cos()) - xi_sinh * b_cosh,
                b_sinh * (theta.sin()),
                -xi_sinh * b_sinh * (theta.cos()) + xi_cosh * b_cosh,
            ]));
        let flipped = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            -edge_as_diameter.coordinates()[0],
            edge_as_diameter.coordinates()[1],
            edge_as_diameter.coordinates()[2],
        ]));
        let sign = -(edge_as_diameter.coordinates()[0]).signum();
        sign * (edge_as_diameter.distance(&flipped)) / 2.0
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
        let alpha = (1.0 + (PI / 6.0).cos()).sqrt();
        let beta = (theta.cos()) * ((2.0 * ((PI / 6.0).cos())).sqrt());
        let gamma = (theta.sin()) * ((2.0 * ((PI / 6.0).cos())).sqrt());
        let p_x = alpha + beta * q_u + gamma * q_v;
        let p_y = beta * q_v - gamma * q_u;
        -2.0 * (p_y.atan2(p_x))
    }
    /// Cusp-to-vertex distance for the {12,12} tiling for Gauss curvature K = -1.
    pub const TWELVETWELVE: f64 = 3.325_771_782_117_242;
    /// Cusp-to-middle-of-edge distance for the {12,12} tiling for Gauss curvature K = -1.
    pub const CUSP_TO_EDGE: f64 = 1.991_652_391_049_437;
    /// Length of one of the sides of the {12,12} tiling for Gauss curvature K = -1.
    pub const EDGE_LENGTH: f64 = 3.983_304_782_098_874;
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
        // Distance to the edge of the {12,12} fundamental domain
        let e = Hyperbolic::<3>::from_polar_coordinates(1.0, 0.1);
        let e_edge_distance = TwelveTwelve::distance_to_boundary(&e);
        let e_edge_distance_numeric = 1.028_489_242_583_61;
        assert_relative_eq!(e_edge_distance, e_edge_distance_numeric, epsilon = 1e-12);

        let f = Hyperbolic::<3>::from_polar_coordinates(0.6, 0.2 + PI / 6.0);
        let f_edge_distance = TwelveTwelve::distance_to_boundary(&f);
        let f_edge_distance_numeric = 1.393_774_804_611_314;
        assert_relative_eq!(f_edge_distance, f_edge_distance_numeric, epsilon = 1e-12);
    }

    #[test]
    fn inside_is_inside() {
        let twelve_twelve = TwelveTwelve {};
        let r = TwelveTwelve::CUSP_TO_EDGE;
        let mut rng = StdRng::seed_from_u64(239);
        let disk = HyperbolicDisk {
            disk_radius: r.try_into().expect("hard-coded positive number"),
            point: Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([0.0, 0.0, 1.0])),
        };
        let random_point: Hyperbolic<3> = disk.sample(&mut rng);
        assert!(twelve_twelve.is_point_inside(&random_point));

        let point_1 = Hyperbolic::<3>::from_polar_coordinates(1.99, PI / 12.0);
        assert!(twelve_twelve.is_point_inside(&point_1));

        let point_2 = Hyperbolic::<3>::from_polar_coordinates(3.32, PI / 6.0);
        assert!(twelve_twelve.is_point_inside(&point_2));
    }

    #[test]
    fn outside_is_outside() {
        let twelve_twelve = TwelveTwelve {};
        let point_1 = Hyperbolic::<3>::from_polar_coordinates(2.0, PI / 12.0);
        assert!((twelve_twelve.is_point_inside(&point_1)).not());

        let point_2 = Hyperbolic::<3>::from_polar_coordinates(3.33, PI / 6.0);
        assert!((twelve_twelve.is_point_inside(&point_2)).not());
    }
}

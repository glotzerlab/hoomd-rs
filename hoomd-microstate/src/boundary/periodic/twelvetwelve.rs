// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement periodic boundary conditions for the {12,12} tiling of hyperbolic
//! space.
//!
//! Specifically, `Periodic<TwelveTwelve>` identifies opposite edges of the
//! dodecagon to implement a genus-3 surface.

use arrayvec::ArrayVec;
use std::f64::consts::PI;

use crate::{
    boundary::{
        Error, GenerateGhosts, MAX_GHOSTS, MaximumAllowableInteractionRange, Periodic, Wrap,
    },
    property::{Point, Position},
};
use hoomd_geometry::shape::TwelveTwelve;
use hoomd_manifold::{Hyperbolic, Minkowski};

impl MaximumAllowableInteractionRange for TwelveTwelve {
    /// The largest value that the maximum interaction range can take.
    ///
    /// This bound is determined by the edge length of the dodecagon.
    #[inline]
    fn maximum_allowable_interaction_range(&self) -> f64 {
        self.skirt * TwelveTwelve::CUSP_TO_EDGE
    }
}

impl Wrap<Point<Hyperbolic<3>>> for Periodic<TwelveTwelve> {
    /// Wrap a point in hyperbolic space to the inside of the {12,12} tile.
    ///
    /// Note that the function fails to wrap points that are outside the
    /// dodecagon and further than `TwelveTwelve::EDGE_LENGTH/2` from any of
    /// the vertices.
    ///
    /// TODO: example
    #[inline]
    #[expect(clippy::too_many_lines, reason = "complicated function")]
    fn wrap(&self, properties: Point<Hyperbolic<3>>) -> Result<Point<Hyperbolic<3>>, Error> {
        let mut properties = properties;
        let r = properties.position_mut();
        let r_coords = r.coordinates();
        assert_eq!(
            r.skirt(),
            self.shape.skirt,
            "point must be wrapped onto a Hyperboloid with the same skirt"
        );
        let angle = r_coords[1]
            .atan2(r_coords[0])
            .rem_euclid(2.0 * PI);

        // distance to the boundary; if positive, r is within the tile and does not need to be wrapped
        let d = TwelveTwelve::distance_to_boundary(r);
        if d >= 0.0 {
            Ok(properties)
        } else if d > -self.maximum_interaction_range {
            // get the sequence of transformations necessary to wrap point back into the tile
            let nearest_vertex_number =
                (((angle + (PI / 12.0)).rem_euclid(PI * 2.0)) / (PI / 6.0)).floor();

            // transform point to frame where relevant vertex is in the center
            let (vertex_boost, vertex_angle) = (
                TwelveTwelve::TWELVETWELVE,
                (nearest_vertex_number * PI / 6.0).rem_euclid(PI * 2.0),
            );
            let (v_cosh, v_sinh, v_cos, v_sin) = (
                (-vertex_boost).cosh(),
                (-vertex_boost).sinh(),
                (-vertex_angle).cos(),
                (-vertex_angle).sin(),
            );
            let transformed_point = Minkowski::from([
                r_coords[0] * v_cosh * v_cos - r_coords[1] * v_cosh * v_sin
                    + r_coords[2] * v_sinh,
                r_coords[0] * v_sin + r_coords[1] * v_cos,
                r_coords[0] * v_sinh * v_cos - r_coords[1] * v_sinh * v_sin
                    + r_coords[2] * v_cosh,
            ]);
            // get coords of point in transformed frame
            let trans_angle =
                transformed_point.coordinates[1].atan2(transformed_point.coordinates[0]);
            // find which sector the transformed point is in
            let sector = (((trans_angle + (PI / 12.0)).rem_euclid(2.0 * PI)) / (PI / 6.0)).floor();

            // transform to tile
            let eta = TwelveTwelve::CUSP_TO_EDGE;
            let wrapped: [f64; 3];
            match sector {
                7.0 => {
                    let theta_1 = (nearest_vertex_number + 5.0).rem_euclid(12.0);
                    wrapped =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r.coordinates());
                }
                8.0 => {
                    let theta_1 = (nearest_vertex_number + 5.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r.coordinates());
                    let theta_2 = (theta_1 + 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                }
                9.0 => {
                    let theta_1 = (nearest_vertex_number + 5.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r.coordinates());
                    let theta_2 = (theta_1 + 5.0).rem_euclid(12.0);
                    let wrapped_2 =
                        TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let theta_3 = (theta_2 + 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                }
                10.0 => {
                    let theta_1 = (nearest_vertex_number + 5.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r.coordinates());
                    let theta_2 = (theta_1 + 5.0).rem_euclid(12.0);
                    let wrapped_2 =
                        TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let theta_3 = (theta_2 + 5.0).rem_euclid(12.0);
                    let wrapped_3 =
                        TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                    let theta_4 = (theta_3 + 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_4 * PI / 6.0 + PI / 12.0, &wrapped_3);
                }
                11.0 => {
                    let theta_1 = (nearest_vertex_number + 5.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r.coordinates());
                    let theta_2 = (theta_1 + 5.0).rem_euclid(12.0);
                    let wrapped_2 =
                        TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let theta_3 = (theta_2 + 5.0).rem_euclid(12.0);
                    let wrapped_3 =
                        TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                    let theta_4 = (theta_3 + 5.0).rem_euclid(12.0);
                    let wrapped_4 =
                        TwelveTwelve::gamma(eta, theta_4 * PI / 6.0 + PI / 12.0, &wrapped_3);
                    let theta_5 = (theta_4 + 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_5 * PI / 6.0 + PI / 12.0, &wrapped_4);
                }
                5.0 => {
                    let theta_1 = (nearest_vertex_number + 6.0).rem_euclid(12.0);
                    wrapped =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r.coordinates());
                }
                4.0 => {
                    let theta_1 = (nearest_vertex_number + 6.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r.coordinates());
                    let theta_2 = (theta_1 - 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                }
                3.0 => {
                    let theta_1 = (nearest_vertex_number + 6.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r.coordinates());
                    let theta_2 = (theta_1 - 5.0).rem_euclid(12.0);
                    let wrapped_2 =
                        TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let theta_3 = (theta_2 - 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                }
                2.0 => {
                    let theta_1 = (nearest_vertex_number + 6.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r.coordinates());
                    let theta_2 = (theta_1 - 5.0).rem_euclid(12.0);
                    let wrapped_2 =
                        TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let theta_3 = (theta_2 - 5.0).rem_euclid(12.0);
                    let wrapped_3 =
                        TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                    let theta_4 = (theta_3 - 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_4 * PI/6.0 - PI/12.0, &wrapped_3);
                }
                1.0 => {
                    let theta_1 = (nearest_vertex_number + 6.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r.coordinates());
                    let theta_2 = (theta_1 - 5.0).rem_euclid(12.0);
                    let wrapped_2 =
                        TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let theta_3 = (theta_2 - 5.0).rem_euclid(12.0);
                    let wrapped_3 =
                        TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                    let theta_4 = (theta_3 - 5.0).rem_euclid(12.0);
                    let wrapped_4 =
                        TwelveTwelve::gamma(eta, theta_4 * PI/6.0 + PI/12.0, &wrapped_3);
                    let theta_5 = (theta_4 - 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_5 * PI/6.0 + 12.0, &wrapped_4);
                }
                0.0 => {
                    if transformed_point.coordinates[1] >= 0.0 {
                        let theta_1 = (nearest_vertex_number + 6.0).rem_euclid(12.0);
                        let wrapped_1 =
                            TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r.coordinates());
                        let theta_2 = (theta_1 - 5.0).rem_euclid(12.0);
                        let wrapped_2 =
                            TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                        let theta_3 = (theta_2 - 5.0).rem_euclid(12.0);
                        let wrapped_3 =
                            TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                        let theta_4 = (theta_3 - 5.0).rem_euclid(12.0);
                        let wrapped_4 =
                            TwelveTwelve::gamma(eta, theta_4 * PI/6.0 + PI/12.0, &wrapped_3);
                        let theta_5 = (theta_4 - 5.0).rem_euclid(12.0);
                        let wrapped_5 =
                        TwelveTwelve::gamma(eta, theta_5 * PI/6.0 + 12.0, &wrapped_4);
                        let theta_6 = (theta_5 - 5.0).rem_euclid(12.0);
                        wrapped = TwelveTwelve::gamma(eta, theta_6*PI/6.0 + PI/12.0, &wrapped_5);
                    } else {
                        let theta_1 = (nearest_vertex_number + 5.0).rem_euclid(12.0);
                        let wrapped_1 =
                            TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r.coordinates());
                        let theta_2 = (theta_1 + 5.0).rem_euclid(12.0);
                        let wrapped_2 =
                            TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                        let theta_3 = (theta_2 + 5.0).rem_euclid(12.0);
                        let wrapped_3 =
                            TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                        let theta_4 = (theta_3 + 5.0).rem_euclid(12.0);
                        let wrapped_4 =
                            TwelveTwelve::gamma(eta, theta_4 * PI / 6.0 + PI / 12.0, &wrapped_3);
                        let theta_5 = (theta_4 + 5.0).rem_euclid(12.0);
                        let wrapped_5 =
                            TwelveTwelve::gamma(eta, theta_5 * PI / 6.0 + PI / 12.0, &wrapped_4);
                        let theta_6 = (theta_5 + 5.0).rem_euclid(12.0);
                        wrapped = TwelveTwelve::gamma(eta, theta_6 * PI/6.0 + PI/12.0, &wrapped_5);
                    }
                }
                _ => return Err(Error::CannotWrapProperties),
            }
            let wrapped_hyperbolic =
                Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from(wrapped), r.skirt());
            *r = wrapped_hyperbolic;
            Ok(properties)
        } else {
            Err(Error::CannotWrapProperties)
        }
    }
}

impl GenerateGhosts<Point<Hyperbolic<3>>> for Periodic<TwelveTwelve> {
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }
    /// Place periodic images of sites near the edges of the periodic boundary
    #[inline]
    fn generate_ghosts(&self, site_properties: &Point<Hyperbolic<3>>) -> ArrayVec<Point<Hyperbolic<3>>, MAX_GHOSTS> {
        let mut result = ArrayVec::new();
        let r = site_properties.position();

        // transform to tile
        let eta = TwelveTwelve::CUSP_TO_EDGE;
        let gamma_pt = |theta: f64, point: &[f64; 3]| {
            let ghost = TwelveTwelve::gamma(eta, theta, point);
            let new_hyperbolic =
                Hyperbolic::from_minkowski_coordinates(Minkowski::from(ghost), r.skirt());
            let mut new_site = *site_properties;
            *new_site.position_mut() = new_hyperbolic;
            new_site
        };
        // identify which octant the point is in
        let angle = (r.coordinates()[1].atan2(r.coordinates()[0])).rem_euclid(2.0 * PI);
        let nearest_vertex_number =
            (((angle + (PI / 12.0)).rem_euclid(PI * 2.0)) / (PI / 6.0)).floor();
        let coords = *r.coordinates();

        let theta_1a = (nearest_vertex_number + 6.0).rem_euclid(12.0);
        let ghost_1a = gamma_pt(theta_1a * PI / 6.0 + PI / 12.0, &coords);
        result.push(ghost_1a);
        let theta_1b = (nearest_vertex_number + 5.0).rem_euclid(12.0);
        let ghost_1b = gamma_pt(theta_1b * PI / 6.0 + PI / 12.0, &coords);
        result.push(ghost_1b);

        let theta_2a = (theta_1a - 5.0).rem_euclid(12.0);
        let ghost_2a = gamma_pt(
            theta_2a * PI / 6.0 + PI / 12.0,
            ghost_1a.position.coordinates(),
        );
        result.push(ghost_2a);
        let theta_2b = (theta_1b + 5.0).rem_euclid(12.0);
        let ghost_2b = gamma_pt(
            theta_2b * PI / 6.0 + PI / 12.0,
            ghost_1b.position.coordinates(),
        );
        result.push(ghost_2b);

        let theta_3a = (theta_2a - 5.0).rem_euclid(12.0);
        let ghost_3a = gamma_pt(
            theta_3a * PI / 6.0 + PI / 12.0,
            ghost_2a.position.coordinates(),
        );
        result.push(ghost_3a);
        let theta_3b = (theta_2b + 5.0).rem_euclid(12.0);
        let ghost_3b = gamma_pt(
            theta_3b * PI / 6.0 + PI / 12.0,
            ghost_2b.position.coordinates(),
        );
        result.push(ghost_3b);

        let theta_4a = (theta_3a - 5.0).rem_euclid(12.0);
        let ghost_4a = gamma_pt(
            theta_4a * PI / 6.0 + PI / 12.0,
            ghost_3a.position.coordinates(),
        );
        result.push(ghost_4a);
        let theta_4b = (theta_3b + 5.0).rem_euclid(12.0);
        let ghost_4b = gamma_pt(
            theta_4b * PI / 6.0 + PI / 12.0,
            ghost_3b.position.coordinates()
        );
        result.push(ghost_4b);

        let theta_5a = (theta_4a - 5.0).rem_euclid(12.0);
        let ghost_5a = gamma_pt(
            theta_5a * PI / 6.0 + PI / 12.0,
            ghost_4a.position.coordinates(),
        );
        result.push(ghost_5a);
        let theta_5b = (theta_4b + 5.0).rem_euclid(12.0);
        let ghost_5b = gamma_pt(
            theta_5b * PI / 6.0 + PI / 12.0,
            ghost_4b.position.coordinates()
        );
        result.push(ghost_5b);

        let theta_6 = (theta_5a - 5.0).rem_euclid(12.0);
        let ghost_6 = gamma_pt(
            theta_6 * PI / 6.0 + PI / 12.0,
            ghost_5a.position.coordinates()
        );
        result.push(ghost_6);

        result
    }
}

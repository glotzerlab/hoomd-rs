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
    property::{Orientation, OrientedHyperbolicPoint, Point, Position},
};
use hoomd_geometry::shape::TwelveTwelve;
use hoomd_manifold::{Hyperbolic, Minkowski};
use hoomd_vector::Angle;

impl MaximumAllowableInteractionRange for TwelveTwelve {
    /// The largest value that the maximum interaction range can take.
    ///
    /// This bound is determined by the edge length of the dodecagon.
    #[inline]
    fn maximum_allowable_interaction_range(&self) -> f64 {
        TwelveTwelve::CUSP_TO_EDGE
    }
}

impl Wrap<Point<Hyperbolic<3>>> for Periodic<TwelveTwelve> {
    /// Wrap a point in hyperbolic space to the inside of the {12,12} tile.
    ///
    /// Note that the function fails to wrap points that are outside the
    /// dodecagon and further than `TwelveTwelve::EDGE_LENGTH/2` from any of
    /// the vertices.
    #[inline]
    #[expect(clippy::too_many_lines, reason = "complicated function")]
    fn wrap(&self, properties: Point<Hyperbolic<3>>) -> Result<Point<Hyperbolic<3>>, Error> {
        let mut properties = properties;
        let r = properties.position_mut();
        let r_coords = r.coordinates();
        let angle = r_coords[1].atan2(r_coords[0]).rem_euclid(2.0 * PI);

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
                r_coords[0] * v_cosh * v_cos - r_coords[1] * v_cosh * v_sin + r_coords[2] * v_sinh,
                r_coords[0] * v_sin + r_coords[1] * v_cos,
                r_coords[0] * v_sinh * v_cos - r_coords[1] * v_sinh * v_sin + r_coords[2] * v_cosh,
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
                    wrapped = TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                }
                8.0 => {
                    let theta_1 = (nearest_vertex_number + 5.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let theta_2 = (theta_1 + 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                }
                9.0 => {
                    let theta_1 = (nearest_vertex_number + 5.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let theta_2 = (theta_1 + 5.0).rem_euclid(12.0);
                    let wrapped_2 =
                        TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let theta_3 = (theta_2 + 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                }
                10.0 => {
                    let theta_1 = (nearest_vertex_number + 5.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
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
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
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
                    wrapped = TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                }
                4.0 => {
                    let theta_1 = (nearest_vertex_number + 6.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let theta_2 = (theta_1 - 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                }
                3.0 => {
                    let theta_1 = (nearest_vertex_number + 6.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let theta_2 = (theta_1 - 5.0).rem_euclid(12.0);
                    let wrapped_2 =
                        TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let theta_3 = (theta_2 - 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                }
                2.0 => {
                    let theta_1 = (nearest_vertex_number + 6.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let theta_2 = (theta_1 - 5.0).rem_euclid(12.0);
                    let wrapped_2 =
                        TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let theta_3 = (theta_2 - 5.0).rem_euclid(12.0);
                    let wrapped_3 =
                        TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                    let theta_4 = (theta_3 - 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_4 * PI / 6.0 - PI / 12.0, &wrapped_3);
                }
                1.0 => {
                    let theta_1 = (nearest_vertex_number + 6.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let theta_2 = (theta_1 - 5.0).rem_euclid(12.0);
                    let wrapped_2 =
                        TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let theta_3 = (theta_2 - 5.0).rem_euclid(12.0);
                    let wrapped_3 =
                        TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                    let theta_4 = (theta_3 - 5.0).rem_euclid(12.0);
                    let wrapped_4 =
                        TwelveTwelve::gamma(eta, theta_4 * PI / 6.0 + PI / 12.0, &wrapped_3);
                    let theta_5 = (theta_4 - 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_5 * PI / 6.0 + 12.0, &wrapped_4);
                }
                0.0 => {
                    if transformed_point.coordinates[1] >= 0.0 {
                        let theta_1 = (nearest_vertex_number + 6.0).rem_euclid(12.0);
                        let wrapped_1 =
                            TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                        let theta_2 = (theta_1 - 5.0).rem_euclid(12.0);
                        let wrapped_2 =
                            TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                        let theta_3 = (theta_2 - 5.0).rem_euclid(12.0);
                        let wrapped_3 =
                            TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                        let theta_4 = (theta_3 - 5.0).rem_euclid(12.0);
                        let wrapped_4 =
                            TwelveTwelve::gamma(eta, theta_4 * PI / 6.0 + PI / 12.0, &wrapped_3);
                        let theta_5 = (theta_4 - 5.0).rem_euclid(12.0);
                        let wrapped_5 =
                            TwelveTwelve::gamma(eta, theta_5 * PI / 6.0 + 12.0, &wrapped_4);
                        let theta_6 = (theta_5 - 5.0).rem_euclid(12.0);
                        wrapped =
                            TwelveTwelve::gamma(eta, theta_6 * PI / 6.0 + PI / 12.0, &wrapped_5);
                    } else {
                        let theta_1 = (nearest_vertex_number + 5.0).rem_euclid(12.0);
                        let wrapped_1 =
                            TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
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
                        wrapped =
                            TwelveTwelve::gamma(eta, theta_6 * PI / 6.0 + PI / 12.0, &wrapped_5);
                    }
                }
                _ => return Err(Error::CannotWrapProperties),
            }
            let wrapped_hyperbolic =
                Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from(wrapped));
            *r = wrapped_hyperbolic;
            Ok(properties)
        } else {
            Err(Error::CannotWrapProperties)
        }
    }
}

impl Wrap<OrientedHyperbolicPoint<3, Angle>> for Periodic<TwelveTwelve> {
    /// Wrap the positions and orientations of oriented bodies in hyperbolic
    /// space under the {12,12} tiling.
    #[inline]
    #[allow(clippy::too_many_lines, reason = "complicated function")]
    fn wrap(
        &self,
        properties: OrientedHyperbolicPoint<3, Angle>,
    ) -> Result<OrientedHyperbolicPoint<3, Angle>, Error> {
        let original_orientation = properties.orientation.theta;
        let mut properties = properties;
        let r = properties.position_mut();
        let r_coords = r.coordinates();
        let angle = r_coords[1].atan2(r_coords[0]).rem_euclid(2.0 * PI);

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
                r_coords[0] * v_cosh * v_cos - r_coords[1] * v_cosh * v_sin + r_coords[2] * v_sinh,
                r_coords[0] * v_sin + r_coords[1] * v_cos,
                r_coords[0] * v_sinh * v_cos - r_coords[1] * v_sinh * v_sin + r_coords[2] * v_cosh,
            ]);
            // get coords of point in transformed frame
            let trans_angle =
                transformed_point.coordinates[1].atan2(transformed_point.coordinates[0]);
            // find which sector the transformed point is in
            let sector = (((trans_angle + (PI / 12.0)).rem_euclid(2.0 * PI)) / (PI / 6.0)).floor();

            // transform to tile
            let eta = TwelveTwelve::CUSP_TO_EDGE;
            let wrapped: [f64; 3];
            let relative_angle: f64;
            match sector {
                7.0 => {
                    let theta_1 = (nearest_vertex_number + 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    relative_angle =
                        TwelveTwelve::reorient(theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                }
                8.0 => {
                    let theta_1 = (nearest_vertex_number + 5.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let relative_angle_1 =
                        TwelveTwelve::reorient(theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let theta_2 = (theta_1 + 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    relative_angle =
                        TwelveTwelve::reorient(theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1)
                            + relative_angle_1;
                }
                9.0 => {
                    let theta_1 = (nearest_vertex_number + 5.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let relative_angle_1 =
                        TwelveTwelve::reorient(theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let theta_2 = (theta_1 + 5.0).rem_euclid(12.0);
                    let wrapped_2 =
                        TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let relative_angle_2 =
                        TwelveTwelve::reorient(theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let theta_3 = (theta_2 + 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                    relative_angle =
                        TwelveTwelve::reorient(theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2)
                            + relative_angle_1
                            + relative_angle_2;
                }
                10.0 => {
                    let theta_1 = (nearest_vertex_number + 5.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let relative_angle_1 =
                        TwelveTwelve::reorient(theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let theta_2 = (theta_1 + 5.0).rem_euclid(12.0);
                    let wrapped_2 =
                        TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let relative_angle_2 =
                        TwelveTwelve::reorient(theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let theta_3 = (theta_2 + 5.0).rem_euclid(12.0);
                    let wrapped_3 =
                        TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                    let relative_angle_3 =
                        TwelveTwelve::reorient(theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                    let theta_4 = (theta_3 + 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_4 * PI / 6.0 + PI / 12.0, &wrapped_3);
                    relative_angle =
                        TwelveTwelve::reorient(theta_4 * PI / 6.0 + PI / 12.0, &wrapped_3)
                            + relative_angle_1
                            + relative_angle_2
                            + relative_angle_3;
                }
                11.0 => {
                    let theta_1 = (nearest_vertex_number + 5.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let relative_angle_1 =
                        TwelveTwelve::reorient(theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let theta_2 = (theta_1 + 5.0).rem_euclid(12.0);
                    let wrapped_2 =
                        TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let relative_angle_2 =
                        TwelveTwelve::reorient(theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let theta_3 = (theta_2 + 5.0).rem_euclid(12.0);
                    let wrapped_3 =
                        TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                    let relative_angle_3 =
                        TwelveTwelve::reorient(theta_3 * PI / 6.0 + PI / 12.0, &wrapped_3);
                    let theta_4 = (theta_3 + 5.0).rem_euclid(12.0);
                    let wrapped_4 =
                        TwelveTwelve::gamma(eta, theta_4 * PI / 6.0 + PI / 12.0, &wrapped_3);
                    let relative_angle_4 =
                        TwelveTwelve::reorient(theta_4 * PI / 6.0 + PI / 12.0, &wrapped_3);
                    let theta_5 = (theta_4 + 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_5 * PI / 6.0 + PI / 12.0, &wrapped_4);
                    relative_angle =
                        TwelveTwelve::reorient(theta_5 * PI / 6.0 + PI / 12.0, &wrapped_4)
                            + relative_angle_1
                            + relative_angle_2
                            + relative_angle_3
                            + relative_angle_4;
                }
                5.0 => {
                    let theta_1 = (nearest_vertex_number + 6.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    relative_angle =
                        TwelveTwelve::reorient(theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                }
                4.0 => {
                    let theta_1 = (nearest_vertex_number + 6.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let relative_angle_1 =
                        TwelveTwelve::reorient(theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let theta_2 = (theta_1 - 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    relative_angle =
                        TwelveTwelve::reorient(theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1)
                            + relative_angle_1;
                }
                3.0 => {
                    let theta_1 = (nearest_vertex_number + 6.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let relative_angle_1 =
                        TwelveTwelve::reorient(theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let theta_2 = (theta_1 - 5.0).rem_euclid(12.0);
                    let wrapped_2 =
                        TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let relative_angle_2 =
                        TwelveTwelve::reorient(theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let theta_3 = (theta_2 - 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                    relative_angle =
                        TwelveTwelve::reorient(theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2)
                            + relative_angle_1
                            + relative_angle_2;
                }
                2.0 => {
                    let theta_1 = (nearest_vertex_number + 6.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let relative_angle_1 =
                        TwelveTwelve::reorient(theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let theta_2 = (theta_1 - 5.0).rem_euclid(12.0);
                    let wrapped_2 =
                        TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let relative_angle_2 =
                        TwelveTwelve::reorient(theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let theta_3 = (theta_2 - 5.0).rem_euclid(12.0);
                    let wrapped_3 =
                        TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                    let relative_angle_3 =
                        TwelveTwelve::reorient(theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                    let theta_4 = (theta_3 - 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_4 * PI / 6.0 - PI / 12.0, &wrapped_3);
                    relative_angle =
                        TwelveTwelve::reorient(theta_4 * PI / 6.0 + PI / 12.0, &wrapped_3)
                            + relative_angle_1
                            + relative_angle_2
                            + relative_angle_3;
                }
                1.0 => {
                    let theta_1 = (nearest_vertex_number + 6.0).rem_euclid(12.0);
                    let wrapped_1 =
                        TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let relative_angle_1 =
                        TwelveTwelve::reorient(theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                    let theta_2 = (theta_1 - 5.0).rem_euclid(12.0);
                    let wrapped_2 =
                        TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let relative_angle_2 =
                        TwelveTwelve::reorient(theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                    let theta_3 = (theta_2 - 5.0).rem_euclid(12.0);
                    let wrapped_3 =
                        TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                    let relative_angle_3 =
                        TwelveTwelve::reorient(theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                    let theta_4 = (theta_3 - 5.0).rem_euclid(12.0);
                    let wrapped_4 =
                        TwelveTwelve::gamma(eta, theta_4 * PI / 6.0 + PI / 12.0, &wrapped_3);
                    let relative_angle_4 =
                        TwelveTwelve::reorient(theta_4 * PI / 6.0 + PI / 12.0, &wrapped_3);
                    let theta_5 = (theta_4 - 5.0).rem_euclid(12.0);
                    wrapped = TwelveTwelve::gamma(eta, theta_5 * PI / 6.0 + PI / 12.0, &wrapped_4);
                    relative_angle =
                        TwelveTwelve::reorient(theta_5 * PI / 6.0 + PI / 12.0, &wrapped_4)
                            + relative_angle_1
                            + relative_angle_2
                            + relative_angle_3
                            + relative_angle_4;
                }
                0.0 => {
                    if transformed_point.coordinates[1] >= 0.0 {
                        let theta_1 = (nearest_vertex_number + 6.0).rem_euclid(12.0);
                        let wrapped_1 =
                            TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                        let relative_angle_1 =
                            TwelveTwelve::reorient(theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                        let theta_2 = (theta_1 - 5.0).rem_euclid(12.0);
                        let wrapped_2 =
                            TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                        let relative_angle_2 =
                            TwelveTwelve::reorient(theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                        let theta_3 = (theta_2 - 5.0).rem_euclid(12.0);
                        let wrapped_3 =
                            TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                        let relative_angle_3 =
                            TwelveTwelve::reorient(theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                        let theta_4 = (theta_3 - 5.0).rem_euclid(12.0);
                        let wrapped_4 =
                            TwelveTwelve::gamma(eta, theta_4 * PI / 6.0 + PI / 12.0, &wrapped_3);
                        let relative_angle_4 =
                            TwelveTwelve::reorient(theta_4 * PI / 6.0 + PI / 12.0, &wrapped_3);
                        let theta_5 = (theta_4 - 5.0).rem_euclid(12.0);
                        let wrapped_5 =
                            TwelveTwelve::gamma(eta, theta_5 * PI / 6.0 + PI / 12.0, &wrapped_4);
                        let relative_angle_5 =
                            TwelveTwelve::reorient(theta_5 * PI / 6.0 + PI / 12.0, &wrapped_4);
                        let theta_6 = (theta_5 - 5.0).rem_euclid(12.0);
                        wrapped =
                            TwelveTwelve::gamma(eta, theta_6 * PI / 6.0 + PI / 12.0, &wrapped_5);
                        relative_angle =
                            TwelveTwelve::reorient(theta_6 * PI / 6.0 + PI / 12.0, &wrapped_5)
                                + relative_angle_1
                                + relative_angle_2
                                + relative_angle_3
                                + relative_angle_4
                                + relative_angle_5;
                    } else {
                        let theta_1 = (nearest_vertex_number + 5.0).rem_euclid(12.0);
                        let wrapped_1 =
                            TwelveTwelve::gamma(eta, theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                        let relative_angle_1 =
                            TwelveTwelve::reorient(theta_1 * PI / 6.0 + PI / 12.0, r_coords);
                        let theta_2 = (theta_1 + 5.0).rem_euclid(12.0);
                        let wrapped_2 =
                            TwelveTwelve::gamma(eta, theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                        let relative_angle_2 =
                            TwelveTwelve::reorient(theta_2 * PI / 6.0 + PI / 12.0, &wrapped_1);
                        let theta_3 = (theta_2 + 5.0).rem_euclid(12.0);
                        let wrapped_3 =
                            TwelveTwelve::gamma(eta, theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                        let relative_angle_3 =
                            TwelveTwelve::reorient(theta_3 * PI / 6.0 + PI / 12.0, &wrapped_2);
                        let theta_4 = (theta_3 + 5.0).rem_euclid(12.0);
                        let wrapped_4 =
                            TwelveTwelve::gamma(eta, theta_4 * PI / 6.0 + PI / 12.0, &wrapped_3);
                        let relative_angle_4 =
                            TwelveTwelve::reorient(theta_4 * PI / 6.0 + PI / 12.0, &wrapped_3);
                        let theta_5 = (theta_4 + 5.0).rem_euclid(12.0);
                        let wrapped_5 =
                            TwelveTwelve::gamma(eta, theta_5 * PI / 6.0 + PI / 12.0, &wrapped_4);
                        let relative_angle_5 =
                            TwelveTwelve::reorient(theta_5 * PI / 6.0 + PI / 12.0, &wrapped_4);
                        let theta_6 = (theta_5 + 5.0).rem_euclid(12.0);
                        wrapped =
                            TwelveTwelve::gamma(eta, theta_6 * PI / 6.0 + PI / 12.0, &wrapped_5);
                        relative_angle =
                            TwelveTwelve::reorient(theta_6 * PI / 6.0 + PI / 12.0, &wrapped_5)
                                + relative_angle_1
                                + relative_angle_2
                                + relative_angle_3
                                + relative_angle_4
                                + relative_angle_5;
                    }
                }
                _ => return Err(Error::CannotWrapProperties),
            }
            let wrapped_hyperbolic =
                Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from(wrapped));
            let mut final_orientation = relative_angle + original_orientation;
            while final_orientation > PI {
                final_orientation -= 2.0 * PI;
            }
            while final_orientation <= -PI {
                final_orientation += 2.0 * PI;
            }
            *r = wrapped_hyperbolic;
            *properties.orientation_mut() = Angle::from(final_orientation);
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
    fn generate_ghosts(
        &self,
        site_properties: &Point<Hyperbolic<3>>,
    ) -> ArrayVec<Point<Hyperbolic<3>>, MAX_GHOSTS> {
        let mut result = ArrayVec::new();
        let r = site_properties.position();

        // transform to tile
        let eta = TwelveTwelve::CUSP_TO_EDGE;
        let gamma_pt = |theta: f64, point: &[f64; 3]| {
            let ghost = TwelveTwelve::gamma(eta, theta, point);
            let new_hyperbolic = Hyperbolic::from_minkowski_coordinates(Minkowski::from(ghost));
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
            ghost_3b.position.coordinates(),
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
            ghost_4b.position.coordinates(),
        );
        result.push(ghost_5b);

        let theta_6 = (theta_5a - 5.0).rem_euclid(12.0);
        let ghost_6 = gamma_pt(
            theta_6 * PI / 6.0 + PI / 12.0,
            ghost_5a.position.coordinates(),
        );
        result.push(ghost_6);

        result
    }
}

impl GenerateGhosts<OrientedHyperbolicPoint<3, Angle>> for Periodic<TwelveTwelve> {
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }
    /// Place periodic images of sites near the edge of the periodic boundary.
    #[inline]
    #[expect(clippy::too_many_lines, reason = "complicated function")]
    fn generate_ghosts(
        &self,
        site_properties: &OrientedHyperbolicPoint<3, Angle>,
    ) -> ArrayVec<OrientedHyperbolicPoint<3, Angle>, MAX_GHOSTS> {
        let mut result = ArrayVec::new();
        let r = site_properties.position();

        // transform to tile
        let eta = TwelveTwelve::CUSP_TO_EDGE;
        let gamma_pt = |theta: f64, point: &[f64; 3]| {
            let ghost = TwelveTwelve::gamma(eta, theta, point);
            let new_hyperbolic = Hyperbolic::from_minkowski_coordinates(Minkowski::from(ghost));
            let mut new_site = *site_properties;
            *new_site.position_mut() = new_hyperbolic;
            new_site
        };
        // identify which octant the point is in
        let angle = (r.coordinates()[1].atan2(r.coordinates()[0])).rem_euclid(2.0 * PI);
        let nearest_vertex_number =
            (((angle + (PI / 12.0)).rem_euclid(PI * 2.0)) / (PI / 6.0)).floor();
        let coords = *r.coordinates();
        let orientation = site_properties.orientation.theta;

        let theta_1a = (nearest_vertex_number + 6.0).rem_euclid(12.0);
        let ghost_1a = gamma_pt(theta_1a * PI / 6.0 + PI / 12.0, &coords);
        let rel_angle_1a = TwelveTwelve::reorient(theta_1a * PI / 6.0 + PI / 12.0, &coords);
        let orientation_1a = rel_angle_1a + orientation;
        let mut final_orientation = orientation_1a;
        while final_orientation > PI {
            final_orientation -= 2.0 * PI;
        }
        while final_orientation <= -PI {
            final_orientation += 2.0 * PI;
        }
        let mut new_site_1a = *site_properties;
        *new_site_1a.position_mut() = ghost_1a.position;
        *new_site_1a.orientation_mut() = Angle::from(final_orientation);
        result.push(new_site_1a);

        let theta_1b = (nearest_vertex_number + 5.0).rem_euclid(12.0);
        let ghost_1b = gamma_pt(theta_1b * PI / 6.0 + PI / 12.0, &coords);
        let rel_angle_1b = TwelveTwelve::reorient(theta_1b * PI / 6.0 + PI / 12.0, &coords);
        let orientation_1b = orientation + rel_angle_1b;
        let mut final_orientation = orientation_1b;
        while final_orientation > PI {
            final_orientation -= 2.0 * PI;
        }
        while final_orientation <= -PI {
            final_orientation += 2.0 * PI;
        }
        let mut new_site_1b = *site_properties;
        *new_site_1b.position_mut() = ghost_1b.position;
        *new_site_1b.orientation_mut() = Angle::from(final_orientation);
        result.push(new_site_1b);

        let theta_2a = (theta_1a - 5.0).rem_euclid(12.0);
        let ghost_2a = gamma_pt(
            theta_2a * PI / 6.0 + PI / 12.0,
            ghost_1a.position.coordinates(),
        );
        let rel_angle_2a = TwelveTwelve::reorient(
            theta_2a * PI / 6.0 + PI / 12.0,
            ghost_1a.position.coordinates(),
        );
        let orientation_2a = orientation_1a + rel_angle_2a;
        let mut final_orientation = orientation_2a;
        while final_orientation > PI {
            final_orientation -= 2.0 * PI;
        }
        while final_orientation <= -PI {
            final_orientation += 2.0 * PI;
        }
        let mut new_site_2a = *site_properties;
        *new_site_2a.position_mut() = ghost_2a.position;
        *new_site_2a.orientation_mut() = Angle::from(final_orientation);
        result.push(new_site_2a);

        let theta_2b = (theta_1b + 5.0).rem_euclid(12.0);
        let ghost_2b = gamma_pt(
            theta_2b * PI / 6.0 + PI / 12.0,
            ghost_1b.position.coordinates(),
        );
        let rel_angle_2b = TwelveTwelve::reorient(
            theta_2b * PI / 6.0 + PI / 12.0,
            ghost_1b.position.coordinates(),
        );
        let orientation_2b = orientation_1b + rel_angle_2b;
        let mut final_orientation = orientation_2b;
        while final_orientation > PI {
            final_orientation -= 2.0 * PI;
        }
        while final_orientation <= -PI {
            final_orientation += 2.0 * PI;
        }
        let mut new_site_2b = *site_properties;
        *new_site_2b.position_mut() = ghost_2b.position;
        *new_site_2b.orientation_mut() = Angle::from(final_orientation);
        result.push(new_site_2b);

        let theta_3a = (theta_2a - 5.0).rem_euclid(12.0);
        let ghost_3a = gamma_pt(
            theta_3a * PI / 6.0 + PI / 12.0,
            ghost_2a.position.coordinates(),
        );
        let rel_angle_3a = TwelveTwelve::reorient(
            theta_3a * PI / 6.0 + PI / 12.0,
            ghost_2a.position.coordinates(),
        );
        let orientation_3a = orientation_2a + rel_angle_3a;
        let mut final_orientation = orientation_3a;
        while final_orientation > PI {
            final_orientation -= 2.0 * PI;
        }
        while final_orientation <= -PI {
            final_orientation += 2.0 * PI;
        }
        let mut new_site_3a = *site_properties;
        *new_site_3a.position_mut() = ghost_3a.position;
        *new_site_3a.orientation_mut() = Angle::from(final_orientation);
        result.push(new_site_3a);

        let theta_3b = (theta_2b + 5.0).rem_euclid(12.0);
        let ghost_3b = gamma_pt(
            theta_3b * PI / 6.0 + PI / 12.0,
            ghost_2b.position.coordinates(),
        );
        let rel_angle_3b = TwelveTwelve::reorient(
            theta_3b * PI / 6.0 + PI / 12.0,
            ghost_2b.position.coordinates(),
        );
        let orientation_3b = rel_angle_3b + orientation_2b;
        let mut final_orientation = orientation_3b;
        while final_orientation > PI {
            final_orientation -= 2.0 * PI;
        }
        while final_orientation <= -PI {
            final_orientation += 2.0 * PI;
        }
        let mut new_site_3b = *site_properties;
        *new_site_3b.position_mut() = ghost_3b.position;
        *new_site_3b.orientation_mut() = Angle::from(final_orientation);
        result.push(new_site_3b);

        let theta_4a = (theta_3a - 5.0).rem_euclid(12.0);
        let ghost_4a = gamma_pt(
            theta_4a * PI / 6.0 + PI / 12.0,
            ghost_3a.position.coordinates(),
        );
        let rel_angle_4a = TwelveTwelve::reorient(
            theta_4a * PI / 6.0 + PI / 12.0,
            ghost_3a.position.coordinates(),
        );
        let orientation_4a = orientation_3a + rel_angle_4a;
        let mut final_orientation = orientation_4a;
        while final_orientation > PI {
            final_orientation -= 2.0 * PI;
        }
        while final_orientation <= -PI {
            final_orientation += 2.0 * PI;
        }
        let mut new_site_4a = *site_properties;
        *new_site_4a.position_mut() = ghost_4a.position;
        *new_site_4a.orientation_mut() = Angle::from(final_orientation);
        result.push(new_site_4a);

        let theta_4b = (theta_3b + 5.0).rem_euclid(12.0);
        let ghost_4b = gamma_pt(
            theta_4b * PI / 6.0 + PI / 12.0,
            ghost_3b.position.coordinates(),
        );
        let rel_angle_4b = TwelveTwelve::reorient(
            theta_4b * PI / 6.0 + PI / 12.0,
            ghost_3b.position.coordinates(),
        );
        let orientation_4b = orientation_3b + rel_angle_4b;
        let mut final_orientation = orientation_4b;
        while final_orientation > PI {
            final_orientation -= 2.0 * PI;
        }
        while final_orientation <= -PI {
            final_orientation += 2.0 * PI;
        }
        let mut new_site_4b = *site_properties;
        *new_site_4b.position_mut() = ghost_4b.position;
        *new_site_4b.orientation_mut() = Angle::from(final_orientation);
        result.push(new_site_4b);

        let theta_5a = (theta_4a - 5.0).rem_euclid(12.0);
        let ghost_5a = gamma_pt(
            theta_5a * PI / 6.0 + PI / 12.0,
            ghost_4a.position.coordinates(),
        );
        let rel_angle_5a = TwelveTwelve::reorient(
            theta_5a * PI / 6.0 + PI / 12.0,
            ghost_4a.position.coordinates(),
        );
        let orientation_5a = orientation_4a + rel_angle_5a;
        let mut final_orientation = orientation_5a;
        while final_orientation > PI {
            final_orientation -= 2.0 * PI;
        }
        while final_orientation <= -PI {
            final_orientation += 2.0 * PI;
        }
        let mut new_site_5a = *site_properties;
        *new_site_5a.position_mut() = ghost_5a.position;
        *new_site_5a.orientation_mut() = Angle::from(final_orientation);
        result.push(new_site_5a);

        let theta_5b = (theta_4b + 5.0).rem_euclid(12.0);
        let ghost_5b = gamma_pt(
            theta_5b * PI / 6.0 + PI / 12.0,
            ghost_4b.position.coordinates(),
        );
        let rel_angle_5b = TwelveTwelve::reorient(
            theta_5b * PI / 6.0 + PI / 12.0,
            ghost_4b.position.coordinates(),
        );
        let orientation_5b = orientation_4a + rel_angle_5b;
        let mut final_orientation = orientation_5b;
        while final_orientation > PI {
            final_orientation -= 2.0 * PI;
        }
        while final_orientation <= -PI {
            final_orientation += 2.0 * PI;
        }
        let mut new_site_5b = *site_properties;
        *new_site_5b.position_mut() = ghost_5b.position;
        *new_site_5b.orientation_mut() = Angle::from(final_orientation);
        result.push(new_site_5b);

        let theta_6 = (theta_5a - 5.0).rem_euclid(12.0);
        let ghost_6 = gamma_pt(
            theta_6 * PI / 6.0 + PI / 12.0,
            ghost_5a.position.coordinates(),
        );
        let rel_angle_6 = TwelveTwelve::reorient(
            theta_6 * PI / 6.0 + PI / 12.0,
            ghost_5a.position.coordinates(),
        );
        let orientation_6 = orientation_5a + rel_angle_6;
        let mut final_orientation = orientation_6;
        while final_orientation > PI {
            final_orientation -= 2.0 * PI;
        }
        while final_orientation <= -PI {
            final_orientation += 2.0 * PI;
        }
        let mut nnew_site_6 = *site_properties;
        *nnew_site_6.position_mut() = ghost_6.position;
        *nnew_site_6.orientation_mut() = Angle::from(final_orientation);
        result.push(nnew_site_6);

        result
    }
}

// TODO!!! Write test code for oriented bodies
#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::Point;
    use approxim::assert_relative_eq;
    use hoomd_manifold::{Hyperbolic, HyperbolicDisk};
    use rand::{RngExt, SeedableRng, distr::Distribution, rngs::StdRng};
    use std::f64::consts::PI;

    #[test]
    fn doesnt_wrap_if_inside() {
        let r = TwelveTwelve::CUSP_TO_EDGE;
        let mut rng = StdRng::seed_from_u64(239);
        let disk = HyperbolicDisk {
            disk_radius: r.try_into().expect("hard-coded positive number"),
            point: Hyperbolic::from_minkowski_coordinates(Minkowski::from([0.0, 0.0, 1.0])),
        };
        for _ in 0..250 {
            let random_point: Hyperbolic<3> = disk.sample(&mut rng);
            let random_point = Point::new(random_point);

            let periodic = Periodic::new(0.5, TwelveTwelve {}).expect("hard-coded positive number");
            let wrapped_point = periodic.wrap(random_point).expect("hard-coded");
            assert_eq!(
                random_point.position.coordinates(),
                wrapped_point.position.coordinates()
            );
        }
    }

    #[test]
    fn wraps_to_opposite_edge() {
        let mut rng = StdRng::seed_from_u64(54321);
        let side = f64::from(rng.random_range(0..12));
        let boost = TwelveTwelve::CUSP_TO_EDGE + 0.5;
        let offset = PI / 12.0;
        let point = Hyperbolic::<3>::from_polar_coordinates(boost, side * PI / 6.0 + offset);
        let point = Point::new(point);
        let periodic = Periodic::new(1.0, TwelveTwelve {}).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(point).expect("hard-coded");

        let wrapped_side = (side + 6.0).rem_euclid(12.0);
        let octant = (((wrapped_point.position.coordinates()[1]
            .atan2(wrapped_point.position.coordinates()[0]))
            / (PI / 6.0))
            .floor())
        .rem_euclid(12.0);

        // Check that point is wrapped to correct octant
        assert_eq!(wrapped_side, octant);

        // Check that point mapping is correct
        let new_boost = 2.0
            * (TwelveTwelve::TWELVETWELVE.tanh() * ((PI / 6.0).sin())
                / (2.0 * ((PI / 12.0).sin())))
            .atanh()
            - boost;
        let ans = Hyperbolic::<3>::from_polar_coordinates(
            new_boost,
            (wrapped_side + 1.0) * (PI / 6.0) - offset,
        );
        assert_relative_eq!(ans, wrapped_point.position, epsilon = 1e-12);
        assert_relative_eq!(
            -TwelveTwelve::distance_to_boundary(&point.position),
            TwelveTwelve::distance_to_boundary(&wrapped_point.position),
            epsilon = 1e-12
        );
    }

    #[test]
    fn wraps_to_opposite_edge_distance() {
        let mut rng = StdRng::seed_from_u64(1);
        let side = f64::from(rng.random_range(0..12));
        let boost = TwelveTwelve::CUSP_TO_EDGE + 0.2;
        let offset = PI / 12.0 + rng.random::<f64>() * (PI / 24.0);
        let point = Hyperbolic::<3>::from_polar_coordinates(boost, side * PI / 6.0 + offset);
        let point = Point::new(point);
        let periodic = Periodic::new(1.0, TwelveTwelve {}).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(point).expect("hard-coded");

        // Check that mapping preserves the distance to the boundary
        assert_relative_eq!(
            -TwelveTwelve::distance_to_boundary(&point.position),
            TwelveTwelve::distance_to_boundary(&wrapped_point.position),
            epsilon = 1e-12
        );
    }

    #[test]
    fn consistency_edge() {
        let mut rng = StdRng::seed_from_u64(5212);
        let side = f64::from(rng.random_range(0..12));
        let boost = TwelveTwelve::CUSP_TO_EDGE + 0.5;
        let offset = PI / 12.0 + 0.15 * rng.random::<f64>();
        let point = Hyperbolic::<3>::from_polar_coordinates(boost, side * PI / 6.0 + offset);
        let point = Point::new(point);
        let periodic = Periodic::new(1.0, TwelveTwelve {}).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(point).expect("hard-coded");
        let wrapped_poincare = wrapped_point.position.to_poincare();

        // Check that mapping is consistent with Poincare transformation
        let (q_u, q_v) = (
            point.position.to_poincare()[0],
            point.position.to_poincare()[1],
        );
        let phi = (side + 6.0).rem_euclid(12.0) * PI / 6.0 + PI / 12.0;
        let a = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi.cos());
        let c = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi.sin());
        let pref = 1.0 / ((b * q_v - c * q_u).powi(2) + (a + b * q_u + c * q_v).powi(2));
        let ans = [
            pref * (q_u * (a.powi(2) + b.powi(2) - c.powi(2))
                + 2.0 * b * c * q_v
                + a * b * (1.0 + q_u.powi(2) + q_v.powi(2))),
            pref * (q_v * (a.powi(2) - b.powi(2) + c.powi(2))
                + 2.0 * b * c * q_u
                + a * c * (1.0 + q_u.powi(2) + q_v.powi(2))),
        ];

        assert_relative_eq!(ans[0], wrapped_poincare[0], epsilon = 1e-12);
        assert_relative_eq!(ans[1], wrapped_poincare[1], epsilon = 1e-12);
    }

    #[test]
    #[allow(clippy::too_many_lines, reason = "complicated function")]
    fn wraps_vertex() {
        let boost = TwelveTwelve::TWELVETWELVE + 0.4;
        let point = Hyperbolic::<3>::from_polar_coordinates(boost, PI / 2.0 - 0.0001);
        let point = Point::new(point);
        let periodic = Periodic::new(0.5, TwelveTwelve {}).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(point).expect("hard-coded");
        let wrapped_poincare = wrapped_point.position.to_poincare();
        let point_poincare = point.position.to_poincare();

        let phi8 = 8.0 * PI / 6.0 + PI / 12.0;
        let a8 = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b8 = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi8.cos());
        let c8 = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi8.sin());
        let pref_8 = 1.0
            / ((b8 * point_poincare[1] - c8 * point_poincare[0]).powi(2)
                + (a8 + b8 * point_poincare[0] + c8 * point_poincare[1]).powi(2));
        let ans_8 = [
            pref_8
                * (point_poincare[0] * (a8.powi(2) + b8.powi(2) - c8.powi(2))
                    + 2.0 * b8 * c8 * point_poincare[1]
                    + a8 * b8 * (1.0 + point_poincare[0].powi(2) + point_poincare[1].powi(2))),
            pref_8
                * (point_poincare[1] * (a8.powi(2) - b8.powi(2) + c8.powi(2))
                    + 2.0 * b8 * c8 * point_poincare[0]
                    + a8 * c8 * (1.0 + point_poincare[0].powi(2) + point_poincare[1].powi(2))),
        ];
        let phi1 = PI / 6.0 + PI / 12.0;
        let a1 = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b1 = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi1.cos());
        let c1 = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi1.sin());
        let pref_1 = 1.0
            / ((b1 * ans_8[1] - c1 * ans_8[0]).powi(2)
                + (a1 + b1 * ans_8[0] + c1 * ans_8[1]).powi(2));
        let ans_1 = [
            pref_1
                * (ans_8[0] * (a1.powi(2) + b1.powi(2) - c1.powi(2))
                    + 2.0 * b1 * c1 * ans_8[1]
                    + a1 * b1 * (1.0 + ans_8[0].powi(2) + ans_8[1].powi(2))),
            pref_1
                * (ans_8[1] * (a1.powi(2) - b1.powi(2) + c1.powi(2))
                    + 2.0 * b1 * c1 * ans_8[0]
                    + a1 * c1 * (1.0 + ans_8[0].powi(2) + ans_8[1].powi(2))),
        ];
        let phi6 = PI + PI / 12.0;
        let a6 = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b6 = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi6.cos());
        let c6 = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi6.sin());
        let pref_6 = 1.0
            / ((b6 * ans_1[1] - c6 * ans_1[0]).powi(2)
                + (a6 + b6 * ans_1[0] + c6 * ans_1[1]).powi(2));
        let ans_6 = [
            pref_6
                * (ans_1[0] * (a6.powi(2) + b6.powi(2) - c6.powi(2))
                    + 2.0 * b6 * c6 * ans_1[1]
                    + a6 * b6 * (1.0 + ans_1[0].powi(2) + ans_1[1].powi(2))),
            pref_6
                * (ans_1[1] * (a6.powi(2) - b6.powi(2) + c6.powi(2))
                    + 2.0 * b6 * c6 * ans_1[0]
                    + a6 * c6 * (1.0 + ans_1[0].powi(2) + ans_1[1].powi(2))),
        ];
        let phi11 = 11.0 * PI / 6.0 + PI / 12.0;
        let a11 = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b11 = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi11.cos());
        let c11 = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi11.sin());
        let pref_11 = 1.0
            / ((b11 * ans_6[1] - c11 * ans_6[0]).powi(2)
                + (a11 + b11 * ans_6[0] + c11 * ans_6[1]).powi(2));
        let ans_11 = [
            pref_11
                * (ans_6[0] * (a11.powi(2) + b11.powi(2) - c11.powi(2))
                    + 2.0 * b11 * c11 * ans_6[1]
                    + a11 * b11 * (1.0 + ans_6[0].powi(2) + ans_6[1].powi(2))),
            pref_11
                * (ans_6[1] * (a11.powi(2) - b11.powi(2) + c11.powi(2))
                    + 2.0 * b11 * c11 * ans_6[0]
                    + a11 * c11 * (1.0 + ans_6[0].powi(2) + ans_6[1].powi(2))),
        ];
        let phi4 = 4.0 * PI / 6.0 + PI / 12.0;
        let a4 = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b4 = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi4.cos());
        let c4 = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi4.sin());
        let pref_4 = 1.0
            / ((b4 * ans_11[1] - c4 * ans_11[0]).powi(2)
                + (a4 + b4 * ans_11[0] + c4 * ans_11[1]).powi(2));
        let ans_4 = [
            pref_4
                * (ans_11[0] * (a4.powi(2) + b4.powi(2) - c4.powi(2))
                    + 2.0 * b4 * c4 * ans_11[1]
                    + a4 * b4 * (1.0 + ans_11[0].powi(2) + ans_11[1].powi(2))),
            pref_4
                * (ans_11[1] * (a4.powi(2) - b4.powi(2) + c4.powi(2))
                    + 2.0 * b4 * c4 * ans_11[0]
                    + a4 * c4 * (1.0 + ans_11[0].powi(2) + ans_11[1].powi(2))),
        ];
        let phi9 = 9.0 * PI / 6.0 + PI / 12.0;
        let a9 = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b9 = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi9.cos());
        let c9 = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi9.sin());
        let pref_9 = 1.0
            / ((b9 * ans_4[1] - c9 * ans_4[0]).powi(2)
                + (a9 + b9 * ans_4[0] + c9 * ans_4[1]).powi(2));
        let ans_9 = [
            pref_9
                * (ans_4[0] * (a9.powi(2) + b9.powi(2) - c9.powi(2))
                    + 2.0 * b9 * c9 * ans_4[1]
                    + a9 * b9 * (1.0 + ans_4[0].powi(2) + ans_4[1].powi(2))),
            pref_9
                * (ans_4[1] * (a9.powi(2) - b9.powi(2) + c9.powi(2))
                    + 2.0 * b9 * c9 * ans_4[0]
                    + a9 * c9 * (1.0 + ans_4[0].powi(2) + ans_4[1].powi(2))),
        ];

        assert_relative_eq!(ans_9[0], wrapped_poincare[0], epsilon = 1e-12);
        assert_relative_eq!(ans_9[1], wrapped_poincare[1], epsilon = 1e-12);
    }

    #[test]
    fn wraps_vertex_non_center() {
        let offset_boost: f64 = 0.4;
        let v: f64 = TwelveTwelve::TWELVETWELVE;
        let point = Hyperbolic::from_minkowski_coordinates(
            [
                (v.sinh()) * (offset_boost.cosh()),
                -offset_boost.sinh(),
                (v.cosh()) * (offset_boost.cosh()),
            ]
            .into(),
        );
        let point = Point::new(point);
        let periodic = Periodic::new(0.5, TwelveTwelve {}).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(point).expect("hard-coded");

        let wrapped_poincare = wrapped_point.position.to_poincare();
        let point_poincare = point.position.to_poincare();

        let phi5 = 5.0 * PI / 6.0 + PI / 12.0;
        let a5 = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b5 = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi5.cos());
        let c5 = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi5.sin());
        let pref_5 = 1.0
            / ((b5 * point_poincare[1] - c5 * point_poincare[0]).powi(2)
                + (a5 + b5 * point_poincare[0] + c5 * point_poincare[1]).powi(2));
        let ans_5 = [
            pref_5
                * (point_poincare[0] * (a5.powi(2) + b5.powi(2) - c5.powi(2))
                    + 2.0 * b5 * c5 * point_poincare[1]
                    + a5 * b5 * (1.0 + point_poincare[0].powi(2) + point_poincare[1].powi(2))),
            pref_5
                * (point_poincare[1] * (a5.powi(2) - b5.powi(2) + c5.powi(2))
                    + 2.0 * b5 * c5 * point_poincare[0]
                    + a5 * c5 * (1.0 + point_poincare[0].powi(2) + point_poincare[1].powi(2))),
        ];
        let phi10 = 10.0 * PI / 6.0 + PI / 12.0;
        let a10 = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b10 = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi10.cos());
        let c10 = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi10.sin());
        let pref_10 = 1.0
            / ((b10 * ans_5[1] - c10 * ans_5[0]).powi(2)
                + (a10 + b10 * ans_5[0] + c10 * ans_5[1]).powi(2));
        let ans_10 = [
            pref_10
                * (ans_5[0] * (a10.powi(2) + b10.powi(2) - c10.powi(2))
                    + 2.0 * b10 * c10 * ans_5[1]
                    + a10 * b10 * (1.0 + ans_5[0].powi(2) + ans_5[1].powi(2))),
            pref_10
                * (ans_5[1] * (a10.powi(2) - b10.powi(2) + c10.powi(2))
                    + 2.0 * b10 * c10 * ans_5[0]
                    + a10 * c10 * (1.0 + ans_5[0].powi(2) + ans_5[1].powi(2))),
        ];
        let phi3 = 3.0 * PI / 6.0 + PI / 12.0;
        let a3 = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b3 = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi3.cos());
        let c3 = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi3.sin());
        let pref_3 = 1.0
            / ((b3 * ans_10[1] - c3 * ans_10[0]).powi(2)
                + (a3 + b3 * ans_10[0] + c3 * ans_10[1]).powi(2));
        let ans_3 = [
            pref_3
                * (ans_10[0] * (a3.powi(2) + b3.powi(2) - c3.powi(2))
                    + 2.0 * b3 * c3 * ans_10[1]
                    + a3 * b3 * (1.0 + ans_10[0].powi(2) + ans_10[1].powi(2))),
            pref_3
                * (ans_10[1] * (a3.powi(2) - b3.powi(2) + c3.powi(2))
                    + 2.0 * b3 * c3 * ans_10[0]
                    + a3 * c3 * (1.0 + ans_10[0].powi(2) + ans_10[1].powi(2))),
        ];

        assert_relative_eq!(ans_3[0], wrapped_poincare[0], epsilon = 1e-12);
        assert_relative_eq!(ans_3[1], wrapped_poincare[1], epsilon = 1e-12);
    }

    #[test]
    fn ghost_near_side() {
        let mut rng = StdRng::seed_from_u64(4593);
        let side = f64::from(rng.random_range(0..12));
        let offset = 0.4;
        let boost = TwelveTwelve::CUSP_TO_EDGE - offset;
        let point = Hyperbolic::<3>::from_polar_coordinates(boost, PI / 12.0 + side * PI / 6.0);
        let point = Point::new(point);

        let periodic = Periodic::new(1.0, TwelveTwelve {}).expect("hard-coded positive number");

        let ghost_array = periodic.generate_ghosts(&point);
        let ghost = match side {
            8.0 => ghost_array[0],
            _ => ghost_array[1],
        };

        let ans = Hyperbolic::<3>::from_polar_coordinates(
            TwelveTwelve::CUSP_TO_EDGE + offset,
            (side + 6.0).rem_euclid(12.0) * PI / 6.0 + PI / 12.0,
        );

        assert_relative_eq!(ans, ghost.position, epsilon = 1e-12);
        assert_relative_eq!(
            TwelveTwelve::distance_to_boundary(&ghost.position),
            -TwelveTwelve::distance_to_boundary(&point.position),
            epsilon = 1e-12
        );
    }

    #[test]
    fn ghost_near_vertex() {
        let offset_boost = 0.3;
        let v: f64 = TwelveTwelve::TWELVETWELVE;
        let point = Hyperbolic::<3>::from_polar_coordinates(v - offset_boost, 0.0);
        let point = Point::new(point);
        let periodic = Periodic::new(0.5, TwelveTwelve {}).expect("hard-coded positive number");

        let ghost_array: ArrayVec<Point<Hyperbolic<3>>, 12> = periodic.generate_ghosts(&point);
        let ghost_10 = ghost_array[10];

        let ans_10 = Hyperbolic::<3>::from_polar_coordinates(v + offset_boost, PI);
        assert_relative_eq!(ans_10, ghost_10.position, epsilon = 1e-10);

        let ghost_4 = ghost_array[4];
        let ans_4 = Hyperbolic::from_minkowski_coordinates(
            [
                -offset_boost.sinh(),
                -(v.sinh()) * (offset_boost.cosh()),
                (v.cosh()) * (offset_boost.cosh()),
            ]
            .into(),
        );
        assert_relative_eq!(ans_4, ghost_4.position, epsilon = 1e-10);

        let ghost_5 = ghost_array[5];
        let ans_5 = Hyperbolic::from_minkowski_coordinates(
            [
                -offset_boost.sinh(),
                (v.sinh()) * (offset_boost.cosh()),
                (v.cosh()) * (offset_boost.cosh()),
            ]
            .into(),
        );
        assert_relative_eq!(ans_5, ghost_5.position, epsilon = 1e-10);
    }

    #[test]
    #[allow(clippy::too_many_lines, reason = "complicated function")]
    fn consistency_vertex() {
        let offset_boost = 0.3;
        let offset_angle = 0.1;
        let edge_boost: f64 = TwelveTwelve::TWELVETWELVE;
        let point =
            Hyperbolic::<3>::from_polar_coordinates(edge_boost - offset_boost, offset_angle);
        let point = Point::new(point);
        let periodic = Periodic::new(0.5, TwelveTwelve {}).expect("hard-coded positive number");

        let ghost_array: ArrayVec<Point<Hyperbolic<3>>, 12> = periodic.generate_ghosts(&point);

        // check double transformations
        let ghost_2_poincare = ghost_array[2].position.to_poincare();
        let (q_u, q_v) = (
            point.position.to_poincare()[0],
            point.position.to_poincare()[1],
        );
        let phi1 = PI / 6.0 + PI / 12.0;
        let phi6 = PI + PI / 12.0;
        let a = (1.0 + (PI / 6.0).cos() + 2.0 * ((PI / 6.0).cos()) * ((phi1 - phi6).cos()))
            / (1.0 - (PI / 6.0).cos());
        let d = (2.0 * ((PI / 6.0).cos()) * (phi1 - phi6).sin()) / (1.0 - (PI / 6.0).cos());
        let b = ((2.0 * ((PI / 6.0).cos()) * (1.0 + (PI / 6.0).cos())).sqrt())
            * (phi6.cos() + phi1.cos())
            / (1.0 - (PI / 6.0).cos());
        let c = ((2.0 * ((PI / 6.0).cos()) * (1.0 + (PI / 6.0).cos())).sqrt())
            * (phi6.sin() + phi1.sin())
            / (1.0 - (PI / 6.0).cos());
        let pref = 1.0 / ((b * q_v - c * q_u - d).powi(2) + (a + b * q_u + c * q_v).powi(2));
        let ans_2 = [
            pref * ((a * b - c * d) * (1.0 + q_u.powi(2) + q_v.powi(2))
                + q_u * (a.powi(2) + b.powi(2) - c.powi(2) - d.powi(2))
                + 2.0 * b * c * q_v
                - 2.0 * a * d * q_v),
            pref * ((a * c + b * d) * (1.0 + q_u.powi(2) + q_v.powi(2))
                + q_v * (a.powi(2) - b.powi(2) + c.powi(2) - d.powi(2))
                + 2.0 * b * c * q_u
                + 2.0 * a * d * q_u),
        ];
        assert_relative_eq!(ans_2[0], ghost_2_poincare[0], epsilon = 1e-12);
        assert_relative_eq!(ans_2[1], ghost_2_poincare[1], epsilon = 1e-12);

        let ghost_3_poincare = ghost_array[3].position.to_poincare();
        let phi5 = 5.0 * PI / 6.0 + PI / 12.0;
        let phi10 = 10.0 * PI / 6.0 + PI / 12.0;
        let a = (1.0 + (PI / 6.0).cos() + 2.0 * ((PI / 6.0).cos()) * ((phi5 - phi10).cos()))
            / (1.0 - (PI / 6.0).cos());
        let d = (2.0 * ((PI / 6.0).cos()) * (phi10 - phi5).sin()) / (1.0 - (PI / 6.0).cos());
        let b = ((2.0 * ((PI / 6.0).cos()) * (1.0 + (PI / 6.0).cos())).sqrt())
            * (phi10.cos() + phi5.cos())
            / (1.0 - (PI / 6.0).cos());
        let c = ((2.0 * ((PI / 6.0).cos()) * (1.0 + (PI / 6.0).cos())).sqrt())
            * (phi10.sin() + phi5.sin())
            / (1.0 - (PI / 6.0).cos());
        let pref = 1.0 / ((b * q_v - c * q_u - d).powi(2) + (a + b * q_u + c * q_v).powi(2));
        let ans_3 = [
            pref * ((a * b - c * d) * (1.0 + q_u.powi(2) + q_v.powi(2))
                + q_u * (a.powi(2) + b.powi(2) - c.powi(2) - d.powi(2))
                + 2.0 * b * c * q_v
                - 2.0 * a * d * q_v),
            pref * ((a * c + b * d) * (1.0 + q_u.powi(2) + q_v.powi(2))
                + q_v * (a.powi(2) - b.powi(2) + c.powi(2) - d.powi(2))
                + 2.0 * b * c * q_u
                + 2.0 * a * d * q_u),
        ];
        assert_relative_eq!(ans_3[0], ghost_3_poincare[0], epsilon = 1e-12);
        assert_relative_eq!(ans_3[1], ghost_3_poincare[1], epsilon = 1e-12);

        // check triple transformations
        let ghost_4_poincare = ghost_array[4].position.to_poincare();
        let phi8 = 8.0 * PI / 6.0 + PI / 12.0;
        let a = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi8.cos());
        let c = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi8.sin());
        let pref = 1.0
            / ((b * ans_2[1] - c * ans_2[0]).powi(2) + (a + b * ans_2[0] + c * ans_2[1]).powi(2));
        let ans_4 = [
            pref * (ans_2[0] * (a.powi(2) + b.powi(2) - c.powi(2))
                + 2.0 * b * c * ans_2[1]
                + a * b * (1.0 + ans_2[0].powi(2) + ans_2[1].powi(2))),
            pref * (ans_2[1] * (a.powi(2) - b.powi(2) + c.powi(2))
                + 2.0 * b * c * ans_2[0]
                + a * c * (1.0 + ans_2[0].powi(2) + ans_2[1].powi(2))),
        ];
        assert_relative_eq!(ans_4[0], ghost_4_poincare[0], epsilon = 1e-12);
        assert_relative_eq!(ans_4[1], ghost_4_poincare[1], epsilon = 1e-12);

        let ghost_5_poincare = ghost_array[5].position.to_poincare();
        let phi3 = 3.0 * PI / 6.0 + PI / 12.0;
        let a = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi3.cos());
        let c = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi3.sin());
        let pref = 1.0
            / ((b * ans_3[1] - c * ans_3[0]).powi(2) + (a + b * ans_3[0] + c * ans_3[1]).powi(2));
        let ans_5 = [
            pref * (ans_3[0] * (a.powi(2) + b.powi(2) - c.powi(2))
                + 2.0 * b * c * ans_3[1]
                + a * b * (1.0 + ans_3[0].powi(2) + ans_3[1].powi(2))),
            pref * (ans_3[1] * (a.powi(2) - b.powi(2) + c.powi(2))
                + 2.0 * b * c * ans_3[0]
                + a * c * (1.0 + ans_3[0].powi(2) + ans_3[1].powi(2))),
        ];
        assert_relative_eq!(ans_5[0], ghost_5_poincare[0], epsilon = 1e-12);
        assert_relative_eq!(ans_5[1], ghost_5_poincare[1], epsilon = 1e-12);

        // check quadruple transformations
        let ghost_6_poincare = ghost_array[6].position.to_poincare();
        let a = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi3.cos());
        let c = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi3.sin());
        let pref = 1.0
            / ((b * ans_4[1] - c * ans_4[0]).powi(2) + (a + b * ans_4[0] + c * ans_4[1]).powi(2));
        let ans_6 = [
            pref * (ans_4[0] * (a.powi(2) + b.powi(2) - c.powi(2))
                + 2.0 * b * c * ans_4[1]
                + a * b * (1.0 + ans_4[0].powi(2) + ans_4[1].powi(2))),
            pref * (ans_4[1] * (a.powi(2) - b.powi(2) + c.powi(2))
                + 2.0 * b * c * ans_4[0]
                + a * c * (1.0 + ans_4[0].powi(2) + ans_4[1].powi(2))),
        ];
        assert_relative_eq!(ans_6[0], ghost_6_poincare[0], epsilon = 1e-12);
        assert_relative_eq!(ans_6[1], ghost_6_poincare[1], epsilon = 1e-12);

        let ghost_7_poincare = ghost_array[7].position.to_poincare();
        let a = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi8.cos());
        let c = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi8.sin());
        let pref = 1.0
            / ((b * ans_5[1] - c * ans_5[0]).powi(2) + (a + b * ans_5[0] + c * ans_5[1]).powi(2));
        let ans_7 = [
            pref * (ans_5[0] * (a.powi(2) + b.powi(2) - c.powi(2))
                + 2.0 * b * c * ans_5[1]
                + a * b * (1.0 + ans_5[0].powi(2) + ans_5[1].powi(2))),
            pref * (ans_5[1] * (a.powi(2) - b.powi(2) + c.powi(2))
                + 2.0 * b * c * ans_5[0]
                + a * c * (1.0 + ans_5[0].powi(2) + ans_5[1].powi(2))),
        ];
        assert_relative_eq!(ans_7[0], ghost_7_poincare[0], epsilon = 1e-12);
        assert_relative_eq!(ans_7[1], ghost_7_poincare[1], epsilon = 1e-12);

        // check quintruple transformations
        let ghost_8_poincare = ghost_array[8].position.to_poincare();
        let a = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi10.cos());
        let c = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi10.sin());
        let pref = 1.0
            / ((b * ans_6[1] - c * ans_6[0]).powi(2) + (a + b * ans_6[0] + c * ans_6[1]).powi(2));
        let ans_8 = [
            pref * (ans_6[0] * (a.powi(2) + b.powi(2) - c.powi(2))
                + 2.0 * b * c * ans_6[1]
                + a * b * (1.0 + ans_6[0].powi(2) + ans_6[1].powi(2))),
            pref * (ans_6[1] * (a.powi(2) - b.powi(2) + c.powi(2))
                + 2.0 * b * c * ans_6[0]
                + a * c * (1.0 + ans_6[0].powi(2) + ans_6[1].powi(2))),
        ];
        assert_relative_eq!(ans_8[0], ghost_8_poincare[0], epsilon = 1e-12);
        assert_relative_eq!(ans_8[1], ghost_8_poincare[1], epsilon = 1e-12);

        let ghost_9_poincare = ghost_array[9].position.to_poincare();
        let a = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi1.cos());
        let c = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi1.sin());
        let pref = 1.0
            / ((b * ans_7[1] - c * ans_7[0]).powi(2) + (a + b * ans_7[0] + c * ans_7[1]).powi(2));
        let ans_9 = [
            pref * (ans_7[0] * (a.powi(2) + b.powi(2) - c.powi(2))
                + 2.0 * b * c * ans_7[1]
                + a * b * (1.0 + ans_7[0].powi(2) + ans_7[1].powi(2))),
            pref * (ans_7[1] * (a.powi(2) - b.powi(2) + c.powi(2))
                + 2.0 * b * c * ans_7[0]
                + a * c * (1.0 + ans_7[0].powi(2) + ans_7[1].powi(2))),
        ];
        assert_relative_eq!(ans_9[0], ghost_9_poincare[0], epsilon = 1e-12);
        assert_relative_eq!(ans_9[1], ghost_9_poincare[1], epsilon = 1e-12);

        // check 6-tuple transformation
        let ghost_10_poincare = ghost_array[10].position.to_poincare();
        let a = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi5.cos());
        let c = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi5.sin());
        let pref = 1.0
            / ((b * ans_8[1] - c * ans_8[0]).powi(2) + (a + b * ans_8[0] + c * ans_8[1]).powi(2));
        let ans_10 = [
            pref * (ans_8[0] * (a.powi(2) + b.powi(2) - c.powi(2))
                + 2.0 * b * c * ans_8[1]
                + a * b * (1.0 + ans_8[0].powi(2) + ans_8[1].powi(2))),
            pref * (ans_8[1] * (a.powi(2) - b.powi(2) + c.powi(2))
                + 2.0 * b * c * ans_8[0]
                + a * c * (1.0 + ans_8[0].powi(2) + ans_8[1].powi(2))),
        ];
        assert_relative_eq!(ans_10[0], ghost_10_poincare[0], epsilon = 1e-12);
        assert_relative_eq!(ans_10[1], ghost_10_poincare[1], epsilon = 1e-12);

        // check single transformations
        let ghost_1_poincare = ghost_array[1].position.to_poincare();
        let a = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi5.cos());
        let c = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi5.sin());
        let pref = 1.0 / ((b * q_v - c * q_u).powi(2) + (a + b * q_u + c * q_v).powi(2));
        let ans_1 = [
            pref * (q_u * (a.powi(2) + b.powi(2) - c.powi(2))
                + 2.0 * b * c * q_v
                + a * b * (1.0 + q_u.powi(2) + q_v.powi(2))),
            pref * (q_v * (a.powi(2) - b.powi(2) + c.powi(2))
                + 2.0 * b * c * q_u
                + a * c * (1.0 + q_u.powi(2) + q_v.powi(2))),
        ];
        assert_relative_eq!(ans_1[0], ghost_1_poincare[0], epsilon = 1e-12);
        assert_relative_eq!(ans_1[1], ghost_1_poincare[1], epsilon = 1e-12);

        let ghost_0_poincare = ghost_array[0].position.to_poincare();
        let a = ((1.0 + (PI / 6.0).cos()) / (1.0 - (PI / 6.0).cos())).sqrt();
        let b = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi6.cos());
        let c = ((2.0 * ((PI / 6.0).cos())) / (1.0 - (PI / 6.0).cos())).sqrt() * (phi6.sin());
        let pref = 1.0 / ((b * q_v - c * q_u).powi(2) + (a + b * q_u + c * q_v).powi(2));
        let ans_0 = [
            pref * (q_u * (a.powi(2) + b.powi(2) - c.powi(2))
                + 2.0 * b * c * q_v
                + a * b * (1.0 + q_u.powi(2) + q_v.powi(2))),
            pref * (q_v * (a.powi(2) - b.powi(2) + c.powi(2))
                + 2.0 * b * c * q_u
                + a * c * (1.0 + q_u.powi(2) + q_v.powi(2))),
        ];
        assert_relative_eq!(ans_0[0], ghost_0_poincare[0], epsilon = 1e-12);
        assert_relative_eq!(ans_0[1], ghost_0_poincare[1], epsilon = 1e-12);
    }

    #[test]
    fn wraps_orientation() {
        let angle_offset: f64 = 0.1;
        let boost = ((TwelveTwelve::TWELVETWELVE).tanh() * ((PI / 6.0).sin())
            / ((angle_offset).sin() + (PI / 6.0 - angle_offset).sin()))
        .atanh()
            + 0.1;
        let point = Hyperbolic::<3>::from_polar_coordinates(boost, angle_offset + PI / 6.0);
        let tangent = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(TwelveTwelve::TWELVETWELVE, PI / 6.0),
            &point,
        );
        let oriented_point = OrientedHyperbolicPoint {
            position: point,
            orientation: Angle::from(13.0 * PI / 12.0 + tangent),
        };
        let periodic = Periodic::new(0.5, TwelveTwelve {}).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(oriented_point).expect("hard-coded");

        let answer = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(TwelveTwelve::TWELVETWELVE, 8.0 * PI / 6.0),
            &wrapped_point.position,
        );

        // Check that orientation maps correctly
        assert_relative_eq!(
            wrapped_point.orientation.theta,
            5.0 * PI / 12.0 + answer,
            epsilon = 1e-12
        );
    }
}

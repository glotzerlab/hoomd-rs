// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement periodic boundary conditions for the {8,8} tiling of hyperbolic
//! space.
//!
//! Specifically, `Periodic<EightEight>` identifies opposite edges of the
//! octagon to implement the Bolza surface.

use arrayvec::ArrayVec;
use std::f64::consts::PI;

use crate::{
    boundary::{
        Error, GenerateGhosts, MAX_GHOSTS, MaximumAllowableInteractionRange, Periodic, Wrap,
    },
    property::{Orientation, OrientedHyperbolicPoint, Point, Position},
};
use hoomd_geometry::shape::EightEight;
use hoomd_manifold::{Hyperbolic, Minkowski};
use hoomd_vector::Angle;

impl MaximumAllowableInteractionRange for EightEight {
    /// The largest value that the maximum interaction range can take.
    ///
    /// This bound is determined by the edge length of the octagon.
    #[inline]
    fn maximum_allowable_interaction_range(&self) -> f64 {
        EightEight::CUSP_TO_EDGE
    }
}

impl Wrap<Point<Hyperbolic<3>>> for Periodic<EightEight> {
    /// Wrap a point in hyperbolic space to the inside of the {8,8} tile.
    ///
    /// Note that the function fails to wrap points that are outside the octagon
    /// and further than `EightEight::EDGE_LENGTH`/2 from any of the vertices. In
    /// this case, the function returns `Error::CannotWrapProperties`
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::shape::EightEight;
    /// use hoomd_manifold::Hyperbolic;
    /// use hoomd_microstate::{
    ///     boundary::{Periodic, Wrap},
    ///     property::Point,
    /// };
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// const EIGHTEIGHT: f64 = EightEight::EIGHTEIGHT;
    /// let offset = PI / 8.0;
    /// let boost = 2.0;
    /// let point =
    ///     Hyperbolic::<3>::from_polar_coordinates(boost, offset + PI / 4.0);
    /// let point = Point::new(point);
    /// let periodic = Periodic::new(0.5, EightEight {})?;
    ///
    /// let wrapped_point = periodic.wrap(point)?;
    ///
    /// let new_boost = 2.0
    ///     * (EIGHTEIGHT.tanh()
    ///         / (offset.cos() - offset.sin() * (1.0 - (2.0_f64).sqrt())))
    ///     .atanh()
    ///     - boost;
    /// let ans = Hyperbolic::<3>::from_polar_coordinates(
    ///     new_boost,
    ///     6.0 * PI / 4.0 - offset,
    /// );
    /// assert_relative_eq!(
    ///     ans.coordinates()[0],
    ///     wrapped_point.position.coordinates()[0],
    ///     epsilon = 1e-12
    /// );
    /// assert_relative_eq!(
    ///     ans.coordinates()[1],
    ///     wrapped_point.position.coordinates()[1],
    ///     epsilon = 1e-12
    /// );
    /// assert_relative_eq!(
    ///     ans.coordinates()[2],
    ///     wrapped_point.position.coordinates()[2],
    ///     epsilon = 1e-12
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[expect(clippy::too_many_lines, reason = "complicated function")]
    fn wrap(&self, properties: Point<Hyperbolic<3>>) -> Result<Point<Hyperbolic<3>>, Error> {
        let mut properties = properties;
        let r = properties.position_mut();
        let angle = r.coordinates()[1]
            .atan2(r.coordinates()[0])
            .rem_euclid(2.0 * PI);

        // distance to the boundary; if positive, r is within the tile and does not need to be wrapped
        let d = EightEight::distance_to_boundary(r);
        if d >= 0.0 {
            Ok(properties)
        } else if d > -self.maximum_interaction_range {
            // get the sequence of transformations necessary to wrap point back into the tile
            let nearest_vertex_number =
                (((angle + (PI / 8.0)).rem_euclid(PI * 2.0)) / (PI / 4.0)).floor();

            // transform point to frame where relevant vertex is in the center
            let (vertex_boost, vertex_angle) = (
                EightEight::EIGHTEIGHT,
                (nearest_vertex_number * PI / 4.0).rem_euclid(PI * 2.0),
            );
            let (v_cosh, v_sinh, v_cos, v_sin) = (
                (-vertex_boost).cosh(),
                (-vertex_boost).sinh(),
                (-vertex_angle).cos(),
                (-vertex_angle).sin(),
            );
            let transformed_point = Minkowski::from([
                r.coordinates()[0] * v_cosh * v_cos - r.coordinates()[1] * v_cosh * v_sin
                    + r.coordinates()[2] * v_sinh,
                r.coordinates()[0] * v_sin + r.coordinates()[1] * v_cos,
                r.coordinates()[0] * v_sinh * v_cos - r.coordinates()[1] * v_sinh * v_sin
                    + r.coordinates()[2] * v_cosh,
            ]);
            // get coords of point in transformed frame
            let trans_angle =
                transformed_point.coordinates[1].atan2(transformed_point.coordinates[0]);
            // find which octant the transformed point is in
            let octant = (((trans_angle + (PI / 8.0)).rem_euclid(2.0 * PI)) / (PI / 4.0)).floor();

            // transform to tile
            let eta = EightEight::CUSP_TO_EDGE;
            let wrapped: [f64; 3];
            match octant {
                5.0 => {
                    let theta_1 = (nearest_vertex_number + 3.0).rem_euclid(8.0);
                    wrapped =
                        EightEight::gamma(eta, theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                }
                6.0 => {
                    let theta_1 = (nearest_vertex_number + 3.0).rem_euclid(8.0);
                    let wrapped_1 =
                        EightEight::gamma(eta, theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                    let theta_2 = (theta_1 + 3.0).rem_euclid(8.0);
                    wrapped = EightEight::gamma(eta, theta_2 * PI / 4.0 + PI / 8.0, &wrapped_1);
                }
                7.0 => {
                    let theta_1 = (nearest_vertex_number + 3.0).rem_euclid(8.0);
                    let wrapped_1 =
                        EightEight::gamma(eta, theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                    let theta_2 = (theta_1 + 3.0).rem_euclid(8.0);
                    let wrapped_2 =
                        EightEight::gamma(eta, theta_2 * PI / 4.0 + PI / 8.0, &wrapped_1);
                    let theta_3 = (theta_2 + 3.0).rem_euclid(8.0);
                    wrapped = EightEight::gamma(eta, theta_3 * PI / 4.0 + PI / 8.0, &wrapped_2);
                }
                3.0 => {
                    let theta_1 = (nearest_vertex_number + 4.0).rem_euclid(8.0);
                    wrapped =
                        EightEight::gamma(eta, theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                }
                2.0 => {
                    let theta_1 = (nearest_vertex_number + 4.0).rem_euclid(8.0);
                    let wrapped_1 =
                        EightEight::gamma(eta, theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                    let theta_2 = (theta_1 - 3.0).rem_euclid(8.0);
                    wrapped = EightEight::gamma(eta, theta_2 * PI / 4.0 + PI / 8.0, &wrapped_1);
                }
                1.0 => {
                    let theta_1 = (nearest_vertex_number + 4.0).rem_euclid(8.0);
                    let wrapped_1 =
                        EightEight::gamma(eta, theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                    let theta_2 = (theta_1 - 3.0).rem_euclid(8.0);
                    let wrapped_2 =
                        EightEight::gamma(eta, theta_2 * PI / 4.0 + PI / 8.0, &wrapped_1);
                    let theta_3 = (theta_2 - 3.0).rem_euclid(8.0);
                    wrapped = EightEight::gamma(eta, theta_3 * PI / 4.0 + PI / 8.0, &wrapped_2);
                }
                0.0 => {
                    if transformed_point.coordinates[1] >= 0.0 {
                        let theta_1 = (nearest_vertex_number + 4.0).rem_euclid(8.0);
                        let wrapped_1 =
                            EightEight::gamma(eta, theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                        let theta_2 = (theta_1 - 3.0).rem_euclid(8.0);
                        let wrapped_2 =
                            EightEight::gamma(eta, theta_2 * PI / 4.0 + PI / 8.0, &wrapped_1);
                        let theta_3 = (theta_2 - 3.0).rem_euclid(8.0);
                        let wrapped_3 =
                            EightEight::gamma(eta, theta_3 * PI / 4.0 + PI / 8.0, &wrapped_2);
                        let theta_4 = (theta_3 - 3.0).rem_euclid(8.0);
                        wrapped = EightEight::gamma(eta, theta_4 * PI / 4.0 + PI / 8.0, &wrapped_3);
                    } else {
                        let theta_1 = (nearest_vertex_number + 3.0).rem_euclid(8.0);
                        let wrapped_1 =
                            EightEight::gamma(eta, theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                        let theta_2 = (theta_1 + 3.0).rem_euclid(8.0);
                        let wrapped_2 =
                            EightEight::gamma(eta, theta_2 * PI / 4.0 + PI / 8.0, &wrapped_1);
                        let theta_3 = (theta_2 + 3.0).rem_euclid(8.0);
                        let wrapped_3 =
                            EightEight::gamma(eta, theta_3 * PI / 4.0 + PI / 8.0, &wrapped_2);
                        let theta_4 = (theta_3 + 3.0).rem_euclid(8.0);
                        wrapped = EightEight::gamma(eta, theta_4 * PI / 4.0 + PI / 8.0, &wrapped_3);
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

impl Wrap<OrientedHyperbolicPoint<3, Angle>> for Periodic<EightEight> {
    /// Wrap the positions and orientations of oriented bodies in two-dimensional
    /// hyperbolic space under the {8,8} tiling.
    #[inline]
    #[allow(clippy::too_many_lines, reason = "complicated function")]
    fn wrap(
        &self,
        properties: OrientedHyperbolicPoint<3, Angle>,
    ) -> Result<OrientedHyperbolicPoint<3, Angle>, Error> {
        let original_orientation = properties.orientation.theta;
        let mut properties = properties;
        // let orientation = properties.orientation_mut();
        let r = properties.position_mut();
        let angle = r.coordinates()[1]
            .atan2(r.coordinates()[0])
            .rem_euclid(2.0 * PI);

        // distance to the boundary; if positive, r is within the tile and does not need to be wrapped
        let d = EightEight::distance_to_boundary(r);
        if d >= 0.0 {
            Ok(properties)
        } else if d > -self.maximum_interaction_range {
            // get the sequence of transformations necessary to wrap point back into the tile
            let nearest_vertex_number =
                (((angle + (PI / 8.0)).rem_euclid(PI * 2.0)) / (PI / 4.0)).floor();

            // transform point to frame where relevant vertex is in the center
            let (vertex_boost, vertex_angle) = (
                EightEight::EIGHTEIGHT,
                (nearest_vertex_number * PI / 4.0).rem_euclid(PI * 2.0),
            );
            let (v_cosh, v_sinh, v_cos, v_sin) = (
                (-vertex_boost).cosh(),
                (-vertex_boost).sinh(),
                (-vertex_angle).cos(),
                (-vertex_angle).sin(),
            );
            let transformed_point = Minkowski::from([
                r.coordinates()[0] * v_cosh * v_cos - r.coordinates()[1] * v_cosh * v_sin
                    + r.coordinates()[2] * v_sinh,
                r.coordinates()[0] * v_sin + r.coordinates()[1] * v_cos,
                r.coordinates()[0] * v_sinh * v_cos - r.coordinates()[1] * v_sinh * v_sin
                    + r.coordinates()[2] * v_cosh,
            ]);
            // get coords of point in transformed frame
            let trans_angle =
                transformed_point.coordinates[1].atan2(transformed_point.coordinates[0]);
            // let trans_boost = (transformed_point.coordinates[2]).acosh();

            // find which octant the transformed point is in
            let octant = (((trans_angle + (PI / 8.0)).rem_euclid(2.0 * PI)) / (PI / 4.0)).floor();

            // transform to tile
            let eta = EightEight::CUSP_TO_EDGE;

            let wrapped: [f64; 3];
            let relative_angle: f64;
            match octant {
                5.0 => {
                    let theta_1 = (nearest_vertex_number + 3.0).rem_euclid(8.0);
                    wrapped =
                        EightEight::gamma(eta, theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                    relative_angle =
                        EightEight::reorient(theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                }
                6.0 => {
                    let theta_1 = (nearest_vertex_number + 3.0).rem_euclid(8.0);
                    let wrapped_1 =
                        EightEight::gamma(eta, theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                    let relative_angle_1 =
                        EightEight::reorient(theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                    let theta_2 = (theta_1 + 3.0).rem_euclid(8.0);
                    wrapped = EightEight::gamma(eta, theta_2 * PI / 4.0 + PI / 8.0, &wrapped_1);
                    relative_angle =
                        EightEight::reorient(theta_2 * PI / 4.0 + PI / 8.0, &wrapped_1)
                            + relative_angle_1;
                }
                7.0 => {
                    let theta_1 = (nearest_vertex_number + 3.0).rem_euclid(8.0);
                    let wrapped_1 =
                        EightEight::gamma(eta, theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                    let relative_angle_1 =
                        EightEight::reorient(theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                    let theta_2 = (theta_1 + 3.0).rem_euclid(8.0);
                    let wrapped_2 =
                        EightEight::gamma(eta, theta_2 * PI / 4.0 + PI / 8.0, &wrapped_1);
                    let relative_angle_2 =
                        EightEight::reorient(theta_2 * PI / 4.0 + PI / 8.0, &wrapped_1);
                    let theta_3 = (theta_2 + 3.0).rem_euclid(8.0);
                    wrapped = EightEight::gamma(eta, theta_3 * PI / 4.0 + PI / 8.0, &wrapped_2);
                    relative_angle =
                        EightEight::reorient(theta_3 * PI / 4.0 + PI / 8.0, &wrapped_2)
                            + relative_angle_1
                            + relative_angle_2;
                }
                3.0 => {
                    let theta_1 = (nearest_vertex_number + 4.0).rem_euclid(8.0);
                    wrapped =
                        EightEight::gamma(eta, theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                    relative_angle =
                        EightEight::reorient(theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                }
                2.0 => {
                    let theta_1 = (nearest_vertex_number + 4.0).rem_euclid(8.0);
                    let wrapped_1 =
                        EightEight::gamma(eta, theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                    let relative_angle_1 =
                        EightEight::reorient(theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                    let theta_2 = (theta_1 - 3.0).rem_euclid(8.0);
                    wrapped = EightEight::gamma(eta, theta_2 * PI / 4.0 + PI / 8.0, &wrapped_1);
                    relative_angle =
                        EightEight::reorient(theta_2 * PI / 4.0 + PI / 8.0, &wrapped_1)
                            + relative_angle_1;
                }
                1.0 => {
                    let theta_1 = (nearest_vertex_number + 4.0).rem_euclid(8.0);
                    let wrapped_1 =
                        EightEight::gamma(eta, theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                    let relative_angle_1 =
                        EightEight::reorient(theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                    let theta_2 = (theta_1 - 3.0).rem_euclid(8.0);
                    let wrapped_2 =
                        EightEight::gamma(eta, theta_2 * PI / 4.0 + PI / 8.0, &wrapped_1);
                    let relative_angle_2 =
                        EightEight::reorient(theta_2 * PI / 4.0 + PI / 8.0, &wrapped_1);
                    let theta_3 = (theta_2 - 3.0).rem_euclid(8.0);
                    wrapped = EightEight::gamma(eta, theta_3 * PI / 4.0 + PI / 8.0, &wrapped_2);
                    relative_angle =
                        EightEight::reorient(theta_3 * PI / 4.0 + PI / 8.0, &wrapped_2)
                            + relative_angle_1
                            + relative_angle_2;
                }
                0.0 => {
                    if transformed_point.coordinates[1] >= 0.0 {
                        let theta_1 = (nearest_vertex_number + 4.0).rem_euclid(8.0);
                        let wrapped_1 =
                            EightEight::gamma(eta, theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                        let relative_angle_1 =
                            EightEight::reorient(theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                        let theta_2 = (theta_1 - 3.0).rem_euclid(8.0);
                        let wrapped_2 =
                            EightEight::gamma(eta, theta_2 * PI / 4.0 + PI / 8.0, &wrapped_1);
                        let relative_angle_2 =
                            EightEight::reorient(theta_2 * PI / 4.0 + PI / 8.0, &wrapped_1);
                        let theta_3 = (theta_2 - 3.0).rem_euclid(8.0);
                        let wrapped_3 =
                            EightEight::gamma(eta, theta_3 * PI / 4.0 + PI / 8.0, &wrapped_2);
                        let relative_angle_3 =
                            EightEight::reorient(theta_3 * PI / 4.0 + PI / 8.0, &wrapped_2);
                        let theta_4 = (theta_3 - 3.0).rem_euclid(8.0);
                        wrapped = EightEight::gamma(eta, theta_4 * PI / 4.0 + PI / 8.0, &wrapped_3);
                        relative_angle =
                            EightEight::reorient(theta_4 * PI / 4.0 + PI / 8.0, &wrapped_3)
                                + relative_angle_1
                                + relative_angle_2
                                + relative_angle_3;
                    } else {
                        let theta_1 = (nearest_vertex_number + 3.0).rem_euclid(8.0);
                        let wrapped_1 =
                            EightEight::gamma(eta, theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                        let relative_angle_1 =
                            EightEight::reorient(theta_1 * PI / 4.0 + PI / 8.0, r.coordinates());
                        let theta_2 = (theta_1 + 3.0).rem_euclid(8.0);
                        let wrapped_2 =
                            EightEight::gamma(eta, theta_2 * PI / 4.0 + PI / 8.0, &wrapped_1);
                        let relative_angle_2 =
                            EightEight::reorient(theta_2 * PI / 4.0 + PI / 8.0, &wrapped_1);
                        let theta_3 = (theta_2 + 3.0).rem_euclid(8.0);
                        let wrapped_3 =
                            EightEight::gamma(eta, theta_3 * PI / 4.0 + PI / 8.0, &wrapped_2);
                        let relative_angle_3 =
                            EightEight::reorient(theta_3 * PI / 4.0 + PI / 8.0, &wrapped_2);
                        let theta_4 = (theta_3 + 3.0).rem_euclid(8.0);
                        wrapped = EightEight::gamma(eta, theta_4 * PI / 4.0 + PI / 8.0, &wrapped_3);
                        relative_angle =
                            EightEight::reorient(theta_4 * PI / 4.0 + PI / 8.0, &wrapped_3)
                                + relative_angle_1
                                + relative_angle_2
                                + relative_angle_3;
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

impl GenerateGhosts<Point<Hyperbolic<3>>> for Periodic<EightEight> {
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }
    /// Place periodic images of sites near the edge of the periodic boundary
    #[inline]
    fn generate_ghosts(
        &self,
        site_properties: &Point<Hyperbolic<3>>,
    ) -> ArrayVec<Point<Hyperbolic<3>>, MAX_GHOSTS> {
        let mut result = ArrayVec::new();
        let r = site_properties.position();

        // transform to tile
        let eta = EightEight::CUSP_TO_EDGE;
        let gamma_pt = |theta: f64, point: &[f64; 3]| {
            let ghost = EightEight::gamma(eta, theta, point);
            let new_hyperbolic = Hyperbolic::from_minkowski_coordinates(Minkowski::from(ghost));
            let mut new_site = *site_properties;
            *new_site.position_mut() = new_hyperbolic;
            new_site
        };
        // identify which octant the point is in
        let angle = (r.coordinates()[1].atan2(r.coordinates()[0])).rem_euclid(2.0 * PI);
        let nearest_vertex_number =
            (((angle + (PI / 8.0)).rem_euclid(PI * 2.0)) / (PI / 4.0)).floor();
        let coords = *r.coordinates();

        let theta_1a = (nearest_vertex_number + 4.0).rem_euclid(8.0);
        let ghost_1a = gamma_pt(theta_1a * PI / 4.0 + PI / 8.0, &coords);
        result.push(ghost_1a);
        let theta_1b = (nearest_vertex_number + 3.0).rem_euclid(8.0);
        let ghost_1b = gamma_pt(theta_1b * PI / 4.0 + PI / 8.0, &coords);
        result.push(ghost_1b);

        let theta_2a = (theta_1a - 3.0).rem_euclid(8.0);
        let ghost_2a = gamma_pt(
            theta_2a * PI / 4.0 + PI / 8.0,
            ghost_1a.position.coordinates(),
        );
        result.push(ghost_2a);
        let theta_2b = (theta_1b + 3.0).rem_euclid(8.0);
        let ghost_2b = gamma_pt(
            theta_2b * PI / 4.0 + PI / 8.0,
            ghost_1b.position.coordinates(),
        );
        result.push(ghost_2b);

        let theta_3a = (theta_2a - 3.0).rem_euclid(8.0);
        let ghost_3a = gamma_pt(
            theta_3a * PI / 4.0 + PI / 8.0,
            ghost_2a.position.coordinates(),
        );
        result.push(ghost_3a);
        let theta_3b = (theta_2b + 3.0).rem_euclid(8.0);
        let ghost_3b = gamma_pt(
            theta_3b * PI / 4.0 + PI / 8.0,
            ghost_2b.position.coordinates(),
        );
        result.push(ghost_3b);

        let theta_4 = (theta_3a - 3.0).rem_euclid(8.0);
        let ghost_4 = gamma_pt(
            theta_4 * PI / 4.0 + PI / 8.0,
            ghost_3a.position.coordinates(),
        );
        result.push(ghost_4);

        result
    }
}

impl GenerateGhosts<OrientedHyperbolicPoint<3, Angle>> for Periodic<EightEight> {
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }
    /// Place periodic images of sites near the edge of the periodic boundary
    #[inline]
    #[expect(clippy::too_many_lines, reason = "complicated function")]
    fn generate_ghosts(
        &self,
        site_properties: &OrientedHyperbolicPoint<3, Angle>,
    ) -> ArrayVec<OrientedHyperbolicPoint<3, Angle>, MAX_GHOSTS> {
        let mut result = ArrayVec::new();
        let r = site_properties.position();
        // identify which octant the point is in
        let angle = (r.coordinates()[1].atan2(r.coordinates()[0])).rem_euclid(2.0 * PI);
        let nearest_vertex_number =
            (((angle + (PI / 8.0)).rem_euclid(PI * 2.0)) / (PI / 4.0)).floor();
        let coords = *r.coordinates();
        let orientation = site_properties.orientation.theta;
        // transform to tile
        let eta = EightEight::CUSP_TO_EDGE;
        let gamma_pt = |theta: f64, point: &[f64; 3]| {
            let ghost = EightEight::gamma(eta, theta, point);
            let new_hyperbolic = Hyperbolic::from_minkowski_coordinates(Minkowski::from(ghost));
            let mut new_site = *site_properties;
            *new_site.position_mut() = new_hyperbolic;
            new_site
        };

        let theta_1a = (nearest_vertex_number + 4.0).rem_euclid(8.0);
        let ghost_1a = gamma_pt(theta_1a * PI / 4.0 + PI / 8.0, &coords);
        let rel_angle_1a = EightEight::reorient(theta_1a * PI / 4.0 + PI / 8.0, &coords);
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

        let theta_1b = (nearest_vertex_number + 3.0).rem_euclid(8.0);
        let ghost_1b = gamma_pt(theta_1b * PI / 4.0 + PI / 8.0, &coords);
        let rel_angle_1b = EightEight::reorient(theta_1b * PI / 4.0 + PI / 8.0, &coords);
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

        let theta_2a = (theta_1a - 3.0).rem_euclid(8.0);
        let ghost_2a = gamma_pt(
            theta_2a * PI / 4.0 + PI / 8.0,
            ghost_1a.position.coordinates(),
        );
        let rel_angle_2a = EightEight::reorient(
            theta_2a * PI / 4.0 + PI / 8.0,
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

        let theta_2b = (theta_1b + 3.0).rem_euclid(8.0);
        let ghost_2b = gamma_pt(
            theta_2b * PI / 4.0 + PI / 8.0,
            ghost_1b.position.coordinates(),
        );
        let rel_angle_2b = EightEight::reorient(
            theta_2b * PI / 4.0 + PI / 8.0,
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

        let theta_3a = (theta_2a - 3.0).rem_euclid(8.0);
        let ghost_3a = gamma_pt(
            theta_3a * PI / 4.0 + PI / 8.0,
            ghost_2a.position.coordinates(),
        );
        let rel_angle_3a = EightEight::reorient(
            theta_3a * PI / 4.0 + PI / 8.0,
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

        let theta_3b = (theta_2b + 3.0).rem_euclid(8.0);
        let ghost_3b = gamma_pt(
            theta_3b * PI / 4.0 + PI / 8.0,
            ghost_2b.position.coordinates(),
        );
        let rel_angle_3b = EightEight::reorient(
            theta_3b * PI / 4.0 + PI / 8.0,
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

        let theta_4 = (theta_3a - 3.0).rem_euclid(8.0);
        let ghost_4 = gamma_pt(
            theta_4 * PI / 4.0 + PI / 8.0,
            ghost_3a.position.coordinates(),
        );
        let rel_angle_4 = EightEight::reorient(
            theta_4 * PI / 4.0 + PI / 8.0,
            ghost_3a.position.coordinates(),
        );
        let orientation_4 = orientation_3a + rel_angle_4;
        let mut final_orientation = orientation_4;
        while final_orientation > PI {
            final_orientation -= 2.0 * PI;
        }
        while final_orientation <= -PI {
            final_orientation += 2.0 * PI;
        }
        let mut new_site_4 = *site_properties;
        *new_site_4.position_mut() = ghost_4.position;
        *new_site_4.orientation_mut() = Angle::from(final_orientation);
        result.push(new_site_4);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::{OrientedHyperbolicPoint, Point};
    use approxim::assert_relative_eq;
    use hoomd_manifold::{Hyperbolic, HyperbolicDisk};
    use hoomd_vector::Metric;
    use rand::{RngExt, SeedableRng, distr::Distribution, rngs::StdRng};
    use std::f64::consts::PI;

    #[test]
    fn doesnt_wrap_if_inside() {
        let r = 1.528_570_919_480_998;
        let mut rng = StdRng::seed_from_u64(239);
        let disk = HyperbolicDisk {
            disk_radius: r.try_into().expect("hard-coded positive number"),
            point: Hyperbolic::from_minkowski_coordinates(Minkowski::from([0.0, 0.0, 1.0])),
        };
        let random_point: Hyperbolic<3> = disk.sample(&mut rng);
        let random_point = Point::new(random_point);

        let periodic = Periodic::new(0.5, EightEight {}).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(random_point).expect("hard-coded");
        assert_eq!(
            random_point.position.coordinates(),
            wrapped_point.position.coordinates()
        );
    }

    #[test]
    fn wraps_to_opposite_edge() {
        let mut rng = StdRng::seed_from_u64(1);
        let side = f64::from(rng.random_range(0..8));
        let boost = EightEight::CUSP_TO_EDGE + 0.5;
        let offset = PI / 8.0;
        let point = Hyperbolic::<3>::from_polar_coordinates(boost, side * PI / 4.0 + offset);
        let point = Point::new(point);
        let periodic = Periodic::new(1.0, EightEight {}).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(point).expect("hard-coded");

        let wrapped_side = (side + 4.0).rem_euclid(8.0);
        let octant = (((wrapped_point.position.coordinates()[1]
            .atan2(wrapped_point.position.coordinates()[0]))
            / (PI / 4.0))
            .floor())
        .rem_euclid(8.0);

        // Check that point is wrapped to correct octant
        assert_eq!(wrapped_side, octant);

        // Check that point mapping is correct
        let new_boost = 2.0
            * (EightEight::EIGHTEIGHT.tanh()
                / (offset.cos() - offset.sin() * (1.0 - (2.0_f64).sqrt())))
            .atanh()
            - boost;
        let ans = Hyperbolic::<3>::from_polar_coordinates(
            new_boost,
            (wrapped_side + 1.0) * (PI / 4.0) - offset,
        );
        assert_relative_eq!(ans, wrapped_point.position, epsilon = 1e-12);
        assert_relative_eq!(
            -EightEight::distance_to_boundary(&point.position),
            EightEight::distance_to_boundary(&wrapped_point.position),
            epsilon = 1e-12
        );
    }

    #[test]
    fn consistency_edge() {
        let mut rng = StdRng::seed_from_u64(1);
        let side = f64::from(rng.random_range(0..8));
        let boost = EightEight::CUSP_TO_EDGE + 0.5;
        let offset = PI / 8.0 + 0.15 * rng.random::<f64>();
        let point = Hyperbolic::<3>::from_polar_coordinates(boost, side * PI / 4.0 + offset);
        let point = Point::new(point);
        let periodic = Periodic::new(1.0, EightEight {}).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(point).expect("hard-coded");
        let wrapped_poincare = wrapped_point.position.to_poincare();

        // Check that mapping is consistent with Poincaré transformation
        let (q_u, q_v) = (
            point.position.to_poincare()[0],
            point.position.to_poincare()[1],
        );
        let phi = (side + 4.0).rem_euclid(8.0) * PI / 4.0 + PI / 8.0;
        let a = ((1.0 + (PI / 4.0).cos()) / (1.0 - (PI / 4.0).cos())).sqrt();
        let b = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi.cos());
        let c = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi.sin());
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
    fn wraps_to_opposite_edge_distance() {
        let mut rng = StdRng::seed_from_u64(1);
        let side = f64::from(rng.random_range(0..8));
        let boost = EightEight::CUSP_TO_EDGE + 0.2;
        let offset = PI / 8.0 + rng.random::<f64>() * (PI / 8.0) * (1.0 - 0.5);
        let point = Hyperbolic::<3>::from_polar_coordinates(boost, side * PI / 4.0 + offset);
        let point = Point::new(point);
        let periodic = Periodic::new(1.0, EightEight {}).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(point).expect("hard-coded");

        // Check that mapping preserves the distance to the boundary
        assert_relative_eq!(
            -EightEight::distance_to_boundary(&point.position),
            EightEight::distance_to_boundary(&wrapped_point.position),
            epsilon = 1e-12
        );
    }

    #[test]
    fn wraps_orientation() {
        let angle_offset: f64 = 0.3;
        let boost = ((EightEight::EIGHTEIGHT).tanh() * ((PI / 4.0).sin())
            / ((angle_offset).sin() + (PI / 4.0 - angle_offset).sin()))
        .atanh()
            + 0.1;
        let point = Hyperbolic::<3>::from_polar_coordinates(boost, angle_offset + PI / 4.0);
        let tangent = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(EightEight::EIGHTEIGHT, PI / 4.0),
            &point,
        );
        let oriented_point = OrientedHyperbolicPoint {
            position: point,
            orientation: Angle::from(9.0 * PI / 8.0 + tangent),
        };
        let periodic = Periodic::new(0.5, EightEight {}).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(oriented_point).expect("hard-coded");

        let answer = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(EightEight::EIGHTEIGHT, 6.0 * PI / 4.0),
            &wrapped_point.position,
        );

        // Check that orientation maps correctly
        assert_relative_eq!(
            wrapped_point.orientation.theta,
            5.0 * PI / 8.0 + answer,
            epsilon = 1e-12
        );
    }

    #[test]
    fn wraps_nearby_orientations() {
        let offset = 0.000_001;
        let angle = PI / 8.0 + offset;
        let boost = ((EightEight::EIGHTEIGHT).tanh() * ((PI / 4.0).sin())
            / ((angle).sin() + (PI / 4.0 - angle).sin()))
        .atanh()
            + 0.5;
        let point_1 =
            Hyperbolic::<3>::from_polar_coordinates(boost, 5.0 * PI / 4.0 + PI / 8.0 - offset);
        let point_2 =
            Hyperbolic::<3>::from_polar_coordinates(boost, 5.0 * PI / 4.0 + PI / 8.0 + offset);

        let oriented_point_1 = OrientedHyperbolicPoint {
            position: point_1,
            orientation: Angle::from(PI),
        };
        let oriented_point_2 = OrientedHyperbolicPoint {
            position: point_2,
            orientation: Angle::from(PI),
        };
        let periodic = Periodic::new(0.5, EightEight {}).expect("hard-coded positive number");
        let wrapped_point_1 = periodic.wrap(oriented_point_1).expect("hard-coded");
        let wrapped_point_2 = periodic.wrap(oriented_point_2).expect("hard-coded");

        // check that positions are nearby
        assert_relative_eq!(
            wrapped_point_1.position.coordinates()[0],
            wrapped_point_2.position.coordinates()[0],
            epsilon = 1e-5
        );
        assert_relative_eq!(
            wrapped_point_1.position.coordinates()[1],
            wrapped_point_2.position.coordinates()[1],
            epsilon = 1e-5
        );
        assert_relative_eq!(
            wrapped_point_1.position.coordinates()[2],
            wrapped_point_2.position.coordinates()[2],
            epsilon = 1e-5
        );

        // check that orientations are nearby
        assert_relative_eq!(
            wrapped_point_1.orientation.theta.rem_euclid(2.0 * PI),
            wrapped_point_2.orientation.theta.rem_euclid(2.0 * PI),
            epsilon = 1e-5
        );
    }

    #[test]
    fn wraps_vertex() {
        let boost = EightEight::EIGHTEIGHT + 0.4;
        let point = Hyperbolic::<3>::from_polar_coordinates(boost, 3.0 * PI / 4.0 - 0.01);
        let point = Point::new(point);
        let periodic = Periodic::new(0.5, EightEight {}).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(point).expect("hard-coded");
        let wrapped_poincare = wrapped_point.position.to_poincare();
        let point_poincare = point.position.to_poincare();

        let phi7 = 7.0 * PI / 4.0 + PI / 8.0;
        let a7 = ((1.0 + (PI / 4.0).cos()) / (1.0 - (PI / 4.0).cos())).sqrt();
        let b7 = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi7.cos());
        let c7 = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi7.sin());
        let pref_7 = 1.0
            / ((b7 * point_poincare[1] - c7 * point_poincare[0]).powi(2)
                + (a7 + b7 * point_poincare[0] + c7 * point_poincare[1]).powi(2));
        let ans_7 = [
            pref_7
                * (point_poincare[0] * (a7.powi(2) + b7.powi(2) - c7.powi(2))
                    + 2.0 * b7 * c7 * point_poincare[1]
                    + a7 * b7 * (1.0 + point_poincare[0].powi(2) + point_poincare[1].powi(2))),
            pref_7
                * (point_poincare[1] * (a7.powi(2) - b7.powi(2) + c7.powi(2))
                    + 2.0 * b7 * c7 * point_poincare[0]
                    + a7 * c7 * (1.0 + point_poincare[0].powi(2) + point_poincare[1].powi(2))),
        ];
        let phi4 = PI + PI / 8.0;
        let a4 = ((1.0 + (PI / 4.0).cos()) / (1.0 - (PI / 4.0).cos())).sqrt();
        let b4 = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi4.cos());
        let c4 = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi4.sin());
        let pref_4 = 1.0
            / ((b4 * ans_7[1] - c4 * ans_7[0]).powi(2)
                + (a4 + b4 * ans_7[0] + c4 * ans_7[1]).powi(2));
        let ans_4 = [
            pref_4
                * (ans_7[0] * (a4.powi(2) + b4.powi(2) - c4.powi(2))
                    + 2.0 * b4 * c4 * ans_7[1]
                    + a4 * b4 * (1.0 + ans_7[0].powi(2) + ans_7[1].powi(2))),
            pref_4
                * (ans_7[1] * (a4.powi(2) - b4.powi(2) + c4.powi(2))
                    + 2.0 * b4 * c4 * ans_7[0]
                    + a4 * c4 * (1.0 + ans_7[0].powi(2) + ans_7[1].powi(2))),
        ];
        let phi1 = PI / 4.0 + PI / 8.0;
        let a1 = ((1.0 + (PI / 4.0).cos()) / (1.0 - (PI / 4.0).cos())).sqrt();
        let b1 = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi1.cos());
        let c1 = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi1.sin());
        let pref_1 = 1.0
            / ((b1 * ans_4[1] - c1 * ans_4[0]).powi(2)
                + (a1 + b1 * ans_4[0] + c1 * ans_4[1]).powi(2));
        let ans_1 = [
            pref_1
                * (ans_4[0] * (a1.powi(2) + b1.powi(2) - c1.powi(2))
                    + 2.0 * b1 * c1 * ans_4[1]
                    + a1 * b1 * (1.0 + ans_4[0].powi(2) + ans_4[1].powi(2))),
            pref_1
                * (ans_4[1] * (a1.powi(2) - b1.powi(2) + c1.powi(2))
                    + 2.0 * b1 * c1 * ans_4[0]
                    + a1 * c1 * (1.0 + ans_4[0].powi(2) + ans_4[1].powi(2))),
        ];
        let phi6 = 6.0 * PI / 4.0 + PI / 8.0;
        let a6 = ((1.0 + (PI / 4.0).cos()) / (1.0 - (PI / 4.0).cos())).sqrt();
        let b6 = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi6.cos());
        let c6 = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi6.sin());
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

        assert_relative_eq!(ans_6[0], wrapped_poincare[0], epsilon = 1e-12);
        assert_relative_eq!(ans_6[1], wrapped_poincare[1], epsilon = 1e-12);
    }

    #[test]
    fn wraps_vertex_orientation() {
        let boost = EightEight::EIGHTEIGHT + 0.5;
        let ve = EightEight::EIGHTEIGHT;
        let point = Hyperbolic::<3>::from_polar_coordinates(boost, PI / 4.0 + 0.00001);

        let (int_1, _or1) =
            OrientedHyperbolicPoint::<3, Angle>::intersection_point(PI / 8.0, ve - boost);
        let intersection = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (ve.cosh()) * (int_1.sinh()) * ((PI / 4.0).cos()) * ((PI / 8.0).cos())
                - (int_1.sinh()) * ((PI / 4.0).sin()) * ((PI / 8.0).sin())
                + (ve.sinh()) * (int_1.cosh()) * ((PI / 4.0).cos()),
            (ve.cosh()) * (int_1.sinh()) * ((PI / 4.0).sin()) * ((PI / 8.0).cos())
                + (int_1.sinh()) * ((PI / 4.0).cos()) * ((PI / 8.0).sin())
                + (ve.sinh()) * (int_1.cosh()) * ((PI / 4.0).sin()),
            (ve.sinh()) * (int_1.sinh()) * ((PI / 8.0).cos()) + (ve.cosh()) * (int_1.cosh()),
        ]));
        let pt_v_to_int = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(ve, PI / 4.0),
            &intersection,
        );
        let pt_int_to_pnt =
            OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(&intersection, &point);

        let oriented_point = OrientedHyperbolicPoint {
            position: point,
            orientation: Angle::from(pt_v_to_int + pt_int_to_pnt + 3.0 * PI / 8.0),
        };
        let periodic = Periodic::new(0.5, EightEight {}).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(oriented_point).expect("hard-coded");

        // check that the orientation maps correctly
        let new_intersection = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (ve.cosh()) * (int_1.sinh()) * ((5.0 * PI / 4.0).cos()) * ((9.0 * PI / 8.0).cos())
                - (int_1.sinh()) * ((5.0 * PI / 4.0).sin()) * ((9.0 * PI / 8.0).sin())
                + (ve.sinh()) * (int_1.cosh()) * ((5.0 * PI / 4.0).cos()),
            (ve.cosh()) * (int_1.sinh()) * ((5.0 * PI / 4.0).sin()) * ((9.0 * PI / 8.0).cos())
                + (int_1.sinh()) * ((5.0 * PI / 4.0).cos()) * ((9.0 * PI / 8.0).sin())
                + (ve.sinh()) * (int_1.cosh()) * ((5.0 * PI / 4.0).sin()),
            (ve.sinh()) * (int_1.sinh()) * ((9.0 * PI / 8.0).cos()) + (ve.cosh()) * (int_1.cosh()),
        ]));
        let pt_v_to_n_int = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(ve, 5.0 * PI / 4.0),
            &new_intersection,
        );
        let pt_n_int_to_w_pnt = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &new_intersection,
            &wrapped_point.position,
        );
        assert_relative_eq!(
            3.0 * PI / 8.0 + pt_v_to_n_int + pt_n_int_to_w_pnt,
            wrapped_point.orientation.theta,
            epsilon = 1e-12
        );
    }
    #[test]
    fn wraps_vertex_non_center() {
        let offset_boost: f64 = 0.4;
        let v: f64 = 2.448_452_447_678_076;
        let point = Hyperbolic::from_minkowski_coordinates(
            [
                (v.sinh()) * (offset_boost.cosh()),
                -offset_boost.sinh(),
                (v.cosh()) * (offset_boost.cosh()),
            ]
            .into(),
        );
        let point = Point::new(point);
        let periodic = Periodic::new(0.5, EightEight {}).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(point).expect("hard-coded");

        let wrapped_poincare = wrapped_point.position.to_poincare();
        let point_poincare = point.position.to_poincare();

        let phi3 = 3.0 * PI / 4.0 + PI / 8.0;
        let a3 = ((1.0 + (PI / 4.0).cos()) / (1.0 - (PI / 4.0).cos())).sqrt();
        let b3 = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi3.cos());
        let c3 = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi3.sin());
        let pref_3 = 1.0
            / ((b3 * point_poincare[1] - c3 * point_poincare[0]).powi(2)
                + (a3 + b3 * point_poincare[0] + c3 * point_poincare[1]).powi(2));
        let ans_3 = [
            pref_3
                * (point_poincare[0] * (a3.powi(2) + b3.powi(2) - c3.powi(2))
                    + 2.0 * b3 * c3 * point_poincare[1]
                    + a3 * b3 * (1.0 + point_poincare[0].powi(2) + point_poincare[1].powi(2))),
            pref_3
                * (point_poincare[1] * (a3.powi(2) - b3.powi(2) + c3.powi(2))
                    + 2.0 * b3 * c3 * point_poincare[0]
                    + a3 * c3 * (1.0 + point_poincare[0].powi(2) + point_poincare[1].powi(2))),
        ];
        let phi6 = 6.0 * PI / 4.0 + PI / 8.0;
        let a6 = ((1.0 + (PI / 4.0).cos()) / (1.0 - (PI / 4.0).cos())).sqrt();
        let b6 = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi6.cos());
        let c6 = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi6.sin());
        let pref_6 = 1.0
            / ((b6 * ans_3[1] - c6 * ans_3[0]).powi(2)
                + (a6 + b6 * ans_3[0] + c6 * ans_3[1]).powi(2));
        let ans_6 = [
            pref_6
                * (ans_3[0] * (a6.powi(2) + b6.powi(2) - c6.powi(2))
                    + 2.0 * b6 * c6 * ans_3[1]
                    + a6 * b6 * (1.0 + ans_3[0].powi(2) + ans_3[1].powi(2))),
            pref_6
                * (ans_3[1] * (a6.powi(2) - b6.powi(2) + c6.powi(2))
                    + 2.0 * b6 * c6 * ans_3[0]
                    + a6 * c6 * (1.0 + ans_3[0].powi(2) + ans_3[1].powi(2))),
        ];

        assert_relative_eq!(ans_6[0], wrapped_poincare[0], epsilon = 1e-12);
        assert_relative_eq!(ans_6[1], wrapped_poincare[1], epsilon = 1e-12);
    }
    #[test]
    fn wraps_orientation_vertex_non_center() {
        let offset_boost: f64 = 0.2;
        let angle_offset = 0.01;
        let v: f64 = 2.448_452_447_678_076;
        let point = Hyperbolic::from_minkowski_coordinates(
            [
                (v.sinh()) * (offset_boost.cosh())
                    + (v.cosh()) * (offset_boost.sinh()) * ((angle_offset + 3.0 * PI / 2.0).cos()),
                (offset_boost.sinh()) * ((angle_offset + 3.0 * PI / 2.0).sin()),
                (v.cosh()) * (offset_boost.cosh())
                    + (v.sinh()) * (offset_boost.sinh()) * ((angle_offset + 3.0 * PI / 2.0).cos()),
            ]
            .into(),
        );

        let (int_1, _or1) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            PI / 8.0 + angle_offset,
            offset_boost,
        );

        let intersection = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * (int_1.sinh()) * ((11.0 * PI / 8.0).cos()) + (v.sinh()) * (int_1.cosh()),
            (int_1.sinh()) * ((11.0 * PI / 8.0).sin()),
            (v.sinh()) * (int_1.sinh()) * ((11.0 * PI / 8.0).cos()) + (v.cosh()) * (int_1.cosh()),
        ]));
        let pt_v_to_int = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, 0.0),
            &intersection,
        );
        let pt_int_to_pnt =
            OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(&intersection, &point);

        let oriented_point = OrientedHyperbolicPoint {
            position: point,
            orientation: Angle::from(pt_v_to_int + pt_int_to_pnt + 11.0 * PI / 8.0),
        };

        let periodic = Periodic::new(0.5, EightEight {}).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(oriented_point).expect("hard-coded");

        // check that wrapping correctly maps orientation
        let new_intersection = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * (int_1.sinh()) * ((3.0 * PI / 2.0).cos()) * ((7.0 * PI / 8.0).cos())
                - (int_1.sinh()) * ((3.0 * PI / 2.0).sin()) * ((7.0 * PI / 8.0).sin())
                + (v.sinh()) * (int_1.cosh()) * ((3.0 * PI / 2.0).cos()),
            (v.cosh()) * (int_1.sinh()) * ((3.0 * PI / 2.0).sin()) * ((7.0 * PI / 8.0).cos())
                + (int_1.sinh()) * ((3.0 * PI / 2.0).cos()) * ((7.0 * PI / 8.0).sin())
                + (v.sinh()) * (int_1.cosh()) * ((3.0 * PI / 2.0).sin()),
            (v.sinh()) * (int_1.sinh()) * ((7.0 * PI / 8.0).cos()) + (v.cosh()) * (int_1.cosh()),
        ]));
        let pt_v_to_n_int = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, 3.0 * PI / 2.0),
            &new_intersection,
        );
        let pt_n_int_to_w_pnt = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &new_intersection,
            &wrapped_point.position,
        );
        assert_relative_eq!(
            3.0 * PI / 8.0 + pt_v_to_n_int + pt_n_int_to_w_pnt,
            wrapped_point.orientation.theta,
            epsilon = 1e-12
        );
    }
    #[test]
    fn ghost_near_side() {
        let mut rng = StdRng::seed_from_u64(1);
        let side = f64::from(rng.random_range(0..8));
        let offset = 0.4;
        let boost = 1.528_570_919_480_998 - offset;
        let point = Hyperbolic::<3>::from_polar_coordinates(boost, PI / 8.0 + side * PI / 4.0);
        let point = Point::new(point);

        let periodic = Periodic::new(1.0, EightEight {}).expect("hard-coded positive number");

        let ghost_array = periodic.generate_ghosts(&point);
        let ghost = match side {
            4.0 => ghost_array[0],
            _ => ghost_array[1],
        };

        let ans = Hyperbolic::<3>::from_polar_coordinates(
            1.528_570_919_480_998 + offset,
            (side + 4.0).rem_euclid(8.0) * PI / 4.0 + PI / 8.0,
        );

        assert_relative_eq!(ans, ghost.position, epsilon = 1e-12);
        assert_relative_eq!(
            EightEight::distance_to_boundary(&ghost.position),
            -EightEight::distance_to_boundary(&point.position),
            epsilon = 1e-12
        );
    }

    #[test]
    fn ghost_near_vertex() {
        let offset_boost = 0.3;
        let v: f64 = 2.448_452_447_678_076;
        let point = Hyperbolic::<3>::from_polar_coordinates(v - offset_boost, 0.0);
        let point = Point::new(point);
        let periodic = Periodic::new(0.5, EightEight {}).expect("hard-coded positive number");

        let ghost_array: ArrayVec<Point<Hyperbolic<3>>, MAX_GHOSTS> =
            periodic.generate_ghosts(&point);
        let ghost_6 = ghost_array[6];

        let ans_6 = Hyperbolic::<3>::from_polar_coordinates(v + offset_boost, PI);
        assert_relative_eq!(ans_6, ghost_6.position, epsilon = 1e-12);

        let ghost_3 = ghost_array[3];

        let ans_3 = Hyperbolic::from_minkowski_coordinates(
            [
                offset_boost.sinh(),
                -(v.sinh()) * (offset_boost.cosh()),
                (v.cosh()) * (offset_boost.cosh()),
            ]
            .into(),
        );
        assert_relative_eq!(ans_3, ghost_3.position, epsilon = 1e-12);
    }

    #[test]
    #[allow(clippy::too_many_lines, reason = "complicated function")]
    fn consistency_vertex() {
        let offset_boost = 0.3;
        let offset_angle = 0.1;
        let edge_boost: f64 = 2.448_452_447_678_076;
        let point =
            Hyperbolic::<3>::from_polar_coordinates(edge_boost - offset_boost, offset_angle);
        let point = Point::new(point);
        let periodic = Periodic::new(0.5, EightEight {}).expect("hard-coded positive number");

        let ghost_array: ArrayVec<Point<Hyperbolic<3>>, MAX_GHOSTS> =
            periodic.generate_ghosts(&point);

        // check double transformations
        let ghost_2_poincare = ghost_array[2].position.to_poincare();
        let (q_u, q_v) = (
            point.position.to_poincare()[0],
            point.position.to_poincare()[1],
        );
        let phi1 = PI / 4.0 + PI / 8.0;
        let phi4 = PI + PI / 8.0;
        let a = (1.0 + (PI / 4.0).cos() + 2.0 * ((PI / 4.0).cos()) * ((phi1 - phi4).cos()))
            / (1.0 - (PI / 4.0).cos());
        let d = (2.0 * ((PI / 4.0).cos()) * (phi1 - phi4).sin()) / (1.0 - (PI / 4.0).cos());
        let b = ((2.0 * ((PI / 4.0).cos()) * (1.0 + (PI / 4.0).cos())).sqrt())
            * (phi4.cos() + phi1.cos())
            / (1.0 - (PI / 4.0).cos());
        let c = ((2.0 * ((PI / 4.0).cos()) * (1.0 + (PI / 4.0).cos())).sqrt())
            * (phi4.sin() + phi1.sin())
            / (1.0 - (PI / 4.0).cos());
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
        let phi3 = 3.0 * PI / 4.0 + PI / 8.0;
        let phi6 = 6.0 * PI / 4.0 + PI / 8.0;
        let a = (1.0 + (PI / 4.0).cos() + 2.0 * ((PI / 4.0).cos()) * ((phi3 - phi6).cos()))
            / (1.0 - (PI / 4.0).cos());
        let d = (2.0 * ((PI / 4.0).cos()) * (phi6 - phi3).sin()) / (1.0 - (PI / 4.0).cos());
        let b = ((2.0 * ((PI / 4.0).cos()) * (1.0 + (PI / 4.0).cos())).sqrt())
            * (phi6.cos() + phi3.cos())
            / (1.0 - (PI / 4.0).cos());
        let c = ((2.0 * ((PI / 4.0).cos()) * (1.0 + (PI / 4.0).cos())).sqrt())
            * (phi6.sin() + phi3.sin())
            / (1.0 - (PI / 4.0).cos());
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
        let a = ((1.0 + (PI / 4.0).cos()) / (1.0 - (PI / 4.0).cos())).sqrt();
        let b = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi6.cos());
        let c = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi6.sin());
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
        let a = ((1.0 + (PI / 4.0).cos()) / (1.0 - (PI / 4.0).cos())).sqrt();
        let b = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi1.cos());
        let c = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi1.sin());
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

        // check quadruple transformation
        let ghost_6_poincare = ghost_array[6].position.to_poincare();
        let a = ((1.0 + (PI / 4.0).cos()) / (1.0 - (PI / 4.0).cos())).sqrt();
        let b = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi3.cos());
        let c = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi3.sin());
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

        // check single transformations
        let ghost_1_poincare = ghost_array[1].position.to_poincare();
        let a = ((1.0 + (PI / 4.0).cos()) / (1.0 - (PI / 4.0).cos())).sqrt();
        let b = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi3.cos());
        let c = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi3.sin());
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
        let a = ((1.0 + (PI / 4.0).cos()) / (1.0 - (PI / 4.0).cos())).sqrt();
        let b = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi4.cos());
        let c = ((2.0 * ((PI / 4.0).cos())) / (1.0 - (PI / 4.0).cos())).sqrt() * (phi4.sin());
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
    #[allow(clippy::too_many_lines, reason = "complicated function")]
    fn ghost_near_zeroth_vertex_orientation() {
        let offset_boost = 0.3;
        let v: f64 = 2.448_452_447_678_076;
        let point = Hyperbolic::<3>::from_polar_coordinates(v - offset_boost, 0.1);
        let point = OrientedHyperbolicPoint {
            position: point,
            orientation: Angle::from(0.0),
        };

        // get nearest boundary point
        let point_in_center = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * point.position.coordinates()[0]
                - (v.sinh()) * point.position.coordinates()[2],
            point.position.coordinates()[1],
            -(v.sinh()) * point.position.coordinates()[0]
                + (v.cosh()) * point.position.coordinates()[2],
        ]));
        let (angle_c, boost_c) = (
            point_in_center.coordinates()[1]
                .atan2(point_in_center.coordinates()[0])
                .rem_euclid(2.0 * PI),
            (point_in_center.coordinates()[2]).acosh(),
        );
        let (int_o, _o_o) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            angle_c - 7.0 * PI / 8.0,
            boost_c,
        );
        // have confirmed that this is the shortest path from the point to the boundary
        let intersection_o = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * (int_o.sinh()) * ((7.0 * PI / 8.0).cos()) + (v.sinh()) * (int_o.cosh()),
            (int_o.sinh()) * ((7.0 * PI / 8.0).sin()),
            (v.sinh()) * (int_o.sinh()) * ((7.0 * PI / 8.0).cos()) + (v.cosh()) * (int_o.cosh()),
        ]));

        let int_to_v_o = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, 0.0),
            &intersection_o,
        );
        let tang_o = 7.0 * PI / 8.0 + int_to_v_o;
        let relative_orientation_upper =
            (OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
                &point.position,
                &intersection_o,
            ) - tang_o)
                .rem_euclid(2.0 * PI);

        let (int_ol, _o_ol) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            9.0 * PI / 8.0 - angle_c,
            boost_c,
        );

        let intersection_ol = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * (int_ol.sinh()) * ((9.0 * PI / 8.0).cos()) + (v.sinh()) * (int_ol.cosh()),
            (int_ol.sinh()) * ((9.0 * PI / 8.0).sin()),
            (v.sinh()) * (int_ol.sinh()) * ((9.0 * PI / 8.0).cos()) + (v.cosh()) * (int_ol.cosh()),
        ]));

        let int_to_v_ol = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, 0.0),
            &intersection_ol,
        );
        let tang_ol = 9.0 * PI / 8.0 + int_to_v_ol;
        let relative_orientation_lower =
            (OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
                &point.position,
                &intersection_ol,
            ) - tang_ol)
                .rem_euclid(2.0 * PI);

        let periodic = Periodic::new(0.5, EightEight {}).expect("hard-coded positive number");

        let ghost_array = periodic.generate_ghosts(&point);

        let ghost_0 = ghost_array[0];
        let ghost_0_center = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((5.0 * PI / 4.0).cos()) * ghost_0.position.coordinates()[0]
                + (v.cosh()) * ((5.0 * PI / 4.0).sin()) * ghost_0.position.coordinates()[1]
                - (v.sinh()) * ghost_0.position.coordinates()[2],
            -((5.0 * PI / 4.0).sin()) * ghost_0.position.coordinates()[0]
                + ((5.0 * PI / 4.0).cos()) * ghost_0.position.coordinates()[1],
            (v.cosh()) * ghost_0.position.coordinates()[2]
                - (v.sinh()) * ((5.0 * PI / 4.0).cos()) * ghost_0.position.coordinates()[0]
                - (v.sinh()) * ((5.0 * PI / 4.0).sin()) * ghost_0.position.coordinates()[1],
        ]));
        let (angle_0, boost_0) = (
            ghost_0_center.coordinates()[1].atan2(ghost_0_center.coordinates()[0]),
            (ghost_0_center.coordinates()[2]).acosh(),
        );
        let (int_0, _o_0) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            9.0 * PI / 8.0 - angle_0,
            boost_0,
        );
        // check that intersection is minimum in center frame
        let int_c = Hyperbolic::<3>::from_polar_coordinates(int_0, 9.0 * PI / 8.0);
        let int_c_plus = Hyperbolic::<3>::from_polar_coordinates(int_0 + 0.05, 9.0 * PI / 8.0);
        let int_c_minus = Hyperbolic::<3>::from_polar_coordinates(int_0 - 0.05, 9.0 * PI / 8.0);
        assert!(ghost_0_center.distance(&int_c) < ghost_0_center.distance(&int_c_minus));
        assert!(ghost_0_center.distance(&int_c) < ghost_0_center.distance(&int_c_plus));

        let intersection_0 = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((5.0 * PI / 4.0).cos()) * (int_0.sinh()) * ((9.0 * PI / 8.0).cos())
                - ((5.0 * PI / 4.0).sin()) * (int_0.sinh()) * ((9.0 * PI / 8.0).sin())
                + (v.sinh()) * ((5.0 * PI / 4.0).cos()) * (int_0.cosh()),
            (v.cosh()) * ((5.0 * PI / 4.0).sin()) * (int_0.sinh()) * ((9.0 * PI / 8.0).cos())
                + ((5.0 * PI / 4.0).cos()) * (int_0.sinh()) * ((9.0 * PI / 8.0).sin())
                + (v.sinh()) * ((5.0 * PI / 4.0).sin()) * (int_0.cosh()),
            (v.sinh()) * (int_0.sinh()) * ((9.0 * PI / 8.0).cos()) + (v.cosh()) * (int_0.cosh()),
        ]));

        // check that intersection point is also a minimum
        let blip = 0.05;
        let intersection_plus = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh())
                * ((5.0 * PI / 4.0).cos())
                * ((int_0 + blip).sinh())
                * ((9.0 * PI / 8.0).cos())
                - ((5.0 * PI / 4.0).sin()) * ((int_0 + blip).sinh()) * ((9.0 * PI / 8.0).sin())
                + (v.sinh()) * ((5.0 * PI / 4.0).cos()) * ((int_0 + blip).cosh()),
            (v.cosh())
                * ((5.0 * PI / 4.0).sin())
                * ((int_0 + blip).sinh())
                * ((9.0 * PI / 8.0).cos())
                + ((5.0 * PI / 4.0).cos()) * ((int_0 + blip).sinh()) * ((9.0 * PI / 8.0).sin())
                + (v.sinh()) * ((5.0 * PI / 4.0).sin()) * ((int_0 + blip).cosh()),
            (v.sinh()) * ((int_0 + blip).sinh()) * ((9.0 * PI / 8.0).cos())
                + (v.cosh()) * ((int_0 + blip).cosh()),
        ]));
        let intersection_minus = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh())
                * ((5.0 * PI / 4.0).cos())
                * ((int_0 - blip).sinh())
                * ((9.0 * PI / 8.0).cos())
                - ((5.0 * PI / 4.0).sin()) * ((int_0 - blip).sinh()) * ((9.0 * PI / 8.0).sin())
                + (v.sinh()) * ((5.0 * PI / 4.0).cos()) * ((int_0 - blip).cosh()),
            (v.cosh())
                * ((5.0 * PI / 4.0).sin())
                * ((int_0 - blip).sinh())
                * ((9.0 * PI / 8.0).cos())
                + ((5.0 * PI / 4.0).cos()) * ((int_0 - blip).sinh()) * ((9.0 * PI / 8.0).sin())
                + (v.sinh()) * ((5.0 * PI / 4.0).sin()) * ((int_0 - blip).cosh()),
            (v.sinh()) * ((int_0 - blip).sinh()) * ((9.0 * PI / 8.0).cos())
                + (v.cosh()) * ((int_0 - blip).cosh()),
        ]));
        assert!(
            ghost_0.position.distance(&intersection_0)
                < ghost_0.position.distance(&intersection_minus)
        );
        assert!(
            ghost_0.position.distance(&intersection_0)
                < ghost_0.position.distance(&intersection_plus)
        );

        let int_to_ghost = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &intersection_0,
            &ghost_0.position,
        );
        let v_to_int = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, 5.0 * PI / 4.0),
            &intersection_0,
        );
        let tang_0 = 3.0 * PI / 8.0 + v_to_int;
        let ans_0 = (relative_orientation_upper + tang_0 + int_to_ghost).rem_euclid(2.0 * PI);
        assert_relative_eq!(
            ans_0.rem_euclid(2.0 * PI),
            ghost_0.orientation.theta.rem_euclid(2.0 * PI),
            epsilon = 1e-12
        );

        // check remaining ghosts in less detail
        // ghost 1
        let ghost_1 = ghost_array[1];
        let ghost_1_center = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((3.0 * PI / 4.0).cos()) * ghost_1.position.coordinates()[0]
                + (v.cosh()) * ((3.0 * PI / 4.0).sin()) * ghost_1.position.coordinates()[1]
                - (v.sinh()) * ghost_1.position.coordinates()[2],
            -((3.0 * PI / 4.0).sin()) * ghost_1.position.coordinates()[0]
                + ((3.0 * PI / 4.0).cos()) * ghost_1.position.coordinates()[1],
            (v.cosh()) * ghost_1.position.coordinates()[2]
                - (v.sinh()) * ((3.0 * PI / 4.0).cos()) * ghost_1.position.coordinates()[0]
                - (v.sinh()) * ((3.0 * PI / 4.0).sin()) * ghost_1.position.coordinates()[1],
        ]));
        let (angle_1, boost_1) = (
            ghost_1_center.coordinates()[1].atan2(ghost_1_center.coordinates()[0]),
            (ghost_1_center.coordinates()[2]).acosh(),
        );
        let (int_1, _o_1) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            angle_1 - 7.0 * PI / 8.0,
            boost_1,
        );
        let intersection_1 = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((3.0 * PI / 4.0).cos()) * (int_1.sinh()) * ((7.0 * PI / 8.0).cos())
                - ((3.0 * PI / 4.0).sin()) * (int_1.sinh()) * ((7.0 * PI / 8.0).sin())
                + (v.sinh()) * ((3.0 * PI / 4.0).cos()) * (int_1.cosh()),
            (v.cosh()) * ((3.0 * PI / 4.0).sin()) * (int_1.sinh()) * ((7.0 * PI / 8.0).cos())
                + ((3.0 * PI / 4.0).cos()) * (int_1.sinh()) * ((7.0 * PI / 8.0).sin())
                + (v.sinh()) * ((3.0 * PI / 4.0).sin()) * (int_1.cosh()),
            (v.sinh()) * (int_1.sinh()) * ((7.0 * PI / 8.0).cos()) + (v.cosh()) * (int_1.cosh()),
        ]));
        let int_to_ghost_1 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &intersection_1,
            &ghost_1.position,
        );
        let v_to_int_1 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, 3.0 * PI / 4.0),
            &intersection_1,
        );
        let tang_1 = 13.0 * PI / 8.0 + v_to_int_1;
        let ans_1 = (relative_orientation_lower + tang_1 + int_to_ghost_1).rem_euclid(2.0 * PI);
        assert_relative_eq!(ans_1, ghost_1.orientation.theta, epsilon = 1e-12);

        // ghost 2
        let ghost_2 = ghost_array[2];
        let ghost_2_center = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((PI / 2.0).cos()) * ghost_2.position.coordinates()[0]
                + (v.cosh()) * ((PI / 2.0).sin()) * ghost_2.position.coordinates()[1]
                - (v.sinh()) * ghost_2.position.coordinates()[2],
            -((PI / 2.0).sin()) * ghost_2.position.coordinates()[0]
                + ((PI / 2.0).cos()) * ghost_2.position.coordinates()[1],
            (v.cosh()) * ghost_2.position.coordinates()[2]
                - (v.sinh()) * ((PI / 2.0).cos()) * ghost_2.position.coordinates()[0]
                - (v.sinh()) * ((PI / 2.0).sin()) * ghost_2.position.coordinates()[1],
        ]));
        let (angle_2, boost_2) = (
            ghost_2_center.coordinates()[1].atan2(ghost_2_center.coordinates()[0]),
            (ghost_2_center.coordinates()[2]).acosh(),
        );
        let (int_2, _o_1) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            angle_2 - 11.0 * PI / 8.0,
            boost_2,
        );
        let intersection_2 = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((PI / 2.0).cos()) * (int_2.sinh()) * ((11.0 * PI / 8.0).cos())
                - ((PI / 2.0).sin()) * (int_2.sinh()) * ((11.0 * PI / 8.0).sin())
                + (v.sinh()) * ((PI / 2.0).cos()) * (int_2.cosh()),
            (v.cosh()) * ((PI / 2.0).sin()) * (int_2.sinh()) * ((11.0 * PI / 8.0).cos())
                + ((PI / 2.0).cos()) * (int_2.sinh()) * ((11.0 * PI / 8.0).sin())
                + (v.sinh()) * ((PI / 2.0).sin()) * (int_2.cosh()),
            (v.sinh()) * (int_2.sinh()) * ((11.0 * PI / 8.0).cos()) + (v.cosh()) * (int_2.cosh()),
        ]));
        let int_to_ghost_2 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &intersection_2,
            &ghost_2.position,
        );
        let v_to_int_2 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, PI / 2.0),
            &intersection_2,
        );
        let tang_2 = 15.0 * PI / 8.0 + v_to_int_2;
        let ans_2 = (relative_orientation_upper + tang_2 + int_to_ghost_2).rem_euclid(2.0 * PI);
        assert_relative_eq!(
            ans_2.rem_euclid(2.0 * PI),
            ghost_2.orientation.theta.rem_euclid(2.0 * PI),
            epsilon = 1e-12
        );

        // ghost 3
        let ghost_3 = ghost_array[3];
        let ghost_3_center = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((3.0 * PI / 2.0).cos()) * ghost_3.position.coordinates()[0]
                + (v.cosh()) * ((3.0 * PI / 2.0).sin()) * ghost_3.position.coordinates()[1]
                - (v.sinh()) * ghost_3.position.coordinates()[2],
            -((3.0 * PI / 2.0).sin()) * ghost_3.position.coordinates()[0]
                + ((3.0 * PI / 2.0).cos()) * ghost_3.position.coordinates()[1],
            (v.cosh()) * ghost_3.position.coordinates()[2]
                - (v.sinh()) * ((3.0 * PI / 2.0).cos()) * ghost_3.position.coordinates()[0]
                - (v.sinh()) * ((3.0 * PI / 2.0).sin()) * ghost_3.position.coordinates()[1],
        ]));
        let (angle_3, boost_3) = (
            ghost_3_center.coordinates()[1].atan2(ghost_3_center.coordinates()[0]),
            (ghost_3_center.coordinates()[2]).acosh(),
        );
        let (int_3, _o_1) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            5.0 * PI / 8.0 - angle_3,
            boost_3,
        );
        let intersection_3 = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((3.0 * PI / 2.0).cos()) * (int_3.sinh()) * ((5.0 * PI / 8.0).cos())
                - ((3.0 * PI / 2.0).sin()) * (int_3.sinh()) * ((5.0 * PI / 8.0).sin())
                + (v.sinh()) * ((3.0 * PI / 2.0).cos()) * (int_3.cosh()),
            (v.cosh()) * ((3.0 * PI / 2.0).sin()) * (int_3.sinh()) * ((5.0 * PI / 8.0).cos())
                + ((3.0 * PI / 2.0).cos()) * (int_3.sinh()) * ((5.0 * PI / 8.0).sin())
                + (v.sinh()) * ((3.0 * PI / 2.0).sin()) * (int_3.cosh()),
            (v.sinh()) * (int_3.sinh()) * ((5.0 * PI / 8.0).cos()) + (v.cosh()) * (int_3.cosh()),
        ]));
        let int_to_ghost_3 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &intersection_3,
            &ghost_3.position,
        );
        let v_to_int_3 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, 3.0 * PI / 2.0),
            &intersection_3,
        );
        let tang_3 = PI / 8.0 + v_to_int_3;
        let ans_3 = (relative_orientation_lower + tang_3 + int_to_ghost_3).rem_euclid(2.0 * PI);
        assert_relative_eq!(
            ans_3.rem_euclid(2.0 * PI),
            ghost_3.orientation.theta.rem_euclid(2.0 * PI),
            epsilon = 1e-12
        );

        // ghost 4
        let ghost_4 = ghost_array[4];
        let ghost_4_center = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((7.0 * PI / 4.0).cos()) * ghost_4.position.coordinates()[0]
                + (v.cosh()) * ((7.0 * PI / 4.0).sin()) * ghost_4.position.coordinates()[1]
                - (v.sinh()) * ghost_4.position.coordinates()[2],
            -((7.0 * PI / 4.0).sin()) * ghost_4.position.coordinates()[0]
                + ((7.0 * PI / 4.0).cos()) * ghost_4.position.coordinates()[1],
            (v.cosh()) * ghost_4.position.coordinates()[2]
                - (v.sinh()) * ((7.0 * PI / 4.0).cos()) * ghost_4.position.coordinates()[0]
                - (v.sinh()) * ((7.0 * PI / 4.0).sin()) * ghost_4.position.coordinates()[1],
        ]));
        let (angle_4, boost_4) = (
            ghost_4_center.coordinates()[1]
                .atan2(ghost_4_center.coordinates()[0])
                .rem_euclid(2.0 * PI),
            (ghost_4_center.coordinates()[2]).acosh(),
        );
        let (int_4, _o_1) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            angle_4 - 13.0 * PI / 8.0,
            boost_4,
        );
        let intersection_4 = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((7.0 * PI / 4.0).cos()) * (int_4.sinh()) * ((13.0 * PI / 8.0).cos())
                - ((7.0 * PI / 4.0).sin()) * (int_4.sinh()) * ((13.0 * PI / 8.0).sin())
                + (v.sinh()) * ((7.0 * PI / 4.0).cos()) * (int_4.cosh()),
            (v.cosh()) * ((7.0 * PI / 4.0).sin()) * (int_4.sinh()) * ((13.0 * PI / 8.0).cos())
                + ((7.0 * PI / 4.0).cos()) * (int_4.sinh()) * ((13.0 * PI / 8.0).sin())
                + (v.sinh()) * ((7.0 * PI / 4.0).sin()) * (int_4.cosh()),
            (v.sinh()) * (int_4.sinh()) * ((13.0 * PI / 8.0).cos()) + (v.cosh()) * (int_4.cosh()),
        ]));
        let int_to_ghost_4 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &intersection_4,
            &ghost_4.position,
        );
        let v_to_int_4 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, 7.0 * PI / 4.0),
            &intersection_4,
        );
        let tang_4 = 11.0 * PI / 8.0 + v_to_int_4;
        let ans_4 = (relative_orientation_upper + tang_4 + int_to_ghost_4).rem_euclid(2.0 * PI);
        assert_relative_eq!(
            ans_4.rem_euclid(2.0 * PI),
            ghost_4.orientation.theta.rem_euclid(2.0 * PI),
            epsilon = 1e-12
        );

        // ghost 5
        let ghost_5 = ghost_array[5];
        let ghost_5_center = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((PI / 4.0).cos()) * ghost_5.position.coordinates()[0]
                + (v.cosh()) * ((PI / 4.0).sin()) * ghost_5.position.coordinates()[1]
                - (v.sinh()) * ghost_5.position.coordinates()[2],
            -((PI / 4.0).sin()) * ghost_5.position.coordinates()[0]
                + ((PI / 4.0).cos()) * ghost_5.position.coordinates()[1],
            (v.cosh()) * ghost_5.position.coordinates()[2]
                - (v.sinh()) * ((PI / 4.0).cos()) * ghost_5.position.coordinates()[0]
                - (v.sinh()) * ((PI / 4.0).sin()) * ghost_5.position.coordinates()[1],
        ]));
        let (angle_5, boost_5) = (
            ghost_5_center.coordinates()[1].atan2(ghost_5_center.coordinates()[0]),
            (ghost_5_center.coordinates()[2]).acosh(),
        );
        let (int_5, _o_1) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            3.0 * PI / 8.0 - angle_5,
            boost_5,
        );
        let intersection_5 = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((PI / 4.0).cos()) * (int_5.sinh()) * ((3.0 * PI / 8.0).cos())
                - ((PI / 4.0).sin()) * (int_5.sinh()) * ((3.0 * PI / 8.0).sin())
                + (v.sinh()) * ((PI / 4.0).cos()) * (int_5.cosh()),
            (v.cosh()) * ((PI / 4.0).sin()) * (int_5.sinh()) * ((3.0 * PI / 8.0).cos())
                + ((PI / 4.0).cos()) * (int_5.sinh()) * ((3.0 * PI / 8.0).sin())
                + (v.sinh()) * ((PI / 4.0).sin()) * (int_5.cosh()),
            (v.sinh()) * (int_5.sinh()) * ((3.0 * PI / 8.0).cos()) + (v.cosh()) * (int_5.cosh()),
        ]));
        let int_to_ghost_5 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &intersection_5,
            &ghost_5.position,
        );
        let v_to_int_5 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, PI / 4.0),
            &intersection_5,
        );
        let tang_5 = 5.0 * PI / 8.0 + v_to_int_5;
        let ans_5 = (relative_orientation_lower + tang_5 + int_to_ghost_5).rem_euclid(2.0 * PI);
        assert_relative_eq!(
            ans_5.rem_euclid(2.0 * PI),
            ghost_5.orientation.theta.rem_euclid(2.0 * PI),
            epsilon = 1e-12
        );

        // ghost 6
        let ghost_6 = ghost_array[6];
        let ghost_6_center = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((PI).cos()) * ghost_6.position.coordinates()[0]
                + (v.cosh()) * ((PI).sin()) * ghost_6.position.coordinates()[1]
                - (v.sinh()) * ghost_6.position.coordinates()[2],
            -((PI).sin()) * ghost_6.position.coordinates()[0]
                + ((PI).cos()) * ghost_6.position.coordinates()[1],
            (v.cosh()) * ghost_6.position.coordinates()[2]
                - (v.sinh()) * ((PI).cos()) * ghost_6.position.coordinates()[0]
                - (v.sinh()) * ((PI).sin()) * ghost_6.position.coordinates()[1],
        ]));
        let (angle_6, boost_6) = (
            ghost_6_center.coordinates()[1]
                .atan2(ghost_6_center.coordinates()[0])
                .rem_euclid(2.0 * PI),
            (ghost_6_center.coordinates()[2]).acosh(),
        );
        let (int_6, _o_1) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            angle_6 - 15.0 * PI / 8.0,
            boost_6,
        );
        let intersection_6 = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((PI).cos()) * (int_6.sinh()) * ((15.0 * PI / 8.0).cos())
                - ((PI).sin()) * (int_6.sinh()) * ((15.0 * PI / 8.0).sin())
                + (v.sinh()) * ((PI).cos()) * (int_6.cosh()),
            (v.cosh()) * ((PI).sin()) * (int_6.sinh()) * ((15.0 * PI / 8.0).cos())
                + ((PI).cos()) * (int_6.sinh()) * ((15.0 * PI / 8.0).sin())
                + (v.sinh()) * ((PI).sin()) * (int_6.cosh()),
            (v.sinh()) * (int_6.sinh()) * ((15.0 * PI / 8.0).cos()) + (v.cosh()) * (int_6.cosh()),
        ]));
        let int_to_ghost_6 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &intersection_6,
            &ghost_6.position,
        );
        let v_to_int_6 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, PI),
            &intersection_6,
        );
        let tang_6 = 7.0 * PI / 8.0 + v_to_int_6;
        let ans_6 = (relative_orientation_upper + tang_6 + int_to_ghost_6).rem_euclid(2.0 * PI);
        assert_relative_eq!(
            ans_6.rem_euclid(2.0 * PI),
            ghost_6.orientation.theta.rem_euclid(2.0 * PI),
            epsilon = 1e-12
        );

        let (int_6_b, _o_1) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            (PI / 8.0 - angle_6).rem_euclid(2.0 * PI),
            boost_6,
        );
        let intersection_6_b = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((PI).cos()) * (int_6_b.sinh()) * ((PI / 8.0).cos())
                - ((PI).sin()) * (int_6_b.sinh()) * ((PI / 8.0).sin())
                + (v.sinh()) * ((PI).cos()) * (int_6_b.cosh()),
            (v.cosh()) * ((PI).sin()) * (int_6_b.sinh()) * ((PI / 8.0).cos())
                + ((PI).cos()) * (int_6_b.sinh()) * ((PI / 8.0).sin())
                + (v.sinh()) * ((PI).sin()) * (int_6_b.cosh()),
            (v.sinh()) * (int_6_b.sinh()) * ((PI / 8.0).cos()) + (v.cosh()) * (int_6_b.cosh()),
        ]));
        let int_to_ghost_6_b = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &intersection_6_b,
            &ghost_6.position,
        );
        let v_to_int_6_b = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, PI),
            &intersection_6_b,
        );
        let tang_6_b = 9.0 * PI / 8.0 + v_to_int_6_b;
        let ans_6_b =
            (relative_orientation_lower + tang_6_b + int_to_ghost_6_b).rem_euclid(2.0 * PI);
        assert_relative_eq!(
            ans_6_b.rem_euclid(2.0 * PI),
            ghost_6.orientation.theta.rem_euclid(2.0 * PI),
            epsilon = 1e-12
        );
    }

    #[test]
    #[allow(clippy::too_many_lines, reason = "complicated function")]
    fn ghost_near_third_vertex_orientation() {
        let offset_boost = 0.25;
        let v: f64 = 2.448_452_447_678_076;
        let point = Hyperbolic::<3>::from_polar_coordinates(v - offset_boost, 3.0 * PI / 4.0 + 0.1);
        let point = OrientedHyperbolicPoint {
            position: point,
            orientation: Angle::from(PI),
        };

        // get nearest boundary point
        let point_in_center = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((3.0 * PI / 4.0).cos()) * point.position.coordinates()[0]
                + (v.cosh()) * ((3.0 * PI / 4.0).sin()) * point.position.coordinates()[1]
                - (v.sinh()) * point.position.coordinates()[2],
            -((3.0 * PI / 4.0).sin()) * point.position.coordinates()[0]
                + ((3.0 * PI / 4.0).cos()) * point.position.coordinates()[1],
            -(v.sinh()) * ((3.0 * PI / 4.0).cos()) * point.position.coordinates()[0]
                - (v.sinh()) * ((3.0 * PI / 4.0).sin()) * point.position.coordinates()[1]
                + (v.cosh()) * point.position.coordinates()[2],
        ]));
        let (angle_c, boost_c) = (
            point_in_center.coordinates()[1]
                .atan2(point_in_center.coordinates()[0])
                .rem_euclid(2.0 * PI),
            (point_in_center.coordinates()[2]).acosh(),
        );
        let (int_o, _o_o) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            angle_c - 7.0 * PI / 8.0,
            boost_c,
        );
        // have confirmed that this is the shortest path from the point to the boundary
        let intersection_o = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((3.0 * PI / 4.0).cos()) * (int_o.sinh()) * ((7.0 * PI / 8.0).cos())
                - ((3.0 * PI / 4.0).sin()) * (int_o.sinh()) * ((7.0 * PI / 8.0).sin())
                + (v.sinh()) * ((3.0 * PI / 4.0).cos()) * (int_o.cosh()),
            (v.cosh()) * ((3.0 * PI / 4.0).sin()) * (int_o.sinh()) * ((7.0 * PI / 8.0).cos())
                + ((3.0 * PI / 4.0).cos()) * (int_o.sinh()) * ((7.0 * PI / 8.0).sin())
                + (v.sinh()) * ((3.0 * PI / 4.0).sin()) * (int_o.cosh()),
            (v.sinh()) * (int_o.sinh()) * ((7.0 * PI / 8.0).cos()) + (v.cosh()) * (int_o.cosh()),
        ]));

        let int_to_v_o = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, 3.0 * PI / 4.0),
            &intersection_o,
        );
        let tang_o = 13.0 * PI / 8.0 + int_to_v_o;
        let relative_orientation_upper = PI
            + (OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
                &point.position,
                &intersection_o,
            ) - tang_o)
                .rem_euclid(2.0 * PI);

        let (int_ol, _o_ol) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            9.0 * PI / 8.0 - angle_c,
            boost_c,
        );

        let intersection_ol = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((3.0 * PI / 4.0).cos()) * (int_ol.sinh()) * ((9.0 * PI / 8.0).cos())
                - ((3.0 * PI / 4.0).sin()) * (int_ol.sinh()) * ((9.0 * PI / 8.0).sin())
                + (v.sinh()) * ((3.0 * PI / 4.0).cos()) * (int_ol.cosh()),
            (v.cosh()) * ((3.0 * PI / 4.0).sin()) * (int_ol.sinh()) * ((9.0 * PI / 8.0).cos())
                + ((3.0 * PI / 4.0).cos()) * (int_ol.sinh()) * ((9.0 * PI / 8.0).sin())
                + (v.sinh()) * ((3.0 * PI / 4.0).sin()) * (int_ol.cosh()),
            (v.sinh()) * (int_ol.sinh()) * ((9.0 * PI / 8.0).cos()) + (v.cosh()) * (int_ol.cosh()),
        ]));

        let int_to_v_ol = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, 3.0 * PI / 4.0),
            &intersection_ol,
        );
        let tang_ol = 15.0 * PI / 8.0 + int_to_v_ol;
        let relative_orientation_lower = PI
            + (OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
                &point.position,
                &intersection_ol,
            ) - tang_ol)
                .rem_euclid(2.0 * PI);

        let periodic = Periodic::new(0.5, EightEight {}).expect("hard-coded positive number");

        let ghost_array = periodic.generate_ghosts(&point);

        let ghost_0 = ghost_array[0];
        let ghost_0_center = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ghost_0.position.coordinates()[0]
                - (v.sinh()) * ghost_0.position.coordinates()[2],
            ghost_0.position.coordinates()[1],
            (v.cosh()) * ghost_0.position.coordinates()[2]
                - (v.sinh()) * ghost_0.position.coordinates()[0],
        ]));
        let (angle_0, boost_0) = (
            ghost_0_center.coordinates()[1].atan2(ghost_0_center.coordinates()[0]),
            (ghost_0_center.coordinates()[2]).acosh(),
        );
        let (int_0, _o_0) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            angle_0 - 9.0 * PI / 8.0,
            boost_0,
        );
        // check that intersection is minimum in center frame
        let int_c = Hyperbolic::<3>::from_polar_coordinates(int_0, 9.0 * PI / 8.0);
        let int_c_plus = Hyperbolic::<3>::from_polar_coordinates(int_0 + 0.05, 9.0 * PI / 8.0);
        let int_c_minus = Hyperbolic::<3>::from_polar_coordinates(int_0 - 0.05, 9.0 * PI / 8.0);
        assert!(ghost_0_center.distance(&int_c) < ghost_0_center.distance(&int_c_minus));
        assert!(ghost_0_center.distance(&int_c) < ghost_0_center.distance(&int_c_plus));

        let intersection_0 = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * (int_0.sinh()) * ((9.0 * PI / 8.0).cos()) + (v.sinh()) * (int_0.cosh()),
            (int_0.sinh()) * ((9.0 * PI / 8.0).sin()),
            (v.sinh()) * (int_0.sinh()) * ((9.0 * PI / 8.0).cos()) + (v.cosh()) * (int_0.cosh()),
        ]));

        // check that intersection point is also a minimum
        let blip = 0.05;
        let intersection_plus = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((int_0 + blip).sinh()) * ((9.0 * PI / 8.0).cos())
                + (v.sinh()) * ((int_0 + blip).cosh()),
            ((int_0 + blip).sinh()) * ((9.0 * PI / 8.0).sin()),
            (v.sinh()) * ((int_0 + blip).sinh()) * ((9.0 * PI / 8.0).cos())
                + (v.cosh()) * ((int_0 + blip).cosh()),
        ]));
        let intersection_minus = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((int_0 - blip).sinh()) * ((9.0 * PI / 8.0).cos())
                + (v.sinh()) * ((int_0 - blip).cosh()),
            ((int_0 - blip).sinh()) * ((9.0 * PI / 8.0).sin()),
            (v.sinh()) * ((int_0 - blip).sinh()) * ((9.0 * PI / 8.0).cos())
                + (v.cosh()) * ((int_0 - blip).cosh()),
        ]));
        assert!(
            ghost_0.position.distance(&intersection_0)
                < ghost_0.position.distance(&intersection_minus)
        );
        assert!(
            ghost_0.position.distance(&intersection_0)
                < ghost_0.position.distance(&intersection_plus)
        );

        let int_to_ghost = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &intersection_0,
            &ghost_0.position,
        );
        let v_to_int = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, 0.0),
            &intersection_0,
        );
        let tang_0 = 9.0 * PI / 8.0 + v_to_int;
        let ans_0 = (relative_orientation_upper + tang_0 + int_to_ghost).rem_euclid(2.0 * PI);
        assert_relative_eq!(ans_0, ghost_0.orientation.theta, epsilon = 1e-12);

        // check remaining ghosts in less detail
        // ghost 1
        let ghost_1 = ghost_array[1];
        let ghost_1_center = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((3.0 * PI / 2.0).cos()) * ghost_1.position.coordinates()[0]
                + (v.cosh()) * ((3.0 * PI / 2.0).sin()) * ghost_1.position.coordinates()[1]
                - (v.sinh()) * ghost_1.position.coordinates()[2],
            -((3.0 * PI / 2.0).sin()) * ghost_1.position.coordinates()[0]
                + ((3.0 * PI / 2.0).cos()) * ghost_1.position.coordinates()[1],
            (v.cosh()) * ghost_1.position.coordinates()[2]
                - (v.sinh()) * ((3.0 * PI / 2.0).cos()) * ghost_1.position.coordinates()[0]
                - (v.sinh()) * ((3.0 * PI / 2.0).sin()) * ghost_1.position.coordinates()[1],
        ]));
        let (angle_1, boost_1) = (
            ghost_1_center.coordinates()[1].atan2(ghost_1_center.coordinates()[0]),
            (ghost_1_center.coordinates()[2]).acosh(),
        );
        let (int_1, _o_1) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            7.0 * PI / 8.0 - angle_1,
            boost_1,
        );
        let intersection_1 = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((3.0 * PI / 2.0).cos()) * (int_1.sinh()) * ((7.0 * PI / 8.0).cos())
                - ((3.0 * PI / 2.0).sin()) * (int_1.sinh()) * ((7.0 * PI / 8.0).sin())
                + (v.sinh()) * ((3.0 * PI / 2.0).cos()) * (int_1.cosh()),
            (v.cosh()) * ((3.0 * PI / 2.0).sin()) * (int_1.sinh()) * ((7.0 * PI / 8.0).cos())
                + ((3.0 * PI / 2.0).cos()) * (int_1.sinh()) * ((7.0 * PI / 8.0).sin())
                + (v.sinh()) * ((3.0 * PI / 2.0).sin()) * (int_1.cosh()),
            (v.sinh()) * (int_1.sinh()) * ((7.0 * PI / 8.0).cos()) + (v.cosh()) * (int_1.cosh()),
        ]));
        let int_to_ghost_1 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &intersection_1,
            &ghost_1.position,
        );
        let v_to_int_1 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, 3.0 * PI / 2.0),
            &intersection_1,
        );
        let tang_1 = 3.0 * PI / 8.0 + v_to_int_1;
        let ans_1 = (relative_orientation_lower + tang_1 + int_to_ghost_1).rem_euclid(2.0 * PI);
        assert_relative_eq!(
            ans_1.rem_euclid(2.0 * PI),
            ghost_1.orientation.theta.rem_euclid(2.0 * PI),
            epsilon = 1e-12
        );

        // ghost 2
        let ghost_2 = ghost_array[2];
        let ghost_2_center = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((5.0 * PI / 4.0).cos()) * ghost_2.position.coordinates()[0]
                + (v.cosh()) * ((5.0 * PI / 4.0).sin()) * ghost_2.position.coordinates()[1]
                - (v.sinh()) * ghost_2.position.coordinates()[2],
            -((5.0 * PI / 4.0).sin()) * ghost_2.position.coordinates()[0]
                + ((5.0 * PI / 4.0).cos()) * ghost_2.position.coordinates()[1],
            (v.cosh()) * ghost_2.position.coordinates()[2]
                - (v.sinh()) * ((5.0 * PI / 4.0).cos()) * ghost_2.position.coordinates()[0]
                - (v.sinh()) * ((5.0 * PI / 4.0).sin()) * ghost_2.position.coordinates()[1],
        ]));
        let (angle_2, boost_2) = (
            ghost_2_center.coordinates()[1].atan2(ghost_2_center.coordinates()[0]),
            (ghost_2_center.coordinates()[2]).acosh(),
        );
        let (int_2, _o_1) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            angle_2 - 11.0 * PI / 8.0,
            boost_2,
        );
        let intersection_2 = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((5.0 * PI / 4.0).cos()) * (int_2.sinh()) * ((11.0 * PI / 8.0).cos())
                - ((5.0 * PI / 4.0).sin()) * (int_2.sinh()) * ((11.0 * PI / 8.0).sin())
                + (v.sinh()) * ((5.0 * PI / 4.0).cos()) * (int_2.cosh()),
            (v.cosh()) * ((5.0 * PI / 4.0).sin()) * (int_2.sinh()) * ((11.0 * PI / 8.0).cos())
                + ((5.0 * PI / 4.0).cos()) * (int_2.sinh()) * ((11.0 * PI / 8.0).sin())
                + (v.sinh()) * ((5.0 * PI / 4.0).sin()) * (int_2.cosh()),
            (v.sinh()) * (int_2.sinh()) * ((11.0 * PI / 8.0).cos()) + (v.cosh()) * (int_2.cosh()),
        ]));
        let int_to_ghost_2 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &intersection_2,
            &ghost_2.position,
        );
        let v_to_int_2 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, 5.0 * PI / 4.0),
            &intersection_2,
        );
        let tang_2 = 5.0 * PI / 8.0 + v_to_int_2;
        let ans_2 = (relative_orientation_upper + tang_2 + int_to_ghost_2).rem_euclid(2.0 * PI);
        assert_relative_eq!(
            ans_2.rem_euclid(2.0 * PI),
            ghost_2.orientation.theta.rem_euclid(2.0 * PI),
            epsilon = 1e-12
        );

        // ghost 3
        let ghost_3 = ghost_array[3];
        let ghost_3_center = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((PI / 4.0).cos()) * ghost_3.position.coordinates()[0]
                + (v.cosh()) * ((PI / 4.0).sin()) * ghost_3.position.coordinates()[1]
                - (v.sinh()) * ghost_3.position.coordinates()[2],
            -((PI / 4.0).sin()) * ghost_3.position.coordinates()[0]
                + ((PI / 4.0).cos()) * ghost_3.position.coordinates()[1],
            (v.cosh()) * ghost_3.position.coordinates()[2]
                - (v.sinh()) * ((PI / 4.0).cos()) * ghost_3.position.coordinates()[0]
                - (v.sinh()) * ((PI / 4.0).sin()) * ghost_3.position.coordinates()[1],
        ]));
        let (angle_3, boost_3) = (
            ghost_3_center.coordinates()[1].atan2(ghost_3_center.coordinates()[0]),
            (ghost_3_center.coordinates()[2]).acosh(),
        );
        let (int_3, _o_1) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            5.0 * PI / 8.0 - angle_3,
            boost_3,
        );
        let intersection_3 = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((PI / 4.0).cos()) * (int_3.sinh()) * ((5.0 * PI / 8.0).cos())
                - ((PI / 4.0).sin()) * (int_3.sinh()) * ((5.0 * PI / 8.0).sin())
                + (v.sinh()) * ((PI / 4.0).cos()) * (int_3.cosh()),
            (v.cosh()) * ((PI / 4.0).sin()) * (int_3.sinh()) * ((5.0 * PI / 8.0).cos())
                + ((PI / 4.0).cos()) * (int_3.sinh()) * ((5.0 * PI / 8.0).sin())
                + (v.sinh()) * ((PI / 4.0).sin()) * (int_3.cosh()),
            (v.sinh()) * (int_3.sinh()) * ((5.0 * PI / 8.0).cos()) + (v.cosh()) * (int_3.cosh()),
        ]));
        let int_to_ghost_3 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &intersection_3,
            &ghost_3.position,
        );
        let v_to_int_3 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, PI / 4.0),
            &intersection_3,
        );
        let tang_3 = 7.0 * PI / 8.0 + v_to_int_3;
        let ans_3 = (relative_orientation_lower + tang_3 + int_to_ghost_3).rem_euclid(2.0 * PI);
        assert_relative_eq!(
            ans_3.rem_euclid(2.0 * PI),
            ghost_3.orientation.theta.rem_euclid(2.0 * PI),
            epsilon = 1e-12
        );

        // ghost 4
        let ghost_4 = ghost_array[4];
        let ghost_4_center = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((PI / 2.0).cos()) * ghost_4.position.coordinates()[0]
                + (v.cosh()) * ((PI / 2.0).sin()) * ghost_4.position.coordinates()[1]
                - (v.sinh()) * ghost_4.position.coordinates()[2],
            -((PI / 2.0).sin()) * ghost_4.position.coordinates()[0]
                + ((PI / 2.0).cos()) * ghost_4.position.coordinates()[1],
            (v.cosh()) * ghost_4.position.coordinates()[2]
                - (v.sinh()) * ((PI / 2.0).cos()) * ghost_4.position.coordinates()[0]
                - (v.sinh()) * ((PI / 2.0).sin()) * ghost_4.position.coordinates()[1],
        ]));
        let (angle_4, boost_4) = (
            ghost_4_center.coordinates()[1].atan2(ghost_4_center.coordinates()[0]),
            (ghost_4_center.coordinates()[2]).acosh(),
        );
        let (int_4, _o_1) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            angle_4 - 13.0 * PI / 8.0,
            boost_4,
        );
        let intersection_4 = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((PI / 2.0).cos()) * (int_4.sinh()) * ((13.0 * PI / 8.0).cos())
                - ((PI / 2.0).sin()) * (int_4.sinh()) * ((13.0 * PI / 8.0).sin())
                + (v.sinh()) * ((PI / 2.0).cos()) * (int_4.cosh()),
            (v.cosh()) * ((PI / 2.0).sin()) * (int_4.sinh()) * ((13.0 * PI / 8.0).cos())
                + ((PI / 2.0).cos()) * (int_4.sinh()) * ((13.0 * PI / 8.0).sin())
                + (v.sinh()) * ((PI / 2.0).sin()) * (int_4.cosh()),
            (v.sinh()) * (int_4.sinh()) * ((13.0 * PI / 8.0).cos()) + (v.cosh()) * (int_4.cosh()),
        ]));
        let int_to_ghost_4 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &intersection_4,
            &ghost_4.position,
        );
        let v_to_int_4 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, PI / 2.0),
            &intersection_4,
        );
        let tang_4 = PI / 8.0 + v_to_int_4;
        let ans_4 = (relative_orientation_upper + tang_4 + int_to_ghost_4).rem_euclid(2.0 * PI);
        assert_relative_eq!(
            ans_4.rem_euclid(2.0 * PI),
            ghost_4.orientation.theta.rem_euclid(2.0 * PI),
            epsilon = 1e-12
        );

        // ghost 5
        let ghost_5 = ghost_array[5];
        let ghost_5_center = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((PI).cos()) * ghost_5.position.coordinates()[0]
                + (v.cosh()) * ((PI).sin()) * ghost_5.position.coordinates()[1]
                - (v.sinh()) * ghost_5.position.coordinates()[2],
            -((PI).sin()) * ghost_5.position.coordinates()[0]
                + ((PI).cos()) * ghost_5.position.coordinates()[1],
            (v.cosh()) * ghost_5.position.coordinates()[2]
                - (v.sinh()) * ((PI).cos()) * ghost_5.position.coordinates()[0]
                - (v.sinh()) * ((PI).sin()) * ghost_5.position.coordinates()[1],
        ]));
        let (angle_5, boost_5) = (
            ghost_5_center.coordinates()[1].atan2(ghost_5_center.coordinates()[0]),
            (ghost_5_center.coordinates()[2]).acosh(),
        );
        let (int_5, _o_1) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            3.0 * PI / 8.0 - angle_5,
            boost_5,
        );
        let intersection_5 = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((PI).cos()) * (int_5.sinh()) * ((3.0 * PI / 8.0).cos())
                - ((PI).sin()) * (int_5.sinh()) * ((3.0 * PI / 8.0).sin())
                + (v.sinh()) * ((PI).cos()) * (int_5.cosh()),
            (v.cosh()) * ((PI).sin()) * (int_5.sinh()) * ((3.0 * PI / 8.0).cos())
                + ((PI).cos()) * (int_5.sinh()) * ((3.0 * PI / 8.0).sin())
                + (v.sinh()) * ((PI).sin()) * (int_5.cosh()),
            (v.sinh()) * (int_5.sinh()) * ((3.0 * PI / 8.0).cos()) + (v.cosh()) * (int_5.cosh()),
        ]));
        let int_to_ghost_5 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &intersection_5,
            &ghost_5.position,
        );
        let v_to_int_5 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, PI),
            &intersection_5,
        );
        let tang_5 = 11.0 * PI / 8.0 + v_to_int_5;
        let ans_5 = (relative_orientation_lower + tang_5 + int_to_ghost_5).rem_euclid(2.0 * PI);
        assert_relative_eq!(
            ans_5.rem_euclid(2.0 * PI),
            ghost_5.orientation.theta.rem_euclid(2.0 * PI),
            epsilon = 1e-12
        );

        // ghost 6
        let ghost_6 = ghost_array[6];
        let ghost_6_center = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((7.0 * PI / 4.0).cos()) * ghost_6.position.coordinates()[0]
                + (v.cosh()) * ((7.0 * PI / 4.0).sin()) * ghost_6.position.coordinates()[1]
                - (v.sinh()) * ghost_6.position.coordinates()[2],
            -((7.0 * PI / 4.0).sin()) * ghost_6.position.coordinates()[0]
                + ((7.0 * PI / 4.0).cos()) * ghost_6.position.coordinates()[1],
            (v.cosh()) * ghost_6.position.coordinates()[2]
                - (v.sinh()) * ((7.0 * PI / 4.0).cos()) * ghost_6.position.coordinates()[0]
                - (v.sinh()) * ((7.0 * PI / 4.0).sin()) * ghost_6.position.coordinates()[1],
        ]));
        let (angle_6, boost_6) = (
            ghost_6_center.coordinates()[1].atan2(ghost_6_center.coordinates()[0]),
            (ghost_6_center.coordinates()[2]).acosh(),
        );
        let (int_6, _o_1) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            15.0 * PI / 8.0 - angle_6,
            boost_6,
        );
        let intersection_6 = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((7.0 * PI / 4.0).cos()) * (int_6.sinh()) * ((15.0 * PI / 8.0).cos())
                - ((7.0 * PI / 4.0).sin()) * (int_6.sinh()) * ((15.0 * PI / 8.0).sin())
                + (v.sinh()) * ((7.0 * PI / 4.0).cos()) * (int_6.cosh()),
            (v.cosh()) * ((7.0 * PI / 4.0).sin()) * (int_6.sinh()) * ((15.0 * PI / 8.0).cos())
                + ((7.0 * PI / 4.0).cos()) * (int_6.sinh()) * ((15.0 * PI / 8.0).sin())
                + (v.sinh()) * ((7.0 * PI / 4.0).sin()) * (int_6.cosh()),
            (v.sinh()) * (int_6.sinh()) * ((15.0 * PI / 8.0).cos()) + (v.cosh()) * (int_6.cosh()),
        ]));
        let int_to_ghost_6 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &intersection_6,
            &ghost_6.position,
        );
        let v_to_int_6 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, 7.0 * PI / 4.0),
            &intersection_6,
        );
        let tang_6 = 13.0 * PI / 8.0 + v_to_int_6;
        let ans_6 = (relative_orientation_upper + tang_6 + int_to_ghost_6).rem_euclid(2.0 * PI);
        assert_relative_eq!(
            ans_6.rem_euclid(2.0 * PI),
            ghost_6.orientation.theta.rem_euclid(2.0 * PI),
            epsilon = 1e-12
        );

        let (int_6, _o_1) = OrientedHyperbolicPoint::<3, Angle>::intersection_point(
            (PI / 8.0 - angle_6).rem_euclid(2.0 * PI),
            boost_6,
        );
        let intersection_6 = Hyperbolic::<3>::from_minkowski_coordinates(Minkowski::from([
            (v.cosh()) * ((7.0 * PI / 4.0).cos()) * (int_6.sinh()) * ((PI / 8.0).cos())
                - ((7.0 * PI / 4.0).sin()) * (int_6.sinh()) * ((PI / 8.0).sin())
                + (v.sinh()) * ((7.0 * PI / 4.0).cos()) * (int_6.cosh()),
            (v.cosh()) * ((7.0 * PI / 4.0).sin()) * (int_6.sinh()) * ((PI / 8.0).cos())
                + ((7.0 * PI / 4.0).cos()) * (int_6.sinh()) * ((PI / 8.0).sin())
                + (v.sinh()) * ((7.0 * PI / 4.0).sin()) * (int_6.cosh()),
            (v.sinh()) * (int_6.sinh()) * ((PI / 8.0).cos()) + (v.cosh()) * (int_6.cosh()),
        ]));
        let int_to_ghost_6 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &intersection_6,
            &ghost_6.position,
        );
        let v_to_int_6 = OrientedHyperbolicPoint::<3, Angle>::parallel_transport_angle(
            &Hyperbolic::<3>::from_polar_coordinates(v, 7.0 * PI / 4.0),
            &intersection_6,
        );
        let tang_6 = 15.0 * PI / 8.0 + v_to_int_6;
        let ans_6 = (relative_orientation_lower + tang_6 + int_to_ghost_6).rem_euclid(2.0 * PI);
        assert_relative_eq!(
            ans_6.rem_euclid(2.0 * PI),
            ghost_6.orientation.theta.rem_euclid(2.0 * PI),
            epsilon = 1e-12
        );
    }
}

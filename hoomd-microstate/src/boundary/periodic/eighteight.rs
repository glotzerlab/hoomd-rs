// Copyright (c) 2024-2025 The Regents of the University of Michigan.
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
    property::Position,
};
use hoomd_geometry::shape::EightEight;
use hoomd_manifold::{Hyperbolic, Minkowski};
use hoomd_vector::Metric;

impl MaximumAllowableInteractionRange for EightEight {
    /// The largest value that the maximum interaction range can take.
    ///
    /// This bound is determined by the edge length of the octagon.
    #[inline]
    fn maximum_allowable_interaction_range(&self) -> f64 {
        self.skirt * EightEight::EDGE_LENGTH / 3.0
    }
}

impl<P> Wrap<P> for Periodic<EightEight>
where
    P: Position<Position = Hyperbolic<3>>,
{
    /// Wrap a point on the Hyperbolic to the inside of the {8,8} tile.
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
    ///     Hyperbolic::<3>::from_polar_coordinates(boost, offset + PI / 4.0, 1.0);
    /// let point = Point::new(point);
    /// let periodic = Periodic::new(0.5, EightEight { skirt: 1.0_f64 })?;
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
    ///     1.0,
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
    #[expect(clippy::cast_possible_truncation, reason = "truncating float to usize")]
    #[expect(clippy::cast_sign_loss, reason = "hard-coded positive numbers")]
    fn wrap(&self, properties: P) -> Result<P, Error> {
        let mut properties = properties;
        let r = properties.position_mut();
        assert_eq!(
            r.skirt(),
            self.shape.skirt,
            "point must be wrapped onto a Hyperbolic with the same skirt"
        );

        let angle = r.coordinates()[1].atan2(r.coordinates()[0]);
        let theta = angle.rem_euclid(PI * 2.0);

        // distance to the boundary; if positive, r is within the tile
        let d = EightEight::distance_to_boundary(r);

        // find out which vertex of the octagon the point is closest to
        let vertex_number = (((theta + (PI / 8.0)).rem_euclid(PI * 2.0)) / (PI / 4.0)).floor();
        let nearest_vertex = Hyperbolic::<3>::from_polar_coordinates(
            EightEight::EIGHTEIGHT,
            PI * vertex_number / 4.0,
            r.skirt(),
        );

        // if point is safely within the tile, do nothing
        if d >= 0.0 {
            Ok(properties)
        } else if r.distance(&nearest_vertex) < EightEight::EDGE_LENGTH / 2.0
            || d > -self.maximum_interaction_range
        {
            // if point is past EIGHTEIGHT and within EDGE_LENGTH/2 of the vertex, figure out which octagon it needs to be wrapped into
            // transform point to frame where relevant vertex is in the center
            let (vertex_boost, vertex_angle) = (
                EightEight::EIGHTEIGHT,
                (vertex_number * PI / 4.0).rem_euclid(PI * 2.0),
            );
            let transformed_point = Minkowski::from([
                r.coordinates()[0] * (-vertex_boost).cosh() * (-vertex_angle).cos()
                    - r.coordinates()[1] * (-vertex_boost).cosh() * (-vertex_angle).sin()
                    + r.coordinates()[2] * (-vertex_boost).sinh(),
                r.coordinates()[0] * (-vertex_angle).sin()
                    + r.coordinates()[1] * (-vertex_angle).cos(),
                r.coordinates()[0] * (-vertex_boost).sinh() * (-vertex_angle).cos()
                    - r.coordinates()[1] * (-vertex_boost).sinh() * (-vertex_angle).sin()
                    + r.coordinates()[2] * (-vertex_boost).cosh(),
            ]);
            // get coords of point in transformed frame
            let trans_angle =
                transformed_point.coordinates[1].atan2(transformed_point.coordinates[0]);
            let octant = (((trans_angle + (PI / 8.0)).rem_euclid(2.0 * PI)) / (PI / 4.0)).floor(); //octant the point is inside of, in the transformed frame, in global coords
            let new_vertex = (octant + (4.0 + 3.0 * vertex_number).rem_euclid(8.0)).rem_euclid(8.0); // in vertex coords
            let vertex_list = [0.0, 3.0, 6.0, 1.0, 4.0, 7.0, 2.0, 5.0];
            let new_vertex_num = vertex_list[new_vertex.floor() as usize]; // new vertex which point should be mapped to the inside of, in global
            // rotate frame in center to get the relevant octant facing up
            let rot = (PI / 2.0 - octant * PI / 4.0).rem_euclid(PI * 2.0);
            let rotated_in_center = Minkowski::from([
                transformed_point.coordinates[0] * (rot.cos())
                    - transformed_point.coordinates[1] * (rot.sin()),
                transformed_point.coordinates[0] * (rot.sin())
                    + transformed_point.coordinates[1] * (rot.cos()),
                transformed_point.coordinates[2],
            ]);
            // now boost and rotate to put the vertex back into the correct spot
            let (new_vertex_boost, new_vertex_angle) = (
                EightEight::EIGHTEIGHT,
                (new_vertex_num * (PI / 4.0) + (PI / 2.0)).rem_euclid(PI * 2.0),
            );
            let wrapped = Minkowski::from([
                rotated_in_center.coordinates[0] * (new_vertex_angle.cos())
                    - rotated_in_center.coordinates[1]
                        * (new_vertex_boost.cosh())
                        * (new_vertex_angle.sin())
                    + rotated_in_center.coordinates[2]
                        * (new_vertex_boost.sinh())
                        * (new_vertex_angle.sin()),
                rotated_in_center.coordinates[0] * (new_vertex_angle.sin())
                    + rotated_in_center.coordinates[1]
                        * (new_vertex_boost.cosh())
                        * (new_vertex_angle.cos())
                    - rotated_in_center.coordinates[2]
                        * (new_vertex_boost.sinh())
                        * (new_vertex_angle.cos()),
                -rotated_in_center.coordinates[1] * (new_vertex_boost.sinh())
                    + rotated_in_center.coordinates[2] * (new_vertex_boost.cosh()),
            ]);
            let wrapped_hyperbolic = Hyperbolic::from_minkowski_coordinates(wrapped, r.skirt());
            *r = wrapped_hyperbolic;
            Ok(properties)
        } else {
            Err(Error::CannotWrapProperties)
        }
    }
}

impl<S> GenerateGhosts<S> for Periodic<EightEight>
where
    S: Position<Position = Hyperbolic<3>> + Copy + Default,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }
    /// Place periodic images of sites near the edge of the periodic boundary
    #[inline]
    #[expect(clippy::too_many_lines, reason = "complicated function")]
    #[expect(clippy::cast_possible_truncation, reason = "truncating float to usize")]
    #[expect(clippy::cast_sign_loss, reason = "hard-coded positive numbers")]
    fn generate_ghosts(&self, site_properties: &S) -> ArrayVec<S, MAX_GHOSTS> {
        let mut result = ArrayVec::new();
        let r = site_properties.position();

        let theta = (r.coordinates()[1].atan2(r.coordinates()[0])).rem_euclid(2.0 * PI);
        let octant = ((theta / (PI / 4.0)).floor()).rem_euclid(8.0);
        let distance_to_bdy = EightEight::distance_to_boundary(r);

        // put a ghost particle near a vertex
        let new_site_vertex = |loc_num: f64, next: i32, point: &Hyperbolic<3>| {
            // boost to frame near vertex, get site position with respect to vertex
            // loc_num is the octagon vertex the point is closest to
            // next is the octagon vertex near which the ghost particle will be placed
            let (loc_boost, loc_angle) = (EightEight::EIGHTEIGHT, loc_num * (PI / 4.0));
            let in_vertex_frame = Minkowski::from([
                point.coordinates()[0] * (-loc_boost).cosh() * (-loc_angle).cos()
                    - point.coordinates()[1] * (-loc_boost).cosh() * (-loc_angle).sin()
                    + point.coordinates()[2] * (-loc_boost).sinh(),
                point.coordinates()[0] * (-loc_angle).sin()
                    + point.coordinates()[1] * (-loc_angle).cos(),
                point.coordinates()[0] * (-loc_boost).sinh() * (-loc_angle).cos()
                    - point.coordinates()[1] * (-loc_boost).sinh() * (-loc_angle).sin()
                    + point.coordinates()[2] * (-loc_boost).cosh(),
            ]);
            // Put a ghost particle near each of the vertices
            let real_to_vertex = [0.0, 3.0, 6.0, 1.0, 4.0, 7.0, 2.0, 5.0];
            let vertex_num = real_to_vertex[(loc_num.rem_euclid(8.0)).floor() as usize];
            // look at the in-center frame of the vertex that is i-th CCW from current one
            let next_vertex_num =
                real_to_vertex[((loc_num + f64::from(next)).rem_euclid(8.0)).floor() as usize];
            // Find which octant our vertex is in, in this frame
            let zeroth = (4.0_f64 + next_vertex_num).rem_euclid(8.0);
            let octant = (vertex_num - zeroth).rem_euclid(8.0);
            // now rotate point in center to put it in the right orientation
            let rot = ((octant - 4.0) * (PI / 4.0)).rem_euclid(PI * 2.0);
            let rotated_in_center = Minkowski::from([
                in_vertex_frame.coordinates[0] * (rot.cos())
                    - in_vertex_frame.coordinates[1] * (rot.sin()),
                in_vertex_frame.coordinates[0] * (rot.sin())
                    + in_vertex_frame.coordinates[1] * (rot.cos()),
                in_vertex_frame.coordinates[2],
            ]);
            // boost and rotate to correct position in global frame
            let (new_vertex_boost, new_vertex_angle) = (
                EightEight::EIGHTEIGHT,
                ((f64::from(next) + loc_num).rem_euclid(8.0) * (PI / 4.0)),
            );
            let ghost = Minkowski::from([
                rotated_in_center.coordinates[0]
                    * (new_vertex_boost.cosh())
                    * (new_vertex_angle.cos())
                    - rotated_in_center.coordinates[1] * (new_vertex_angle.sin())
                    + rotated_in_center.coordinates[2]
                        * (new_vertex_boost.sinh())
                        * (new_vertex_angle.cos()),
                rotated_in_center.coordinates[0]
                    * (new_vertex_boost.cosh())
                    * (new_vertex_angle.sin())
                    + rotated_in_center.coordinates[1] * (new_vertex_angle.cos())
                    + rotated_in_center.coordinates[2]
                        * (new_vertex_boost.sinh())
                        * (new_vertex_angle.sin()),
                rotated_in_center.coordinates[0] * (new_vertex_boost.sinh())
                    + rotated_in_center.coordinates[2] * (new_vertex_boost.cosh()),
            ]);
            let new_hyperbolic = Hyperbolic::from_minkowski_coordinates(ghost, r.skirt());
            let mut new_site = *site_properties;
            *new_site.position_mut() = new_hyperbolic;
            new_site
        };

        // put a ghost particle near an edge
        let new_site_edge = |edge_num: f64, point: &Hyperbolic<3>| {
            let vertex_number = (((theta + (PI / 8.0)).rem_euclid(PI * 2.0)) / (PI / 4.0)).floor();
            let mut new_site = *site_properties;
            if vertex_number == edge_num {
                new_site = new_site_vertex(vertex_number, 5, point);
            } else if vertex_number == (edge_num + 1.0_f64).rem_euclid(8.0) {
                new_site = new_site_vertex(vertex_number, 3, point);
            }
            new_site
        };

        let toggle: f64 = EightEight::EDGE_LENGTH / 3.0;

        let near_vertex_0 = r.distance(&Hyperbolic::<3>::from_polar_coordinates(
            EightEight::EIGHTEIGHT,
            0.0,
            r.skirt(),
        )) < toggle;
        let near_vertex_1 = r.distance(&Hyperbolic::<3>::from_polar_coordinates(
            EightEight::EIGHTEIGHT,
            PI / 4.0,
            r.skirt(),
        )) < toggle;
        let near_vertex_2 = r.distance(&Hyperbolic::<3>::from_polar_coordinates(
            EightEight::EIGHTEIGHT,
            2.0 * PI / 4.0,
            r.skirt(),
        )) < toggle;
        let near_vertex_3 = r.distance(&Hyperbolic::<3>::from_polar_coordinates(
            EightEight::EIGHTEIGHT,
            3.0 * PI / 4.0,
            r.skirt(),
        )) < toggle;
        let near_vertex_4 = r.distance(&Hyperbolic::<3>::from_polar_coordinates(
            EightEight::EIGHTEIGHT,
            4.0 * PI / 4.0,
            r.skirt(),
        )) < toggle;
        let near_vertex_5 = r.distance(&Hyperbolic::<3>::from_polar_coordinates(
            EightEight::EIGHTEIGHT,
            5.0 * PI / 4.0,
            r.skirt(),
        )) < toggle;
        let near_vertex_6 = r.distance(&Hyperbolic::<3>::from_polar_coordinates(
            EightEight::EIGHTEIGHT,
            6.0 * PI / 4.0,
            r.skirt(),
        )) < toggle;
        let near_vertex_7 = r.distance(&Hyperbolic::<3>::from_polar_coordinates(
            EightEight::EIGHTEIGHT,
            7.0 * PI / 4.0,
            r.skirt(),
        )) < toggle;

        let near_side_0 = octant == 0.0
            && distance_to_bdy < self.maximum_interaction_range
            && r.distance(&Hyperbolic::<3>::from_polar_coordinates(
                EightEight::EIGHTEIGHT,
                0.0,
                r.skirt(),
            )) > toggle
            && r.distance(&Hyperbolic::<3>::from_polar_coordinates(
                EightEight::EIGHTEIGHT,
                PI / 4.0,
                r.skirt(),
            )) > toggle;
        let near_side_1 = octant == 1.0
            && distance_to_bdy < self.maximum_interaction_range
            && r.distance(&Hyperbolic::<3>::from_polar_coordinates(
                EightEight::EIGHTEIGHT,
                PI / 4.0,
                r.skirt(),
            )) > toggle
            && r.distance(&Hyperbolic::<3>::from_polar_coordinates(
                EightEight::EIGHTEIGHT,
                2.0 * PI / 4.0,
                r.skirt(),
            )) > toggle;
        let near_side_2 = octant == 2.0
            && distance_to_bdy < self.maximum_interaction_range
            && r.distance(&Hyperbolic::<3>::from_polar_coordinates(
                EightEight::EIGHTEIGHT,
                2.0 * PI / 4.0,
                r.skirt(),
            )) > toggle
            && r.distance(&Hyperbolic::<3>::from_polar_coordinates(
                EightEight::EIGHTEIGHT,
                3.0 * PI / 4.0,
                r.skirt(),
            )) > toggle;
        let near_side_3 = octant == 3.0
            && distance_to_bdy < self.maximum_interaction_range
            && r.distance(&Hyperbolic::<3>::from_polar_coordinates(
                EightEight::EIGHTEIGHT,
                3.0 * PI / 4.0,
                r.skirt(),
            )) > toggle
            && r.distance(&Hyperbolic::<3>::from_polar_coordinates(
                EightEight::EIGHTEIGHT,
                PI,
                r.skirt(),
            )) > toggle;
        let near_side_4 = octant == 4.0
            && distance_to_bdy < self.maximum_interaction_range
            && r.distance(&Hyperbolic::<3>::from_polar_coordinates(
                EightEight::EIGHTEIGHT,
                PI,
                r.skirt(),
            )) > toggle
            && r.distance(&Hyperbolic::<3>::from_polar_coordinates(
                EightEight::EIGHTEIGHT,
                5.0 * PI / 4.0,
                r.skirt(),
            )) > toggle;
        let near_side_5 = octant == 5.0
            && distance_to_bdy < self.maximum_interaction_range
            && r.distance(&Hyperbolic::<3>::from_polar_coordinates(
                EightEight::EIGHTEIGHT,
                5.0 * PI / 4.0,
                r.skirt(),
            )) > toggle
            && r.distance(&Hyperbolic::<3>::from_polar_coordinates(
                EightEight::EIGHTEIGHT,
                6.0 * PI / 4.0,
                r.skirt(),
            )) > toggle;
        let near_side_6 = octant == 6.0
            && distance_to_bdy < self.maximum_interaction_range
            && r.distance(&Hyperbolic::<3>::from_polar_coordinates(
                EightEight::EIGHTEIGHT,
                6.0 * PI / 4.0,
                r.skirt(),
            )) > toggle
            && r.distance(&Hyperbolic::<3>::from_polar_coordinates(
                EightEight::EIGHTEIGHT,
                7.0 * PI / 4.0,
                r.skirt(),
            )) > toggle;
        let near_side_7 = octant == 7.0
            && distance_to_bdy < self.maximum_interaction_range
            && r.distance(&Hyperbolic::<3>::from_polar_coordinates(
                EightEight::EIGHTEIGHT,
                7.0 * PI / 4.0,
                r.skirt(),
            )) > toggle
            && r.distance(&Hyperbolic::<3>::from_polar_coordinates(
                EightEight::EIGHTEIGHT,
                0.0,
                r.skirt(),
            )) > toggle;

        if near_vertex_0 {
            for i in 1..8 {
                result.push(new_site_vertex(0.0, i, r));
            }
        } else if near_vertex_1 {
            for i in 1..8 {
                result.push(new_site_vertex(1.0, i, r));
            }
        } else if near_vertex_2 {
            for i in 1..8 {
                result.push(new_site_vertex(2.0, i, r));
            }
        } else if near_vertex_3 {
            for i in 1..8 {
                result.push(new_site_vertex(3.0, i, r));
            }
        } else if near_vertex_4 {
            for i in 1..8 {
                result.push(new_site_vertex(4.0, i, r));
            }
        } else if near_vertex_5 {
            for i in 1..8 {
                result.push(new_site_vertex(5.0, i, r));
            }
        } else if near_vertex_6 {
            for i in 1..8 {
                result.push(new_site_vertex(6.0, i, r));
            }
        } else if near_vertex_7 {
            for i in 1..8 {
                result.push(new_site_vertex(7.0, i, r));
            }
        } else if near_side_0 {
            result.push(new_site_edge(0.0, r));
        } else if near_side_1 {
            result.push(new_site_edge(1.0, r));
        } else if near_side_2 {
            result.push(new_site_edge(2.0, r));
        } else if near_side_3 {
            result.push(new_site_edge(3.0, r));
        } else if near_side_4 {
            result.push(new_site_edge(4.0, r));
        } else if near_side_5 {
            result.push(new_site_edge(5.0, r));
        } else if near_side_6 {
            result.push(new_site_edge(6.0, r));
        } else if near_side_7 {
            result.push(new_site_edge(7.0, r));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::Point;
    use approxim::assert_relative_eq;
    use hoomd_manifold::{Hyperbolic, HyperbolicDisk};
    use rand::{Rng, SeedableRng, distr::Distribution, rngs::StdRng};
    use std::f64::consts::PI;

    #[test]
    fn doesnt_wrap_if_inside() {
        let r = 1.528_570_919_480_998;
        let mut rng = StdRng::seed_from_u64(239);
        let disk = HyperbolicDisk {
            disk_radius: r.try_into().expect("hard-coded positive number"),
            point: Hyperbolic::from_minkowski_coordinates(Minkowski::from([0.0, 0.0, 1.0]), 1.0),
        };
        let random_point: Hyperbolic<3> = disk.sample(&mut rng);
        let random_point = Point::new(random_point);

        let periodic =
            Periodic::new(0.5, EightEight { skirt: 1.0_f64 }).expect("hard-coded positive number");
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
        let boost = 1.6;
        let offset = PI / 8.0;
        let point = Hyperbolic::<3>::from_polar_coordinates(boost, side * PI / 4.0 + offset, 1.0);
        let point = Point::new(point);
        let periodic =
            Periodic::new(0.5, EightEight { skirt: 1.0_f64 }).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(point).expect("hard-coded");

        let wrapped_side = (side + 4.0).rem_euclid(8.0);
        let octant = (((wrapped_point.position.coordinates()[1]
            .atan2(wrapped_point.position.coordinates()[0]))
            / (PI / 4.0))
            .floor())
        .rem_euclid(8.0);

        // Check that point is wraped to correct octant
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
            1.0,
        );
        assert_relative_eq!(ans, wrapped_point.position, epsilon = 1e-12);
    }

    #[test]
    fn wraps_vertex() {
        let boost = 2.45;
        let point = Hyperbolic::<3>::from_polar_coordinates(boost, PI / 4.0, 1.0);
        let point = Point::new(point);
        let periodic =
            Periodic::new(0.5, EightEight { skirt: 1.0_f64 }).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(point).expect("hard-coded");

        let distance_from_vertex = -EightEight::distance_to_boundary(&point.position);
        let v = 2.0 * 2.448_452_447_678_076 - boost;
        let ans = Hyperbolic::<3>::from_polar_coordinates(v, 5.0 * PI / 4.0, 1.0);
        assert_relative_eq!(
            EightEight::distance_to_boundary(&ans),
            distance_from_vertex,
            epsilon = 1e-12
        );

        assert_relative_eq!(ans, wrapped_point.position, epsilon = 1e-12);
    }
    #[test]
    fn wraps_vertex_non_center() {
        let offset_boost: f64 = 0.3;
        let v: f64 = 2.448_452_447_678_076;
        let point = Hyperbolic::from_minkowski_coordinates(
            [
                (v.sinh()) * (offset_boost.cosh()),
                -offset_boost.sinh(),
                (v.cosh()) * (offset_boost.cosh()),
            ]
            .into(),
            1.0,
        );
        let point = Point::new(point);
        let periodic =
            Periodic::new(0.5, EightEight { skirt: 1.0_f64 }).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(point).expect("hard-coded");

        let new_boost = 2.448_452_447_678_076 - offset_boost;
        let ans = Hyperbolic::<3>::from_polar_coordinates(new_boost, 3.0 * PI / 2.0, 1.0);

        assert_relative_eq!(ans, wrapped_point.position, epsilon = 1e-12);
    }
    #[test]
    fn ghost_near_side() {
        let mut rng = StdRng::seed_from_u64(1);
        let side = f64::from(rng.random_range(0..8));
        let offset = 0.1;
        let boost = 1.528_570_919_480_998 - offset;
        let point = Hyperbolic::<3>::from_polar_coordinates(boost, PI / 8.0 + side * PI / 4.0, 1.0);
        let point = Point::new(point);

        let periodic =
            Periodic::new(0.5, EightEight { skirt: 1.0_f64 }).expect("hard-coded positive number");

        let ghost_array = periodic.generate_ghosts(&point);
        let ghost = ghost_array[0];

        let ans = Hyperbolic::<3>::from_polar_coordinates(
            1.528_570_919_480_998 + offset,
            (side + 4.0).rem_euclid(8.0) * PI / 4.0 + PI / 8.0,
            1.0,
        );

        assert_relative_eq!(ans, ghost.position, epsilon = 1e-12);
    }

    #[test]
    fn ghost_near_vertex() {
        let offset_boost = 0.1;
        let v: f64 = 2.448_452_447_678_076;
        let point = Hyperbolic::<3>::from_polar_coordinates(v - offset_boost, 0.0, 1.0);
        let point = Point::new(point);
        let periodic =
            Periodic::new(0.5, EightEight { skirt: 1.0_f64 }).expect("hard-coded positive number");

        let ghost_array = periodic.generate_ghosts(&point);
        let ghost_3 = ghost_array[3];

        let ans_3 = Hyperbolic::<3>::from_polar_coordinates(v + offset_boost, PI, 1.0);
        assert_relative_eq!(ans_3, ghost_3.position, epsilon = 1e-12);

        let ghost_5 = ghost_array[5];

        let ans_5 = Hyperbolic::from_minkowski_coordinates(
            [
                offset_boost.sinh(),
                -(v.sinh()) * (offset_boost.cosh()),
                (v.cosh()) * (offset_boost.cosh()),
            ]
            .into(),
            1.0,
        );
        assert_relative_eq!(ans_5, ghost_5.position, epsilon = 1e-12);
    }
}

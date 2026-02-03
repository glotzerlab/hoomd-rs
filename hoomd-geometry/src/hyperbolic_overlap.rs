// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Overlap Check for Hyperbolic Surfaces

use crate::BoundingSphereRadius;
use hoomd_geometry::shape::HyperbolicConvexPolytope;
use hoomd_manifold::{Hyperbolic, Minkowski};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Angle, Metric};
use robust::{Coord, orient2d};
use std::f64::consts::PI;

/// Test whether two shapes overlap.
pub trait SeparatingPlanes<S, M, R> {
    /// Test whether the set of points in one shape intersects with the set of another.
    /// Method works by iterating through adjacent vertices of `self`, constructing the
    /// hyperplane which pass through the two vertices, and checking if all the
    /// points of `other` are on the opposite side. If a separating plane exists, then
    /// the two shapes do not overlap and the method returns `false`.
    fn intersects_at(&self, x_i: &M, r_i: &R, x_j: &M, r_j: &R) -> bool;
}

impl SeparatingPlanes<HyperbolicConvexPolytope<3>, Hyperbolic<3>, Angle> for HyperbolicConvexPolytope<3> {
    #[inline]
    #[allow(clippy::too_many_lines, reason = "complicated function")]
    fn intersects_at(
        &self,
        x_i: &Hyperbolic<3>,
        r_i: &Angle,
        x_j: &Hyperbolic<3>,
        r_j: &Angle,
    ) -> bool {
        let d = x_i.distance(x_j);
        if d > 2.0 * self.bounding_radius {
            return false;
        }
        let mut result = true;
        let mut v_count = 0_usize;
        let n = self.vertices.len();
        while result && (v_count < 2 * n) {
            if v_count < n {
                let v_num = (v_count) % n;
                let v_next = (v_num + 1) % n;
                let v_next_next = (v_num + 2) % n;
                // translate all vertices
                // need to do this for every other vertex
                let v_1 = Self::vertex_to_system_frame(&self.vertices[v_num], r_i, x_i);
                let v_2 = Self::vertex_to_system_frame(&self.vertices[v_next], r_i, x_i);
                let v_3 = Self::vertex_to_system_frame(&self.vertices[v_next_next], r_i, x_i);
                let other = self
                    .vertices
                    .iter()
                    .map(|vertex| Self::vertex_to_system_frame(vertex, r_j, x_j))
                    .collect::<Vec<Hyperbolic<3>>>();
                let self_translated = Self::to_vertex_frame_oriented(
                    x_i,
                    r_i,
                    v_next,
                    self.bounding_radius,
                    &[v_1, v_2, v_3],
                    n,
                );
                let other_translated = Self::to_vertex_frame_oriented(
                    x_i,
                    r_i,
                    v_next,
                    self.bounding_radius,
                    &other,
                    n,
                );
                // convert to poincare coordinates to perform orientation checks.
                let self_coord = self_translated
                    .iter()
                    .map(|pt: &Hyperbolic<3>| {
                        let poincare = pt.to_poincare();
                        Coord {
                            x: poincare[0],
                            y: poincare[1],
                        }
                    })
                    .collect::<Vec<Coord<f64>>>();
                let other_coord = other_translated
                    .iter()
                    .map(|pt: &Hyperbolic<3>| {
                        let poincare = pt.to_poincare();
                        Coord {
                            x: poincare[0],
                            y: poincare[1],
                        }
                    })
                    .collect::<Vec<Coord<f64>>>();
                // then do edge check
                let mut overlap = false;
                let mut counter = 0_usize;
                while !overlap && (counter < other.len()) {
                    if orient2d(self_coord[0], self_coord[1], other_coord[counter]) >= 0.0 {
                        // break out of loop once one of the vertices is on the wrong side of the line
                        overlap = true;
                        break;
                    }
                    counter += 1;
                }
                counter = 0_usize;
                // if overlap is false, then no need to check next edge FIX THIS!
                if overlap {
                    overlap = false;
                    while !overlap && (counter < other.len()) {
                        if orient2d(self_coord[1], self_coord[2], other_coord[counter]) >= 0.0 {
                            // break out of loop once one of the vertices is on the wrong side of the line
                            overlap = true;
                            break;
                        }
                        counter += 1;
                    }
                }
                result = overlap;
                v_count += 2;
                // need to add a check for when the counter moves onto the next square
            } else {
                let v_num = (v_count) % n;
                let v_next = (v_num + 1) % n;
                let v_next_next = (v_num + 2) % n;
                // translate all vertices
                let v_1 = Self::vertex_to_system_frame(&self.vertices[v_num], r_j, x_j);
                let v_2 = Self::vertex_to_system_frame(&self.vertices[v_next], r_j, x_j);
                let v_3 = Self::vertex_to_system_frame(&self.vertices[v_next_next], r_j, x_j);
                let other = self
                    .vertices
                    .iter()
                    .map(|vertex| Self::vertex_to_system_frame(vertex, r_i, x_i))
                    .collect::<Vec<Hyperbolic<3>>>();
                let self_translated = Self::to_vertex_frame_oriented(
                    x_j,
                    r_j,
                    v_next,
                    self.bounding_radius,
                    &[v_1, v_2, v_3],
                    n,
                );
                let other_translated = Self::to_vertex_frame_oriented(
                    x_j,
                    r_j,
                    v_next,
                    self.bounding_radius,
                    &other,
                    n,
                );
                // convert to poincare coordinates to perform orientation checks.
                let self_coord = self_translated
                    .iter()
                    .map(|pt: &Hyperbolic<3>| {
                        let poincare = pt.to_poincare();
                        Coord {
                            x: poincare[0],
                            y: poincare[1],
                        }
                    })
                    .collect::<Vec<Coord<f64>>>();
                let other_coord = other_translated
                    .iter()
                    .map(|pt: &Hyperbolic<3>| {
                        let poincare = pt.to_poincare();
                        Coord {
                            x: poincare[0],
                            y: poincare[1],
                        }
                    })
                    .collect::<Vec<Coord<f64>>>();
                // then do edge check
                let mut overlap = false;
                let mut counter = 0_usize;
                while !overlap && (counter < other.len()) {
                    if orient2d(self_coord[0], self_coord[1], other_coord[counter]) >= 0.0 {
                        // break out of loop once one of the vertices is on the wrong side of the line
                        overlap = true;
                        break;
                    }
                    counter += 1;
                }
                counter = 0_usize;
                if overlap {
                    overlap = false;
                    while !overlap && (counter < other.len()) {
                        if orient2d(self_coord[1], self_coord[2], other_coord[counter]) >= 0.0 {
                            // break out of loop once one of the vertices is on the wrong side of the line
                            overlap = true;
                        }
                        counter += 1;
                    }
                }
                result = overlap;
                v_count += 2;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::{HyperbolicConvexPolygon, HyperbolicConvexPolytope};
    use approxim::assert_relative_eq;
    use hoomd_manifold::{Hyperbolic, Minkowski};
    use hoomd_vector::Angle;

    #[test]
    fn no_square_overlap() {
        let square = HyperbolicConvexPolytope::<3>::regular(4, 0.5, 1.0);
        let boost: f64 = 3.0;
        let rotation: f64 = 2.3;
        let orientation: f64 = 0.4;
        let x_j = Hyperbolic::<3>::from_polar_coordinates(boost, rotation, 1.0);
        assert!(!SeparatingPlanes::intersects_at(
            &square,
            &Hyperbolic::<3>::default(),
            &Angle::default(),
            &x_j,
            &Angle::from(orientation)
        ));
    }
    #[test]
    fn square_overlap() {
        let square = HyperbolicConvexPolytope::<3>::regular(4, 0.5, 1.0);
        let boost: f64 = 0.49;
        let rotation: f64 = 2.3;
        let orientation: f64 = 0.4;
        let x_j = Hyperbolic::<3>::from_polar_coordinates(boost, rotation, 1.0);
        assert!(square.intersects_at(
            &Hyperbolic::<3>::default(),
            &Angle::default(),
            &x_j,
            &Angle::from(orientation)
        ));
    }
    #[test]
    fn overlap_translation_check() {
        let r_0 = 0.5;
        let square = HyperbolicConvexPolytope::<3>::regular(4, r_0, 1.0);
        let com = Hyperbolic::<3>::from_polar_coordinates(1.0, 0.0, 1.0);
        let distance = 2.0;
        let num_spaces: usize = 10;
        let num_trials: usize = 15;
        let spacing = 1.321_592_891_727_355;
        let trials = (0..num_trials)
            .map(|n| -(n as f64) * spacing / (num_spaces as f64))
            .collect::<Vec<f64>>();

        let nudged_centers = trials
            .iter()
            .map(|inch| {
                let original_center = [
                    (1.0_f64 + distance).sinh(),
                    0.0_f64,
                    (1.0_f64 + distance).cosh(),
                ];
                let translated = Minkowski::from([
                    original_center[0] * (inch.cosh()) + original_center[2] * (inch.sinh()),
                    original_center[1],
                    original_center[0] * (inch.sinh()) + original_center[2] * (inch.cosh()),
                ]);
                Hyperbolic::from_minkowski_coordinates(translated, 1.0)
            })
            .collect::<Vec<Hyperbolic<3>>>();
        // Check over overlaps
        for translated in nudged_centers.iter().take(num_spaces) {
            assert!(!square.intersects_at(
                &com,
                &Angle::from(PI / 4.0),
                translated,
                &Angle::from(PI / 4.0)
            ));
        }
        for translated in nudged_centers.iter().take(num_trials).skip(num_spaces) {
            assert!(square.intersects_at(
                &com,
                &Angle::from(PI / 4.0),
                translated,
                &Angle::from(PI / 4.0)
            ));
        }
    }
    #[test]
    fn overlap_rotation_check() {
        let r_0 = 0.5;
        let boost: f64 = 0.339_203_554_136_322;
        let distance: f64 = 0.45;
        let square = HyperbolicConvexPolytope::<3>::regular(4, r_0, 1.0);
        let num_spaces: usize = 10;
        let num_trials: usize = 15;
        let spacing = 0.365_106_058_818_114;
        let trials = (0..num_trials)
            .map(|n| (n as f64) * spacing / (num_spaces as f64))
            .collect::<Vec<f64>>();
        let center_1 = Hyperbolic::<3>::from_polar_coordinates(-boost, 0.0, 1.0);
        let center_2 = Hyperbolic::<3>::from_polar_coordinates(distance, 0.0, 1.0);
        // Check over overlaps
        for ep in trials.iter().take(num_spaces) {
            assert!(!square.intersects_at(
                &center_1,
                &Angle::from(PI / 4.0),
                &center_2,
                &Angle::from(ep + PI / 4.0)
            ));
        }
        for ep in trials.iter().take(num_trials).skip(num_spaces) {
            assert!(square.intersects_at(
                &center_1,
                &Angle::from(PI / 4.0),
                &center_2,
                &Angle::from(ep + PI / 4.0)
            ));
        }
    }
}

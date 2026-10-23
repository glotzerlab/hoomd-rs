// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Overlap Check for Spherical Surfaces

use crate::{BoundingSphereRadius, IntersectsAtGlobal};
use hoomd_manifold::Spherical;
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Angle, Cartesian, Metric};
use robust::{Coord, orient2d};
use std::f64::consts::PI;

/// A faceted spherical solid defined by the convex hull of its vertices.
///
/// TODO: examples
#[derive(Clone, Debug, PartialEq)]
pub struct SphericalConvexPolytope<const N: usize> {
    /// The vertices of the shape
    vertices: Vec<Spherical<N>>,
    /// The radius of the bounding sphere of the shape in the Spherical metric.
    bounding_radius: f64,
}

impl<const N: usize> SphericalConvexPolytope<N> {
    /// Get the vertices of the shape.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> &[Spherical<N>] {
        &self.vertices
    }
}

impl<const N: usize> BoundingSphereRadius for SphericalConvexPolytope<N> {
    #[inline]
    fn bounding_sphere_radius(&self) -> PositiveReal {
        self.bounding_radius
            .try_into()
            .expect("hard coded constant should be positive")
    }
}

/// A two-dimensional spherical convex polytope.
/// ```rust
/// use hoomd_geometry::shape::SphericalConvexPolygon;
///
/// # fn main() -> Result<(), hoomd_geometry::Error> {
/// let spherical_pentagon = SphericalConvexPolygon::regular(5, 1.0_f64);
/// # Ok(())
/// # }
/// ```
pub type SphericalConvexPolygon = SphericalConvexPolytope<3>;

impl SphericalConvexPolytope<3> {
    /// Create a regular *n*-gon with *n* vertices and a given circumradius on
    /// the surface of a sphere.
    #[inline]
    #[must_use]
    pub fn regular(n: usize, circumradius: f64) -> SphericalConvexPolytope<3> {
        SphericalConvexPolytope {
            vertices: (0..n)
                .map(|x| {
                    let theta = 2.0 * PI * (x as f64) / (n as f64);
                    Spherical::<3>::from_polar_coordinates(circumradius, theta)
                })
                .collect::<Vec<_>>(),
            bounding_radius: circumradius,
        }
    }
    /// Transform a vertex of a spherical polygon into the system frame.
    #[inline]
    pub fn vertex_to_system_frame(
        vertex: &Spherical<3>,
        body_orientation: Angle,
        body_position: &Spherical<3>,
    ) -> Spherical<3> {
        let body_pos = body_position.coordinates();
        let body_pos_phi = body_pos[1].atan2(body_pos[0]);
        let body_pos_theta = (body_pos[2]).acos();
        let body_angle = body_orientation.theta;
        let site_pos = vertex.coordinates();
        let (theta_cos, theta_sin) = (body_pos_theta.cos(), body_pos_theta.sin());
        let (phi_cos, phi_sin, diff_cos, diff_sin) = (
            body_pos_phi.cos(),
            body_pos_phi.sin(),
            (body_angle - body_pos_phi).cos(),
            (body_angle - body_pos_phi).sin(),
        );
        let transformed_point = Cartesian::from([
            site_pos[0] * (theta_cos * phi_cos * diff_cos - phi_sin * diff_sin)
                - site_pos[1] * (theta_cos * phi_cos * diff_sin + phi_sin * diff_cos)
                + site_pos[2] * theta_sin * phi_cos,
            site_pos[0] * (theta_cos * phi_sin * diff_cos + phi_cos * diff_sin)
                + site_pos[1] * (phi_cos * diff_cos - theta_cos * phi_sin * diff_sin)
                + site_pos[2] * theta_sin * phi_sin,
            -site_pos[0] * theta_sin * diff_cos
                + site_pos[1] * theta_sin * diff_sin
                + site_pos[2] * theta_cos,
        ]);
        Spherical::from_cartesian_coordinates(transformed_point)
    }
    #[inline]
    /// Transform `points` to the frame where the `vertex_num`-th vertex of an
    /// oriented spherical polygon with with bounding radius `bounding_radius`
    /// is at the origin.
    fn to_vertex_frame_oriented(
        body_position: &Spherical<3>,
        body_orientation: Angle,
        vertex_num: usize,
        bounding_radius: f64,
        points: &[Spherical<3>],
        num_of_sides: usize,
    ) -> Vec<Spherical<3>> {
        let phi = body_position.coordinates()[1].atan2(body_position.coordinates()[0]);
        let theta = (body_position.coordinates()[2]).acos();
        let alpha =
            body_orientation.theta - phi + 2.0 * PI * (vertex_num as f64) / (num_of_sides as f64);
        let r = bounding_radius;
        let vertex_translate = |point: &Spherical<3>| -> Spherical<3> {
            let pt = point.point().coordinates;
            let (theta_sin, theta_cos, r_sin, r_cos) = (theta.sin(), theta.cos(), r.sin(), r.cos());
            let (alpha_sin, alpha_cos, phi_sin, phi_cos) =
                (alpha.sin(), alpha.cos(), phi.sin(), phi.cos());
            let translated = Cartesian::from([
                pt[0]
                    * (r_cos * alpha_cos * theta_cos * phi_cos
                        - r_cos * alpha_sin * phi_sin
                        - r_sin * theta_sin * phi_cos)
                    + pt[1]
                        * (r_cos * alpha_cos * theta_cos * phi_sin + r_cos * alpha_sin * phi_cos
                            - r_sin * theta_sin * phi_sin)
                    + pt[2] * (-r_cos * alpha_cos * theta_sin - r_sin * theta_cos),
                pt[0] * (-alpha_sin * theta_cos * phi_cos - alpha_cos * phi_sin)
                    + pt[1] * (-alpha_sin * theta_cos * phi_sin + alpha_cos * phi_cos)
                    + pt[2] * alpha_sin * theta_sin,
                pt[0]
                    * (r_sin * alpha_cos * theta_cos * phi_cos - r_sin * alpha_sin * phi_sin
                        + r_cos * theta_sin * phi_cos)
                    + pt[1]
                        * (r_sin * alpha_cos * theta_cos * phi_sin
                            + r_sin * alpha_sin * phi_cos
                            + r_cos * theta_sin * phi_sin)
                    + pt[2] * (-r_sin * alpha_cos * theta_sin + r_cos * theta_cos),
            ]);
            Spherical::from_cartesian_coordinates(translated)
        };
        points.iter().map(vertex_translate).collect::<Vec<_>>()
    }
}

impl IntersectsAtGlobal<SphericalConvexPolytope<3>, Spherical<3>, Angle>
    for SphericalConvexPolytope<3>
{
    #[inline]
    #[allow(clippy::too_many_lines, reason = "complicated function")]
    fn intersects_at_global(
        &self,
        other: &SphericalConvexPolytope<3>,
        r_self: &Spherical<3>,
        o_self: &Angle,
        r_other: &Spherical<3>,
        o_other: &Angle,
    ) -> bool {
        let d = r_self.distance(r_other);
        if d > self.bounding_radius + other.bounding_radius {
            return false;
        }
        let mut result = true;
        let mut v_count = 0_usize;
        let n_self = self.vertices.len();
        let n_other = other.vertices.len();

        // for stability: first generate and translate all vertices
        let self_vertices = self
            .vertices
            .iter()
            .map(|v| Self::vertex_to_system_frame(v, *o_self, r_self))
            .collect::<Vec<Spherical<3>>>();
        let other_vertices = other
            .vertices
            .iter()
            .map(|v| Self::vertex_to_system_frame(v, *o_other, r_other))
            .collect::<Vec<Spherical<3>>>();
        let self_translated = Self::to_vertex_frame_oriented(
            r_self,
            *o_self,
            0_usize,
            self.bounding_radius,
            &self_vertices,
            n_self,
        );
        let other_translated = Self::to_vertex_frame_oriented(
            r_self,
            *o_self,
            0_usize,
            self.bounding_radius,
            &other_vertices,
            n_self,
        );
        // gnomonic projection through north pole
        let self_coord = self_translated
            .iter()
            .map(|pt| {
                let coords = pt.point().coordinates;
                Coord {
                    x: coords[0] / coords[2],
                    y: coords[1] / coords[2],
                }
            })
            .collect::<Vec<Coord<f64>>>();
        let other_coord = other_translated
            .iter()
            .map(|pt| {
                let coords = pt.point().coordinates;
                Coord {
                    x: coords[0] / coords[2],
                    y: coords[1] / coords[2],
                }
            })
            .collect::<Vec<Coord<f64>>>();
        while result && (v_count < n_self + n_other) {
            if v_count < n_self {
                let v_next = (v_count + 1) % n_self;
                // check edges on self shape
                let mut overlap = false;
                let mut counter = 0_usize;
                while !overlap && (counter < other_vertices.len()) {
                    if orient2d(
                        self_coord[v_count],
                        self_coord[v_next],
                        other_coord[counter],
                    ) >= 0.0
                    {
                        // break out of loop once one of the vertices is on the wrong side of the line
                        overlap = true;
                        break;
                    }
                    counter += 1;
                }
                result = overlap;
                v_count += 1;
            } else {
                let v_next = (v_count - n_self + 1) % n_other;
                // check edges on other shape
                let mut overlap = false;
                let mut counter = 0_usize;
                while !overlap && (counter < other_vertices.len()) {
                    if orient2d(
                        other_coord[v_count - n_self],
                        other_coord[v_next],
                        self_coord[counter],
                    ) >= 0.0
                    {
                        // break out of loop once one of the vertices is on the wrong side of the line
                        overlap = true;
                        break;
                    }
                    counter += 1;
                }
                result = overlap;
                v_count += 1;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::shape::SphericalConvexPolygon;

    use super::*;
    use approxim::assert_relative_eq;
    use hoomd_manifold::Spherical;
    use hoomd_vector::Angle;
    use std::f64::consts::PI;

    #[test]
    fn vertex_to_system() {
        let cube_edgelength = 2.0 / f64::sqrt(3.0);
        let square = SphericalConvexPolygon::regular(4, f64::acos(1.0 / f64::sqrt(3.0)));
        let v1 = square.vertices()[0];

        let face_1_center = Spherical::<3>::from_cartesian_coordinates([0.0, 0.0, 1.0].into());
        let v1_sys = SphericalConvexPolytope::<3>::vertex_to_system_frame(
            &v1,
            Angle::from(0.0),
            &face_1_center,
        );
        let ans_1 = [cube_edgelength / f64::sqrt(2.0), 0.0, cube_edgelength / 2.0];
        assert_relative_eq!(ans_1[0], v1_sys.coordinates()[0]);
        assert_relative_eq!(ans_1[1], v1_sys.coordinates()[1]);
        assert_relative_eq!(ans_1[2], v1_sys.coordinates()[2]);

        let face_2_center = Spherical::<3>::from_cartesian_coordinates(
            [1.0 / f64::sqrt(2.0), -1.0 / f64::sqrt(2.0), 0.0].into(),
        );
        let v2_sys = SphericalConvexPolytope::<3>::vertex_to_system_frame(
            &v1,
            Angle::from(0.0),
            &face_2_center,
        );
        let ans_2 = [
            cube_edgelength / f64::sqrt(2.0),
            0.0,
            -cube_edgelength / 2.0,
        ];
        assert_relative_eq!(ans_2[0], v2_sys.coordinates()[0]);
        assert_relative_eq!(ans_2[1], v2_sys.coordinates()[1]);
        assert_relative_eq!(ans_2[2], v2_sys.coordinates()[2]);

        // rotated cube
        let v1_rot_sys = SphericalConvexPolygon::vertex_to_system_frame(
            &v1,
            Angle::from(PI / 4.0),
            &face_1_center,
        );
        let ans_1_rot = [
            cube_edgelength / 2.0,
            cube_edgelength / 2.0,
            cube_edgelength / 2.0,
        ];
        assert_relative_eq!(ans_1_rot[0], v1_rot_sys.coordinates()[0]);
        assert_relative_eq!(ans_1_rot[1], v1_rot_sys.coordinates()[1]);
        assert_relative_eq!(ans_1_rot[2], v1_rot_sys.coordinates()[2]);

        let face_2_center_rot = Spherical::<3>::from_cartesian_coordinates([1.0, 0.0, 0.0].into());
        let v2_rot_sys = SphericalConvexPolygon::vertex_to_system_frame(
            &v1,
            Angle::from(PI / 4.0),
            &face_2_center_rot,
        );
        let ans_2_rot = [
            cube_edgelength / 2.0,
            cube_edgelength / 2.0,
            -cube_edgelength / 2.0,
        ];
        assert_relative_eq!(ans_2_rot[0], v2_rot_sys.coordinates()[0]);
        assert_relative_eq!(ans_2_rot[1], v2_rot_sys.coordinates()[1]);
        assert_relative_eq!(ans_2_rot[2], v2_rot_sys.coordinates()[2]);
    }

    #[test]
    fn to_vertex_frame_oriented() {
        let r_0 = f64::acos(1.0 / f64::sqrt(3.0));
        let square = SphericalConvexPolygon::regular(4, r_0);
        let face_1_center = Spherical::<3>::from_cartesian_coordinates([0.0, 0.0, 1.0].into());
        let face_orientation = Angle::from(PI / 4.0);
        let vertices: Vec<Spherical<3>> = square
            .vertices
            .iter()
            .map(|pt| {
                SphericalConvexPolygon::vertex_to_system_frame(pt, face_orientation, &face_1_center)
            })
            .collect();
        let transformed_vertices = SphericalConvexPolygon::to_vertex_frame_oriented(
            &face_1_center,
            face_orientation,
            1,
            r_0,
            &vertices,
            4,
        );
        // check that query vertex is at NP
        assert_relative_eq!(transformed_vertices[1].coordinates()[0], 0.0);
        assert_relative_eq!(transformed_vertices[1].coordinates()[1], 0.0);
        assert_relative_eq!(transformed_vertices[1].coordinates()[2], 1.0);
        // other vertices are funny
        let geodesic_edgelength = 2.0 * f64::asin(1.0 / f64::sqrt(3.0));
        let geodesic_diag = 2.0 * r_0;
        let v0_ans = Spherical::<3>::from_polar_coordinates(geodesic_edgelength, -2.0 * PI / 3.0);
        let v2_ans = Spherical::<3>::from_polar_coordinates(geodesic_edgelength, 2.0 * PI / 3.0);
        let v3_ans = Spherical::<3>::from_polar_coordinates(geodesic_diag, PI);

        assert_relative_eq!(
            transformed_vertices[0].coordinates()[0],
            v0_ans.coordinates()[0],
            epsilon = 1e-12
        );
        assert_relative_eq!(
            transformed_vertices[0].coordinates()[1],
            v0_ans.coordinates()[1],
            epsilon = 1e-12
        );
        assert_relative_eq!(
            transformed_vertices[0].coordinates()[2],
            v0_ans.coordinates()[2],
            epsilon = 1e-12
        );

        assert_relative_eq!(
            transformed_vertices[2].coordinates()[0],
            v2_ans.coordinates()[0],
            epsilon = 1e-12
        );
        assert_relative_eq!(
            transformed_vertices[2].coordinates()[1],
            v2_ans.coordinates()[1],
            epsilon = 1e-12
        );
        assert_relative_eq!(
            transformed_vertices[2].coordinates()[2],
            v2_ans.coordinates()[2],
            epsilon = 1e-12
        );

        assert_relative_eq!(
            transformed_vertices[3].coordinates()[0],
            v3_ans.coordinates()[0],
            epsilon = 1e-12
        );
        assert_relative_eq!(
            transformed_vertices[3].coordinates()[1],
            v3_ans.coordinates()[1],
            epsilon = 1e-12
        );
        assert_relative_eq!(
            transformed_vertices[3].coordinates()[2],
            v3_ans.coordinates()[2],
            epsilon = 1e-12
        );
    }

    #[test]
    fn non_overlapping_squares() {
        let r_0 = 0.5;
        let edge_dist = f64::atan(f64::tan(r_0) / f64::sqrt(2.0));
        let square = SphericalConvexPolygon::regular(4, r_0);
        let position_1 = Spherical::<3>::from_cartesian_coordinates([0.0, 0.0, 1.0].into());
        let orientation_1 = Angle::from(PI / 4.0);
        let position_2 = Spherical::<3>::from_polar_coordinates(2.0 * edge_dist + 0.01, 0.0);
        let orientation_2 = Angle::from(3.0 * PI / 4.0);
        assert!(!square.intersects_at_global(
            &square,
            &position_1,
            &orientation_1,
            &position_2,
            &orientation_2
        ));
    }

    #[test]
    fn overlapping_squares() {
        let r_0 = 0.5;
        let square = SphericalConvexPolygon::regular(4, r_0);
        let position_1 = Spherical::<3>::from_cartesian_coordinates([0.0, 0.0, 1.0].into());
        let orientation_1 = Angle::from(PI / 4.0);
        let position_2 = Spherical::<3>::from_polar_coordinates(0.1, 0.01);
        let orientation_2 = Angle::from(1.23);
        assert!(square.intersects_at_global(
            &square,
            &position_1,
            &orientation_1,
            &position_2,
            &orientation_2
        ));
    }

    #[test]
    fn overlap_translation_check() {
        let r_0 = 0.5;
        let edge_dist = f64::atan(f64::tan(r_0) / f64::sqrt(2.0));
        let square = SphericalConvexPolygon::regular(4, r_0);
        let position_1 = Spherical::<3>::from_cartesian_coordinates([0.0, 0.0, 1.0].into());
        let orientation = Angle::from(PI / 4.0);

        let num_trials: usize = 15;
        let num_spaces: usize = (num_trials - 1) / 2;
        let spacing = 2.0 * edge_dist / (num_spaces as f64);
        let trials = (0..num_trials)
            .map(|n| (n as f64) * spacing)
            .collect::<Vec<f64>>();
        let nudged_centers = trials
            .iter()
            .map(|theta| Spherical::<3>::from_polar_coordinates(*theta, 0.0))
            .collect::<Vec<Spherical<3>>>();
        for translated in nudged_centers.iter().take(num_spaces + 1) {
            assert!(square.intersects_at_global(
                &square,
                &position_1,
                &orientation,
                translated,
                &orientation
            ));
        }
        for translated in nudged_centers.iter().take(num_trials).skip(num_spaces + 1) {
            assert!(!square.intersects_at_global(
                &square,
                &position_1,
                &orientation,
                translated,
                &orientation
            ));
        }
    }

    #[test]
    fn overlap_rotation_check() {
        let r_0 = 0.5;
        let edge_dist = f64::atan(f64::tan(r_0) / f64::sqrt(2.0));
        let square = SphericalConvexPolygon::regular(4, r_0);
        let position_1 = Spherical::<3>::from_polar_coordinates(edge_dist, PI);
        let orientation_1 = Angle::from(PI / 4.0);
        let position_2 = Spherical::<3>::from_polar_coordinates(edge_dist + 0.1, 0.0);

        let num_trials: usize = 15;
        let num_spaces: usize = (num_trials - 1) / 2;
        let crit_angle = 0.400_096_996_710_237_6;
        let spacing = crit_angle / (num_spaces as f64);
        let trials = (0..num_trials)
            .map(|n| (n as f64) * spacing - 0.001)
            .collect::<Vec<f64>>();
        for angle in trials.iter().take(num_spaces + 1) {
            let orientation_2 = Angle::from(*angle + PI / 4.0);
            assert!(!square.intersects_at_global(
                &square,
                &position_1,
                &orientation_1,
                &position_2,
                &orientation_2
            ));
        }
        for angle in trials.iter().take(num_trials).skip(num_spaces + 1) {
            let orientation_2 = Angle::from(*angle + PI / 4.0);
            assert!(square.intersects_at_global(
                &square,
                &position_1,
                &orientation_1,
                &position_2,
                &orientation_2
            ));
        }
    }
}

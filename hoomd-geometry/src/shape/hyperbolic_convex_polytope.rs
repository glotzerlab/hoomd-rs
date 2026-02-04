// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Overlap Check for Hyperbolic Surfaces

use crate::{BoundingSphereRadius, IntersectsAtGlobal};
use hoomd_manifold::{Hyperbolic, Minkowski};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Angle, Metric};
use robust::{Coord, orient2d};
use std::f64::consts::PI;

/// A faceted hyperbolic solid defined by the convex hull of its vertices.
///
/// # Examples
///
/// Construction and basic methods:
/// ```
/// use hoomd_geometry::{BoundingSphereRadius, shape::HyperbolicConvexPolytope};
///
/// # fn main() -> Result<(), hoomd_geometry::Error> {
/// let hyperbolic_square = HyperbolicConvexPolytope::<3>::regular(4, 0.5, 1.0);
///
/// let bounding_radius = hyperbolic_square.bounding_sphere_radius();
///
/// assert_eq!(bounding_radius.get(), 0.5);
/// # Ok(())
/// # }
/// ```
///
/// Overlap check:
/// /// ```
/// use hoomd_geometry::shape::HyperbolicConvexPolytope;
/// use hoomd_manifold::Hyperbolic;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let square = HyperbolicConvexPolytope::<3>::regular(4, 0.5, 1.0);
/// assert!(square.intersects_at_global(
///     &square,
///     &Hyperbolic::<3>::default(),
///     &Angle::default(),
///     &Hyperbolic::<3>::from_polar_coordinates(0.49, 2.3, 1.0);,
///     &Angle::from(0.4)
/// ));
///
/// assert!(!square.intersects_at_global(
///     &square,
///     &Hyperbolic::<3>::default(),
///     &Angle::default(),
///     &Hyperbolic::<3>::from_polar_coordinates(3.0, 2.3, 1.0),
///     &Angle::from(orientation)
/// ));
/// #Ok(())
/// #}
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct HyperbolicConvexPolytope<const N: usize> {
    /// The vertices of the shape
    vertices: Vec<Hyperbolic<N>>,
    /// The radius of the bounding sphere of the shape in the Hyperbolic metric.
    bounding_radius: f64,
}

impl<const N: usize> HyperbolicConvexPolytope<N> {
    /// Get the vertices of the shape.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> &[Hyperbolic<N>] {
        &self.vertices
    }
    /// Compute the bounding radius from a set of vertices.
    #[inline]
    #[must_use]
    #[allow(dead_code, reason = "function useful to user")]
    fn bounding_radius(vertices: &[Hyperbolic<N>]) -> f64 {
        vertices
            .iter()
            .map(hoomd_manifold::Hyperbolic::distance_from_cusp)
            .fold(0.0, f64::max)
    }
}

impl<const N: usize> BoundingSphereRadius for HyperbolicConvexPolytope<N> {
    #[inline]
    fn bounding_sphere_radius(&self) -> PositiveReal {
        self.bounding_radius
            .try_into()
            .expect("hard coded constant should be positive")
    }
}

/// A two-dimensional hyperbolic convex polytope.
/// ```rust
/// use hoomd_geometry::shape::HyperbolicConvexPolygon;
///
/// # fn main() -> Result<(), hoomd_geometry::Error> {
/// let hyperbolic_pentagon = HyperbolicConvexPolygon::regular(5, 1.0_f64, 1.0_f64);
/// # Ok(())
/// # }
/// ```
pub type HyperbolicConvexPolygon = HyperbolicConvexPolytope<3>;

impl HyperbolicConvexPolytope<3> {
    /// Create a regular *n*-gon with *n* vertices and a given circumradius in
    /// hyperbolic space.
    #[inline]
    #[must_use]
    pub fn regular(n: usize, circumradius: f64, skirt: f64) -> HyperbolicConvexPolytope<3> {
        HyperbolicConvexPolytope {
            vertices: (0..n)
                .map(|x| {
                    let theta = 2.0 * PI * (x as f64) / (n as f64);
                    Hyperbolic::<3>::from_polar_coordinates(circumradius, theta, skirt)
                })
                .collect::<Vec<_>>(),
            bounding_radius: circumradius,
        }
    }
    /// Calculate the distance between the center of a `HyperbolicConvexPolytope<3>`
    /// centered at the origin and the edge at angle phi. The calculation works by
    /// computing the intersection of the the hyperboloid with a plane passing
    /// through the origin and the two adjacent vertices of the polygon.
    ///
    /// # Example
    /// ```
    /// use hoomd_geometry::shape::HyperbolicConvexPolytope;
    /// use std::f64::consts::PI;
    /// use approxim::assert_relative_eq;
    ///
    /// let bounding_radius = 0.5;
    /// let hyperbolic_square = HyperbolicConvexPolytope::<3>::regular(4, bounding_radius, 1.0);
    ///
    /// //calculation using hyperbolic trigonometry
    /// let pi_fourths_distance = (((PI/4.0).cos())*(bounding_radius.tanh())).atanh();
    ///
    ///
    /// let edge_distance = hyperbolic_square.edge_distance(PI/4.0);
    ///
    /// assert_relative_eq!(pi_fourths_distance, edge_distance);
    ///
    /// ```
    #[inline]
    #[must_use]
    pub fn edge_distance(&self, phi: f64) -> f64 {
        let n = self.vertices.len() as f64;
        let phi_mod = phi.rem_euclid(2.0 * PI / n) - PI / n;
        let eta_tanh = self.bounding_radius.tanh();
        let arg = (eta_tanh * ((2.0 * PI / n).sin()))
            / ((PI / n - phi_mod).sin() + (PI / n + phi_mod).sin());
        arg.atanh()
    }
    #[inline]
    fn to_vertex_frame_oriented(
        body_position: &Hyperbolic<3>,
        body_orientation: &Angle,
        vertex_num: usize,
        bounding_radius: f64,
        points: &[Hyperbolic<3>],
        num_of_sides: usize,
    ) -> Vec<Hyperbolic<3>> {
        let theta = body_position.coordinates()[1].atan2(body_position.coordinates()[0]);
        let eta = (body_position.coordinates()[2]).acosh();
        let tau_over_two = -eta / 2.0;
        let poincare = body_position.to_poincare();
        let body_angle_body = -2.0
            * (((tau_over_two.sinh())
                * ((theta.cos()) * poincare[1] - (theta.sin()) * poincare[0]))
                .atan2(
                    (tau_over_two.cosh())
                        + (tau_over_two.sinh())
                            * ((theta.cos()) * poincare[0] + (theta.sin()) * poincare[1]),
                ))
            + body_orientation.theta;

        let alpha =
            body_angle_body - theta + 2.0 * (vertex_num as f64) * PI / (num_of_sides as f64);
        let r = bounding_radius;
        let vertex_translate = |point: &Hyperbolic<3>| -> Hyperbolic<3> {
            let pt = point.point().coordinates;
            let (eta_sinh, eta_cosh, r_sinh, r_cosh) = (eta.sinh(), eta.cosh(), r.sinh(), r.cosh());
            let (alpha_cos, alpha_sin, theta_cos, theta_sin) =
                (alpha.cos(), alpha.sin(), theta.cos(), theta.sin());
            let translated = Minkowski::from([
                (eta_cosh * r_cosh * alpha_cos * theta_cos - r_cosh * alpha_sin * theta_sin
                    + eta_sinh * r_sinh * theta_cos)
                    * pt[0]
                    + (eta_cosh * r_cosh * alpha_cos * theta_sin
                        + r_cosh * alpha_sin * theta_cos
                        + eta_sinh * r_sinh * theta_sin)
                        * pt[1]
                    - (eta_sinh * r_cosh * alpha_cos + eta_cosh * r_sinh) * pt[2],
                -(eta_cosh * alpha_sin * theta_cos + alpha_cos * theta_sin) * pt[0]
                    + (-eta_cosh * alpha_sin * theta_sin + alpha_cos * theta_cos) * pt[1]
                    + eta_sinh * alpha_sin * pt[2],
                (-eta_cosh * r_sinh * alpha_cos * theta_cos + r_sinh * alpha_sin * theta_sin
                    - eta_sinh * r_cosh * theta_cos)
                    * pt[0]
                    - (eta_cosh * r_sinh * alpha_cos * theta_sin
                        + r_sinh * alpha_sin * theta_cos
                        + eta_sinh * r_cosh * theta_sin)
                        * pt[1]
                    + (eta_sinh * r_sinh * alpha_cos + eta_cosh * r_cosh) * pt[2],
            ]);
            Hyperbolic::from_minkowski_coordinates(translated, point.skirt())
        };
        points.iter().map(vertex_translate).collect::<Vec<_>>()
    }
    #[inline]
    fn vertex_to_system_frame(
        vertex: &Hyperbolic<3>,
        body_orientation: &Angle,
        body_position: &Hyperbolic<3>,
    ) -> Hyperbolic<3> {
        let theta = body_position.coordinates()[1].atan2(body_position.coordinates()[0]);
        let nu = (body_position.coordinates()[2] / body_position.skirt()).acosh();
        let tau_over_two = -nu / 2.0;
        let poincare = body_position.to_poincare();
        let body_angle_body = (-2.0
            * (((tau_over_two.sinh())
                * ((theta.cos()) * poincare[1] - (theta.sin()) * poincare[0]))
                .atan2(
                    (tau_over_two.cosh())
                        + (tau_over_two.sinh())
                            * ((theta.cos()) * poincare[0] + (theta.sin()) * poincare[1]),
                ))
            + body_orientation.theta)
            .rem_euclid(2.0 * PI);
        let phi = body_angle_body - theta;
        let pt = vertex.point().coordinates;
        let (nu_sinh, nu_cosh) = (nu.sinh(), nu.cosh());
        let (theta_sin, theta_cos, phi_sin, phi_cos) =
            (theta.sin(), theta.cos(), phi.sin(), phi.cos());
        let transformed = Minkowski::from([
            (nu_cosh * phi_cos * theta_cos - phi_sin * theta_sin) * pt[0]
                - (nu_cosh * phi_sin * theta_cos + phi_cos * theta_sin) * pt[1]
                + nu_sinh * theta_cos * pt[2],
            (nu_cosh * phi_cos * theta_sin + phi_sin * theta_cos) * pt[0]
                + (-nu_cosh * phi_sin * theta_sin + phi_cos * theta_cos) * pt[1]
                + nu_sinh * theta_sin * pt[2],
            nu_sinh * phi_cos * pt[0] - nu_sinh * phi_sin * pt[1] + nu_cosh * pt[2],
        ]);
        Hyperbolic::from_minkowski_coordinates(transformed, vertex.skirt())
    }
}

impl IntersectsAtGlobal<HyperbolicConvexPolytope<3>, Hyperbolic<3>, Angle> for HyperbolicConvexPolytope<3> {
    #[inline]
    #[allow(clippy::too_many_lines, reason = "complicated function")]
    fn intersects_at_global(
        &self,
        other: &HyperbolicConvexPolytope<3>,
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
        let n_self = self.vertices.len();
        let n_other = other.vertices.len();
        while result && (v_count < n_self + n_other) {
            if v_count < n_self {
                let v_num = (v_count) % n_self;
                let v_next = (v_num + 1) % n_self;
                let v_next_next = (v_num + 2) % n_self;
                // translate all vertices
                // need to do this for every other vertex
                let v_1 = Self::vertex_to_system_frame(&self.vertices[v_num], r_i, x_i);
                let v_2 = Self::vertex_to_system_frame(&self.vertices[v_next], r_i, x_i);
                let v_3 = Self::vertex_to_system_frame(&self.vertices[v_next_next], r_i, x_i);
                let other_vertices = self
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
                    n_self,
                );
                let other_translated = Self::to_vertex_frame_oriented(
                    x_i,
                    r_i,
                    v_next,
                    self.bounding_radius,
                    &other_vertices,
                    n_self,
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
                while !overlap && (counter < other_vertices.len()) {
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
                    while !overlap && (counter < other_vertices.len()) {
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
            } else {
                let v_num = (v_count) % n_other;
                let v_next = (v_num + 1) % n_other;
                let v_next_next = (v_num + 2) % n_other;
                // translate all vertices
                let v_1 = Self::vertex_to_system_frame(&other.vertices[v_num], r_j, x_j);
                let v_2 = Self::vertex_to_system_frame(&other.vertices[v_next], r_j, x_j);
                let v_3 = Self::vertex_to_system_frame(&other.vertices[v_next_next], r_j, x_j);
                let other_vertices = self
                    .vertices
                    .iter()
                    .map(|vertex| Self::vertex_to_system_frame(vertex, r_i, x_i))
                    .collect::<Vec<Hyperbolic<3>>>();
                let self_translated = Self::to_vertex_frame_oriented(
                    x_j,
                    r_j,
                    v_next,
                    other.bounding_radius,
                    &[v_1, v_2, v_3],
                    n_other,
                );
                let other_translated = Self::to_vertex_frame_oriented(
                    x_j,
                    r_j,
                    v_next,
                    other.bounding_radius,
                    &other_vertices,
                    n_other,
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
                while !overlap && (counter < other_vertices.len()) {
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
                    while !overlap && (counter < other_vertices.len()) {
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
    use crate::shape::EightEight;
    use approxim::assert_relative_eq;
    use hoomd_manifold::{Hyperbolic, Minkowski};
    use hoomd_vector::Angle;

    #[test]
    fn octagon_edges() {
        let center_dist = 1.528_570_919_480_998;
        let quarter_dist = 1.643_866_837_922_488;
        let octagon = HyperbolicConvexPolytope::<3>::regular(8, EightEight::EIGHTEIGHT, 1.0);
        assert_relative_eq!(
            center_dist,
            octagon.edge_distance(-3.0 * PI / 8.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            quarter_dist,
            octagon.edge_distance(PI / 16.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            EightEight::EIGHTEIGHT,
            octagon.edge_distance(PI / 4.0),
            epsilon = 1e-12
        );
    }
    #[test]
    fn square_edges() {
        let center_dist = 0.602_080_559_268_716;
        let quarter_dist = 0.666_842_324_123_307;
        let square = HyperbolicConvexPolytope::<3>::regular(4, 1.0, 1.0);
        assert_relative_eq!(center_dist, square.edge_distance(PI / 4.0), epsilon = 1e-12);
        assert_relative_eq!(
            quarter_dist,
            square.edge_distance(-PI / 8.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(1.0, square.edge_distance(PI / 2.0), epsilon = 1e-12);
    }
     #[test]
    fn center_at_oriented_vertex() {
        let square = HyperbolicConvexPolytope::<3>::regular(4, 0.5, 1.0);
        let (boost, rotation, orientation) = (0.5, PI / 4.0, 0.4);
        let body_position = Hyperbolic::<3>::from_polar_coordinates(boost, rotation, 1.0);
        let square_system = square
            .vertices()
            .iter()
            .map(|v| {
                HyperbolicConvexPolygon::vertex_to_system_frame(
                    v,
                    &Angle::from(orientation),
                    &body_position,
                )
            })
            .collect::<Vec<Hyperbolic<3>>>();
        let translated = HyperbolicConvexPolygon::to_vertex_frame_oriented(
            &body_position,
            &Angle::from(orientation),
            2_usize,
            0.5,
            &square_system,
            4_usize,
        );
        assert_relative_eq!(0.0, translated[2].coordinates()[0], epsilon = 1e-12);
        assert_relative_eq!(0.0, translated[2].coordinates()[1], epsilon = 1e-12);
        assert_relative_eq!(1.0, translated[2].coordinates()[2], epsilon = 1e-12);
    }
    #[test]
    fn no_square_overlap() {
        let square = HyperbolicConvexPolytope::<3>::regular(4, 0.5, 1.0);
        let boost: f64 = 3.0;
        let rotation: f64 = 2.3;
        let orientation: f64 = 0.4;
        let x_j = Hyperbolic::<3>::from_polar_coordinates(boost, rotation, 1.0);
        assert!(!square.intersects_at_global(
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
        assert!(square.intersects_at_global(
            &square,
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
            assert!(!square.intersects_at_global(
                &square,
                &com,
                &Angle::from(PI / 4.0),
                translated,
                &Angle::from(PI / 4.0)
            ));
        }
        for translated in nudged_centers.iter().take(num_trials).skip(num_spaces) {
            assert!(square.intersects_at_global(
                &square,
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
            assert!(!square.intersects_at_global(
                &square,
                &center_1,
                &Angle::from(PI / 4.0),
                &center_2,
                &Angle::from(ep + PI / 4.0)
            ));
        }
        for ep in trials.iter().take(num_trials).skip(num_spaces) {
            assert!(square.intersects_at_global(
                &square,
                &center_1,
                &Angle::from(PI / 4.0),
                &center_2,
                &Angle::from(ep + PI / 4.0)
            ));
        }
    }
}

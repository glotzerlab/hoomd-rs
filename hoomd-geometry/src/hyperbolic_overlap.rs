//! Implement Overlap Check for Hyperbolic Surfaces

use hoomd_manifold::{Minkowski, Hyperbolic};
use hoomd_vector::Angle;
use std::f64::consts::PI;
use robust::{Coord, orient2d};

/// TODO
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
    pub fn bounding_radius(vertices: &[Hyperbolic<N>]) -> f64 {
        vertices.iter().map(hoomd_manifold::Hyperbolic::distance_from_cusp)
            .fold(0.0, f64::max)
    }
}

/// TODO
pub type HyperbolicConvexPolygon = HyperbolicConvexPolytope<3>;

impl HyperbolicConvexPolytope<3> {
    /// Create a regular *n*-gon with *n* vertices and a given circumradius in 
    /// hyperbolic space.
    #[inline]
    #[must_use]
    pub fn regular(n: usize, circumradius: f64, skirt: f64) -> HyperbolicConvexPolytope<3> {
        HyperbolicConvexPolytope { 
            vertices:  (0..n).map(|x| {
                let theta = 2.0*PI*(x as f64)/(n as f64);
                Hyperbolic::<3>::from_polar_coordinates(circumradius, theta, skirt)
            })
            .collect::<Vec<_>>(),
            bounding_radius: circumradius,
        }
    }
    /// Calculate the distance between the center of a `HyperbolicConvexPolytope<3>` 
    /// and the edge at angle phi. The calculation works by computing the intersection 
    /// of the the hyperboloid with a plane passing through the origin and the two 
    /// adjacent vertices of the polygon. 
    #[inline]
    #[must_use]
    pub fn edge_distance(&self, phi: f64) -> f64 {
        let n = self.vertices.len() as f64;
        let phi_mod = phi.rem_euclid(2.0*PI/n) - PI/n;
        let eta_tanh = self.bounding_radius.tanh();
        let arg = (eta_tanh * ((2.0*PI/n).sin()))/((PI/n - phi_mod).sin() + (PI/n + phi_mod).sin());
        arg.atanh()
    }
}

/// Test whether two shapes overlap.
pub trait SeparatingPlanes<S, M, R> {
    /// Test whether the set of points in one shape intersects with the set of another.
    /// Method works by iterating through adjacent vertices of `self`, constructing the 
    /// hyperplane which pass through the two vertices, and checking if all the 
    /// points of `other` are on the opposite side. If a separating plane exists, then 
    /// the two shapes do not overlap and the method returns `false`.  
    fn intersects_at(&self, other: &[M],  x_i: &M, r_i: &R) -> bool;
    /// Translate vector of vertices to frame where query vertex is at origin.
    fn to_vertex_frame_oriented(body_position: &M, body_orientation: &Angle, vertex_num: usize, bounding_radius: f64, points: &[M], num_of_sides: usize) -> Vec<M>;
    /// Translate a vertex stored in `HyperbolicConvexPolygon` into the system frame.
    fn vertex_to_system_frame(vertex: &Hyperbolic<3>, body_orientation: &Angle, body_position: &Hyperbolic<3>) -> Hyperbolic<3>;
}

impl SeparatingPlanes<HyperbolicConvexPolygon, Hyperbolic<3>, Angle> for HyperbolicConvexPolygon {
    #[inline]
    fn intersects_at(&self, other: &[Hyperbolic<3>], x_i: &Hyperbolic<3>, r_i: &Angle) -> bool {
        let mut result = true;
        let mut v_count = 0_usize;
        let n = self.vertices.len();
        while result && (v_count < n) {
            let v_num = v_count;
            let v_next = (v_num + 1)%n;
            // translate all vertices  
            let v_1 = Self::vertex_to_system_frame(&self.vertices[v_num], r_i, x_i);
            let v_2 = Self::vertex_to_system_frame(&self.vertices[v_next], r_i, x_i);
            let self_translated = Self::to_vertex_frame_oriented(x_i, r_i, v_num, self.bounding_radius, &[v_1,v_2], n);
            let other_translated = Self::to_vertex_frame_oriented(x_i, r_i, v_num, self.bounding_radius, other, n);
            // convert to poincare coordinates to perform orientation checks. 
            let self_coord = self_translated
                .iter()
                .map(|pt| {
                    let poincare = pt.to_poincare();
                    Coord {x: poincare[0], y: poincare[1]}
                })
                .collect::<Vec<Coord<f64>>>();
            let other_coord = other_translated
                .iter()
                .map(|pt| {
                    let poincare = pt.to_poincare();
                    Coord {x: poincare[0], y: poincare[1]}
                })
                .collect::<Vec<Coord<f64>>>();
            // then do edge check
            let mut overlap = false;
            let mut counter = 0_usize;
            while !overlap && (counter < other.len()) {
                if orient2d(self_coord[0], self_coord[1], other_coord[counter]) >= 0.0 {
                    // break out of loop once one of the vertices is on the wrong side of the line
                    overlap = true;
                }
                counter += 1;
            }
            result = overlap;
            v_count += 1;
        }
        result
    }
    #[inline]
    fn to_vertex_frame_oriented(body_position: &Hyperbolic<3>, body_orientation: &Angle, vertex_num: usize, bounding_radius: f64, points: &[Hyperbolic<3>], num_of_sides: usize) -> Vec<Hyperbolic<3>> {
        let phi = body_orientation.theta + 2.0*PI*(vertex_num as f64)/(num_of_sides as f64);
        let theta = body_position.coordinates()[1].atan2(body_position.coordinates()[0]);
        let nu = (body_position.coordinates()[2]/body_position.skirt()).acosh();
        let ep = bounding_radius;
        let vertex_translate = |point: &Hyperbolic<3>| -> Hyperbolic<3> {
            let pt = point.point().coordinates;
            let translated = Minkowski::from([
                ((nu.cosh()) * (ep.cosh()) * (theta.cos()) * (phi.cos()) - (ep.cosh()) * (theta.sin()) * (phi.sin()) + (nu.sinh()) * (ep.sinh()) * (theta.cos()))*pt[0]
                + ((nu.cosh())*(ep.cosh())*(theta.sin())*(phi.cos()) + (ep.cosh())*(theta.cos())*(phi.sin()) + (nu.sinh())*(ep.sinh())*(theta.sin())) * pt[1]
                + (-(nu.sinh())*(ep.cosh())*(phi.cos()) - (nu.cosh())*(ep.sinh())) * pt[2],

                (-(nu.cosh())*(theta.cos())*(phi.sin()) - (theta.sin())*(phi.cos())) * pt[0]
                + (-(nu.cosh())*(theta.sin())*(phi.sin()) + (theta.cos())*(phi.cos())) * pt[1]
                + ((nu.sinh())*(phi.sin()))*pt[2],

                (-(nu.cosh())*(ep.sinh())*(theta.cos())*(phi.cos()) + (ep.sinh())*(theta.sin())*(phi.sin()) - (nu.sinh())*(ep.cosh())*(theta.cos()))*pt[0]
                +(-(nu.cosh())*ep.sinh()*theta.sin()*phi.cos() - ep.sinh()*theta.cos()*phi.sin() - nu.sinh()*ep.cosh()*theta.sin())*pt[1]
                +((nu.sinh())*(ep.sinh())*(phi.cos()) + (nu.cosh())*(ep.cosh()))*pt[2]
            ]);
            Hyperbolic::from_minkowski_coordinates(translated, point.skirt())
            };
            points.iter().map(vertex_translate).collect::<Vec<_>>()
    }
    #[inline]
    fn vertex_to_system_frame(vertex: &Hyperbolic<3>, body_orientation: &Angle, body_position: &Hyperbolic<3>) -> Hyperbolic<3> {
        let phi = body_orientation.theta;
        let theta = body_position.coordinates()[1].atan2(body_position.coordinates()[0]);
        let nu = (body_position.coordinates()[2]/body_position.skirt()).acosh();
        let pt = vertex.point().coordinates;
        let transformed = Minkowski::from([
            pt[0]*(nu.cosh())*(theta.cos())*(phi.cos()) - pt[1]*(nu.cosh())*(theta.cos())*(phi.sin()) + pt[2]*(nu.sinh())*(theta.cos()) - pt[0]*(theta.sin())*(phi.sin()) - pt[1]*(theta.sin())*(phi.cos()),
            pt[0]*(nu.cosh())*(theta.sin())*(phi.cos()) - pt[1]*(nu.cosh())*(theta.sin())*(phi.sin()) +pt[2]*(nu.sinh())*(theta.sin())+pt[0]*(theta.cos())*(phi.sin())+pt[1]*(theta.cos())*(phi.cos()),
            pt[0]*(nu.sinh())*(phi.cos()) - pt[1]*(nu.sinh())*(phi.sin())+pt[2]*(nu.cosh())
        ]);
        Hyperbolic::from_minkowski_coordinates(transformed, vertex.skirt())
        }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use crate::shape::EightEight;
    use hoomd_manifold::{Hyperbolic, Minkowski};
    use hoomd_vector::Angle;

    fn generate_square_pair(global_position: &Hyperbolic<3>, r_0: f64, sq_2_trans: f64, sq_2_rot: f64, sq_2_orientation: Angle) -> (Vec<Hyperbolic<3>>,Vec<Hyperbolic<3>>) {
        let square = HyperbolicConvexPolytope::<3>::regular(4, r_0, 1.0);
        let square_1: Vec<Hyperbolic<3>> = square.clone().vertices
            .iter()
            .map(|point| {
                let site_pos = point.coordinates();
                let rotated_site_pos = Minkowski::from([
                    site_pos[0]*((PI/4.0).cos()) - site_pos[1]*((PI/4.0).sin()),
                    site_pos[0]*((PI/4.0).sin()) + site_pos[1]*((PI/4.0).cos()),
                    site_pos[2]
                ]);
                Hyperbolic::from_minkowski_coordinates(rotated_site_pos, 1.0)
            })
            .collect::<Vec<Hyperbolic<3>>>();
        let square_2: Vec<Hyperbolic<3>> = square.clone().vertices
            .iter()
            .map(|point| {
                let site_pos = point.coordinates();
                let rotated_site_pos = Minkowski::from([
                    site_pos[0]*(sq_2_orientation.theta.cos()) - site_pos[1]*(sq_2_orientation.theta.sin()),
                    site_pos[0]*(sq_2_orientation.theta.sin()) + site_pos[1]*(sq_2_orientation.theta.cos()),
                    site_pos[2]
                ]);
                let transformed_point = Minkowski::from([
                    rotated_site_pos[0] * (sq_2_trans.cosh()) * (sq_2_rot.cos())
                    - rotated_site_pos[1] * (sq_2_rot.sin())
                    + rotated_site_pos[2] * (sq_2_trans.sinh()) * (sq_2_rot.cos()),
            
                    rotated_site_pos[0] * (sq_2_trans.cosh()) * (sq_2_rot.sin())
                    + rotated_site_pos[1] * (sq_2_rot.cos())
                    + rotated_site_pos[2] * (sq_2_trans.sinh()) * (sq_2_rot.sin()),

                    rotated_site_pos[0] * (sq_2_trans.sinh()) 
                    + rotated_site_pos[2] * (sq_2_trans.cosh()),
                ]);
                Hyperbolic::from_minkowski_coordinates(transformed_point, 1.0)
            })
            .collect::<Vec<Hyperbolic<3>>>();
        let global_rot = global_position.coordinates()[1].atan2(global_position.coordinates()[0]);
        let global_boost = (global_position.coordinates()[2]/global_position.skirt()).acosh();
        let translate = |point: &Hyperbolic<3>| {
            let site_pos = point.coordinates();
            let transformed_point = Minkowski::from([
                site_pos[0] * (global_boost.cosh()) * (global_rot.cos())
                - site_pos[1] * (global_rot.sin())
                + site_pos[2] * (global_boost.sinh()) * (global_rot.cos()),
        
                site_pos[0] * (global_boost.cosh()) * (global_rot.sin())
                + site_pos[1] * (global_rot.cos())
                + site_pos[2] * (global_boost.sinh()) * (global_rot.sin()),

                site_pos[0] * (global_boost.sinh()) 
                + site_pos[2] * (global_boost.cosh()),
            ]);
            Hyperbolic::from_minkowski_coordinates(transformed_point, 1.0)
            };
        let translated_square_1 = square_1
            .iter()
            .map(translate)
            .collect::<Vec<Hyperbolic<3>>>();
        let translated_square_2 = square_2
            .iter()
            .map(translate)
            .collect::<Vec<Hyperbolic<3>>>();
        (translated_square_1, translated_square_2)
    }

    #[test]
    fn octagon_edges() {
        let center_dist = 1.528_570_919_480_998;
        let quarter_dist = 1.643_866_837_922_488;
        let octagon = HyperbolicConvexPolytope::<3>::regular(8, EightEight::EIGHTEIGHT, 1.0);
        assert_relative_eq!(center_dist, octagon.edge_distance(-3.0*PI/8.0), epsilon=1e-12);
        assert_relative_eq!(quarter_dist, octagon.edge_distance(PI/16.0), epsilon=1e-12);
        assert_relative_eq!(EightEight::EIGHTEIGHT, octagon.edge_distance(PI/4.0), epsilon=1e-12);
    }
    #[test]
    fn square_edges() {
        let center_dist = 0.602_080_559_268_716;
        let quarter_dist = 0.666_842_324_123_307;
        let square = HyperbolicConvexPolytope::<3>::regular(4, 1.0, 1.0);
        assert_relative_eq!(center_dist, square.edge_distance(PI/4.0), epsilon=1e-12);
        assert_relative_eq!(quarter_dist, square.edge_distance(-PI/8.0), epsilon=1e-12);
        assert_relative_eq!(1.0, square.edge_distance(PI/2.0), epsilon=1e-12);
    }
    #[test]
    fn center_at_oriented_vertex() {
        let square = HyperbolicConvexPolytope::<3>::regular(4, 0.5, 1.0);
        let body_position = Hyperbolic::<3>::default();
        let translated = HyperbolicConvexPolygon::to_vertex_frame_oriented(&body_position, &Angle::from(0.0), 2_usize, 0.5, square.vertices(), 4_usize);
        assert_relative_eq!(0.0, translated[2].coordinates()[0], epsilon=1e-12);
        assert_relative_eq!(0.0, translated[2].coordinates()[1], epsilon=1e-12);
        assert_relative_eq!(1.0, translated[2].coordinates()[2], epsilon=1e-12);

        let boost:f64 = 1.2;
        let rotation:f64 = 0.5;
        let new_body = Hyperbolic::<3>::from_polar_coordinates(boost, rotation, 1.0);
        let orientation: f64 = 0.6;
        let transformed_square: Vec<Hyperbolic<3>> = square.vertices
            .iter()
            .map(|point| {
                let site_pos = point.coordinates();
                let rotated_site_pos = Minkowski::from([
                    site_pos[0]*(orientation.cos()) - site_pos[1]*(orientation.sin()),
                    site_pos[0]*(orientation.sin()) + site_pos[1]*(orientation.cos()),
                    site_pos[2]
                ]);
                let transformed_point = Minkowski::from([
                    rotated_site_pos[0] * (boost.cosh()) * (rotation.cos())
                    - rotated_site_pos[1] * (rotation.sin())
                    + rotated_site_pos[2] * (boost.sinh()) * (rotation.cos()),
            
                    rotated_site_pos[0] * (boost.cosh()) * (rotation.sin())
                    + rotated_site_pos[1] * (rotation.cos())
                    + rotated_site_pos[2] * (boost.sinh()) * (rotation.sin()),

                    rotated_site_pos[0] * (boost.sinh()) 
                    + rotated_site_pos[2] * (boost.cosh()),
                ]);
                Hyperbolic::from_minkowski_coordinates(transformed_point, 1.0)
            })
            .collect::<Vec<Hyperbolic<3>>>();
        let translated_again = HyperbolicConvexPolygon::to_vertex_frame_oriented(&new_body, &Angle::from(orientation), 1_usize, 0.5, &transformed_square, 4_usize);
        assert_relative_eq!(0.0, translated_again[1].coordinates()[0], epsilon=1e-12);
        assert_relative_eq!(0.0, translated_again[1].coordinates()[1], epsilon=1e-12);
        assert_relative_eq!(1.0, translated_again[1].coordinates()[2], epsilon=1e-12);
    }
    #[test]
    fn no_square_overlap() {
        let square = HyperbolicConvexPolytope::<3>::regular(4, 0.5, 1.0);
        let boost:f64 = 3.0;
        let rotation:f64 = 2.3;
        let orientation: f64 = 0.4;
        let transformed_square: Vec<Hyperbolic<3>> = square.vertices
            .iter()
            .map(|point| {
                let site_pos = point.coordinates();
                let rotated_site_pos = Minkowski::from([
                    site_pos[0]*(orientation.cos()) - site_pos[1]*(orientation.sin()),
                    site_pos[0]*(orientation.sin()) + site_pos[1]*(orientation.cos()),
                    site_pos[2]
                ]);
                let transformed_point = Minkowski::from([
                    rotated_site_pos[0] * (boost.cosh()) * (rotation.cos())
                    - rotated_site_pos[1] * (rotation.sin())
                    + rotated_site_pos[2] * (boost.sinh()) * (rotation.cos()),
            
                    rotated_site_pos[0] * (boost.cosh()) * (rotation.sin())
                    + rotated_site_pos[1] * (rotation.cos())
                    + rotated_site_pos[2] * (boost.sinh()) * (rotation.sin()),

                    rotated_site_pos[0] * (boost.sinh()) 
                    + rotated_site_pos[2] * (boost.cosh()),
                ]);
                Hyperbolic::from_minkowski_coordinates(transformed_point, 1.0)
            })
            .collect::<Vec<Hyperbolic<3>>>();
        assert!(!square.intersects_at(&transformed_square, &Hyperbolic::<3>::default(), &Angle::default()));
    }
    #[test]
    fn square_overlap() {
        let square = HyperbolicConvexPolytope::<3>::regular(4, 0.5, 1.0);
        let boost:f64 = 0.49;
        let rotation:f64 = 2.3;
        let orientation: f64 = 0.4;
        let transformed_square: Vec<Hyperbolic<3>> = square.vertices
            .iter()
            .map(|point| {
                let site_pos = point.coordinates();
                let rotated_site_pos = Minkowski::from([
                    site_pos[0]*(orientation.cos()) - site_pos[1]*(orientation.sin()),
                    site_pos[0]*(orientation.sin()) + site_pos[1]*(orientation.cos()),
                    site_pos[2]
                ]);
                let transformed_point = Minkowski::from([
                    rotated_site_pos[0] * (boost.cosh()) * (rotation.cos())
                    - rotated_site_pos[1] * (rotation.sin())
                    + rotated_site_pos[2] * (boost.sinh()) * (rotation.cos()),
            
                    rotated_site_pos[0] * (boost.cosh()) * (rotation.sin())
                    + rotated_site_pos[1] * (rotation.cos())
                    + rotated_site_pos[2] * (boost.sinh()) * (rotation.sin()),

                    rotated_site_pos[0] * (boost.sinh()) 
                    + rotated_site_pos[2] * (boost.cosh()),
                ]);
                Hyperbolic::from_minkowski_coordinates(transformed_point, 1.0)
            })
            .collect::<Vec<Hyperbolic<3>>>();
        assert!(square.intersects_at(&transformed_square, &Hyperbolic::<3>::default(), &Angle::default()));
    }
    #[test]
    fn overlap_translation_check() {
        let r_0 = 0.5;
        let square = HyperbolicConvexPolytope::<3>::regular(4, r_0, 1.0);
        let com = Hyperbolic::<3>::from_polar_coordinates(0.0, 0.0, 1.0);
        let distance = 2.0;
        let (_square_1, square_2) = generate_square_pair(&com, r_0, distance, 0.0, Angle::from(PI/4.0));
        let num_spaces: usize = 10;
        let num_trials: usize = 15;
        let spacing = 1.321_592_891_727_355;
        let trials = (0..num_trials).map(|n| -(n as f64)*spacing/(num_spaces as f64)).collect::<Vec<f64>>();
        let nudged_squares: Vec<Vec<Hyperbolic<3>>>  = trials.iter().map(|inch| {
                square_2.iter().map(|sq| {
                    let site_pos = sq.coordinates();
                    let transformed_point = Minkowski::from([
                        site_pos[0] * (inch.cosh()) + site_pos[2] * (inch.sinh()),
                        site_pos[1],
                        site_pos[0] * (inch.sinh()) + site_pos[2] * (inch.cosh()),
                    ]);
                Hyperbolic::from_minkowski_coordinates(transformed_point, 1.0)
                })
                .collect::<Vec<Hyperbolic<3>>>()
            })
            .collect::<Vec<Vec<Hyperbolic<3>>>>();

        // Check over overlaps
        for translated in nudged_squares.iter().take(num_spaces) {
            assert!(!square.intersects_at(&translated, &com, &Angle::from(PI/4.0)));
        }
        for translated in nudged_squares.iter().take(num_trials).skip(num_spaces) {
            assert!(square.intersects_at(&translated, &com, &Angle::from(PI/4.0)));
        }
    }
    #[test]
    fn overlap_rotation_check() {
        let r_0 = 0.5;
        let boost: f64 = 0.339_203_554_136_322;
        let distance: f64 = 0.45;
        let square = HyperbolicConvexPolytope::<3>::regular(4, r_0, 1.0);
        let com = Hyperbolic::<3>::from_polar_coordinates(0.0, 0.0, 1.0);
        let (_square_1_u, square_2_u) = generate_square_pair(&com, r_0, 0.0, 0.0, Angle::from(PI/4.0));
        let square_2 = square_2_u.iter().map(|point| {
            let pt = point.coordinates();
            let transformed_point = Minkowski::from([
                pt[0]*(distance.cosh()) + pt[2]*(distance.sinh()),
                pt[1],
                pt[0]*(distance.sinh()) + pt[2]*(distance.cosh())
            ]);
            Hyperbolic::from_minkowski_coordinates(transformed_point, 1.0)
            })
            .collect::<Vec<Hyperbolic<3>>>();
        let num_spaces: usize = 10;
        let num_trials: usize = 15;
        let spacing = 0.365_106_058_818_114;
        let trials = (0..num_trials).map(|n| (n as f64)*spacing/(num_spaces as f64)).collect::<Vec<f64>>();
        let nudged_squares: Vec<Vec<Hyperbolic<3>>>  = trials.iter().map(|inch| {
                square_2.iter().map(|sq| {
                    let pt = sq.coordinates();
                    let transformed_point = Minkowski::from([
                        pt[0]*(distance.cosh())*(distance.cosh())*(inch.cos()) - pt[2]*(distance.cosh())*(distance.sinh())*(inch.cos())- pt[1]*(distance.cosh())*(inch.sin())-pt[0]*(distance.sinh())*(distance.sinh()) + pt[2]*(distance.sinh())*(distance.cosh()),
                        pt[0]*(distance.cosh())*(inch.sin()) - pt[2]*(distance.sinh())*(inch.sin()) + pt[1]*(inch.cos()),
                        pt[0]*(distance.sinh())*(distance.cosh())*(inch.cos()) - pt[2]*(distance.sinh())*(distance.sinh())*(inch.cos())- pt[1]*(distance.sinh())*(inch.sin())-pt[0]*(distance.cosh())*(distance.sinh()) + pt[2]*(distance.cosh())*(distance.cosh()),
                    ]);
                Hyperbolic::from_minkowski_coordinates(transformed_point, 1.0)
                })
                .collect::<Vec<Hyperbolic<3>>>()
            })
            .collect::<Vec<Vec<Hyperbolic<3>>>>();

        let center = Hyperbolic::<3>::from_polar_coordinates(-boost, 0.0, 1.0);
        // Check over overlaps
        for rotated in nudged_squares.iter().take(num_spaces) {
            assert!(!square.intersects_at(&rotated, &center, &Angle::from(PI/4.0)));
        }
        for rotated in nudged_squares.iter().take(num_trials).skip(num_spaces) {
            assert!(square.intersects_at(&rotated, &center, &Angle::from(PI/4.0)));
        }
    }
}
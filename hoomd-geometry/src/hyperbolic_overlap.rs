//! Implement Overlap Check for Hyperbolic Surfaces

use hoomd_manifold::{Minkowski, Hyperbolic};
use hoomd_vector::{Angle, Metric};
use std::f64::consts::PI;
use robust::{Coord, orient2d};

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
        vertices.iter().map(|hyp| hyp.distance_from_cusp())
            .fold(0.0, f64::max)
    }
}

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
    /// TODO
    fn intersects_at(&self, other: &[M],  x_i: &M, r_i: &R) -> bool;
    /// Translate vector of vertices to frame where query vertex is at origin.
    fn to_vertex_frame(body_position: &M, body_orientation: &Angle, vertex_num: usize, bounding_radius: f64, points: &[M], num_of_sides: usize) -> Vec<M>;
}

impl SeparatingPlanes<HyperbolicConvexPolygon, Hyperbolic<3>, Angle> for HyperbolicConvexPolygon {
    #[inline]
    fn intersects_at(&self, other: &[Hyperbolic<3>], x_i: &Hyperbolic<3>, r_i: &Angle) -> bool {
        let mut result: bool = true;
        let mut v_count = 0_usize;
        let n = self.vertices.len();
        while (result == true) && (v_count < n) {
            let v_num = v_count;
            let v_next = (v_num + 1)%n;
            // translate all vertices 
            let self_translated = Self::to_vertex_frame(x_i, r_i, v_num, self.bounding_radius, &[self.vertices()[v_num],self.vertices()[v_next]], self.vertices.len());
            let other_translated = Self::to_vertex_frame(x_i, r_i, v_num, self.bounding_radius, other, self.vertices.len());
            // convert to poincare
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
            while (overlap == false) && (counter < other.len()) {
                if orient2d(self_coord[0], self_coord[1], other_coord[counter]) >= 0.0 {
                    // break out of loop once one of the vertices is on the wrong side of the line
                    overlap = true;
                };
                counter += 1;
            }
            result = overlap;
            v_count += 1;
        }
        result
    }
    #[inline]
    fn to_vertex_frame(body_position: &Hyperbolic<3>, body_orientation: &Angle, vertex_num: usize, bounding_radius: f64, points: &[Hyperbolic<3>], num_of_sides: usize) -> Vec<Hyperbolic<3>> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use crate::shape::EightEight;
    use hoomd_manifold::{Hyperbolic, Minkowski};
    use hoomd_vector::Angle;

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
    fn center_at_vertex() {
        let square = HyperbolicConvexPolytope::<3>::regular(4, 0.5, 1.0);
        let body_position = Hyperbolic::<3>::default();
        let translated = HyperbolicConvexPolygon::to_vertex_frame(&body_position, &Angle::from(0.0), 2_usize, 0.5, square.vertices(), 4_usize);
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
        let translated_again = HyperbolicConvexPolygon::to_vertex_frame(&new_body, &Angle::from(orientation), 1_usize, 0.5, &transformed_square, 4_usize);
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
}
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
                let theta = PI*(x as f64)/(n as f64);
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
    fn intersects_at(&self, other: &S, r_i: &R, r_j: &R) -> bool;
}

impl SeparatingPlanes<HyperbolicConvexPolygon, Hyperbolic<3>, Angle> for HyperbolicConvexPolygon {
    #[inline]
    fn intersects_at(&self, other: &HyperbolicConvexPolygon, r_i: &Angle, r_j: &Angle) -> bool {
        // closure for checking one edge
        let edge_check = |v1: usize, v2: usize| -> bool {
            false
        };
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use crate::shape::EightEight;

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
}
// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement the {8,8} tiling of hyperbolic space
*/

use tinyvec::ArrayVec;
use libm::{atan2, acosh, cos, cosh, sin, sinh};
use std::f64::consts::PI;

use crate::{
    boundary::{
        Error, GenerateGhosts, MAX_GHOSTS, MaximumAllowableInteractionRange, Periodic, Wrap,
    },
    property::Position,
};
use hoomd_geometry::IsPointInside;
use hoomd_manifold::{FundamentalDomain, Hyperboloid, Minkowski};

/** The {8,8} tile of hyperbolic space
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EightEight {
    /// Skirt width of the hyperboloid
    pub skirt: f64,
}

/// Cusp-to-vertex distance for {8,8} tiling for Gauss curvature K = -1
const EIGHTEIGHT: f64 = 2.448_452_447_678_076;

impl IsPointInside<Hyperboloid<3>> for EightEight {
    #[inline]
    fn is_point_inside(&self, point: &Hyperboloid<3>) -> bool {
        point.distance_to_boundary() >= 0.0
    }
}

impl MaximumAllowableInteractionRange for EightEight {
    #[inline]
    fn maximum_allowable_interaction_range(&self) -> f64 {
        self.skirt * 1.528_570_919_480_998
    }
}

impl<P> Wrap<P> for Periodic<EightEight> 
where
    P: Position<Metric = Hyperboloid<3>> 
{
    /** Wrap a point on the hyperboloid to the inside of the {8,8} tile. Function only works if the point is within two tile radii of the hyperboloid cusp.
     */
    #[inline]
    fn wrap(&self, properties: P) -> Result<P, Error> {
        let mut properties = properties;
        let r = properties.position_mut();

        let angle = atan2(r.point.coordinates[1], r.point.coordinates[0]);
        let theta = angle.rem_euclid(PI * 2.0);
        let (side, remainder) = ((theta/(PI/4.0)).floor(), theta.rem_euclid(PI/4.0));
        let v = acosh(r.point.coordinates[2]/r.skirt);

        // if point is safely within the tile, do nothing
        if v < 1.528_570_919_480_998 {
            Ok(properties)
        } else if v < EIGHTEIGHT {
            let d = r.distance_to_boundary();
            if d >= 0.0 {
                // do nothing if point is within tile
                Ok(properties)
            } else {
                // if point is less than EIGHTEIGHT away from cusp, just wrap point around to other side
                let new_side = (side + 4_f64) % 8_f64;
                let new_angle = PI/4.0 + PI*(new_side as f64)/4.0 - remainder;
                let wrapped_hyperboloid = Hyperboloid::<3>::from_polar(v/r.skirt, new_angle, r.skirt);
                r.point = wrapped_hyperboloid.point;
                Ok (properties)
            }
        } else if v < 2_f64 * EIGHTEIGHT{
            // if point is past EIGHTEIGHT, figure out which octagon it needs to be reflected into
            // NOTE: because of all the transformations, this part is numerically unstable
            let vertex_number: f64 = (((theta + (PI/8.0)).rem_euclid(PI*2.0))/(PI/4.0)).floor(); // vertex the point is closest to
            let (vertex_boost, vertex_angle) = (EIGHTEIGHT, (vertex_number * PI/4.0 - PI/8.0).rem_euclid(PI*2.0));
            // transform point to frame where relevant vertex is in the center
            let transformed_point = Minkowski::from([
                r.point.coordinates[0] * cosh(-vertex_boost) * cos(-vertex_angle) - r.point.coordinates[1] * sin(-vertex_angle)
                    + r.point.coordinates[2] * sinh(-vertex_boost) * cos(-vertex_angle),
                r.point.coordinates[0] * cosh(-vertex_boost) * sin(-vertex_angle)
                    + r.point.coordinates[1] * cos(-vertex_angle)
                    + r.point.coordinates[2] * sinh(-vertex_boost) * sin(-vertex_angle),
                r.point.coordinates[0] * sinh(-vertex_boost) + r.point.coordinates[2] * cosh(-vertex_boost),
            ]);
            // get coords of point in transformed frame
            let trans_angle = atan2(transformed_point.coordinates[1], transformed_point.coordinates[0]);
            let new_vertex = ((trans_angle / (PI/4.0)).floor() + (4.0 + 3.0*vertex_number).rem_euclid(8.0)).rem_euclid(8.0); 
            let vertex_list = [0.0, 3.0, 6.0, 1.0, 4.0, 7.0, 2.0, 5.0];
            let new_vertex_num = vertex_list[new_vertex.floor() as usize]; // new vertex which point should be maped to the inside of
            let (new_vertex_boost, new_vertex_angle) = (EIGHTEIGHT, new_vertex_num * (PI/4.0));
            let wrapped = Minkowski::from([
                transformed_point.coordinates[0] * cosh(new_vertex_boost) * cos(new_vertex_angle) - transformed_point.coordinates[1] * sin(new_vertex_angle)
                    + transformed_point.coordinates[2] * sinh(new_vertex_boost) * cos(new_vertex_angle),
                transformed_point.coordinates[0] * cosh(new_vertex_boost) * sin(new_vertex_angle)
                    + transformed_point.coordinates[1] * cos(new_vertex_angle)
                    + transformed_point.coordinates[2] * sinh(new_vertex_boost) * sin(new_vertex_angle),
                transformed_point.coordinates[0] * sinh(new_vertex_boost) + transformed_point.coordinates[2] * cosh(new_vertex_boost),
            ]);
            let wrapped_hyperboloid = Hyperboloid::<3>::from(&wrapped);
            r.point = wrapped_hyperboloid.point;            
            Ok(properties)
        } else {
            Err(Error::CannotWrapProperties)
        }
    }
}

impl<S> GenerateGhosts<S> for Periodic<EightEight> 
where
    S: Position<Metric = Hyperboloid<3>> + Copy + Default,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }
    /** Place periodic images of sites near the edge of the periodic boundary
    */
    #[inline]
    fn generate_ghosts(&self, _site_properties: &S) -> ArrayVec<[S; MAX_GHOSTS]> {
        let mut result = ArrayVec::new();
        let r = _site_properties.position();
        
        let angle = atan2(r.point.coordinates[1], r.point.coordinates[0]);
        let theta = angle.rem_euclid(PI * 2.0);
        let v = (r.point.coordinates[2]/r.skirt).acosh();

        // put a ghost particle near an edge
        let new_site_edge = |edge_num : f64, boost: f64, angle_mod_pi_fourths: f64| {
            let offset = (PI/8.0) - angle_mod_pi_fourths.rem_euclid(PI/4.0);
            let new_edge = (edge_num + 4.0).rem_euclid(8.0);
            let new_boost = boost + (EIGHTEIGHT.tanh() / (angle_mod_pi_fourths.cos() - angle_mod_pi_fourths.sin() * (1.0 - (2.0_f64).sqrt()))).atanh();
            let new_angle = ((PI/4.0)*new_edge + PI/8.0 + offset).rem_euclid(2.0*PI);
            Hyperboloid::<3>::from_polar(new_boost,new_angle, r.skirt)
        };

        // put a ghost particle near a vertex
        let new_site_vertex = |loc: f64, vertex_num: f64, point: &Hyperboloid<3>| {
            // boost to frame near vertex, calculate how to get to site from the vertex
            let (loc_boost, loc_angle) = (EIGHTEIGHT, vertex_num*(PI/4.0));
            let in_vertex_frame = Minkowski::from([
                point.point.coordinates[0] * cosh(-loc_boost) * cos(-loc_angle) - point.point.coordinates[1] * sin(-loc_angle)
                    + point.point.coordinates[2] * sinh(-loc_boost) * cos(-loc_angle),
                point.point.coordinates[0] * cosh(-loc_boost) * sin(-loc_angle)
                    + point.point.coordinates[1] * cos(-loc_angle)
                    + point.point.coordinates[2] * sinh(-loc_boost) * sin(-loc_angle),
                point.point.coordinates[0] * sinh(-loc_boost) + point.point.coordinates[2] * cosh(-loc_boost),
            ]);
            let (in_vertex_fram_boost, in_vertex_frame_angle) = (acosh(in_vertex_frame.coordinates[2]/point.skirt), atan2(in_vertex_frame.coordinates[1], in_vertex_frame.coordinates[0]));

        };
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_manifold::{HyperbolicDisk, Hyperboloid};
    use rand::{SeedableRng, Rng, rngs::StdRng, distr::Distribution};
    use std::f64::consts::PI;
    use approx::assert_relative_eq;

    #[test]
    fn doesnt_wrap_if_inside() {
        let r = 1.528_570_919_480_998;
        let mut rng = StdRng::seed_from_u64(239);
        let disk = HyperbolicDisk {r: r.try_into().expect("hard-coded positive number"), point: Hyperboloid::<3>::default().point, skirt: 1.0};
        let random_point: Hyperboloid<3> = disk.sample(&mut rng);

        let periodic = Periodic::new(1.528_570_919_480_998, EightEight {skirt: 1.0_f64}).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(random_point).expect("hard-coded");
        assert_eq!(random_point.point.coordinates, wrapped_point.point.coordinates)
    }

    #[test]
    fn wraps_to_opposite_edge() {
        let mut rng = rand::rng();
        let side = rng.random_range(0..8) as f64;
        let offset = PI/16.0;
        let boost = 2.0;
        let point = Hyperboloid::<3>::from_polar(boost, offset + side*PI/4.0, 1.0);
        let periodic = Periodic::new(1.528_570_919_480_998, EightEight {skirt: 1.0_f64}).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(point).expect("hard-coded");

        let wrapped_side = (side+4.0).rem_euclid(8.0);
        let octant = ((atan2(wrapped_point.point.coordinates[1], wrapped_point.point.coordinates[0])/(PI/4.0)).floor()).rem_euclid(8.0);

        // Check that point is wraped to correct octant
        assert_eq!(wrapped_side, octant);

        // Check that point mapping is correct
        let ans = Hyperboloid::<3>::from_polar(boost, (wrapped_side+1.0)*(PI/4.0) - offset, 1.0);
        assert_relative_eq!(ans.point.coordinates[0], wrapped_point.point.coordinates[0], epsilon=1e-12);
        assert_relative_eq!(ans.point.coordinates[1], wrapped_point.point.coordinates[1], epsilon=1e-12);
        assert_relative_eq!(ans.point.coordinates[2], wrapped_point.point.coordinates[2], epsilon=1e-12);
    }

    #[test]
    fn wraps_far_away_point_no_angle() {
        // NOTE: this is pretty unstable
        let boost = 2.448_452_448;
        let offset = 0.0;
        let point = Hyperboloid::<3>::from_polar(boost,offset, 1.0);
        let periodic = Periodic::new(1.528_570_919_480_998, EightEight {skirt: 1.0_f64}).expect("hard-coded positive number");
        let wrapped_point = periodic.wrap(point).expect("hard-coded");

        let distance_from_vertex = -point.distance_to_boundary();
        let v = 2.0*2.448_452_447_678_076 - boost;
        let ans = Hyperboloid::<3>::from_polar(v, PI, 1.0);
        assert_relative_eq!(ans.distance_to_boundary(), distance_from_vertex, epsilon=1e-12);

        assert_relative_eq!(ans.point.coordinates[0], wrapped_point.point.coordinates[0], epsilon=1e-8);
        assert_relative_eq!(ans.point.coordinates[1], wrapped_point.point.coordinates[1], epsilon=1e-8);
        assert_relative_eq!(ans.point.coordinates[2], wrapped_point.point.coordinates[2], epsilon=1e-8);
    }
}
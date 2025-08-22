// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement the {8,8} tiling of hyperbolic space
*/

use tinyvec::ArrayVec;

use crate::{
    boundary::{
        Error, GenerateGhosts, MAX_GHOSTS, MaximumAllowableInteractionRange, Periodic, Wrap,
    },
    property::Position,
};
use hoomd_geometry::IsPointInside;
use hoomd_utility::valid::PositiveReal;
use hoomd_manifold::{FundamentalDomain, Hyperboloid};

/** The {8,8} tile of hyperbolic space
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EightEight {
    /// Skirt width of the hyperboloid
    pub skirt: f64,
}

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
    #[inline]
    fn wrap(&self, properties: P) -> Result<P, Error> {
        let mut properties = properties;
        let r = properties.position_mut();

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
    #[inline]
    fn generate_ghosts(&self, _site_properties: &S) -> ArrayVec<[S; MAX_GHOSTS]> {
        
    }
}
// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![allow(clippy::all)]
#![allow(clippy::pedantic)]

use crate::local::GeneratorHyperbolic;
use crate::voronoi::Generator;
use glam::DVec3;
use rstar::{AABB, PointDistance, RTree, RTreeObject};

pub(crate) fn build_rtree(generators: &[Generator]) -> RTree<Generator> {
    RTree::bulk_load(generators.to_vec())
}

pub(crate) fn build_rtree_hyperbolic(
    generators: &[GeneratorHyperbolic],
) -> RTree<GeneratorHyperbolic> {
    RTree::bulk_load(generators.to_vec())
}

pub fn nn_iter<'a>(
    rtree: &'a RTree<Generator>,
    loc: DVec3,
) -> Box<dyn Iterator<Item = (usize, Option<DVec3>)> + 'a> {
    Box::new(
        rtree
            .nearest_neighbor_iter(&[loc.x, loc.y, loc.z])
            .map(|g| (g.id(), None)),
    )
}

pub fn nn_iter_hyperbolic<'a>(
    rtree: &'a RTree<GeneratorHyperbolic>,
    loc: Vec<f64>,
) -> Box<dyn Iterator<Item = (usize, Option<DVec3>)> + 'a> {
    Box::new(
        rtree
            .nearest_neighbor_iter(&[loc[0], loc[1], loc[2]])
            .map(|g| (g.site_tag, None)),
    )
}

impl RTreeObject for Generator {
    type Envelope = AABB<[f64; 3]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.loc().x, self.loc().y, self.loc().z])
    }
}

impl RTreeObject for GeneratorHyperbolic {
    type Envelope = AABB<[f64; 3]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.loc[0], self.loc[1], self.loc[2]])
    }
}

impl PointDistance for Generator {
    fn distance_2(
        &self,
        point: &<Self::Envelope as rstar::Envelope>::Point,
    ) -> <<Self::Envelope as rstar::Envelope>::Point as rstar::Point>::Scalar {
        self.loc().distance_squared(DVec3 {
            x: point[0],
            y: point[1],
            z: point[2],
        })
    }
}

//Squared Poincare disk distance
impl PointDistance for GeneratorHyperbolic {
    fn distance_2(
        &self,
        point: &<Self::Envelope as rstar::Envelope>::Point,
    ) -> <<Self::Envelope as rstar::Envelope>::Point as rstar::Point>::Scalar {
        let point_0 = self.loc();
        let point_1 = DVec3 {
            x: point[0],
            y: point[1],
            z: point[2],
        };
        let zero = DVec3::from_array([0.0, 0.0, 0.0]);
        // TODO: check that this is correct scaling for poincare metric
        let arg = (2.0 * (point_1 - point_0).distance_squared(zero))
            / ((self.skirt().powi(2) - point_1.distance_squared(zero))
                * (self.skirt().powi(2) - point_0.distance_squared(zero)));
        let dist = self.skirt() * (1.0 + arg).acosh();
        dist * dist
    }
}

use crate::meshless_voro::{Dimensionality, voronoi::Generator};
use glam::DVec3;
use rstar::{Envelope, ParentNode, Point, PointDistance, RTree, RTreeNode, RTreeObject, AABB};
use std::collections::BinaryHeap;
use crate::local::PDGenerator;
use std::iter::zip;

pub(crate) fn build_rtree(generators: &[Generator]) -> RTree<Generator> {
    RTree::bulk_load(generators.to_vec())
}

pub(crate) fn build_rtree_from_pd(pd_generators: &[PDGenerator]) -> RTree<PDGenerator> {
    RTree::bulk_load(pd_generators.to_vec())
}

pub fn nn_iter<'a>(
    rtree: &'a RTree<Generator>,
    loc: DVec3,
) -> Box<dyn Iterator<Item = (usize, Option<DVec3>)> + 'a> {
    Box::new(rtree.nearest_neighbor_iter(&[loc.x, loc.y, loc.z]).map(|g| (g.id(), None)))
}

pub fn nn_iter_from_pd<'a>(
    rtree: &'a RTree<PDGenerator>,
    loc: DVec3,
) -> Box<dyn Iterator<Item = (usize, Option<DVec3>)> + 'a> {
    Box::new(rtree.nearest_neighbor_iter(&[loc.x, loc.y, loc.z]).map(|g| (g.site_tag, None)))
}

pub(crate) fn wrapping_nn_iter<'a>(
    rtree: &'a RTree<Generator>,
    loc: DVec3,
    width: DVec3,
    dimensionality: Dimensionality,
) -> Box<dyn Iterator<Item = (usize, Option<DVec3>)> + 'a> {
    let query_point = [loc.x, loc.y, loc.z];
    let width = [width.x, width.y, width.z];
    Box::new(
        RTreeWrappingNearestNeighbourIter::new(rtree.root(), query_point, width, dimensionality)
            .map(move |(g, _distance, shift)| {
                let shift = if shift[0] == 0. && shift[1] == 0. && shift[2] == 0. {
                    None
                } else {
                    Some(-DVec3::from_array(shift))
                };
                (g.id(), shift)
            }),
    )
}

pub(crate) fn wrapping_nn_iter_from_pd<'a>(
    rtree: &'a RTree<PDGenerator>,
    loc: DVec3,
    width: DVec3,
    dimensionality: Dimensionality,
) -> Box<dyn Iterator<Item = (usize, Option<DVec3>)> + 'a> {
    let query_point = [loc.x, loc.y, loc.z];
    let width = [width.x, width.y, width.z];
    Box::new(
        RTreeWrappingNearestNeighbourIter::new_from_pd(rtree.root(), query_point, width, dimensionality)
            .map(move |(g, _distance, shift)| {
                let shift = if shift[0] == 0. && shift[1] == 0. && shift[2] == 0. {
                    None
                } else {
                    Some(-DVec3::from_array(shift))
                };
                (g.site_tag, shift)
            }),
    )
}

macro_rules! point {
    ($Self:ident) => {
        <<$Self as RTreeObject>::Envelope as Envelope>::Point
    };
}

pub struct RTreeNodeDistanceWrapper<'a, T>
where
    T: WrappingPointDistance + 'a,
{
    node: &'a RTreeNode<T>,
    distance: <point!(T) as Point>::Scalar,
    shift: point!(T),
}

pub struct RTreeWrappingNearestNeighbourIter<'a, T>
where
    T: WrappingPointDistance + 'a,
{
    nodes: BinaryHeap<RTreeNodeDistanceWrapper<'a, T>>,
    query_point: point!(T),
}

impl<'a> RTreeWrappingNearestNeighbourIter<'a, Generator> {
    pub fn new(
        root: &'a ParentNode<Generator>,
        query_point: [f64; 3],
        width: [f64; 3],
        dimensionality: Dimensionality,
    ) -> Self {
        let mut result = RTreeWrappingNearestNeighbourIter {
            nodes: BinaryHeap::with_capacity(27),
            query_point,
        };

        // Add the children of this node to the heap and also shifted versions of it for
        // all directions
        let j_range = match dimensionality {
            Dimensionality::TwoD | Dimensionality::ThreeD => -1..=1,
            Dimensionality::OneD => 0..=0,
        };
        let k_range = match dimensionality {
            Dimensionality::ThreeD => -1..=1,
            Dimensionality::OneD | Dimensionality::TwoD => 0..=0,
        };
        for i in -1..=1 {
            for j in j_range.clone() {
                for k in k_range.clone() {
                    let shift = [i as f64 * width[0], j as f64 * width[1], k as f64 * width[2]];
                    result.extend_heap(root.children(), shift);
                }
            }
        }
        result
    }

    fn extend_heap(&mut self, children: &'a [RTreeNode<Generator>], shift: [f64; 3]) {
        let &mut RTreeWrappingNearestNeighbourIter {
            ref mut nodes,
            ref query_point,
        } = self;
        nodes.extend(children.iter().map(|child: &RTreeNode<Generator>| {
            let distance = match child {
                RTreeNode::Parent(data) => {
                    data.envelope().wrapping_distance_2(query_point, &shift)
                }
                RTreeNode::Leaf(t) => t.wrapping_distance_2(query_point, &shift),
            };

            RTreeNodeDistanceWrapper {
                node: child,
                distance,
                shift,
            }
        }));
    }
}

impl<'a> RTreeWrappingNearestNeighbourIter<'a, PDGenerator> {
    pub fn new_from_pd(
        root: &'a ParentNode<PDGenerator>,
        query_point: [f64; 3],
        width: [f64; 3],
        dimensionality: Dimensionality,
    ) -> Self {
        let mut result = RTreeWrappingNearestNeighbourIter {
            nodes: BinaryHeap::with_capacity(27),
            query_point,
        };

        // Add the children of this node to the heap and also shifted versions of it for
        // all directions
        let j_range = match dimensionality {
            Dimensionality::TwoD | Dimensionality::ThreeD => -1..=1,
            Dimensionality::OneD => 0..=0,
        };
        let k_range = match dimensionality {
            Dimensionality::ThreeD => -1..=1,
            Dimensionality::OneD | Dimensionality::TwoD => 0..=0,
        };
        for i in -1..=1 {
            for j in j_range.clone() {
                for k in k_range.clone() {
                    let shift = [i as f64 * width[0], j as f64 * width[1], k as f64 * width[2]];
                    result.extend_heap_from_pd(root.children(), shift);
                }
            }
        }
        result
    }

    fn extend_heap_from_pd(&mut self, children: &'a [RTreeNode<PDGenerator>], shift: [f64; 3]) {
        let &mut RTreeWrappingNearestNeighbourIter {
            ref mut nodes,
            ref query_point,
        } = self;
        nodes.extend(children.iter().map(|child: &RTreeNode<PDGenerator>| {
            let distance = match child {
                RTreeNode::Parent(data) => {
                    data.envelope().wrapping_distance_2(query_point, &shift)
                }
                RTreeNode::Leaf(t) => t.wrapping_distance_2(query_point, &shift),
            };

            RTreeNodeDistanceWrapper {
                node: child,
                distance,
                shift,
            }
        }));
    }
}

impl RTreeObject for Generator {
    type Envelope = AABB<[f64; 3]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.loc().x, self.loc().y, self.loc().z])
    }
}

impl RTreeObject for PDGenerator {
    type Envelope = AABB<[f64; 3]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.center[0], self.center[1], self.center[2]])
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

//Squared power distance
impl PointDistance for PDGenerator {
    fn distance_2(
        &self,
        point: &<Self::Envelope as rstar::Envelope>::Point,
    ) -> <<Self::Envelope as rstar::Envelope>::Point as rstar::Point>::Scalar {
        let euclidean_distance_squared = self.center.distance_squared(DVec3 {
            x: point[0],
            y: point[1],
            z: point[2],
        });
        euclidean_distance_squared - self.radius
    }
}

impl WrappingPointDistance for Generator {
    fn wrapping_distance_2(&self, point: &[f64; 3], shift: &[f64; 3]) -> f64 {
        let dx = [
            point[0] + shift[0] - self.loc().x,
            point[1] + shift[1] - self.loc().y,
            point[2] + shift[2] - self.loc().z,
        ];

        dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]
    }
}

impl WrappingEnvelope for AABB<[f64; 3]> {
    fn wrapping_distance_2(&self, point: &[f64; 3], shift: &[f64; 3]) -> f64 {
        fn clamp(x: f64, min: f64, max: f64) -> f64 {
            x.max(min).min(max)
        }

        let lower = self.lower();
        let upper = self.upper();
        let mut dx = [0., 0., 0.];
        for i in 0..3 {
            dx[i] = clamp(point[i] + shift[i], lower[i], upper[i]) - point[i] - shift[i];
        }

        dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]
    }
}

//this might be wrong. Right now this is the Euclidean distance
impl WrappingPointDistance for PDGenerator {
    fn wrapping_distance_2(&self, point: &[f64; 3], shift: &[f64; 3]) -> f64 {
        let dx = [
            point[0] + shift[0] - self.center[0],
            point[1] + shift[1] - self.center[1],
            point[2] + shift[2] - self.center[2],
        ];

        dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] - self.radius*self.radius
    }
}

pub trait WrappingPointDistance: PointDistance {
    fn wrapping_distance_2(
        &self,
        point: &point!(Self),
        shift: &point!(Self),
    ) -> <point!(Self) as Point>::Scalar;
}

pub trait WrappingEnvelope: Envelope {
    fn wrapping_distance_2(
        &self,
        point: &Self::Point,
        shift: &Self::Point,
    ) -> <Self::Point as Point>::Scalar;
}

impl<'a, T> PartialEq for RTreeNodeDistanceWrapper<'a, T>
where
    T: WrappingPointDistance,
{
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl<'a, T> PartialOrd for RTreeNodeDistanceWrapper<'a, T>
where
    T: WrappingPointDistance,
{
    fn partial_cmp(&self, other: &Self) -> Option<::core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, T> Eq for RTreeNodeDistanceWrapper<'a, T> where T: WrappingPointDistance {}

impl<'a, T> Ord for RTreeNodeDistanceWrapper<'a, T>
where
    T: WrappingPointDistance,
{
    fn cmp(&self, other: &Self) -> ::core::cmp::Ordering {
        // Inverse comparison creates a min heap
        other
            .distance
            .partial_cmp(&self.distance)
            .expect("Distances to RTree nodes must be finite")
    }
}

impl<'a> Iterator for RTreeWrappingNearestNeighbourIter<'a, Generator> {
    type Item = (&'a Generator, f64, [f64; 3]);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(current) = self.nodes.pop() {
            match current {
                RTreeNodeDistanceWrapper {
                    node: RTreeNode::Parent(data),
                    shift,
                    ..
                } => {
                    self.extend_heap(data.children(), shift);
                }
                RTreeNodeDistanceWrapper {
                    node: RTreeNode::Leaf(t),
                    distance,
                    shift,
                } => {
                    return Some((t, distance, shift));
                }
            }
        }

        None
    }
}

impl<'a> Iterator for RTreeWrappingNearestNeighbourIter<'a, PDGenerator> {
    type Item = (&'a PDGenerator, f64, [f64; 3]);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(current) = self.nodes.pop() {
            match current {
                RTreeNodeDistanceWrapper {
                    node: RTreeNode::Parent(data),
                    shift,
                    ..
                } => {
                    self.extend_heap_from_pd(data.children(), shift);
                }
                RTreeNodeDistanceWrapper {
                    node: RTreeNode::Leaf(t),
                    distance,
                    shift,
                } => {
                    return Some((t, distance, shift));
                }
            }
        }

        None
    }
}
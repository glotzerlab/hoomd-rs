// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use crate::predicates::{self, distance_point_to_line_squared};
use hoomd_vector::{Cartesian, InnerProduct};
use itertools::Zip;

use crate::shape::ConvexPolyhedron;

use core::f64;
use std::iter::Iterator;
/// TODO
pub trait VertexIterator<const N: usize>: Iterator<Item = Cartesian<N>> {}
impl<I, const N: usize> VertexIterator<N> for I where I: Iterator<Item = Cartesian<N>> {}

/// TODO
fn calculate_extents<const N: usize>(points: &[Cartesian<N>]) -> ([usize; N], [usize; N]) {
    let (mut mins, mut maxs) = ([f64::INFINITY; N], [f64::NEG_INFINITY; N]);

    points.iter().enumerate().fold(
        ([usize::MAX; N], [usize::MAX; N]),
        |(mut min_idx, mut max_idx), (i, pt)| {
            for (j, v) in pt.into_iter().enumerate() {
                if v < mins[j] {
                    mins[j] = v;
                    min_idx[j] = i;
                }
                if v > maxs[j] {
                    maxs[j] = v;
                    max_idx[j] = i;
                }
            }
            (min_idx, max_idx)
        },
    )
}
/// Find the indices of the pair of points with the largest magnitude difference.
///
/// This method can be used to find two points on a hull.
fn find_furthest_pair<const N: usize>(points: &[Cartesian<N>]) -> (usize, usize) {
    let (min_extent_idx, max_extent_idx) = calculate_extents(points);
    let mut max_distance = f64::NEG_INFINITY;

    max_extent_idx.iter().zip(min_extent_idx).fold(
        (usize::MAX, usize::MAX),
        |(a, b), (&max_idx_i, min_idx_i)| {
            let distance = (points[max_idx_i] - points[min_idx_i]).norm_squared();
            if distance > max_distance {
                max_distance = distance;
                (min_idx_i, max_idx_i)
            } else {
                (a, b)
            }
        },
    )
}

/// Find the index and distance to the point `c` furthest from the line defined by indices `a` and `b`.
///
/// This method can be used to find the third point on a hull.
fn find_furthest_point_from_line<const N: usize>(
    points: &[Cartesian<N>],
    a: usize,
    b: usize,
) -> (usize, f64) {
    let edge = points[b] - points[a];
    let segment_start = points[a];

    points.iter().enumerate().fold(
        (usize::MAX, f64::NEG_INFINITY),
        |(furthest_point_idx, furthest_distance_squared), (i, pt)| {
            let dist = distance_point_to_line_squared(pt, &segment_start, &edge);
            if dist > furthest_distance_squared {
                (i, dist)
            } else {
                (furthest_point_idx, furthest_distance_squared)
            }
        },
    )
}

///
fn hull(vertices: &[Cartesian<3>]) -> ConvexPolyhedron {
    ConvexPolyhedron {
        vertices: vec![],
        bounding_radius: 0.0,
    }
}

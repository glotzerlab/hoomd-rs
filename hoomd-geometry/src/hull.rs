use std::collections::VecDeque;

use crate::Error;
use hoomd_vector::{Cartesian, InnerProduct};
use itertools::Itertools;

/// Find the lowest, leftmost point from a slice of Cartesian vectors.
#[inline]
fn find_lowest_leftmost(vertices: &[Cartesian<2>]) -> (usize, Cartesian<2>) {
    vertices.iter().enumerate().fold(
        (usize::MAX, Cartesian::from([f64::INFINITY, f64::INFINITY])),
        |(i, p), (min_i, &min_p)| {
            if p[1] < min_p[1] || (p[1] == min_p[1] && p[0] < min_p[0]) {
                (i, p)
            } else {
                (min_i, min_p)
            }
        },
    )
}

#[derive(Debug, PartialEq, PartialOrd)]
/// Helper struct for ordering points with respect to an anchor.
struct SortKey {
    /// The angle a point makes with another.
    angle: f64,
    /// The distance of the point from the reference.
    distance: f64,
}

#[inline]
fn get_graham_key(p: Cartesian<2>, anchor: Cartesian<2>) -> (f64, f64) {
    let diff = p - anchor;
    (f64::atan2(diff[1], diff[0]), diff.dot(&diff))
}
/// Determines whether a point `test` is to the left, right, or colinear with `edge`.
///
/// # Warning
///
/// This predicate is not robust: points very close to the line may be misclassified
/// due to floating-point precision limits. For all practical inputs, this will not
/// result in issues.
///
/// # Note
///
/// This formulation (often referred to as the shoelace formula) guarantees
/// **cyclic invariance**, or the property that the orientation sign is identical
/// for any ordering of the three points `(e0, e1, t)`, `(e1, t, e0)`, and
/// `(t, e0, e1)`. As a result, the result is antisymmetric about the edge `e`
/// such that `p == -p'` for any `p'` reflected over `e`.
///
/// These properties do *not* prevent misclassification of points near the line—they
/// only ensure consistent behavior for related inputs.
///
/// **Source:** [Robust Arithmetic in Computational Geometry](https://observablehq.com/@mourner/non-robust-arithmetic-as-art)
fn predicate_orient2d(edge: (Cartesian<2>, Cartesian<2>), test: Cartesian<2>) -> i64 {
    let (p, q) = edge;
    let orientation = (p[0] * q[1] - p[1] * q[0])
        + (q[0] * test[1] - q[1] * test[0])
        + (test[0] * p[1] - test[1] * p[0]);
    if orientation > 0.0 {
        // clockwise
        return 1;
    } else if orientation < 0.0 {
        // counterclockwise
        return -1;
    }
    0
}

/// Compute the convex hull of points in two dimensions using a Graham scan.
/// # Errors
///
/// `[Error::PolytopeNotConvex]` when the provided vertices do not form a convex set.
pub fn hull_2d_grahamscan<I>(vertices: &mut [Cartesian<2>]) -> Option<bool> {
    // vertices.sort_by(|a, b| {});
    let (anchor_idx, anchor) = find_lowest_leftmost(&vertices[..]);

    // Move the anchor to the front of the list of vertices, as it is always in the hull
    vertices.swap(0, anchor_idx);
    let anchor = vertices[0];

    // Sort the remainder of the slice in-place
    vertices[1..].sort_unstable_by(|&a, &b| {
        let (a0, a1) = get_graham_key(a, anchor);
        let (b0, b1) = get_graham_key(b, anchor);
        a0.total_cmp(&b0).then(a1.total_cmp(&b1))
    });

    // TODO: no allocation
    let mut angle_order = vertices[1..].iter().copied().collect::<VecDeque<_>>();

    // The anchor and the first vertex in the sorted list are always on the hull
    let mut hull_vertices = vec![anchor, angle_order.pop_front()?];
    while !angle_order.is_empty() {
        let c = angle_order.pop_front()?;
        while hull_vertices.len() >= 2 {
            // Backtrack while hull_indices[-1] is inside hull
            let &[p, n] = hull_vertices.last_chunk::<2>()?;

            if predicate_orient2d((p, n), c) <= 0 {
                hull_vertices.pop(); // point n is inside the hull, so we remove it
            } else {
                break;
            }
        }
        hull_vertices.push(c);
    }
    Some(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use rstest::*;

    #[rstest]
    #[case::single_point(vec![[1.0, 2.0]], 0)]
    #[case::two_points_first_lower(vec![[1.0, 1.0], [2.0, 3.0]], 0)]
    #[case::two_points_second_lower(vec![[1.0, 3.0], [2.0, 1.0]], 1)]
    #[case::same_y_leftmost_wins(vec![[3.0, 1.0], [1.0, 1.0], [2.0, 1.0]], 1)]
    #[case::same_y_negative(vec![[0.0, -5.0], [-3.0, -5.0], [2.0, -5.0]], 1)]
    #[case::multiple_points(vec![[3.0, 5.0], [1.0, 1.0], [4.0, 2.0], [2.0, 3.0]], 1)]
    #[case::negative_coords(vec![[0.0, 0.0], [-1.0, -1.0], [1.0, -1.0]], 1)]
    #[case::same_y_all_negative_x(vec![[5.0, 0.0], [-10.0, 0.0], [-5.0, 0.0]], 1)]
    #[case::lowest_is_only_point(vec![[100.0, -100.0]], 0)]
    #[case::diagonal_tiebreak(vec![[5.0, 5.0], [4.0, 4.0], [3.0, 3.0], [2.0, 2.0], [1.0, 1.0]], 4)]
    fn test_find_lowest_leftmost(#[case] vertices: Vec<[f64; 2]>, #[case] expected_idx: usize) {
        let vertices: Vec<Cartesian<2>> = vertices.into_iter().map(Cartesian::from).collect();
        let (idx, point) = find_lowest_leftmost(&vertices);
        assert_eq!(idx, expected_idx);
        assert_relative_eq!(point, vertices[expected_idx]);
    }

    #[rstest]
    fn test_single_point_various_coords(
        #[values([0.0, 0.0], [-1.0, -1.0], [100.0, -50.0], [f64::MIN_POSITIVE, f64::MIN_POSITIVE])]
        coords: [f64; 2],
    ) {
        let vertices = vec![Cartesian::from(coords)];
        let (idx, point) = find_lowest_leftmost(&vertices);
        assert_eq!(idx, 0);
        assert_relative_eq!(point, Cartesian::from(coords));
    }
}

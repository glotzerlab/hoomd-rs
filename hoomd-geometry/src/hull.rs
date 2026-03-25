use crate::Error;
use hoomd_vector::Cartesian;

struct _SortKey {
    angle: f64,
    distance: f64,
}

/// Find the lowest, leftmost point from a slice of Cartesian vectors.
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

/// Compute the convex hull of points in two dimensions using a Graham scan.
/// # Errors
///
/// `[Error::PolytopeNotConvex]` when the provided vertices do not form a convex set.
pub fn hull_2d_grahamscan<I>(vertices: Vec<Cartesian<2>>) -> Result<bool, Error> {
    // vertices.sort_by(|a, b| {});
    let (anchor_idx, anchor) = find_lowest_leftmost(&vertices[..]);
    Ok(true)
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

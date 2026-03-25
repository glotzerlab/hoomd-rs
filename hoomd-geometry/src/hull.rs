use crate::Error;
use hoomd_vector::Cartesian;

struct _SortKey {
    angle: f64,
    distance: f64,
}

/// Find the lowest, leftmost point from a slice of Cartesian vectors.
fn _find_lowest_leftmost(vertices: &[Cartesian<2>]) -> (usize, Cartesian<2>) {
    vertices.iter().enumerate().fold(
        (usize::MAX, Cartesian::from([f64::INFINITY, f64::INFINITY])),
        |(i, p), (min_i, &min_p)| {
            if p[1] < min_p[1] || (p[1] == min_p[0] && p[0] < min_p[0]) {
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
    Ok(true)
}

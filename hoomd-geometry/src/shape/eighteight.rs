// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`EightEight`] */

use crate::IsPointInside;
use hoomd_manifold::Hyperboloid;
use std::f64::consts::PI;

/** [`EightEight`] implements a single regular octagon in the {8,8} tiling of two-dimensional hyperbolic space. The scaling of the octagon is set such that each of the angles is 2 pi/ 8, i.e., so that eight equivalent octagons may meet at each vertex.
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EightEight {
    /// Skirt width of the hyperboloid
    pub skirt: f64,
}

impl IsPointInside<Hyperboloid<3>> for EightEight {
    /** Checks if a given hyperboloid point is inside [`EightEight`]

    # Example
    ```
    use hoomd_geometry::{shape::EightEight, IsPointInside};
    use hoomd_manifold::Hyperboloid;
    use std::f64::consts::PI;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let eight_eight = EightEight {skirt:1.0};

    let point = Hyperboloid::<3>::from_polar(1.0, PI/8.0, 1.0);
    assert!(eight_eight.is_point_inside(&point));
    # Ok(())
    # }

    ```
     */
    #[inline]
    fn is_point_inside(&self, point: &Hyperboloid<3>) -> bool {
        EightEight::distance_to_boundary(point) >= 0.0
    }
}

/// Cusp-to-vertex distance for {8,8} tiling for Gauss curvature K = -1
const EIGHTEIGHT: f64 = 2.448_452_447_678_076;

impl EightEight {
    /** Computes the shortest distance between a given point and the boundary of `EightEight`. The shortest distance is along the radial path, i.e., the geodesic passing between the hyperboloid cusp and the query point.

    # Example
    ```
    use hoomd_geometry::shape::EightEight;
    use hoomd_manifold::{Hyperboloid, Minkowski};
    use std::f64::consts::PI;
    use approx::assert_relative_eq;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let v : f64 = 2.448_452_447_678_076;
    let rho : f64 = 1.0;
    let theta: f64 = PI/4.0;
    let x = Hyperboloid::from(&Minkowski::from([rho*(v.sinh())*(theta.cos()),rho*(v.sinh())*(theta.sin()),rho*(v.cosh())]));
    assert_relative_eq!(EightEight::distance_to_boundary(&x),0.0, epsilon=1e-12);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn distance_to_boundary(point: &Hyperboloid<3>) -> f64 {
        let theta = point.point.coordinates[1].atan2(point.point.coordinates[0]);
        let angle = theta.rem_euclid(PI / 4.0);
        let boost = (point.point.coordinates[2] / point.skirt).acosh();
        let tile_size = EIGHTEIGHT;
        let eta =
            (tile_size.tanh() / (angle.cos() - angle.sin() * (1.0 - (2.0_f64).sqrt()))).atanh();
        point.skirt * (eta - boost)
    }
    /** Outputs vector of points on the boundary of the fundamental domain
     */
    #[inline]
    #[must_use]
    pub fn boundary_points(m: usize, skirt: f64) -> Vec<(f64, f64)> {
        let mut coords = Vec::<(f64, f64)>::new();
        for n in 0..m {
            let angle = (n as f64) * 2.0 * PI / (m as f64);
            let tile_size = EIGHTEIGHT;
            let eta =
                (tile_size.tanh() / (angle.cos() - angle.sin() * (1.0 - (2.0_f64).sqrt()))).atanh();
            let x = (skirt * eta.sinh()) / (1.0 + eta.cosh());
            for k in 0..8 {
                coords.push((
                    x * (angle + f64::from(k) * PI / 4.0).cos(),
                    x * (angle + f64::from(k) * PI / 4.0).sin(),
                ));
            }
        }
        coords
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use hoomd_manifold::{HyperbolicDisk, Hyperboloid};
    use rand::{SeedableRng, distr::Distribution, rngs::StdRng};
    use std::ops::Not;

    #[test]
    fn boundary_distance() {
        // Distance to the edge of the {8,8} fundamental domain
        let e = Hyperboloid::<3>::from_polar(1.0, 0.1, 1.0);
        let e_edge_distance = EightEight::distance_to_boundary(&e);
        let e_edge_distance_numeric = 0.838_080_324_331_728;
        assert_relative_eq!(e_edge_distance, e_edge_distance_numeric, epsilon = 1e-12);

        let f = Hyperboloid::<3>::from_polar(1.0, 1.1, 1.0);
        let f_edge_distance = EightEight::distance_to_boundary(&f);
        let f_edge_distance_numeric = 0.545_034_457_278_499_5;
        assert_relative_eq!(f_edge_distance, f_edge_distance_numeric, epsilon = 1e-12);
    }

    #[test]
    fn inside_is_inside() {
        let eight_eight = EightEight { skirt: 1.0 };
        let r = 1.528_570_919_480_998;
        let mut rng = StdRng::seed_from_u64(239);
        let disk = HyperbolicDisk {
            r: r.try_into().expect("hard-coded positive number"),
            point: Hyperboloid::<3>::default().point,
            skirt: 1.0,
        };
        let random_point: Hyperboloid<3> = disk.sample(&mut rng);
        assert!(eight_eight.is_point_inside(&random_point));

        let point_1 = Hyperboloid::<3>::from_polar(1.52, PI / 8.0, 1.0);
        assert!(eight_eight.is_point_inside(&point_1));

        let point_2 = Hyperboloid::<3>::from_polar(2.44, PI / 4.0, 1.0);
        assert!(eight_eight.is_point_inside(&point_2));
    }

    #[test]
    fn outside_is_outside() {
        let eight_eight = EightEight { skirt: 1.0 };
        let point_1 = Hyperboloid::<3>::from_polar(1.53, PI / 8.0, 1.0);
        assert!((eight_eight.is_point_inside(&point_1)).not());

        let point_2 = Hyperboloid::<3>::from_polar(2.45, PI / 4.0, 1.0);
        assert!((eight_eight.is_point_inside(&point_2)).not());
    }
}

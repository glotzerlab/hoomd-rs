use crate::{Convex, Shape, Volume};
use hoomd_vector::Cartesian;

use std::f64::consts::PI;

fn factorial(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        (1..=n).reduce(|acc, x| acc * x).unwrap()
    }
}
fn double_factorial(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        (1..=n).step_by(2).reduce(|acc, x| acc * x).unwrap()
    }
}

/// An n-hypersphere ===================================================================
#[derive(Clone, Copy, Debug)]
pub struct Sphere<const N: usize> {
    /// Radius of the sphere
    pub r: f64,
    /// Centroid of the sphere
    pub c: Cartesian<N>,
}

impl<const N: usize> Default for Sphere<N> {
    fn default() -> Self {
        Sphere {
            r: 1.0,
            c: Cartesian::default(),
        }
    }
}

// Const generic params :(
// impl<const N: usize> From<[f64; N+1]> for Sphere<{N}> {
impl<const N: usize> From<(f64, [f64; N])> for Sphere<{ N }> {
    /** Construct a [`Sphere`] from 4 values.

    The first value is the radius. The 2nd through 4th are the center of mass:
    `[r, c_x, c_y, c_z]`.

    # Example
    ```
    use hoomd_geom::Sphere;

    let q = Sphere::from(1.0, [2.0, 3.0, 4.0]);
    assert_eq!(q.r, 1.0);
    assert_eq!(q.c, [2.0, 3.0, 4.0].into());
    ```
    */

    #[inline]
    fn from(value: (f64, [f64; N])) -> Self {
        let (r, c) = value;
        Self { r, c: c.into() }
    }
}
impl<const N: usize> From<f64> for Sphere<{ N }> {
    /** Construct a [`Sphere`] of radius `r` centered at the origin

    # Example
    ```
    use hoomd_geom::Sphere;

    let q = Sphere::from(1.0);
    assert_eq!(q.r, 1.0);
    assert_eq!(q.c, [0.0, 0.0, 0.0].into());
    ```
    */

    #[inline]
    fn from(r: f64) -> Self {
        Self {
            r,
            c: Cartesian::<N>::default(),
        }
    }
}

// TRAITS

impl<const N: usize> Convex for Sphere<N> {}

/// Redundant in this case, but helps me test the trait bounds
impl<const N: usize> Shape<N> for Sphere<N> {
    type V = Cartesian<N>;
    fn centroid(&self) -> Self::V {
        self.c
    }
}
impl<const N: usize> Volume for Sphere<N> {
    fn volume(&self) -> f64 {
        let dim_factor = (if N.rem_euclid(2) == 0 { N } else { N - 1 } / 2)
            .try_into()
            .unwrap();
        if N.rem_euclid(2) == 0 {
            PI.powi(dim_factor) / (factorial(N / 2) as f64)
        } else {
            2.0 * (2.0 * PI).powi(dim_factor) / (double_factorial(N) as f64)
        } // TODO: replace with std::f64::gamma when its in main
    }
}

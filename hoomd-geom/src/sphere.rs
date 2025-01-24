// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use crate::{Convex, Shape, Volume};
use hoomd_vector::{Cartesian, Vector};

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
pub struct Sphere<const N: usize, V>
where
    V: Vector + Copy,
{
    /// Radius of the sphere
    pub r: f64,
    /// Centroid of the sphere
    pub c: V,
}

impl<const N: usize> Default for Sphere<N, Cartesian<N>> {
    fn default() -> Self {
        Sphere {
            r: 1.0,
            c: Cartesian::default(),
        }
    }
}

// Const generic params :(
// impl<const N: usize> From<[f64; N+1]> for Sphere<{N}> {
impl<const N: usize, V: Vector + From<[f64; N]>> From<(f64, [f64; N])> for Sphere<{ N }, V> {
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
impl<const N: usize, V: Vector + Default> From<f64> for Sphere<{ N }, V> {
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
        Self { r, c: V::default() }
    }
}

// TRAITS

impl<const N: usize, V: Vector> Convex for Sphere<N, V> {}

/// Redundant in this case, but helps me test the trait bounds
impl<const N: usize, V: Vector> Shape<N, V> for Sphere<N, V> {
    // type V = Vector;
    fn centroid(&self) -> V {
        self.c
    }
    fn bounding_sphere(&self) -> Sphere<N, V> {
        *self
    }
}
impl<const N: usize, V: Vector> Volume for Sphere<N, V> {
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

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use crate::{Shape, Volume};
use hoomd_vector::Vector;

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
}

impl<const N: usize> Default for Sphere<N> {
    fn default() -> Self {
        Sphere { r: 1.0 }
    }
}

impl<const N: usize> From<f64> for Sphere<N> {
    #[inline]
    fn from(r: f64) -> Self {
        Sphere { r }
    }
}

// TRAITS

impl<const N: usize, V: Vector> Shape<N, V> for Sphere<N> {
    fn euler_characteristic(&self) -> i32 {
        2
    }
    fn bounding_sphere(&self) -> Sphere<N> {
        *self
    }
}

// impl<const N: usize, V: Vector> Particle<N, V> for Sphere<N, V> {
//     fn position(self, vec: Vec<V>) -> V {
//         // "Extrinsic" position of the shape - doesn't have to be the center of mass
//         vec[self.id]
//     }
// }

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

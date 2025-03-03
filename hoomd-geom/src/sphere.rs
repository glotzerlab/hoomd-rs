// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.
use crate::{Shape, Volume};
use hoomd_vector::Vector;
use std::f64::consts::PI;

/// The (single, double, ...)-factorial function
fn factorial(n: usize, ntuple: usize) -> usize {
    assert!(ntuple > 0);
    if n == 0 {
        1
    } else {
        (1..=n).step_by(ntuple).reduce(|acc, x| acc * x).unwrap()
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
    fn bounding_sphere(&self) -> Sphere<N> {
        *self
    }
}


impl<const N: usize> Volume for Sphere<N> {
    fn volume(&self) -> f64 {
        let dim_factor = (if N.rem_euclid(2) == 0 { N } else { N - 1 } / 2)
            .try_into()
            .unwrap();
        if N.rem_euclid(2) == 0 {
            PI.powi(dim_factor) / (factorial(N / 2, 1) as f64)
        } else {
            2.0 * (2.0 * PI).powi(dim_factor) / (factorial(N, 2) as f64)
        } // TODO: replace with std::f64::gamma when its in main
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;
    use approx::assert_relative_eq;
    use paste::paste;

    fn volume_map(N: usize) -> f64 {
        match N {
            0 => 1.0,
            1 => 2.0,
            2 => PI,
            3 => 4.0 / 3.0 * PI,
            4 => PI.powi(2) / 2.0,
            5 => 8.0 * PI.powi(2) / 15.0,
            _ => unreachable!(),
        }
    }
    // Parameterize a test function over an array of vector lengths
    macro_rules! parameterize_vector_length {
        ($test_body:ident, [$($dim:expr),*]) => {
            $(
                paste! {
                    #[test]
                    fn [< $test_body "_" $dim>]() {
                        const DIM: usize = $dim;
                        $test_body::<DIM>();
                    }
                }
            )*
        };
    }
    
    fn volume_and_radius<const N: usize>() {
        let s = Sphere::<N>::default();
        assert_eq!(s.r, 1.0);
        let ans = volume_map(N);
        assert_relative_eq!(s.volume(), ans)
    }
    parameterize_vector_length!(volume_and_radius, [0, 1, 2, 3, 4, 5]);

}

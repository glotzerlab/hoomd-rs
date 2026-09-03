// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `k_atic_psi`

use num_complex::Complex;

use hoomd_vector::Cartesian;

/// Compute the 2D *k-atic* order parameter $` \psi_k `$.
///
/// Let the `neighbors` iterator produce `n` points; $` \vec{r}_j `$.
/// 
/// The *k-atic* order parameter is given by:
/// ```math
/// \psi_k = \frac{1}{n} \sum \limits_j e^{k i \theta_{ij}}
/// ```
/// where $` \theta_{ij} `$ is the polar angle of the vector
/// $`\vec{r}_{ij} = \vec{r}_j - \vec{r}_i`$.
#[inline]
pub fn k_atic_psi<I: IntoIterator<Item=Cartesian<2>>>(k: f64, r_i: Cartesian<2>, neighbors: I) -> Complex<f64> {
    Complex::default()
}

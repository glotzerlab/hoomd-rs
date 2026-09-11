// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `k_atic_psi`

use num_complex::Complex;

use hoomd_vector::Cartesian;

/// Compute the 2D *k-atic* order parameter $` \psi_k `$.
///
/// Let the `neighbors` iterator produce `n` points $` \vec{r}_j `$ and
/// $` \theta_j `$ be the polar angle of the vector $`\vec{r}_j - \vec{r}`$.
/// The *k-atic* order parameter is then:
/// ```math
/// \psi_k = \frac{1}{n} \sum_j e^{i k \theta_j}
/// ```
///
/// # Example
///
/// ```
/// use num_complex::Complex;
/// use approxim::assert_relative_eq;
///
/// let r = [0.0, 0.0].into();
/// let neighbors = [[1.0, 0.0].into(), [0.0, 1.0].into(), [-1.0, 0.0].into(), [0.0, -1.0].into()];
/// let psi = hoomd_order::k_atic_psi(4.0, &r, neighbors);
///
/// assert_relative_eq!(psi, Complex::new(1.0, 0.0), epsilon = 1e-12);
/// ```
///
/// TODO: Example with microstate and `k_atic_psi` on all sites using `SitesInBall`.
#[inline]
pub fn k_atic_psi<I: IntoIterator<Item=Cartesian<2>>>(k: f64, r: &Cartesian<2>, neighbors: I) -> Complex<f64> {
    let mut total: Complex<f64> = Complex::default();
    let mut count: usize = 0;
    
    for r_j in neighbors {
        let delta_r = r_j - *r;
        let theta = delta_r[1].atan2(delta_r[0]);
        let (sin, cos) = (k * theta).sin_cos();
        total += Complex::new(cos, sin);
        count += 1;
    }

    total / Complex::new(count as f64, 0.0)
}

 #[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;

    #[test]
    fn single() {
        let r = [0.0, 0.0].into();
        let neighbors = [[1.0, 0.0].into()];

        assert_relative_eq!(k_atic_psi(1.0, &r, neighbors), Complex::new(1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(2.0, &r, neighbors), Complex::new(1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(3.0, &r, neighbors), Complex::new(1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(4.0, &r, neighbors), Complex::new(1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(5.0, &r, neighbors), Complex::new(1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(6.0, &r, neighbors), Complex::new(1.0, 0.0), epsilon = 1e-12);
    }

    #[test]
    fn shifted() {
        let r = [-5.0, 4.0].into();
        let neighbors = [[-4.0, 4.0].into()];

        assert_relative_eq!(k_atic_psi(1.0, &r, neighbors), Complex::new(1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(2.0, &r, neighbors), Complex::new(1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(3.0, &r, neighbors), Complex::new(1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(4.0, &r, neighbors), Complex::new(1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(5.0, &r, neighbors), Complex::new(1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(6.0, &r, neighbors), Complex::new(1.0, 0.0), epsilon = 1e-12);
    }

    #[test]
    fn rotated() {
        let r = [0.0, 0.0].into();

        assert_relative_eq!(k_atic_psi(2.0, &r, [[1.0, 0.0].into()]), Complex::new(1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(2.0, &r, [[-1.0, 0.0].into()]), Complex::new(1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(2.0, &r, [[0.0, 1.0].into()]), Complex::new(-1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(2.0, &r, [[0.0, -1.0].into()]), Complex::new(-1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(2.0, &r, [[1.0, 1.0].into()]), Complex::new(0.0, 1.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(2.0, &r, [[1.0, -1.0].into()]), Complex::new(0.0, -1.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(2.0, &r, [[-1.0, 1.0].into()]), Complex::new(0.0, -1.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(2.0, &r, [[-1.0, -1.0].into()]), Complex::new(0.0, 1.0), epsilon = 1e-12);

        assert_relative_eq!(k_atic_psi(4.0, &r, [[1.0, 0.0].into()]), Complex::new(1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(4.0, &r, [[0.0, 1.0].into()]), Complex::new(1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(4.0, &r, [[-1.0, 0.0].into()]), Complex::new(1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(4.0, &r, [[0.0, -1.0].into()]), Complex::new(1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(4.0, &r, [[1.0, 1.0].into()]), Complex::new(-1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(4.0, &r, [[1.0, -1.0].into()]), Complex::new(-1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(4.0, &r, [[-1.0, 1.0].into()]), Complex::new(-1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(4.0, &r, [[-1.0, -1.0].into()]), Complex::new(-1.0, 0.0), epsilon = 1e-12);
    }

    #[test]
    fn average() {
        let r = [0.0, 0.0].into();

        assert_relative_eq!(k_atic_psi(2.0, &r, [[1.0, 0.0].into(), [-1.0, 0.0].into()]), Complex::new(1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(2.0, &r, [[0.0, 1.0].into(), [0.0, -1.0].into()]), Complex::new(-1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(2.0, &r, [[-1.0, 1.0].into(), [-1.0, -1.0].into()]), Complex::new(0.0, 0.0), epsilon = 1e-12);

        assert_relative_eq!(k_atic_psi(4.0, &r, [[1.0, 0.0].into(), [0.0, 1.0].into(), [-1.0, 0.0].into(), [0.0, -1.0].into()]), Complex::new(1.0, 0.0), epsilon = 1e-12);
        assert_relative_eq!(k_atic_psi(4.0, &r, [[1.0, 1.0].into(), [1.0, -1.0].into(), [-1.0, 1.0].into(), [-1.0, -1.0].into()]), Complex::new(-1.0, 0.0), epsilon = 1e-12);
    }
}

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `Steinhardt`

use std::f64::consts::PI;

use hoomd_vector::{Cartesian, InnerProduct};

use crate::math::{SphericalHarmonic, SphericalHarmonicOutputs};

/// Compute the 3D Steinhardt order parameters $` q_{Lm} `$ and $` q_L `$.
///
/// [Steinhardt, Nelson, and Ronchetti 1983](https://doi.org/10.1103/PhysRevB.28.784)
/// introduces the order parameter $` q_{Lm} `$ and its rotationally invariant counterpart,
/// $` q_L `$.
///
/// Let the `neighbors` iterator produce `n` points $` \vec{r}_j `$.
/// The order parameters are then:
/// ```math
/// \begin{align*}
/// q_{Lm} &= \frac{1}{n} \sum_j Y_L^m \left( \frac{\vec{r}_j - \vec{r}}{\lvert \vec{r}_j - \vec{r} \rvert} \right) \\
/// q_L &= \sqrt{\frac{4 \pi}{2L + 1} \sum_{m=-L}^L \lvert q_{Lm} \rvert^2} \\
/// \end{align*}
/// ```
#[derive(Copy, Clone, Debug, Default)]
pub struct Steinhardt<const L: usize> {
    /// Compute the spherical harmonics.
    spherical_harmonic: SphericalHarmonic<L>
}

impl<const L: usize> Steinhardt<L> {
    /// Construct a new Steinhardt order parameter evaluator with degree `L`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_order::Steinhardt;
    ///
    /// let steinhardt = Steinhardt::<6>::new();
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            spherical_harmonic: SphericalHarmonic::new()
        }
    }

    /// Compute $` q_{Lm} `$.
    ///
    /// ```math
    /// q_{Lm} = \frac{1}{n} \sum_j Y_L^m \left( \frac{\vec{r}_j - \vec{r}}{\lvert \vec{r}_j - \vec{r} \rvert} \right)
    /// ```
    /// where the `neighbors` iterator produces `n` points $` \vec{r}_j `$.
    ///
    /// # Errors
    ///
    /// [`hoomd_vector::Error::InvalidVectorMagnitude`] when any $` |\vec{r}_j - \vec{r}| = 0 `$.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_order::Steinhardt;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let steinhardt = Steinhardt::<6>::new();
    /// let fcc_bonds: Vec<Cartesian<3>> = [
    ///     [-1.0, -1.0,  0.0], [-1.0,  1.0,  0.0], [1.0, -1.0,  0.0], [1.0,  1.0,  0.0],
    ///     [-1.0,  0.0, -1.0], [-1.0,  0.0,  1.0], [1.0,  0.0, -1.0], [1.0,  0.0,  1.0],
    ///     [ 0.0, -1.0, -1.0], [ 0.0, -1.0,  1.0], [0.0,  1.0, -1.0], [0.0,  1.0,  1.0],
    /// ].map(Cartesian::<3>::from).to_vec();
    ///
    /// let q_lm = steinhardt.q_lm(&Cartesian::default(), fcc_bonds)?;
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn q_lm<I: IntoIterator<Item=Cartesian<3>>>(&self, r: &Cartesian<3>, neighbors: I) -> Result<SphericalHarmonicOutputs<L>, hoomd_vector::Error> {
        let mut total = SphericalHarmonicOutputs::default();
        let mut count: usize = 0;

        for r_j in neighbors {
            let (delta_r_unit, _) = (r_j - *r).to_unit()?;
            total += self.spherical_harmonic.evaluate(&delta_r_unit);
            count += 1;
        }

        Ok(total / (count as f64))
    }

    /// Compute $` q_L `$ given $` q_{Lm} `$.
    ///
    /// ```math
    /// q_L = \sqrt{\frac{4 \pi}{2L + 1} \sum_{m=-L}^L \lvert q_{Lm} \rvert^2}
    /// ```
    #[inline]
    pub fn make_rotationally_invariant(q_lm: &SphericalHarmonicOutputs<L>) -> f64 {
        let mut sum_squared = q_lm[0].norm_sqr();
        for i in 1..=L {
            sum_squared += 2.0 * q_lm[i].norm_sqr();
        }
        (4.0 * PI / (2 * L + 1) as f64 * sum_squared).sqrt()
    }

    /// Compute $` q_L `$ given $` \vec{r} `$ and a set of neighbors $` \vec{r}_j `$.
    ///
    /// ```math
    /// q_L = \sqrt{\frac{4 \pi}{2L + 1} \sum_{m=-L}^L \lvert q_{Lm} \rvert^2}
    /// ```
    /// where
    /// ```math
    /// q_{Lm} = \frac{1}{n} \sum_j Y_L^m \left( \frac{\vec{r}_j - \vec{r}}{\lvert \vec{r}_j - \vec{r} \rvert} \right)
    /// ```
    /// and the `neighbors` iterator produces `n` points $` \vec{r}_j `$.
    ///
    /// # Errors
    ///
    /// [`hoomd_vector::Error::InvalidVectorMagnitude`] when any $` |\vec{r}_j = \vec{r}| = 0 `$.
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    ///
    /// use hoomd_order::Steinhardt;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let steinhardt = Steinhardt::<6>::new();
    /// let fcc_bonds: Vec<Cartesian<3>> = [
    ///     [-1.0, -1.0,  0.0], [-1.0,  1.0,  0.0], [1.0, -1.0,  0.0], [1.0,  1.0,  0.0],
    ///     [-1.0,  0.0, -1.0], [-1.0,  0.0,  1.0], [1.0,  0.0, -1.0], [1.0,  0.0,  1.0],
    ///     [ 0.0, -1.0, -1.0], [ 0.0, -1.0,  1.0], [0.0,  1.0, -1.0], [0.0,  1.0,  1.0],
    /// ].map(Cartesian::<3>::from).to_vec();
    ///
    /// let q_l = steinhardt.q_l(&Cartesian::default(), fcc_bonds)?;
    /// assert_relative_eq!(q_l, 0.57452416, epsilon = 1e-6);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn q_l<I: IntoIterator<Item=Cartesian<3>>>(&self, r: &Cartesian<3>, neighbors: I) -> Result<f64, hoomd_vector::Error> {
        let q_lm = self.q_lm(r, neighbors)?;
        Ok(Self::make_rotationally_invariant(&q_lm))
    }
}

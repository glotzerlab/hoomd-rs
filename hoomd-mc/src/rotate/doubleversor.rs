// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`DoubleVersor`] trial moves for 4-dimensional orientable bodies.

use std::f64::consts::PI;

use hoomd_microstate::property::Orientation;
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{DoubleVersor, Rotation};
use rand::Rng;
use rand_distr::Distribution;

use crate::{Adjust, LocalTrial, Rotate, rotate::versor::VersorDisplacement};

/// A normal distribution of random [`DoubleVersor`] displacements, centered on
/// the identity with some standard deviation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DoubleVersorDisplacement {
    /// The standard deviation of the normal distribution of quaternions around the identity.
    std_dev: f64,
}

impl From<f64> for DoubleVersorDisplacement {
    #[inline]
    fn from(value: f64) -> Self {
        Self { std_dev: value }
    }
}
impl Distribution<DoubleVersor> for DoubleVersorDisplacement {
    /// Sample a random [`DoubleVersor`] displacement centered on the identity.
    ///
    /// Mathematically, we sample two independent 3-dimensional Normal
    /// distributions in the tangent spaces of the left- and right-isoclinic
    /// components (together, a 6-dimensional Normal distribution in the tangent
    /// space of SO(4)) and lift them to the manifold with the exponential map.
    /// The result is a small displacement from the identity, with fast decay in
    /// the tails that makes large displacements unlikely. This is desirable for
    /// Monte Carlo, as large moves are very likely to be rejected.
    ///
    /// When used in a trial move, the displacement is combined with (centered
    /// on) a body's existing orientation; see [`LocalTrial::propose`].
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> DoubleVersor {
        let single_displacement = VersorDisplacement::from(self.std_dev);
        // Sample two independent small rotations
        let q_l = single_displacement.sample(rng);
        let q_r = single_displacement.sample(rng);

        // Combine them into a DoubleVersor
        (q_l, q_r).into()
    }
}

impl<B> LocalTrial<B> for Rotate<DoubleVersor>
where
    B: Orientation<Rotation = DoubleVersor>,
{
    /// Perturb a body's orientation by a random amount.
    #[inline]
    fn propose<R: Rng>(&self, rng: &mut R, body_properties: B) -> B {
        let mut trial = body_properties;
        let displacement = DoubleVersorDisplacement {
            std_dev: self.maximum_rotation.get(),
        };

        let delta_quat = displacement.sample(rng);
        *trial.orientation_mut() = delta_quat.combine(trial.orientation());

        trial
    }
}

impl Adjust for Rotate<DoubleVersor> {
    /// Change the maximum trial move size by the given scale factor.
    #[inline]
    fn adjust(&mut self, factor: PositiveReal) {
        self.maximum_rotation *= factor;

        if self.maximum_rotation.get() > PI / 2.0 {
            self.maximum_rotation = (PI / 2.0)
                .try_into()
                .expect("PI/2.0 should be a positive real");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::check;
    use hoomd_microstate::property::OrientedPoint;
    use hoomd_vector::{Cartesian, DoubleVersor, Metric};
    use rand::{SeedableRng, rngs::StdRng};
    use rstest::*;

    /// Number of trial moves to test.
    const N: usize = 262_144;

    /// Expected geodesic distance for small a:E[χ₆] = sqrt2*Γ(7/2)/Γ(3) = 15sqrt(2π)/16
    #[inline]
    fn expected_chi6_mean() -> f64 {
        15.0 * (2.0 * PI).sqrt() / 16.0
    }

    #[rstest]
    fn rotate(#[values(0.1, 0.5)] a: f64) {
        let mut rng = StdRng::seed_from_u64(1);
        let body = OrientedPoint {
            position: Cartesian::from([0.0, 0.0, 0.0]),
            orientation: DoubleVersor::default(),
        };
        let rotate = Rotate::with_maximum_rotation(
            a.try_into()
                .expect("hard-coded constant should be a positive real"),
        );

        let mut delta_thetas = Vec::with_capacity(N);
        for _ in 0..N {
            let trial = rotate.propose(&mut rng, body);
            delta_thetas.push(trial.orientation.distance(&body.orientation));
        }

        let mean = delta_thetas.iter().sum::<f64>() / N as f64;
        let expected = expected_chi6_mean() * a;
        assert!(
            (mean - expected).abs() < 0.01 * a, // Should be within a few percent
            "mean = {mean}, expected = {expected}, diff = {}",
            (mean - expected).abs(),
        );
    }

    #[rstest]
    fn rotate_mean_distance(#[values(1e-3, 0.1)] a: f64) {
        // Two independent versor displacements give distance:
        // sqrt(|sl|² + |sr|²) = a*\chi_6 for small displacements, so
        // E[distance] = a * sqrt(2) * (15*sqrt(pi)/8)/2 = a * 15sqrt(2π)/16.
        let expected = expected_chi6_mean() * a;

        let mut rng = StdRng::seed_from_u64(1);
        let body = OrientedPoint {
            position: Cartesian::from([0.0, 0.0, 0.0]),
            orientation: DoubleVersor::default(),
        };
        let rotate = Rotate::with_maximum_rotation(
            a.try_into()
                .expect("hard-coded constant should be a positive real"),
        );

        let sum: f64 = (0..N)
            .map(|_| {
                let trial = rotate.propose(&mut rng, body);
                trial.orientation.distance(&body.orientation)
            })
            .sum();

        let mean = sum / N as f64;
        assert!(
            (mean - expected).abs() < 0.001 * a,
            "mean = {mean}, expected = {expected}, diff = {}",
            (mean - expected).abs(),
        );
    }

    #[test]
    fn test_adjust() -> anyhow::Result<()> {
        let mut rotate = Rotate::<DoubleVersor>::with_maximum_rotation(0.5.try_into()?);

        rotate.adjust(2.0.try_into()?);
        check!(rotate.maximum_rotation().get() == 1.0);

        rotate.adjust(0.5.try_into()?);
        check!(rotate.maximum_rotation().get() == 0.5);

        rotate.adjust(10.0.try_into()?);
        check!(rotate.maximum_rotation().get() == PI / 2.0);

        Ok(())
    }
}

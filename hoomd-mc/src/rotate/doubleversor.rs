// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::f64::consts::PI;

use hoomd_microstate::property::Orientation;
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{DoubleVersor, Rotation};
use rand::{Rng, RngExt};
use rand_distr::Distribution;

use crate::{Adjust, LocalTrial, Rotate, rotate::versor::VersorDisplacement};

/// A normal distribution of random [`DoubleVersor`]s, centered on a mean with
/// some standard deviation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DoubleVersorDisplacement {
    /// The standard deviation of the normal distribution of quaternions around the mean.
    std_dev: f64,
}

impl From<f64> for DoubleVersorDisplacement {
    #[inline]
    fn from(value: f64) -> Self {
        Self { std_dev: value }
    }
}
impl Distribution<DoubleVersor> for DoubleVersorDisplacement {
    /// Sample a random [`DoubleVersor`] displacement from a provided mean.
    ///
    /// Mathematically, we sample from a 6-dimensional Normal distribution
    /// in the tangent space of SO(4), lift to the manifold, then rotate to center on
    /// the mean of the [`DoubleVersorDisplacement`]. The result is a small displacement
    /// from a rotational input, with fast decay in the tails that make large
    /// displacements unlikely. This is desirable for Monte Carlo, as large moves are
    /// very likely to be rejected.
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

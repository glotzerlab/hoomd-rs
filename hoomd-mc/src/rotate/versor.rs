// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Rotate for Versor

use rand::{Rng, distr::Distribution};
use rand_distr::Normal;
use std::f64::consts::PI;

use hoomd_microstate::property::Orientation;
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, InnerProduct, Quaternion, Rotation, Versor};

use crate::{Adjust, LocalTrial, Rotate};

/// A normal distribution of random Versors, centered on a mean with
/// some standard deviation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VersorDisplacement {
    /// The standard deviation of the normal distribution of quaternions around the mean.
    std_dev: f64,
}

impl From<f64> for VersorDisplacement {
    #[inline]
    fn from(value: f64) -> Self {
        Self { std_dev: value }
    }
}
impl Distribution<Versor> for VersorDisplacement {
    /// Sample a random [`Versor`] displacement from a provided mean.
    ///
    /// Mathematically, we sample from a 3-dimensional Normal distribution
    /// in the tangent space of SO(3), lift to the manifold, then rotate to center on
    /// the mean of the [`VersorDisplacement`]. The result is a small displacement from
    /// a quaternion input, with fast decay in the tails that make large displacements
    /// unlikely. This is desirable for Monte Carlo, as large moves are very likely to
    /// be rejected.
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Versor {
        // Based on Karney 2007: doi.org/10.1016/j.jmgm.2006.04.002
        const _SAFE_THETA: f64 = 1e-12;
        loop {
            // As in section 7, we select `s` from a 3D Gaussian distribution.
            let normal =
                Normal::new(0.0, self.std_dev).expect("Failed to create normal distribution.");

            let s = Cartesian::from([(); 3].map(|()| normal.sample(rng)));
            let theta = s.norm();

            // Reject moves with |s| > pi to ensure detailed balance. Otherwise, there
            // is a shorter path between the proposed move and the start (theta - pi),
            // which has a different rejection probability.
            if theta > PI {
                continue;
            }

            // Lift the normally distributed values to SO(3) with the exponential map
            let half_theta = 0.5 * theta;
            let w = half_theta.cos();

            let mut v_factor = half_theta.sin() / theta;
            if !v_factor.is_finite() {
                v_factor = 0.5;
            }

            let v = s * v_factor;

            // We are normalized by construction, so call the tuple initializer
            return (Quaternion::from([w, v[0], v[1], v[2]])).to_versor_unchecked();
        }
    }
}

impl<B> LocalTrial<B> for Rotate<Versor>
where
    B: Orientation<Rotation = Versor>,
{
    /// Perturb a body's orientation by a random amount.
    ///
    /// In three dimensions, we design this perturbation as a versor whose
    /// distribution is centered on the existing orientation and whose distribution is
    /// narrow. To do so, we sample from a 3-dimensional Normal distribution
    /// in the tangent space of SO(3), lift to the manifold, then rotate to center on
    /// the current orientation. The result is a small displacement from
    /// a quaternion input, with fast decay in the tails that make large displacements
    /// unlikely. This is desirable for Monte Carlo, as large moves are very likely to
    /// be rejected. This sampling obeys detailed balance.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_mc::{LocalTrial, Rotate};
    /// use hoomd_microstate::property::OrientedPoint;
    /// use hoomd_vector::{Cartesian, Versor};
    /// use rand::{Rng, SeedableRng, rngs::StdRng};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut rng = StdRng::seed_from_u64(1);
    /// let initial_orientation = Versor::default();
    /// let body_properties = OrientedPoint {
    ///     position: Cartesian::from([0.0, 0.0, 0.0]),
    ///     orientation: initial_orientation,
    /// };
    /// let rotate = Rotate::with_maximum_rotation(0.1.try_into()?);
    ///
    /// let new_body_properties = rotate.propose(&mut rng, body_properties);
    /// assert!(new_body_properties.orientation != initial_orientation);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn propose<R: Rng>(&self, rng: &mut R, body_properties: B) -> B {
        let mut trial = body_properties;
        let displacement = VersorDisplacement {
            std_dev: self.maximum_rotation.get(),
        };

        let delta_quat = displacement.sample(rng);
        *trial.orientation_mut() = delta_quat.combine(trial.orientation());

        trial
    }
}

impl Adjust for Rotate<Versor> {
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
    use hoomd_vector::{Cartesian, Versor};
    use rand::{SeedableRng, rngs::StdRng};
    use rstest::*;

    /// Number of trial moves to test.
    const N: usize = 65536;

    #[rstest]
    fn rotate(#[values(0.1, 1.0, 1.53)] a: f64) {
        // Ensure that `Rotate` proposes moves that rotate the body with a valid
        // range of maximum distances.

        let mut delta_thetas = Vec::with_capacity(N);

        let mut rng = StdRng::seed_from_u64(1);
        let body = OrientedPoint {
            position: Cartesian::from([0.0, 0.0, 0.0]),
            orientation: Versor::default(),
        };
        let rotate = Rotate::with_maximum_rotation(
            a.try_into()
                .expect("hard-coded constant should be a positive real"),
        );

        for _ in 0..N {
            let trial = rotate.propose(&mut rng, body);
            let delta_theta = trial.orientation.arc_distance(&body.orientation);
            delta_thetas.push(delta_theta);
        }

        // Validate our distribution's mean is relatively close to 0.75 * a.
        // Why this is the correct choice? I'm not sure. But most values of `a` result
        // in a distribution whose mean is near 0.75 * a
        let mean = (delta_thetas.iter().sum::<f64>() / N as f64).abs();
        assert!(
            (mean - (0.75 * a)).abs() < (0.1 * a),
            "{} !< {}",
            (mean - (0.75 * a)).abs(),
            (0.1 * a)
        );
    }

    #[test]
    fn test_adjust() -> anyhow::Result<()> {
        let mut rotate = Rotate::<Versor>::with_maximum_rotation(0.5.try_into()?);

        rotate.adjust(2.0.try_into()?);
        check!(rotate.maximum_rotation().get() == 1.0);

        rotate.adjust(0.5.try_into()?);
        check!(rotate.maximum_rotation().get() == 0.5);

        rotate.adjust(10.0.try_into()?);
        check!(rotate.maximum_rotation().get() == PI / 2.0);

        Ok(())
    }
}

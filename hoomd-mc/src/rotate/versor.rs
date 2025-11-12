// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Rotate for Versor

use std::f64::consts::PI;

use rand::{
    Rng,
    distr::Distribution,
};
use rand_distr::Normal;

use super::Rotate;
use crate::LocalTrial;
use hoomd_microstate::property::Orientation;
use hoomd_vector::{Cartesian, InnerProduct, Quaternion, Rotation, Versor};
/// A normal distribution of random Versors, centered on a mean with
/// some standard deviation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VersorDisplacement {
    /// The center of the distribution on the 3-Sphere.
    mean: Versor,
    /// The standard deviation of the normal distribution of quaternions around the mean.
    // normal: Normal<f64>,
    std_dev: f64,
}

impl From<(Versor, f64)> for VersorDisplacement {
    #[inline]
    fn from(value: (Versor, f64)) -> Self {
        Self {
            mean: value.0,
            std_dev: value.1,
        }
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

            // If theta is near 0, do not compute the value. TODO: detailed balance?
            let v_factor = if theta < _SAFE_THETA {
                0.5
            } else {
                half_theta.sin() / theta
            };

            let v = s * v_factor * half_theta.sin();

            // We are normalized by construction, so call the tuple initializer
            // Then, rotate by the current quaternion we displace from
            return (Quaternion::from([w, v[0], v[1], v[2]]))
                .to_versor_unchecked() // TODO: no need to re-normalize!
                .combine(&self.mean);
        }
    }
}

impl<B> LocalTrial<B> for Rotate<Versor>
where
    B: Orientation<Rotation = Versor>,
{
    /// Perturb a body's orientation by a random amount.
    ///
    /// In three dimensions, we design this perturbation as a quaternion whose
    /// distribution is centered on the existing orientation and whose distribition is
    /// narrow. To do so, we sample from a 3-dimensional Normal distribution
    /// in the tangent space of SO(3), lift to the manifold, then rotate to center on
    /// the mean of the [`VersorDisplacement`]. The result is a small displacement from
    /// a quaternion input, with fast decay in the tails that make large displacements
    /// unlikely. This is desirable for Monte Carlo, as large moves are very likely to
    /// be rejected.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_mc::{LocalTrial, Rotate};
    /// use hoomd_microstate::property::OrientedPoint;
    /// use hoomd_vector::{Angle, Cartesian};
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

        let a = self.maximum_rotation.get();

        let displacement = VersorDisplacement::from((*trial.orientation(), a));

        let delta_quat = displacement.sample(rng);
        // TODO: this is q * rand_q * q, do we need both rotations?
        *trial.orientation_mut() = trial.orientation().combine(&delta_quat);

        trial
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_microstate::property::OrientedPoint;
    use hoomd_vector::{Angle, Cartesian, Versor};
    use rand::{SeedableRng, rngs::StdRng};
    use rstest::*;

    /// Number of trial moves to test.
    const N: usize = 1024;

    // #[rstest]
    // fn rotate(#[values(0.1, 1.0)] a: f64) {
    //     // Ensure that `Rotate` proposes moves that rotate the body with a valid
    //     // range of maximum distances.

    //     let mut total = 0.0;
    //     let mut min_norm = f64::INFINITY;
    //     let mut max_norm = 0.0_f64;

    //     let mut rng = StdRng::seed_from_u64(1);
    //     let body = OrientedPoint {
    //         position: Cartesian::from([0.0, 0.0, 0.0]),
    //         orientation: Versor::default(),
    //     };
    //     let rotate = Rotate::with_maximum_rotation(
    //         a.try_into()
    //             .expect("hard-coded constant should be a positive real"),
    //     );

    //     for _ in 0..N {
    //         let trial = rotate.propose(&mut rng, body);

    //         let delta_theta = trial.orientation.theta - body.orientation.theta;
    //         total += delta_theta;
    //         min_norm = min_norm.min(delta_theta.abs());
    //         max_norm = max_norm.max(delta_theta.abs());
    //     }

    //     assert!(max_norm <= a);

    //     let average = total / N as f64;

    //     // Validate with appropriately loose tolerances to account for the small sample size.
    //     assert!(min_norm < a * 0.1);
    //     assert!(max_norm > a * 0.9);
    //     assert!(average.abs() < a * 0.1);
    // }
}

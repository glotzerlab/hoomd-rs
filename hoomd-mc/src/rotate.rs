// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Rotate
*/

use super::LocalTrial;
use hoomd_microstate::property::Orientation;
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Angle;

use rand::Rng;
use rand::distr::{Distribution, Uniform};

/** Change the orientation of a body by a small amount.

[`Rotate`] proposes local trial moves that rotate the orientation of a body
by a small amount. The [`maximum_rotation`] parameter sets the largest possible
rotation. For the 2D [`Angle`], [`maximum_rotation`] is measured in radians.
TODO: document 3d maximum.

# Example

```
use hoomd_mc::Rotate;
use hoomd_vector::Angle;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let a = 0.1;
let rotate = Rotate { maximum_rotation: a.try_into()? };
# Ok(())
# }
```
*/
pub struct Rotate {
    /// Limit the maximum rotation applied during a single trial move.
    pub maximum_rotation: PositiveReal,
}

impl<B> LocalTrial<B> for Rotate
where
    B: Orientation<Rotation = Angle>,
{
    /** Randomly translate a body's orientation.

    # Example

    ```
    use hoomd_mc::{LocalTrial, Rotate};
    use hoomd_microstate::property::OrientedPoint;
    use hoomd_vector::{Angle, Cartesian};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(1);
    let body_properties = OrientedPoint {
        position: Cartesian::from([0.0, 0.0]),
        orientation: Angle::from(0.0),
    };
    let rotate = Rotate { maximum_rotation: 0.1.try_into()? };

    let new_body_properties = rotate.propose(&mut rng, body_properties);
    assert!(new_body_properties.orientation.theta.abs() < 0.1);
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn propose<R: Rng>(&self, rng: &mut R, body_properties: B) -> B {
        let mut trial = body_properties;

        let a = self.maximum_rotation.get();
        let uniform = Uniform::new(-a, a).expect("a should be positive");

        let delta_theta = uniform.sample(rng);
        trial.orientation_mut().theta += delta_theta;

        trial
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_microstate::property::OrientedPoint;
    use hoomd_vector::{Angle, Cartesian};
    use rand::{SeedableRng, rngs::StdRng};
    use rstest::*;

    /// Number of trial moves to test.
    const N: usize = 1024;

    #[rstest]
    fn rotate(#[values(0.1, 1.0)] a: f64) {
        // Ensure that `Rotate` proposes moves that rotate the body with a valid
        // range of maximum distances.

        let mut total = 0.0;
        let mut min_norm = f64::INFINITY;
        let mut max_norm = 0.0_f64;

        let mut rng = StdRng::seed_from_u64(1);
        let body = OrientedPoint {
            position: Cartesian::from([0.0, 0.0]),
            orientation: Angle::default() };
        let rotate= Rotate{
            maximum_rotation: a
                .try_into()
                .expect("hard-coded constant should be a positive real"),
        };

        for _ in 0..N {
            let trial = rotate.propose(&mut rng, body);

            let delta_theta = trial.orientation.theta - body.orientation.theta;
            total += delta_theta;
            min_norm = min_norm.min(delta_theta.abs());
            max_norm = max_norm.max(delta_theta.abs());
        }

        assert!(max_norm <= a);

        let average = total / N as f64;

        // Validate with appropriately loose tolerances to account for the small sample size.
        assert!(min_norm < a * 0.1);
        assert!(max_norm > a * 0.9);
        assert!(average.abs() < a * 0.1);
    }
}

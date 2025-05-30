// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Translate
*/

use super::LocalTrial;
use hoomd_microstate::property::Position;
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Vector, distribution::Ball};

use rand::Rng;
use rand::distr::Distribution;

/** Move the position of a body by a small distance.

`Translate` proposes local trial moves that translate the position of a body
in space by a random vector up a given maximum length.

# Example

```
use hoomd_mc::Translate;
use hoomd_vector::Cartesian;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let d = 0.1;
let translate = Translate { maximum_distance: d.try_into()? };
# Ok(())
# }
```
*/
pub struct Translate {
    /// The maximum distance a body can be translated in one trial move.
    pub maximum_distance: PositiveReal,
}

impl<V, B> LocalTrial<B> for Translate
where
    B: Position<Vector = V>,
    V: Vector,
    Ball: Distribution<V>,
{
    /** Randomly translate a body's position.

    # Example

    ```
    use hoomd_mc::{LocalTrial, Translate};
    use hoomd_microstate::property::Point;
    use hoomd_vector::{Cartesian, Vector};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(1);
    let body_properties = Point::new(Cartesian::from([0.0, 0.0]));
    let d = 1.0;
    let translate = Translate { maximum_distance: d.try_into()? };

    let new_body_properties = translate.propose(&mut rng, body_properties);
    assert!(new_body_properties.position.norm() < 1.0);
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn propose<R: Rng>(&self, rng: &mut R, body_properties: B) -> B {
        let mut trial = body_properties;

        let ball = Ball {
            r: self.maximum_distance,
        };

        let delta_r = ball.sample(rng);
        *trial.position_mut() += delta_r;

        trial
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_microstate::property::Point;
    use hoomd_vector::{Cartesian, Vector};
    use rand::{SeedableRng, rngs::StdRng};
    use rstest::*;

    /// Number of trial moves to test.
    const N: usize = 1024;

    #[rstest]
    fn translate(#[values(0.1, 1.0)] d: f64) {
        // Ensure that `Translate` proposes moves that translate the body with a valid
        // range of maximum distances.

        let mut total = Cartesian::from([0.0, 0.0, 0.0]);
        let mut min_norm = f64::INFINITY;
        let mut max_norm = 0.0_f64;

        let mut rng = StdRng::seed_from_u64(1);
        let a = Point::new(Cartesian::from([1.0, -5.0, 2.5]));
        let translate = Translate {
            maximum_distance: d
                .try_into()
                .expect("hard-coded constant should be a positive real"),
        };

        for _ in 0..N {
            let b = translate.propose(&mut rng, a);

            let delta_r = b.position - a.position;
            total += delta_r;
            min_norm = min_norm.min(delta_r.norm());
            max_norm = max_norm.max(delta_r.norm());
        }

        assert!(max_norm <= d);

        let average = total / N as f64;

        // Validate with appropriately loose tolerances to account for the small sample size.
        assert!(min_norm < d * 0.1);
        assert!(max_norm > d * 0.9);
        for x in average.coordinates {
            assert!(x.abs() < d * 0.1);
        }
    }
}

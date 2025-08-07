// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Random distributions of vectors.
 */

use super::{Cartesian, InnerProduct};
use hoomd_utility::valid::PositiveReal;

use rand::Rng;
use rand::distr::{Distribution, Uniform};
use std::array;

/** A uniform distribution of all points inside or on a sphere with radius `r`.

# Example

```
use hoomd_vector::{Cartesian, distribution::Ball};
use rand::{distr::Distribution, Rng, rngs::StdRng, SeedableRng};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut rng = StdRng::seed_from_u64(1);

let r = 5.0;
let ball = Ball { radius: r.try_into()? };
let v: Cartesian<3> = ball.sample(&mut rng);
# Ok(())
# }
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ball {
    /// The radius of the ball *(\[length\])*.
    pub radius: PositiveReal,
}

impl<const N: usize> Distribution<Cartesian<N>> for Ball {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Cartesian<N> {
        let r = self.radius.get();

        let uniform = Uniform::new_inclusive(-r, r).expect("r should be a positive real value");

        loop {
            let v = Cartesian {
                coordinates: array::from_fn(|_| uniform.sample(rng)),
            };

            if v.norm_squared() < r * r {
                return v;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Number of random vectors to sample.
    const N: usize = 1024;
    use approx::assert_abs_diff_eq;
    use rand::{SeedableRng, rngs::StdRng};
    use rstest::*;

    #[rstest(
        r => [0.5, 1.0, 12.0])]
    fn ball(r: f64) {
        let ball = Ball {
            radius: r.try_into()
                .expect("hard-coded constant should be positive"),
        };
        let mut rng = StdRng::seed_from_u64(1);

        let mut total = Cartesian::default();
        for _ in 0..N {
            let v: Cartesian<3> = ball.sample(&mut rng);
            assert!(v.norm_squared() < r * r);

            total += v;
        }

        let average = total / N as f64;
        for x in average.coordinates {
            assert_abs_diff_eq!(x, 0.0, epsilon = r * 0.1);
        }
    }
}

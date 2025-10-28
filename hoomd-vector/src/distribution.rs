// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Random distributions of vectors.

use super::{Cartesian, InnerProduct};
use hoomd_utility::valid::PositiveReal;

use rand::{Rng, distr::Distribution};
use rand_distr::StandardNormal;

/// A uniform distribution of all points inside or on a sphere with radius `r`.
///
/// # Example
///
/// ```
/// use hoomd_vector::{Cartesian, distribution::Ball};
/// use rand::{Rng, SeedableRng, distr::Distribution, rngs::StdRng};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut rng = StdRng::seed_from_u64(1);
///
/// let r = 5.0;
/// let ball = Ball {
///     radius: r.try_into()?,
/// };
/// let v: Cartesian<3> = ball.sample(&mut rng);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Ball {
    /// The radius of the ball *(\[length\])*.
    pub radius: PositiveReal,
}

impl<const N: usize> Distribution<Cartesian<N>> for Ball {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Cartesian<N> {
        let r = self.radius.get();

        // Muller/Marsaglia 'normalized Gaussians' approach: see the following source.
        // https://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        let mut point = Cartesian {
            coordinates: std::array::from_fn::<_, N, _>(|_| rng.sample(StandardNormal)),
        };
        point /= point.norm();
        match N {
            2 => point * (r * rng.random::<f64>().sqrt()),
            3 => point * (r * rng.random::<f64>().cbrt()),
            _ => point * (r * rng.random::<f64>().powf(1.0 / N as f64)),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Number of random vectors to sample.
    const N: usize = 1024;
    use approxim::assert_abs_diff_eq;
    use rand::{SeedableRng, rngs::StdRng};
    use rstest::*;

    #[rstest(
        r => [0.5, 1.0, 12.0])]
    fn ball(r: f64) {
        let ball = Ball {
            radius: r
                .try_into()
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

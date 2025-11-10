// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Helpers that enable consistent use of random numbers throughout hoomd-rs.

use crate::SFC64Rng;
use rand::Rng;

/// Conveniently construct counter based random number generators.
///
/// Counter based random number generators produce a stream of random numbers
/// that is a reproducible function of a counter value. They are efficient to
/// use, even when only one or a few random numbers are needed. [`Counter`]
/// allows callers to conveniently construct RNGS that are independent, as well
/// as ones that produce identical values.
///
/// There are 3 required elements of each counter.
/// * `step` (8 bytes) is the current simulation step and ensures that random
///   number streams are not correlated from one simulation step to the next.
/// * `substep` (4 bytes) similarly ensures that different parts of the
///   algorithm that advance the simulation are not correlated within a single
///   step.
/// * `seed` (4 bytes) is a value that allows users to execute replicate
///   simulations that are identical except for the random numbers applied.
///
/// There is an additional 8-byte index. Generally, many simulation algorithms
/// will set this to particle indices so that RNG streams are independent from
/// one particle (or pair of particles) to the next. The [`indices`] method
/// treats the index as two 4-byte indices. To generate the same random numbers
/// (e.g. for use in a DPD thermostat) in independent threads, set the first
/// index to `min(i,j)` and the second to `max(i,j)`.
///
/// [`indices`]: Self::indices
///
/// # Performance
///
/// The current implementation uses [`SFC64Rng`], which generates
/// one 64-bit word at at time. Benchmarks show that executing
/// `Counter.new(...).make_rng()` and sampling values that fall in the first
/// batch runs at approximately 100 million operations per second (run `cargo
/// bench` to see the measured performance on your architecture).
///
/// [`SFC64`]: crate::SFC64Rng
///
/// # Example
///
/// ```
/// use hoomd_rand::Counter;
/// use rand::Rng;
///
/// # let step = 100_000;
/// # let substep = 10;
/// # let seed = 100;
/// let mut rng = Counter::new(step, substep, seed).make_rng();
///
/// let r: f64 = rng.random();
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Counter {
    /// The current simulation step.
    step: u64,
    /// The current substep.
    substep: u32,
    /// User-chosen random seed.
    seed: u32,
    /// The index.
    index: u64,
}

impl Counter {
    /// Construct a new counter.
    ///
    /// On constructions, the index defaults to 0.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_rand::Counter;
    ///
    /// let step = 100_000;
    /// let substep = 10;
    /// let seed = 100;
    ///
    /// let counter = Counter::new(step, substep, seed);
    /// ```
    #[must_use]
    #[inline]
    pub fn new(step: u64, substep: u32, seed: u32) -> Self {
        Counter {
            step,
            substep,
            seed,
            index: 0,
        }
    }

    /// Set the index with two 4-byte values.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_rand::Counter;
    ///
    /// let step = 100_000;
    /// let substep = 10;
    /// let seed = 100;
    /// let i = 12;
    /// let j = 152;
    ///
    /// let counter = Counter::new(step, substep, seed).indices(i.max(j), i.min(j));
    /// ```
    #[must_use]
    #[inline]
    pub fn indices(mut self, a: u32, b: u32) -> Self {
        self.index = u64::from(a) << 32 | u64::from(b);
        self
    }

    /// Set the index.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_rand::Counter;
    ///
    /// let step = 100_000;
    /// let substep = 10;
    /// let seed = 100;
    /// let index = 1_000_000_000_000u64;
    ///
    /// let counter = Counter::new(step, substep, seed).index(index);
    /// ```
    #[must_use]
    #[inline]
    pub fn index(mut self, index: u64) -> Self {
        self.index = index;
        self
    }

    /// Seed a [`Rng`] with the counter.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_rand::Counter;
    /// use rand::Rng;
    ///
    /// let step = 100_000;
    /// let substep = 10;
    /// let seed = 100;
    ///
    /// let mut rng = Counter::new(step, substep, seed).make_rng();
    ///
    /// let r: f64 = rng.random();
    /// ```
    #[must_use]
    #[inline]
    pub fn make_rng(self) -> impl Rng + use<> {
        SFC64Rng::initialize(self.step, u64::from(self.substep) << 32 | u64::from(self.seed),
            self.index, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::check;

    /// Number of stream elements to sample.
    const N: usize = 256;

    #[test]
    fn independent() {
        // This test is not exhaustive, but serves as a quick check that the
        // different elements in Counter indeed produce different random
        // number streams.

        let counters = [
            Counter::new(0, 0, 0),
            Counter::new(1, 0, 0),
            Counter::new(0, 1, 0),
            Counter::new(0, 0, 1),
            Counter::new(0, 0, 0).indices(1, 0),
            Counter::new(0, 0, 0).indices(0, 1),
            Counter::new(0, 0, 0).index(2),
        ];

        for (i, counter_i) in counters.iter().enumerate() {
            for (j, counter_j) in counters.iter().enumerate() {
                let mut rng_i = counter_i.clone().make_rng();
                let values_i = core::array::from_fn::<_, N, _>(|_| rng_i.random::<f64>());

                let mut rng_j = counter_j.clone().make_rng();
                let values_j = core::array::from_fn::<_, N, _>(|_| rng_j.random::<f64>());

                if i == j {
                    check!(values_i == values_j);
                } else {
                    check!(values_i != values_j);
                }
            }
        }
    }
}

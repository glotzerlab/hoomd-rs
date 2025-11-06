// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Helpers that enable consistent use of random numbers throughout hoomd-rs.

// use chacha20::ChaCha8Rng;
use hoomd_rand::SFC64Rng;
use rand::{Rng, SeedableRng};

/// Conveniently construct counter based random number generators.
///
/// Counter based random number generators produce a stream of random numbers
/// that is a reproducible function of a counter value. They are efficient to
/// use, even when only one or a few random numbers are needed. [`Counter`]
/// allows callers to conveniently construct RNGS that are independent, as well
/// as ones that produce identical values.
///
/// There are 3 required elements of each counter.
/// * `step` is the current simulation step and ensures that random number
///   streams are not correlated from one simulation step to the next.
/// * `substep` similarly ensures that different parts of the algorithm that
///   advance the simulation are not correlated within a single step.
/// * `seed` is a value that allows users to execute replicate simulations that
///   are identical except for the random numbers applied.
///
/// There are two optional indices. Generally, many simulation algorithms will
/// set these to particle indices so that RNG streams are independent from one
/// particle (or pair of particles) to the next. To generate the same random
/// numbers (e.g. for use in a DPD thermostat) in independent threads, set the
/// first index to `min(i,j)` and the second to `max(i,j)`.
///
/// There are also three general purpose counters. Simulation algorithms can
/// use these as needed when many independent streams are needed per particle,
/// per substep.
///
/// # Performance
///
/// The current implementation uses `SFC64`, which generates one 64-bit word at at time.
/// Benchmarks show that executing `Counter.new(...).make_rng()`
/// and sampling values that fall in the first batch runs at approximately
/// 100 million operations per second (run `cargo bench` to see the measured
/// performance on your architecture). If performance is somehow an issue, `AESRand` may
/// be slightly faster on some platforms.
///
/// # Example
///
/// ```
/// use hoomd_utility::random::Counter;
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
    /// First index.
    index_a: u32,
    /// Second index.
    index_b: u32,
}

impl Counter {
    /// Construct a new counter.
    ///
    /// On constructions, all indices and counters default to 0.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_utility::random::Counter;
    ///
    /// # let step = 100_000;
    /// # let substep = 10;
    /// # let seed = 100;
    /// let counter = Counter::new(step, substep, seed);
    /// ```
    #[must_use]
    #[inline]
    pub fn new(step: u64, substep: u32, seed: u32) -> Self {
        Counter {
            step,
            substep,
            seed,
            index_a: 0,
            index_b: 0,
        }
    }

    /// Set indices.
    ///
    /// There are only 2 indices. Calling `indices` (or [`index`](Self::index)) more
    /// than once will overwrite existing values.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_utility::random::Counter;
    ///
    /// # let step = 100_000;
    /// # let substep = 10;
    /// # let seed = 100;
    /// # let i = 12;
    /// # let j = 152;
    /// let counter = Counter::new(step, substep, seed).indices(i.max(j), i.min(j));
    /// ```
    #[must_use]
    #[inline]
    pub fn indices(mut self, a: u32, b: u32) -> Self {
        self.index_a = a;
        self.index_b = b;
        self
    }

    /// Set indices from a 64-bit integer, splitting to fill both items.
    ///
    /// There are only 2 indices. Calling `indices` (or [`index`](Self::index)) more
    /// than once will overwrite existing values.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_utility::random::Counter;
    ///
    /// # let step = 100_000;
    /// # let substep = 10;
    /// # let seed = 100;
    /// let long_int = 1_000_000_000_000u64;
    /// let counter = Counter::new(step, substep, seed).indices_from_u64(long_int);
    /// ```
    #[must_use]
    #[inline]
    pub fn indices_from_u64(mut self, combined_index: u64) -> Self {
        self.index_a = (combined_index >> 32) as u32;
        self.index_b = (combined_index & 0xFFFF_FFFF) as u32;
        self
    }

    /// Set one index.
    ///
    /// Equivalent to `indices(a, 0)`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_utility::random::Counter;
    ///
    /// # let step = 100_000;
    /// # let substep = 10;
    /// # let seed = 100;
    /// # let i = 12;
    /// let counter = Counter::new(step, substep, seed).index(i);
    /// ```
    #[must_use]
    #[inline]
    pub fn index(mut self, a: u32) -> Self {
        self.index_a = a;
        self
    }

    /// Seed a [`Rng`] with the counter.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_utility::random::Counter;
    /// use rand::Rng;
    ///
    /// # let step = 100_000;
    /// # let substep = 10;
    /// # let seed = 100;
    /// let mut rng = Counter::new(step, substep, seed).make_rng();
    ///
    /// let r: f64 = rng.random();
    /// ```
    #[must_use]
    #[inline]
    pub fn make_rng(self) -> impl Rng + use<> {
        let mut seed = [0u8; 24];
        seed[..8].copy_from_slice(&self.step.to_le_bytes());
        seed[8..12].copy_from_slice(&self.substep.to_le_bytes());
        seed[12..16].copy_from_slice(&self.seed.to_le_bytes());

        seed[16..20].copy_from_slice(&self.index_a.to_le_bytes());
        seed[20..].copy_from_slice(&self.index_b.to_le_bytes());

        SFC64Rng::from_seed(seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

                assert_eq!(values_i == values_j, i == j);
            }
        }
    }
}

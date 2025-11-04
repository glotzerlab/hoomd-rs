// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Helpers that enable consistent use of random numbers throughout hoomd-rs.

// use chacha20::ChaCha8Rng;
use rand::{Rng, SeedableRng};
use threefry::{Squares64, Squares128, ThreeFry2x64Rng, Tyche4x32Rng, XSM64Rng};

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
/// The current implementation uses `ChaCha8`. `ChaCha` generates random numbers
/// in 64 word batches. Benchmarks show that `Counter.new(...).make_rng()`
/// and sampling values that fall in the first batch runs at approximately
/// 10 million operations per second (run `cargo bench` to see the measured
/// performance on your architecture). This is slow enough that serial
/// algorithms should make ONE random generator and sample from it repeatedly
/// (instead of e.g. making one random generator per particle). Parallel
/// algorithms by necessity must make many different random generators from
/// different counters. Should `ChaCha` prove to be a bottleneck in practice,
/// this implementation may be switched to a more efficient RNG.
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
#[expect(
    clippy::struct_field_names,
    reason = "The counters must be distinguishable from the indices."
)]
pub struct Counter {
    /// The current simulation step.
    step: u64,
    /// The current substep.
    substep: u32,
    /// User-chosen random seed.
    seed: u32,
    /// First index.
    index_a: u64,
    /// Second index.
    index_b: u64,
    /// First counter.
    counter_a: u32,
    /// Second counter.
    counter_b: u32,
    /// Third counter.
    counter_c: u32,
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
            counter_a: 0,
            counter_b: 0,
            counter_c: 0,
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
    pub fn indices(mut self, a: u64, b: u64) -> Self {
        self.index_a = a;
        self.index_b = b;
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
    pub fn index(mut self, a: u64) -> Self {
        self.index_a = a;
        self
    }

    /// Set counters.
    ///
    /// There are only 3 counters. Calling `counters` (or
    /// [`counter`](Self::counter)) more than once will overwrite existing values.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_utility::random::Counter;
    ///
    /// # let step = 100_000;
    /// # let substep = 10;
    /// # let seed = 100;
    /// # let a = 12;
    /// # let b = 54;
    /// # let c = 62;
    /// let counter = Counter::new(step, substep, seed).counters(a, b, c);
    /// ```
    #[must_use]
    #[inline]
    pub fn counters(mut self, a: u32, b: u32, c: u32) -> Self {
        self.counter_a = a;
        self.counter_b = b;
        self.counter_c = c;
        self
    }

    /// Set one counter.
    ///
    /// Equivalent to `counters(a, 0, 0)`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_utility::random::Counter;
    ///
    /// # let step = 100_000;
    /// # let substep = 10;
    /// # let seed = 100;
    /// let counter = Counter::new(step, substep, seed).counter(1);
    /// ```
    #[must_use]
    #[inline]
    pub fn counter(mut self, a: u32) -> Self {
        self.counter_a = a;
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
        // ChaCha separates the input into a seed and a stream id. As a hash,
        // it shouldn't matter where different parts of the counter are placed.
        // However, best practice in the encryption community is to put the
        // fastest-varying parts in the stream id. For Counter, this means
        // placing the first index and counter in the stream id and everything
        // else in the seed.

        // let mut stream = [0u8; 32];
        // stream[..8].copy_from_slice(&self.index_a.to_le_bytes());
        // stream[8..16].copy_from_slice(&self.index_b.to_le_bytes());
        // stream[16..24].copy_from_slice(&self.step.to_le_bytes());
        // stream[24..28].copy_from_slice(&self.substep.to_le_bytes());
        // stream[28..32].copy_from_slice(&self.seed.to_le_bytes());
        let mut stream = [0u8; 16];
        stream[..8].copy_from_slice(&self.index_a.to_le_bytes());
        stream[8..16].copy_from_slice(&self.index_b.to_le_bytes());
        // stream[8..12].copy_from_slice(&self.counter_a.to_le_bytes());

        let mut seed = [0u8; 16];
        // let mut seed = [0u8; 32]; // ChaCha and Xoshiro256
        seed[..8].copy_from_slice(&(self.step).to_le_bytes());
        seed[8..12].copy_from_slice(&self.substep.to_le_bytes());
        seed[12..16].copy_from_slice(&(self.seed).to_le_bytes());

        // seed[16..24].copy_from_slice(&self.index_b.to_le_bytes());
        // seed[28..].copy_from_slice(&self.counter_c.to_le_bytes());

        // let mut rng = ThreeFry2x64Rng::<13>::from_seed(seed);

        // let mut rng = chacha20::ChaCha8Rng::from_seed(seed);
        // rng
        // rng.set_stream(stream);
        // rng
        // Squares64::seed_from_u64(0x16d7358fe8d9a17b)
        // Squares128::from_seed(seed)
        // rand_xoshiro::Xoshiro256Plus::from_seed(seed)
        Tyche4x32Rng::from_seed(seed)
        // XSM64Rng::from_seed(seed)
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

        let counters = vec![
            Counter::new(0, 0, 0),
            Counter::new(1, 0, 0),
            Counter::new(0, 1, 0),
            Counter::new(0, 0, 1),
            Counter::new(0, 0, 0).indices(1, 0),
            Counter::new(0, 0, 0).indices(0, 1),
            Counter::new(0, 0, 0).index(2),
            Counter::new(0, 0, 0).counters(1, 0, 0),
            Counter::new(0, 0, 0).counters(0, 1, 0),
            Counter::new(0, 0, 0).counters(0, 0, 1),
            Counter::new(0, 0, 0).counter(2),
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

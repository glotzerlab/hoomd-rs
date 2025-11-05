// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use rand::{RngCore, SeedableRng};
use rand_core::impls;

use crate::backends::{read_u64_le_unchecked, rotl_u64};

/// The "Small Fast Chaotic" PRNG, originally designed by Chris Doty-Humphrey.
///
/// This PRNG holds 256 bits of state (including a 64-bit counter) and generates 64 bits
/// of output with each step. The minimum cycle length is $`2^64`$, and the expected
/// period is $`~2^255`$. Independent seeds are guaranteed to not collide within the
/// first $`2^64`$ steps, although the actual time to collision may be much longer.
pub struct SFC64Rng {
    /// The internal state of the PRNG.
    state: [u64; 3],
    /// The current step of the PRNG. This value is incremented with each output, but
    /// can be set independently without issues.
    counter: u64,
}

impl SFC64Rng {
    /// Set up the PRNG from four u64 values, discarding the first 12 outputs to avoid
    /// correlations between similar seeds.
    #[inline]
    pub(crate) fn initialize(a: u64, b: u64, c: u64, counter: u64) -> Self {
        let mut rng = Self {
            state: [a, b, c],
            counter,
        };
        (0..12).for_each(|_| {
            rng.step();
        });
        rng
    }
    /// Increment the RNG forward, returning the generated result.
    #[inline]
    fn step(&mut self) -> u64 {
        self.counter += 1;
        let out = self.state[0] + self.state[1] + self.counter;
        self.state[0] = self.state[1] ^ (self.state[1] >> 9);
        self.state[1] = self.state[2] + (self.state[2] << 3);
        self.state[2] = rotl_u64(self.state[2], 21) + out;
        out
    }

    /// Initialize the PRNG from 192 bits of state and a u64 counter.
    #[inline]
    #[must_use]
    pub fn from_state_and_counter(state: [u64; 3], counter: u64) -> Self {
        Self::initialize(state[0], state[1], state[2], counter)
    }
}

impl RngCore for SFC64Rng {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.step()
    }
    #[inline]
    fn next_u32(&mut self) -> u32 {
        impls::next_u32_via_fill(self)
    }
    #[inline]
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        impls::fill_bytes_via_next(self, dst);
    }
}
impl SeedableRng for SFC64Rng {
    type Seed = [u8; 32];
    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        Self::initialize(
            read_u64_le_unchecked(seed, 0..8),
            read_u64_le_unchecked(seed, 8..16),
            read_u64_le_unchecked(seed, 16..24),
            read_u64_le_unchecked(seed, 24..32),
        )
    }
    #[inline]
    fn seed_from_u64(state: u64) -> Self {
        Self::initialize(0, 0, 0, state)
    }
}

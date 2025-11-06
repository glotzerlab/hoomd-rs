// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![cfg(feature = "extras")]
use rand::SeedableRng;
use rand_core::{
    RngCore,
    block::{BlockRng64, BlockRngCore},
    impls,
};

/// Generate a u32 from a Squares64 PRNG.
fn squares32(seed: u64, counter: &mut u64) -> u32 {
    let mut x = seed.wrapping_mul(*counter);
    let y = x;
    let z = y.wrapping_add(seed);
    // Round 1
    x = (x.wrapping_mul(x).wrapping_add(y)).rotate_left(32);
    // Round 2
    x = (x.wrapping_mul(x).wrapping_add(z)).rotate_left(32);
    // Round 3
    x = (x.wrapping_mul(x).wrapping_add(y)).rotate_left(32);
    // Round 4
    *counter += 1;
    ((x.wrapping_mul(x).wrapping_add(z)) >> 32) as u32
}

/// A 64-bit variant of the counter-based Middle Square Weyl Sequence PRNG.
pub struct Squares64 {
    /// The seed for the PRNG.
    seed: u64,
    /// The current counter state.
    counter: u64,
}

impl RngCore for Squares64 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.seed.wrapping_mul(self.counter);
        let y = x;
        let z = y.wrapping_add(self.seed);
        // Round 1
        x = (x.wrapping_mul(x).wrapping_add(y)).rotate_left(32);
        // Round 2
        x = (x.wrapping_mul(x).wrapping_add(z)).rotate_left(32);
        // Round 3
        x = (x.wrapping_mul(x).wrapping_add(y)).rotate_left(32);
        // Round 4
        let t = x.wrapping_mul(x).wrapping_add(z);
        x = t.rotate_left(32);
        self.counter += 1;
        // Round 5
        t ^ ((x.wrapping_mul(x).wrapping_add(y)) >> 32)
    }
    #[inline]
    fn next_u32(&mut self) -> u32 {
        squares32(self.seed, &mut self.counter)
    }
    #[inline]
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        impls::fill_bytes_via_next(self, dst);
    }
}

impl SeedableRng for Squares64 {
    type Seed = [u8; 8];
    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        Self {
            seed: u64::from_le_bytes(seed),
            counter: 0,
        }
    }
    #[inline]
    fn seed_from_u64(state: u64) -> Self {
        Self {
            seed: state,
            counter: 0,
        }
    }
}

/// A 128-bit variant of the counter-based Middle Square Weyl Sequence PRNG.
struct Squares128Core {
    /// The seed for the PRNG.
    seed: u128,
    /// The current state of the counter.
    counter: u128,
}

impl BlockRngCore for Squares128Core {
    type Item = u64;
    type Results = [u64; 2];
    #[expect(clippy::cast_possible_truncation, reason = "Truncation is expected.")]
    #[inline]
    fn generate(&mut self, results: &mut Self::Results) {
        // NOTE: corsika rotates right, but it's by half the width of the value so the
        // direction does not matter. We rotate left to match Squares64
        //
        // NOTE: corsika also suggests only 3-4 rounds are needed for Squares128, but
        // it's possible this could be a 'too big to fail' situation. We maintain
        // the same 5 rounds as the Squares64 implementation, with the final xor at
        // the end.
        let mut x = self.seed.wrapping_mul(self.counter);
        let y = x;
        let z = y.wrapping_add(self.seed);
        // Round 1a
        // NOTE: corsika rotates right, but because it's by half the width the direction
        // does not matter. We rotate left to match the original authors.
        x = (x.wrapping_mul(x).wrapping_add(y)).rotate_left(64);
        // Round 2
        x = (x.wrapping_mul(x).wrapping_add(z)).rotate_left(64);
        // Round 3
        x = (x.wrapping_mul(x).wrapping_add(y)).rotate_left(64);
        // Round 4
        let t = x.wrapping_mul(x).wrapping_add(z);
        x = t.rotate_left(64);
        self.counter += 1;
        // Round 5
        let final_val = t ^ (x.wrapping_mul(x).wrapping_add(y));
        results[0] = (final_val >> 64) as u64;
        results[1] = final_val as u64;
    }
}
/// A 128-bit variant of the counter-based Middle Square Weyl Sequence PRNG.
pub struct Squares128(BlockRng64<Squares128Core>);

impl SeedableRng for Squares128 {
    type Seed = [u8; 16];
    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        // Step is u64, substep and seed are u32.
        // First word must be odd in 1-15, so 8 choices
        // Words 9-15 (6 words, 12 bytes) are random from 1-15 ODD | EVEN, with n != n-1
        // See openrand commented out code
        let seed = (u128::from_le_bytes(seed) + 1) * 0xc58e_fd15_4ce3_2f6d;
        Self(BlockRng64::new(Squares128Core { seed, counter: 0 }))
    }
    #[inline]
    fn seed_from_u64(state: u64) -> Self {
        let seed = (u128::from(state) + 1) * 0xc58e_fd15_4ce3_2f6d;
        Self(BlockRng64::new(Squares128Core { seed, counter: 0 }))
    }
}

impl RngCore for Squares128 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }
    #[inline]
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        self.0.fill_bytes(dst);
    }
}

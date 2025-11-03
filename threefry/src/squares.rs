// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use rand::SeedableRng;
use rand_core::{RngCore, impls};
///..
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

/// .
pub struct Squares64 {
    seed: u64,
    counter: u64,
}
pub struct Squares128 {
    seed: u128,
    counter: u128,
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
        let t = (x.wrapping_mul(x).wrapping_add(z)).rotate_left(32);
        x = t;
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
impl RngCore for Squares128 {
    #[expect(clippy::cast_possible_truncation, reason = "Truncation is intended.")]
    #[inline]
    fn next_u64(&mut self) -> u64 {
        /*
         * NOTE: corsika rotates right, but it's by half the width of the value so the
         * direction does not matter. We rotate left to match Squares64
         *
         * NOTE: corsika also suggests only 3-4 rounds are needed for Squares128, but
         * it's possible this could be a 'too big to fail' situation. We maintain
         * the same 5 rounds as the Squares64 implementation, with the final xor at
         * the end.
         */
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
        let t = (x.wrapping_mul(x).wrapping_add(z)).rotate_left(64);
        x = t;
        self.counter += 1;
        // Round 5
        (t ^ (x.wrapping_mul(x).wrapping_add(y))) as u64
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
impl SeedableRng for Squares128 {
    type Seed = [u8; 16];
    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        Self {
            seed: u128::from_le_bytes(seed), // TODO:
            counter: 0,
        }
    }
    #[inline]
    fn seed_from_u64(state: u64) -> Self {
        Self {
            seed: u128::from(state),
            counter: 0,
        }
    }
}

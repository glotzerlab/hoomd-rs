// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Asdf.

/// asdf
pub(crate) mod backends;

use backends::C240;
use rand::SeedableRng;
use rand_core::{
    RngCore,
    block::{BlockRng64, BlockRngCore},
};
/// Key schedule for ``ThreeFry2x64``.
const ROTATION_2X64: [u32; 8] = [16, 42, 12, 31, 16, 32, 24, 21];

/// TODO: unsafe if N < slice size
#[inline]
fn read_u64_le_unchecked<const N: usize>(stream: [u8; N], range: std::ops::Range<usize>) -> u64 {
    u64::from_le_bytes(stream[range].try_into().unwrap_or_else(|_| unreachable!()))
}

/// TODO.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ThreeFry2x64Core {
    /// .
    seed: [u64; 3],
    /// .
    counter: [u64; 2],
}
const TF2X64_ROUNDS: usize = 20;
impl BlockRngCore for ThreeFry2x64Core {
    type Item = u64;
    type Results = [u64; 2];

    #[inline]
    fn generate(&mut self, results: &mut Self::Results) {
        // for d in 0..TF2X64_ROUNDS {
        //     if (d % 4) == 0 {
        //         let s = d / 4;
        //         self.counter[0] = self.counter[0].wrapping_add(self.seed[s % 3]);
        //         self.counter[1] = self.counter[1].wrapping_add(self.seed[(s + 1) % 3] + s as u64);
        //     }
        //     backends::mix(&mut self.counter, ROTATION_2X64[d % 8]);
        // }

        (0..TF2X64_ROUNDS).for_each(|d| {
            if d % 4 == 0 {
                let s = d / 4;
                self.counter[0] = self.counter[0].wrapping_add(self.seed[s % 3]);
                self.counter[1] = self.counter[1].wrapping_add(self.seed[(s + 1) % 3] + s as u64);
            }
            backends::mix(&mut self.counter, ROTATION_2X64[d % 8]);
        });
        self.counter[0] = self.counter[0].wrapping_add(self.seed[(TF2X64_ROUNDS / 4) % 3]);
        self.counter[1] = self.counter[1]
            .wrapping_add(self.seed[((TF2X64_ROUNDS / 4) + 1) % 3] + (TF2X64_ROUNDS / 4) as u64);
        *results = self.counter;
    }
}

impl SeedableRng for ThreeFry2x64Rng {
    type Seed = [u8; 16];
    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        let (k0, k1) = (
            read_u64_le_unchecked(seed, 0..8),
            read_u64_le_unchecked(seed, 8..16),
        );
        Self(BlockRng64::new(ThreeFry2x64Core {
            seed: [k0, k1, C240 ^ k0 ^ k1],
            counter: [0u64, 0u64],
        }))
    }
    #[inline]
    fn seed_from_u64(state: u64) -> Self {
        Self(BlockRng64::new(ThreeFry2x64Core {
            seed: [0, state, C240 ^ state],
            counter: [0u64, 0u64],
        }))
    }
}

/// TODO.
pub struct ThreeFry2x64Rng(BlockRng64<ThreeFry2x64Core>);
impl ThreeFry2x64Rng {
    /// TODO
    #[inline]
    pub fn set_stream(&mut self, stream: [u8; 16]) {
        self.0.core.counter[0] = read_u64_le_unchecked(stream, 0..8);
        self.0.core.counter[1] = read_u64_le_unchecked(stream, 8..16);
    }
    ///.
    #[inline]
    pub fn set_stream_from_u64(&mut self, stream: u64) {
        self.0.core.counter = [0, stream];
    }
}
impl RngCore for ThreeFry2x64Rng {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }
    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill_bytes(dest);
    }
}

fn main() {
    let mut x = ThreeFry2x64Rng::seed_from_u64(0);
    x.set_stream_from_u64(0);
    assert_eq!(x.next_u64(), 14_030_652_003_081_164_901);
    assert_eq!(x.next_u64(), 8_034_964_082_011_408_461);
}

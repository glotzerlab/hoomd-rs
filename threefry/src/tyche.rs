// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Tyche random number generator.

use crate::backends::{read_u32_le_unchecked, read_u64_le_unchecked};
use rand::SeedableRng;
use rand_core::{
    block::{BlockRng, BlockRngCore},
    RngCore,
};
use wide::u32x4;

#[inline]
pub(crate) fn rotl_u32x4(x: u32x4, d: u32) -> u32x4 {
    (x << d) | (x >> (32 - d))
}

/// ⌊2**32⌋/π
const D_INIT: u32 = 1_367_130_551;

/// Core of the Tyche4x32 RNG.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tyche4x32Core {
    a: u32x4,
    b: u32x4,
    c: u32x4,
    d: u32x4,
}

impl Tyche4x32Core {
    /// The ChaCha quarter round.
    #[inline]
    fn chacha_quarterround(&mut self) {
        self.a += self.b;
        self.d = rotl_u32x4(self.d ^ self.a, 16);
        self.c += self.d;
        self.b = rotl_u32x4(self.b ^ self.c, 12);
        self.a += self.b;
        self.d = rotl_u32x4(self.d ^ self.a, 8);
        self.c += self.d;
        self.b = rotl_u32x4(self.b ^ self.c, 7);
    }
}

impl BlockRngCore for Tyche4x32Core {
    type Item = u32;
    type Results = [u32; 4];

    #[inline]
    fn generate(&mut self, results: &mut Self::Results) {
        self.chacha_quarterround();
        *results = self.b.into();
    }
}

/// A Tyche4x32 random number generator.
pub struct Tyche4x32Rng(BlockRng<Tyche4x32Core>);

impl SeedableRng for Tyche4x32Rng {
    type Seed = [u8; 16];

    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        let step = read_u64_le_unchecked(seed, 0..8);
        let substep = read_u32_le_unchecked(seed, 8..12);
        let key = read_u32_le_unchecked(seed, 12..16);

        let mut core = Tyche4x32Core {
            a: u32x4::from((step >> 32) as u32),
            b: u32x4::from((step & 0xFFFF_FFFF) as u32),
            c: key.into(),
            d: u32x4::from(D_INIT ^ substep),
        };

        // Decorrelate the initial state.
        for _ in 0..20 {
            core.chacha_quarterround();
        }

        Self(BlockRng::new(core))
    }

    #[inline]
    fn seed_from_u64(state: u64) -> Self {
        // `state` is used as `step`. `substep` is 0, `key` is a constant.
        let mut core = Tyche4x32Core {
            a: u32x4::from((state >> 32) as u32),
            b: u32x4::from((state & 0xFFFF_FFFF) as u32),
            c: 0xAAAA_AAAA.into(),
            d: u32x4::from(D_INIT),
        };
        for _ in 0..20 {
            core.chacha_quarterround();
        }
        Self(BlockRng::new(core))
    }
}

impl RngCore for Tyche4x32Rng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.0.next_u32()
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.0.fill_bytes(dest)
    }
}

impl Tyche4x32Rng {
    /// Generate the next block of 4 `u32`s.
    #[inline]
    pub fn next_block(&mut self) -> [u32; 4] {
        let mut results = [0u32; 4];
        self.0.core.generate(&mut results);
        results
    }

    /// Set the stream for the RNG.
    #[inline]
    pub fn set_stream(&mut self, stream: u64, substream: u32) {
        let core = &mut self.0.core;
        core.a = u32x4::from((stream >> 32) as u32);
        core.b = u32x4::from((stream & 0xFFFF_FFFF) as u32);
        core.d = u32x4::from(D_INIT ^ substream);
        // `c` (the key) is unchanged.
        // Decorrelate again.
        for _ in 0..20 {
            core.chacha_quarterround();
        }
    }
}

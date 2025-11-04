use rand::{RngCore, SeedableRng};
use rand_core::{impls, le::read_u64_into};

use crate::backends::{read_u64_le_unchecked, rotl_u64};

pub struct SFC64Rng {
    state: [u64; 4],
}

impl SFC64Rng {
    #[inline]
    fn initialize(a: u64, b: u64, c: u64, d: u64) -> Self {
        let mut rng = Self {
            state: [a, b, c, d],
        };
        (0..12).for_each(|_| {
            rng.step();
        });
        rng
    }
    /// Increment the RNG forward, returning the generated result.
    #[inline]
    fn step(&mut self) -> u64 {
        self.state[3] += 1;
        let tmp = self.state[0] + self.state[1] + self.state[3];
        self.state[0] = self.state[1] ^ (self.state[1] >> 9);
        self.state[1] = self.state[2] + (self.state[2] << 3);
        self.state[2] = rotl_u64(self.state[2], 21) + tmp;
        tmp
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

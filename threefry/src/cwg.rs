use rand::{RngCore, SeedableRng};
use rand_core::impls;

use crate::backends::read_u64_le_unchecked;

pub struct CWG64Rng {
    /// State variable 0
    x: u64,
    /// The
    a: u64,

    /// Current value of our position in the Weyl sequence
    weyl_ctr: u64,
    /// Index of our current stream. Must be odd
    stream_index: u64, // must be odd
}

impl CWG64Rng {
    /// Increment the generator
    #[inline]
    fn step(&mut self) {
        self.a = self.a.wrapping_add(self.x);
        self.weyl_ctr = self.weyl_ctr.wrapping_add(self.stream_index);

        self.x = (self.x >> 1).wrapping_mul(self.a | 1) ^ self.weyl_ctr;
    }
    /// Initialize the generator, randomizing the state
    #[inline]
    fn initialize(x: u64, a: u64, weyl_ctr: u64, stream_index: u64) -> Self {
        let mut rng = Self {
            x,
            a,
            weyl_ctr,
            stream_index,
        };
        (0..48).for_each(|_| {
            rng.step();
        });
        rng
    }
}

impl RngCore for CWG64Rng {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.step();
        self.a >> 48 ^ self.x
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
impl SeedableRng for CWG64Rng {
    type Seed = [u8; 32];
    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        let stream_index = read_u64_le_unchecked(seed, 24..32);
        Self::initialize(
            read_u64_le_unchecked(seed, 0..8),
            read_u64_le_unchecked(seed, 8..16),
            read_u64_le_unchecked(seed, 16..24),
            if stream_index.is_multiple_of(2) {
                stream_index + 1
            } else {
                stream_index
            },
        )
    }
    #[inline]
    fn seed_from_u64(state: u64) -> Self {
        Self::initialize(
            0,
            0,
            0,
            if state.is_multiple_of(2) {
                state + 1
            } else {
                state
            },
        )
    }
}

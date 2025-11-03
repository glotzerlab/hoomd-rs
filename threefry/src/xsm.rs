use rand::{RngCore, SeedableRng};
use rand_core::impls;

use crate::backends::read_u64_le_unchecked;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XSM64Rng {
    seed: [u64; 2],    //state
    counter: [u64; 2], // adder
}

impl XSM64Rng {
    /// .
    fn step_forward(&mut self) {
        // 'Increment' the counter by one step of a LCG.
        // S_n+1 = S_n * (1 + 2**64) + A = S_n + A + (S_n & u64::MAX) << 64
        let tmp = self.seed[1].wrapping_add(self.counter[0]);
        self.seed[1] = self.seed[1].wrapping_add(self.counter[1]);

        self.seed[0] = self.seed[0]
            .wrapping_add(tmp)
            .wrapping_add(u64::from(self.seed[1] < self.counter[1]));
    }

    /// Process seeds such that no two seeds are closer than 2**127 apart on a cycle.
    fn from_u64_pair(seed_low: u64, seed_high: u64) -> Self {
        let lcg_adder_low = (seed_low << 1) | 1;

        //every bit of seed except the highest bit gets used in the adder
        let lcg_adder_high = (seed_low >> 63) | (seed_high << 1);

        let lcg_low = lcg_adder_low;
        let lcg_high = lcg_adder_high ^ ((seed_high >> 63) << 63); //and the highest bit of seed is used to determine which end of the cycle we start at
        let seed = [lcg_high, lcg_low];
        let counter = [lcg_adder_high, lcg_adder_low];

        let mut xsm = XSM64Rng { seed, counter };
        xsm.step_forward();
        xsm
    }
}

impl SeedableRng for XSM64Rng {
    type Seed = [u8; 16];
    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        XSM64Rng::from_u64_pair(
            read_u64_le_unchecked(seed, 0..8),
            read_u64_le_unchecked(seed, 8..16),
        )
    }
    #[inline]
    fn seed_from_u64(state: u64) -> Self {
        XSM64Rng::from_u64_pair(state, 0)
    }
}
impl RngCore for XSM64Rng {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        const K: u64 = 0xA3E_C647_6593_59AC;
        let mut tmp = self.seed[0] ^ (self.seed[0].wrapping_add(self.seed[1]).rotate_left(16));
        tmp ^= tmp.wrapping_add(self.counter[0]).rotate_left(40);
        tmp = tmp.wrapping_mul(K);
        self.step_forward();
        tmp = tmp ^ tmp.wrapping_add(self.seed[0]).rotate_left(32);
        tmp = tmp.wrapping_mul(K);
        tmp ^= tmp >> 32;
        tmp
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

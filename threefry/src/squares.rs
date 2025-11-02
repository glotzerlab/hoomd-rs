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
pub struct Squares {
    seed: u64,
    counter: u64,
}
impl RngCore for Squares {
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
impl SeedableRng for Squares {
    type Seed = [u8; 8];
    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        Squares {
            seed: u64::from_le_bytes(seed),
            counter: 0,
        }
    }
    #[inline]
    fn seed_from_u64(state: u64) -> Self {
        Squares {
            seed: state,
            counter: 0,
        }
    }
}

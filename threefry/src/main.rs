//! Asdf.

/// asdf
pub(crate) mod backends;
use std::ops::DerefMut;

use backends::C240;
pub use cipher::{
    IvSizeUser, KeyIvInit, KeySizeUser, StreamCipherCoreWrapper,
    array::Array,
    consts::{U12, U32, U64},
};
use rand::SeedableRng;
use rand_core::{
    RngCore,
    block::{BlockRng64, BlockRngCore},
};

/// TODO: unsafe if N < slice size
fn read_u64_le_unchecked<const N: usize>(stream: [u8; N], range: std::ops::Range<usize>) -> u64 {
    u64::from_le_bytes(stream[range].try_into().unwrap_or_else(|_| unreachable!()))
}

// use crate::{ChaChaCore, R8, R12, R20, Rounds, variants::Ietf};

// /// Key type used by all ChaCha variants.
// pub type Key = Array<u8, U32>;

// /// Nonce type used by ChaCha variants.
// pub type Nonce = Array<u8, U12>;

// /// ChaCha8 stream cipher (reduced-round variant of [`ChaCha20`] with 8 rounds)
// pub type ChaCha8 = StreamCipherCoreWrapper<ChaChaCore<R8, Ietf>>;

/// TODO.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ThreeFry2x64Core {
    /// .
    seed: (u64, u64, u64),
    /// .
    counter: (u64, u64),
}

impl ThreeFry2x64Core {
    /// TODO
    fn set_stream(&mut self, stream: [u8; 16]) {
        self.counter.0 = read_u64_le_unchecked(stream, 0..8);
        self.counter.1 = read_u64_le_unchecked(stream, 8..16);
    }
}

/// TODO;
pub struct ThreeFry2x64Rng {
    /// .
    pub core: BlockRng64<ThreeFry2x64Core>,
}
impl From<ThreeFry2x64Core> for ThreeFry2x64Rng {
    fn from(core: ThreeFry2x64Core) -> Self {
        Self { core }
    }
}

impl BlockRngCore for ThreeFry2x64Core {
    type Item = u64;
    // Results is a 128-bit (16-byte) block, viewed as 2 x u64.
    type Results = [u64; 2];

    fn generate(&mut self, results: &mut Self::Results) {
        unimplemented!()
    }
}

impl RngCore for ThreeFry2x64Rng {
    #[expect(clippy::cast_possible_truncation, reason = "Truncation is intended")]
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32 // TODO: efficient
    }
    fn next_u64(&mut self) -> u64 {
        backends::step()
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        rand_core::impls::fill_bytes_via_next(self, dest);
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl SeedableRng for ThreeFry2x64Rng {
    type Seed = [u8; 16];
    fn from_seed(seed: Self::Seed) -> Self {
        let (k0, k1) = (
            read_u64_le_unchecked(seed, 0..8),
            read_u64_le_unchecked(seed, 8..16),
        );
        Self::from(ThreeFry2x64Core {
            seed: (k0, k1, C240 ^ k0 ^ k1),
            counter: (0u64, 0u64),
        })
    }
    fn seed_from_u64(state: u64) -> Self {
        Self::from(ThreeFry2x64Core {
            seed: (0, state, C240 ^ state),
            counter: (0u64, 0u64),
        })
    }
}

fn main() {
    let mut x = ThreeFry2x64Rng::seed_from_u64(0);
    println!("Hello, world: {}", x.next_u64());
}

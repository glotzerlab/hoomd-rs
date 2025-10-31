//! Asdf.
use num_traits::{PrimInt, Unsigned};

/// Key schedule for ``ThreeFry2x64``.
const ROTATION_2X64: [u32; 8] = [16, 42, 12, 31, 16, 32, 24, 21];

/// Key schedule constant C240.
///
/// This increases the randomness of outputs when keys are mostly zero. C240 is the AES
/// encryption of the plaintext "240" (in decimal), under the all 0 AES256 key.
/// In the Random123 library, this constant is named ``SKEIN_KS_PARITY64``
const C240: u64 = 0x1_BD1_1BD_AA9_FC1_A22;

/// Rotate a 64 bit unsigned integer left by `r bits`
fn rotl(x: u64, d: u32) -> u64 {
    (x << d) | (x >> (u64::BITS - d))
}
/// TODO.
fn mix<const N: usize>(state: &mut (u64, u64), round_key: u32) {
    state.0 += state.1;
    state.1 = rotl(state.1, round_key) ^ state.0;
}

/// TODO: unsafe if N < slice size
fn read_u64_le_unchecked<const N: usize>(stream: [u8; N], range: std::ops::Range<usize>) -> u64 {
    u64::from_le_bytes(stream[range].try_into().unwrap_or_else(|_| unreachable!()))
}

/// TODO.
struct ThreeFry2x64 {
    seed: (u64, u64, u64),
    counter: (u64, u64),
}

impl ThreeFry2x64 {
    /// TODO.
    fn from_seed(seed: [u8; 16]) -> ThreeFry2x64 {
        let (k0, k1) = (
            read_u64_le_unchecked(seed, 0..8),
            read_u64_le_unchecked(seed, 8..16),
        );
        ThreeFry2x64 {
            seed: (k0, k1, C240 ^ k0 ^ k1),
            counter: (0u64, 0u64),
        }
    }
    /// TODO:
    fn seed_from_u64(seed: u64) -> ThreeFry2x64 {
        ThreeFry2x64 {
            seed: (0, seed, C240 ^ seed),
            counter: (0u64, 0u64),
        }
    }
    /// TODO
    fn set_stream(&mut self, stream: [u8; 16]) {
        self.counter.0 = read_u64_le_unchecked(stream, 0..8);
        self.counter.1 = read_u64_le_unchecked(stream, 8..16);
    }
}

fn main() {
    println!("Hello, world!");
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/// Key schedule constant C240.
///
/// This increases the randomness of outputs when keys are mostly zero. C240 is the AES
/// encryption of the plaintext "240" (in decimal), under the all 0 AES256 key.
/// In the Random123 library, this constant is named ``SKEIN_KS_PARITY64``
pub const C240: u64 = 0x1_BD1_1BD_AA9_FC1_A22;

/// Rotate a 64 bit unsigned integer left by `r bits`
fn rotl(x: u64, d: u32) -> u64 {
    (x << d) | (x >> (u64::BITS - d))
}
/// TODO.
pub(crate) fn mix(state: &mut (u64, u64), round_key: u32) {
    state.0 = state.0.wrapping_add(state.1);
    state.1 = rotl(state.1, round_key) ^ state.0;
}

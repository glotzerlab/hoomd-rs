// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/// Key schedule constant C240.
///
/// This increases the randomness of outputs when keys are mostly zero. C240 is the AES
/// encryption of the plaintext "240" (in decimal), under the all 0 AES256 key.
/// In the Random123 library, this constant is named ``SKEIN_KS_PARITY64``
pub const C240: u64 = 0x1_BD1_1BD_AA9_FC1_A22;

/// Rotate a 64 bit unsigned integer left by `r bits`
#[inline]
pub(crate) fn rotl_u64(x: u64, d: u32) -> u64 {
    (x << d) | (x >> (64 - d))
}
/// Rotate a 32 bit unsigned integer left by `r bits`
#[inline]
pub(crate) fn rotl_u32(x: u32, d: u32) -> u32 {
    (x << d) | (x >> (32 - d))
}
/// Mixing function for the ThreeFry2x64 PRNG.
#[inline]
pub(crate) fn mix2x64(state: &mut [u64; 2], round_key: u32) {
    state[0] = state[0].wrapping_add(state[1]);
    state[1] = rotl_u64(state[1], round_key) ^ state[0];
}
/// Mixing function for the ThreeFry4x64 PRNG.
#[inline]
pub(crate) fn mix4x64(state: &mut [u64; 4], rotations: (u32, u32), round: usize) {
    if round.is_multiple_of(2) {
        state[0] = state[0].wrapping_add(state[1]);
        state[1] = rotl_u64(state[1], rotations.0) ^ state[0];
        state[2] = state[2].wrapping_add(state[3]);
        state[3] = rotl_u64(state[3], rotations.1) ^ state[2];
    } else {
        state[0] = state[0].wrapping_add(state[3]);
        state[3] = rotl_u64(state[3], rotations.0) ^ state[0];
        state[2] = state[2].wrapping_add(state[1]);
        state[1] = rotl_u64(state[1], rotations.1) ^ state[2];
    }
}
/// Read a little-endian u64 from a byte array.
///
/// # Panics
///
/// This function will panic if the provided range is not 8 bytes long or is
/// out of bounds for the array.
#[inline]
pub(crate) fn read_u64_le_unchecked<const N: usize>(
    stream: [u8; N],
    range: std::ops::Range<usize>,
) -> u64 {
    u64::from_le_bytes(stream[range].try_into().unwrap_or_else(|_| unreachable!()))
}
/// Read a little-endian u32 from a byte array.
///
/// # Panics
///
/// This function will panic if the provided range is not 4 bytes long or is
/// out of bounds for the array.
#[inline]
pub(crate) fn read_u32_le_unchecked<const N: usize>(
    stream: [u8; N],
    range: std::ops::Range<usize>,
) -> u32 {
    u32::from_le_bytes(stream[range].try_into().unwrap_or_else(|_| unreachable!()))
}

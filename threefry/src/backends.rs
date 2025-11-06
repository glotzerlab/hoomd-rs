// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/// Rotate a 64 bit unsigned integer left by `r bits`
#[inline]
pub(crate) fn rotl_u64(x: u64, d: u32) -> u64 {
    (x << d) | (x >> (64 - d))
}
/// Mixing function for the `ThreeFry2x64` PRNG.
#[inline]
pub(crate) fn mix2x64(state: &mut [u64; 2], round_key: u32) {
    state[0] = state[0].wrapping_add(state[1]);
    state[1] = state[1].rotate_left(round_key) ^ state[0];
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
    u64::from_le_bytes(
        stream[range]
            .try_into()
            .expect("Cannot read bytes into target!"),
    )
}

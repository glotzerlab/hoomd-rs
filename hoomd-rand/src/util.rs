// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

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

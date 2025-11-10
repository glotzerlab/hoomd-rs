// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/// Read the next 8 bytes from a slice as a u64
///
/// # Panics
///
/// This function will panic when the slice is shorter than 8 bytes.
pub(crate) fn read_ne_u64(input: &mut &[u8]) -> u64 {
    let (int_bytes, rest) = input.split_at(size_of::<u64>());
    *input = rest;
    u64::from_ne_bytes(
        int_bytes
            .try_into()
            .expect("input slice should be a multiple of 8 bytes"),
    )
}

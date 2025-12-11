// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Ensure that values are in well-defined ranges.

mod positive_real;
mod open_unit_interval_number;

pub use positive_real::PositiveReal;
pub use open_unit_interval_number::OpenUnitIntervalNumber;

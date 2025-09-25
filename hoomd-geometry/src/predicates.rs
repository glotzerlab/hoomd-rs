// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use hoomd_vector::Cartesian;

pub(crate) fn distance_point_to_line_squared<const N: usize>(
    point: &Cartesian<N>,
    segment_start: &Cartesian<N>,
    segment_direction: &Cartesian<N>,
) -> f64 {
    0.0
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! hoomd-rs benchmarking framework.

mod benchmark;
pub mod mc;
mod place;

pub use benchmark::Benchmark;
pub use place::place_hard_hyperspheres;

/// Track the amount of work a benchmark completed.
///
/// Not all methods perform the same amount of effort per step. `Effort`
/// allows benchmarks to return comparable results.
pub trait Effort {
    /// Units of the effort.
    fn units() -> String;
    /// The amount of effort.
    fn effort(&self) -> f64;
}

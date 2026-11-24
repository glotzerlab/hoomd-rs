// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! hoomd-rs benchmarking framework.
//!
//! Requires the `parallel` feature, enabled by default.

#[cfg(feature = "parallel")]
mod benchmark;
#[cfg(feature = "parallel")]
pub mod mc;
#[cfg(feature = "parallel")]
pub mod md;
#[cfg(feature = "parallel")]
pub mod place;

#[cfg(feature = "parallel")]
pub use benchmark::Benchmark;

/// Track the amount of work a benchmark completed.
///
/// Not all methods perform the same amount of effort per step. `Effort`
/// allows benchmarks to return comparable results.
#[cfg(feature = "parallel")]
pub trait Effort {
    /// Units of the effort.
    fn units() -> String;
    /// The amount of effort.
    fn effort(&self) -> f64;
}

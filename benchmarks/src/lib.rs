// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! hoomd-rs benchmarking framework.

mod benchmark;
pub mod mc;
mod place;

pub use benchmark::Benchmark;
pub use place::place_hard_hyperspheres;

pub trait Effort {
    fn units() -> String;
    fn effort(&self) -> f64;
}

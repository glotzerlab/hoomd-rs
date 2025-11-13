// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Test rayon overhead with various workloads

use std::fmt;

use rand::Rng;
use rayon::prelude::*;

use hoomd_simulation::Simulation;
use hoomd_rand::Counter;

/// Benchmark Rayon overhead with a compute-only workload that generates many random numbers
pub struct ThreadedRng {
    /// Number of random number streams
    n: usize,

    /// Number of random floats to generate per stream
    m: usize,

    /// The current step of the "simulation"
    step: u64,

    /// Results
    results: Vec<f64>,
}

impl ThreadedRng {
    #[inline]
    fn body(m: usize, step: u64, i: usize) -> f64 {
        let mut rng = Counter::new(step, 0, 0).index(i as u64).make_rng();
        let mut value = 0.0f64;
        for _ in 0..m {
            value += rng.random::<f64>();
        }
        value
    }

    pub fn new(n: usize, m: usize) -> Self {
        Self {
            n,
            m,
            step: 0,
            results: Vec::new(),
        }   
    }
}

impl Simulation for ThreadedRng {
    #[inline]
    fn advance(&mut self) -> anyhow::Result<()> {

        (0..self.n).into_par_iter()
            .map(|i| Self::body(self.m, self.step, i))
            .collect_into_vec(&mut self.results);

        // self.results.clear();
        // for i in 0..self.n {
        //     self.results.push(Self::body(self.m, self.step, i));
        // }

        self.step += 1;
        Ok(())
    }

    #[inline]
    fn step(&self) -> u64 {
        self.step
    }
}

impl fmt::Display for ThreadedRng {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ThreadedRng")
    }
}

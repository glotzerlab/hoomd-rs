// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement benchmark timing methods.

use std::time::{Duration, Instant};

use log::{debug, info};

use hoomd_simulation::Simulation;

use crate::Effort;

pub struct Benchmark {
    /// Time to warm up.
    pub warmup_time: Duration,

    /// Time to benchmark.
    pub benchmark_time: Duration,
}

const INFO_TIME: Duration = Duration::new(0, 500_000_000);

impl Default for Benchmark {
    fn default() -> Self {
        Self {
            warmup_time: Duration::new(2, 0),
            benchmark_time: Duration::new(4, 0),
        }
    }
}

impl Benchmark {
    /// Measure the average run time of a simulation.
    ///
    /// Return the average time per step (in milliseconds) when the simulation is successful.
    ///
    /// # Errors
    ///
    /// Returns any error reported by `simulation.advance`.
    pub fn measure<S>(&self, simulation: &mut S) -> anyhow::Result<f64>
    where
        S: Simulation + Effort,
    {
        let total_time = Instant::now();

        let mut start_time = total_time;
        let mut start_effort = simulation.effort();
        let mut warmup = true;

        let mut last_info_instant = start_time;
        let mut last_info_effort = simulation.effort();
        loop {
            simulation.advance()?;

            let time = Instant::now();
            let chunk_duration = time.duration_since(last_info_instant);
            if chunk_duration >= INFO_TIME {
                let run_time = chunk_duration.as_secs_f64();
                let effort = simulation.effort() - last_info_effort;

                debug!("{} {}/s", effort / run_time, S::units());

                last_info_effort = simulation.effort();
                last_info_instant = time;
            }
            if !warmup && time.duration_since(total_time) >= self.warmup_time {
                warmup = false;
                start_time = time;
                start_effort = simulation.effort();
            }
            if time.duration_since(total_time) >= self.warmup_time + self.benchmark_time {
                break;
            }
        }

        let run_time = start_time.elapsed().as_secs_f64();
        let effort = simulation.effort() - start_effort;
        let seconds_per_effort = run_time / effort;

        info!(
            "Average: {} {}/s",
            effort / start_time.elapsed().as_secs_f64(),
            S::units()
        );

        Ok(seconds_per_effort)
    }
}

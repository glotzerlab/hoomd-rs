// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement benchmark timing methods.

use std::time::{Duration, Instant};

use log::{debug, info};

use hoomd_simulation::Simulation;

pub struct Benchmark {
    /// Time to warm up.
    warmup_time: Duration,

    /// Time to benchmark.
    benchmark_time: Duration,
}

const INFO_TIME: Duration = Duration::new(0, 500_000_000);

impl Default for Benchmark {
    fn default() -> Self {
        Self {
            warmup_time: Duration::new(1,0),
            benchmark_time: Duration::new(2,0),
        }
    }
}

impl Benchmark {

    /// Benchmark a simulation
    ///
    /// Return the average time per step (in milliseconds) when the simulation is successful.
    ///
    /// # Errors
    ///
    /// Returns any error reported by `simulation.advance`.    
    pub fn benchmark_one<S>(&self, simulation: &mut S) -> anyhow::Result<f64>
    where S: Simulation
    {
        let total_time = Instant::now();

        let mut start_time = total_time;
        let mut start_step = simulation.step();
        let mut warmup = true;

        let mut last_info_instant = start_time;
        let mut last_info_step = simulation.step();
        loop {
            simulation.advance()?;

            let time = Instant::now();
            let chunk_duration = time.duration_since(last_info_instant);
            if chunk_duration >= INFO_TIME {
                let run_time = chunk_duration.as_secs_f64();
                let steps = simulation.step() - last_info_step;
    
                debug!("{} steps/s", steps as f64 / run_time);

                last_info_step = simulation.step();
                last_info_instant = time;
            }
            if !warmup && time.duration_since(total_time) >= self.warmup_time {
                warmup = false;
                start_time = time;
                start_step = simulation.step();
            }
            if time.duration_since(total_time) >= self.warmup_time + self.benchmark_time {
                break;
            }
        }

        let run_time = start_time.elapsed().as_secs_f64() / 1e-3;
        let steps = simulation.step() - start_step;
        let milliseconds_per_step = run_time / steps as f64;

        info!("Average: {} steps/s", steps as f64 / start_time.elapsed().as_secs_f64());

        Ok(milliseconds_per_step)
    }
}

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Command line tool that benchmarks hoomd-rs performance.

use std::{fmt, fs::File, sync::Arc, time::Duration};

use anyhow::anyhow;
use clap::Parser;
use clap_verbosity_flag::{InfoLevel, Verbosity, log::LevelFilter};
use log::{info, trace};
use parquet::{
    file::{properties::WriterProperties, writer::SerializedFileWriter},
    record::RecordWriter,
};
use parquet_derive::ParquetRecordWriter;

use benchmarks::{Benchmark, Effort, mc};
use hoomd_microstate::{SiteKey, property::OrientedPoint};
use hoomd_simulation::Simulation;
use hoomd_spatial::VecCell;
use hoomd_vector::{Angle, Cartesian, Versor};

use rayon::ThreadPoolBuilder;
use wildmatch::WildMatch;

/// Command line options.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Options {
    /// Execute benchmarks that match a wildcard pattern.
    #[arg(short, long, value_name = "pattern", default_value_t=String::from("*"))]
    benchmarks: String,

    /// The smallest system size to benchmark.
    #[arg(short, long, default_value_t = 4096)]
    n_min: usize,

    /// Largest system size to benchmark (inclusive, defaults to the smallest).
    #[arg(long)]
    n_max: Option<usize>,

    /// The smallest number of threads to benchmark on.
    #[arg(short, long, default_value_t = 1)]
    threads_min: usize,

    /// Largest number of threads to benchmark on (inclusive, defaults to the smallest).
    #[arg(long)]
    threads_max: Option<usize>,

    /// Use `ParallelSweep` even when running on 1 thread.
    #[arg(long)]
    parallel_sweep: bool,

    /// Writes the parquet file
    #[arg(long, value_name = "filename")]
    parquet: Option<String>,

    /// Time to warm up each benchmark (seconds)
    #[arg(long, default_value_t = 2.0)]
    warmup_time: f64,

    /// Time to run each benchmark (seconds)
    #[arg(long, default_value_t = 4.0)]
    benchmark_time: f64,

    /// Verbosity.
    #[command(flatten)]
    verbose: Verbosity<InfoLevel>,
}

/// A single entry in the performance table.
#[derive(ParquetRecordWriter)]
struct Performance {
    /// Name of the benchmark.
    benchmark: String,

    /// The units of the benchmark result.
    unit: String,

    /// Number of bodies or sites in the benchmark.
    n: usize,

    /// Number of threads the benchmark executed on.
    threads: usize,

    /// Time (in seconds) for each unit of effort.
    time_per_operation: f64,
}

/// Execute a single benchmark.
fn execute<S>(
    simulation: &mut S,
    benchmark: &Benchmark,
    name: &str,
    n: usize,
    threads: usize,
) -> anyhow::Result<Performance>
where
    S: Simulation + fmt::Display + Effort,
{
    info!("{name}: {n} bodies");
    info!("{threads} threads");
    let time_per_operation = benchmark.measure(simulation)?;
    trace!("{simulation}");

    Ok(Performance {
        benchmark: name.to_string(),
        unit: "step".to_string(),
        n,
        threads,
        time_per_operation,
    })
}

/// Execute all benchmarks that match a given glob.
fn execute_matching(
    results: &mut Vec<Performance>,
    n: usize,
    threads: usize,
    options: &Options,
) -> anyhow::Result<()> {
    let benchmark_matcher = WildMatch::new(&options.benchmarks);
    let number_density = 0.8;
    let benchmark = Benchmark {
        warmup_time: Duration::from_secs_f64(options.warmup_time),
        benchmark_time: Duration::from_secs_f64(options.benchmark_time),
    };

    let needs_microstate_2d = benchmark_matcher.matches("mc_2d_sphere")
        || benchmark_matcher.matches("mc_2d_lennard_jones")
        || benchmark_matcher.matches("mc_2d_hexagon");

    let needs_microstate_3d = benchmark_matcher.matches("mc_3d_sphere")
        || benchmark_matcher.matches("mc_3d_lennard_jones")
        || benchmark_matcher.matches("mc_3d_octahedron");

    let maybe_microstate_2d = if needs_microstate_2d {
        Some(benchmarks::place_hard_hyperspheres::<
            OrientedPoint<Cartesian<2>, Angle>,
            OrientedPoint<Cartesian<2>, Angle>,
            2,
        >(n, number_density)?)
    } else {
        None
    };

    let name = "mc_2d_sphere";
    if benchmark_matcher.matches(name) {
        let microstate_2d = &maybe_microstate_2d
            .as_ref()
            .expect("microstate_2d should be initialized");
        let mut simulation = mc::HardSphereSim::<2, VecCell<SiteKey, 2>>::new(
            n,
            options.parallel_sweep || threads > 1,
        )?;
        results.push(execute(&mut simulation, &benchmark, name, n, threads)?);
    }

    let name = "mc_2d_lennard_jones";
    if benchmark_matcher.matches(name) {
        let microstate_2d = &maybe_microstate_2d
            .as_ref()
            .expect("microstate_2d should be initialized");
        let mut simulation = mc::LennardJones::<2, VecCell<SiteKey, 2>>::new(
            microstate_2d,
            options.parallel_sweep || threads > 1,
        )?;
        results.push(execute(&mut simulation, &benchmark, name, n, threads)?);
    }

    let name = "mc_2d_hexagon";
    if benchmark_matcher.matches(name) {
        let microstate_2d = &maybe_microstate_2d
            .as_ref()
            .expect("microstate_2d should be initialized");
        let mut simulation = mc::RegularPolygon::<VecCell<SiteKey, 2>>::new(
            microstate_2d,
            options.parallel_sweep || threads > 1,
        )?;
        results.push(execute(&mut simulation, &benchmark, name, n, threads)?);
    }

    let maybe_microstate_3d = if needs_microstate_3d {
        Some(benchmarks::place_hard_hyperspheres::<
            OrientedPoint<Cartesian<3>, Versor>,
            OrientedPoint<Cartesian<3>, Versor>,
            3,
        >(n, number_density)?)
    } else {
        None
    };

    let name = "mc_3d_sphere";
    if benchmark_matcher.matches(name) {
        let mut simulation = mc::HardSphereSim::<3, VecCell<SiteKey, 3>>::new(
            n,
            options.parallel_sweep || threads > 1,
        )?;
        results.push(execute(&mut simulation, &benchmark, name, n, threads)?);
    }

    let name = "mc_3d_lennard_jones";
    if benchmark_matcher.matches(name) {
        let microstate_3d = &maybe_microstate_3d
            .as_ref()
            .expect("microstate_3d should be initialized");
        let mut simulation = mc::LennardJones::<3, VecCell<SiteKey, 3>>::new(
            microstate_3d,
            options.parallel_sweep || threads > 1,
        )?;
        results.push(execute(&mut simulation, &benchmark, name, n, threads)?);
    }

    let name = "mc_3d_octahedron";
    if benchmark_matcher.matches(name) {
        let microstate_3d = &maybe_microstate_3d
            .as_ref()
            .expect("microstate_3d should be initialized");
        let mut simulation = mc::Octahedron::<VecCell<SiteKey, 3>>::new(
            microstate_3d,
            options.parallel_sweep || threads > 1,
        )?;
        results.push(execute(&mut simulation, &benchmark, name, n, threads)?);
    }

    Ok(())
}

#[expect(clippy::print_stdout, reason = "benchmark should provide output")]
fn main() -> anyhow::Result<()> {
    let options = Options::parse();

    let log_level = match options.verbose.log_level_filter() {
        LevelFilter::Off => "off",
        LevelFilter::Error => "error",
        LevelFilter::Warn => "warn",

        LevelFilter::Info => "info",
        LevelFilter::Debug => "debug",
        LevelFilter::Trace => "trace",
    };

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .format_timestamp(None)
        .init();

    let mut results = Vec::new();

    let mut threads = options.threads_min;
    let threads_max = options.threads_max.unwrap_or(options.threads_min);

    if threads_max != threads && options.parquet.is_none() {
        return Err(anyhow!(
            "Parquet output is required when threads_min ({threads}) is not equal to threads_max ({threads_max})"
        ));
    }

    loop {
        let mut n = options.n_min;
        let n_max = options.n_max.unwrap_or(options.n_min);

        if n_max != n && options.parquet.is_none() {
            return Err(anyhow!(
                "Parquet output is required when n_min ({n}) is not equal to n_max ({n_max})"
            ));
        }

        loop {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("the thread pool should be valid");
            pool.install(|| execute_matching(&mut results, n, threads, &options))?;

            n *= 2;
            if n > n_max {
                break;
            }
        }

        threads *= 2;
        if threads > threads_max {
            break;
        }
    }

    if let Some(filename) = options.parquet {
        let schema = results.as_slice().schema()?;
        let props = Arc::new(WriterProperties::builder().build());
        let file = File::create(filename)?;
        let mut writer = SerializedFileWriter::new(file, schema, props)?;
        let mut row_group = writer.next_row_group()?;

        results.as_slice().write_to_row_group(&mut row_group)?;

        row_group.close()?;
        writer.close()?;
    } else {
        for result in results {
            let operations_per_second = 1.0 / result.time_per_operation;
            let name = result.benchmark;
            let unit = result.unit;
            println!("{name:20}: {operations_per_second:9.3} {unit}s/s");
        }
    }

    Ok(())
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::{collections::BTreeMap, fmt, fs::File, io::Write, time::Duration};

use anyhow::anyhow;
use clap::{Parser, ValueEnum};
use clap_verbosity_flag::{InfoLevel, Verbosity, log::LevelFilter};
use log::{info, trace};
use serde::Serialize;

use benchmarks::{Benchmark, mc};
use hoomd_microstate::{SiteKey, property::OrientedPoint};
use hoomd_simulation::Simulation;
use hoomd_spatial::{AllPairs, HashCell, VecCell};
use hoomd_vector::{Angle, Cartesian, Versor};

use wildmatch::WildMatch;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum SpatialData {
    /// Cell list, backed by Vec storage.
    VecCell,
    /// Cell list, backed by HashMap storage.
    HashCell,
    /// Loop over all sites.
    AllPairs,
}

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

    /// The spatial data structure to use.
    #[arg(short, long, default_value_t = SpatialData::VecCell, value_enum)]
    spatial_data: SpatialData,

    /// Write the json files {benchmark}-{suffix}.json
    #[arg(long, default_value_t = false)]
    json: bool,

    /// Time to warm up each benchmark (seconds)
    #[arg(long, default_value_t = 2.0)]
    warmup_time: f64,

    /// Time to run each benchmark (seconds)
    #[arg(long, default_value_t = 4.0)]
    benchmark_time: f64,

    /// Suffix to use in json file names.
    #[arg(long)]
    suffix: Option<String>,

    #[command(flatten)]
    pub verbose: Verbosity<InfoLevel>,
}

#[derive(Serialize)]
struct Performance {
    unit: String,
    n: Vec<usize>,
    time_per_operation: Vec<f64>,
}

impl Performance {
    fn with_unit(unit: String) -> Self {
        Self {
            unit,
            n: Vec::new(),
            time_per_operation: Vec::new(),
        }
    }
}

fn execute<'a, S>(
    results: &mut BTreeMap<&'a str, Performance>,
    simulation: &mut S,
    benchmark: &Benchmark,
    name: &'a str,
    n: usize,
) -> anyhow::Result<()>
where
    S: Simulation + fmt::Display,
{
    info!("{name}: {n} bodies");
    let performance = benchmark.measure(simulation)?;
    trace!("{simulation}");

    let entry = results
        .entry(name)
        .or_insert_with(|| Performance::with_unit("step".to_string()));
    entry.n.push(n);
    entry.time_per_operation.push(performance);

    Ok(())
}

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

    let benchmark_matcher = WildMatch::new(&options.benchmarks);

    let mut results: BTreeMap<&str, Performance> = BTreeMap::new();

    let number_density = 0.8;
    let benchmark = Benchmark {
        warmup_time: Duration::from_secs_f64(options.warmup_time),
        benchmark_time: Duration::from_secs_f64(options.benchmark_time),
    };

    let mut n = options.n_min;
    let n_max = options.n_max.unwrap_or(options.n_min);

    if n_max != n && !options.json {
        return Err(anyhow!(
            "JSON output is required when n_min ({n}) is not equal to n_max ({n_max})"
        ));
    }

    let needs_microstate_2d = benchmark_matcher.matches("mc_2d_sphere")
        || benchmark_matcher.matches("mc_2d_lennard_jones")
        || benchmark_matcher.matches("mc_2d_hexagon");

    let needs_microstate_3d = benchmark_matcher.matches("mc_3d_sphere")
        || benchmark_matcher.matches("mc_3d_lennard_jones")
        || benchmark_matcher.matches("mc_3d_octahedron");

    loop {
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
            match options.spatial_data {
                SpatialData::VecCell => {
                    let mut simulation =
                        mc::HardSphereSim::<2, VecCell<SiteKey, 2>>::with_microstate(
                            microstate_2d,
                        )?;
                    execute(&mut results, &mut simulation, &benchmark, name, n)?;
                }
                SpatialData::HashCell => {
                    let mut simulation =
                        mc::HardSphereSim::<2, HashCell<SiteKey, 2>>::with_microstate(
                            microstate_2d,
                        )?;
                    execute(&mut results, &mut simulation, &benchmark, name, n)?;
                }
                SpatialData::AllPairs => {
                    let mut simulation =
                        mc::HardSphereSim::<2, AllPairs<SiteKey>>::with_microstate(microstate_2d)?;
                    execute(&mut results, &mut simulation, &benchmark, name, n)?;
                }
            }
        }

        let name = "mc_2d_lennard_jones";
        if benchmark_matcher.matches(name) {
            let microstate_2d = &maybe_microstate_2d
                .as_ref()
                .expect("microstate_2d should be initialized");
            match options.spatial_data {
                SpatialData::VecCell => {
                    let mut simulation =
                        mc::LennardJones::<2, VecCell<SiteKey, 2>>::with_microstate(microstate_2d)?;
                    execute(&mut results, &mut simulation, &benchmark, name, n)?;
                }
                SpatialData::HashCell => {
                    let mut simulation =
                        mc::LennardJones::<2, HashCell<SiteKey, 2>>::with_microstate(
                            microstate_2d,
                        )?;
                    execute(&mut results, &mut simulation, &benchmark, name, n)?;
                }
                SpatialData::AllPairs => {
                    let mut simulation =
                        mc::LennardJones::<2, AllPairs<SiteKey>>::with_microstate(microstate_2d)?;
                    execute(&mut results, &mut simulation, &benchmark, name, n)?;
                }
            }
        }

        let name = "mc_2d_hexagon";
        if benchmark_matcher.matches(name) {
            let microstate_2d = &maybe_microstate_2d
                .as_ref()
                .expect("microstate_2d should be initialized");
            match options.spatial_data {
                SpatialData::VecCell => {
                    let mut simulation =
                        mc::RegularPolygon::<VecCell<SiteKey, 2>>::with_microstate(microstate_2d)?;
                    execute(&mut results, &mut simulation, &benchmark, name, n)?;
                }
                SpatialData::HashCell => {
                    let mut simulation =
                        mc::RegularPolygon::<HashCell<SiteKey, 2>>::with_microstate(microstate_2d)?;
                    execute(&mut results, &mut simulation, &benchmark, name, n)?;
                }
                SpatialData::AllPairs => {
                    let mut simulation =
                        mc::RegularPolygon::<AllPairs<SiteKey>>::with_microstate(microstate_2d)?;
                    execute(&mut results, &mut simulation, &benchmark, name, n)?;
                }
            }
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
            let microstate_3d = &maybe_microstate_3d
                .as_ref()
                .expect("microstate_3d should be initialized");
            match options.spatial_data {
                SpatialData::VecCell => {
                    let mut simulation =
                        mc::HardSphereSim::<3, VecCell<SiteKey, 3>>::with_microstate(
                            microstate_3d,
                        )?;
                    execute(&mut results, &mut simulation, &benchmark, name, n)?;
                }
                SpatialData::HashCell => {
                    let mut simulation =
                        mc::HardSphereSim::<3, HashCell<SiteKey, 3>>::with_microstate(
                            microstate_3d,
                        )?;
                    execute(&mut results, &mut simulation, &benchmark, name, n)?;
                }
                SpatialData::AllPairs => {
                    let mut simulation =
                        mc::HardSphereSim::<3, AllPairs<SiteKey>>::with_microstate(microstate_3d)?;
                    execute(&mut results, &mut simulation, &benchmark, name, n)?;
                }
            }
        }

        let name = "mc_3d_lennard_jones";
        if benchmark_matcher.matches(name) {
            let microstate_3d = &maybe_microstate_3d
                .as_ref()
                .expect("microstate_3d should be initialized");
            match options.spatial_data {
                SpatialData::VecCell => {
                    let mut simulation =
                        mc::LennardJones::<3, VecCell<SiteKey, 3>>::with_microstate(microstate_3d)?;
                    execute(&mut results, &mut simulation, &benchmark, name, n)?;
                }
                SpatialData::HashCell => {
                    let mut simulation =
                        mc::LennardJones::<3, HashCell<SiteKey, 3>>::with_microstate(
                            microstate_3d,
                        )?;
                    execute(&mut results, &mut simulation, &benchmark, name, n)?;
                }
                SpatialData::AllPairs => {
                    let mut simulation =
                        mc::LennardJones::<3, AllPairs<SiteKey>>::with_microstate(microstate_3d)?;
                    execute(&mut results, &mut simulation, &benchmark, name, n)?;
                }
            }
        }

        let name = "mc_3d_octahedron";
        if benchmark_matcher.matches(name) {
            let microstate_3d = &maybe_microstate_3d
                .as_ref()
                .expect("microstate_3d should be initialized");
            match options.spatial_data {
                SpatialData::VecCell => {
                    let mut simulation =
                        mc::Octahedron::<VecCell<SiteKey, 3>>::with_microstate(microstate_3d)?;
                    execute(&mut results, &mut simulation, &benchmark, name, n)?;
                }
                SpatialData::HashCell => {
                    let mut simulation =
                        mc::Octahedron::<HashCell<SiteKey, 3>>::with_microstate(microstate_3d)?;
                    execute(&mut results, &mut simulation, &benchmark, name, n)?;
                }
                SpatialData::AllPairs => {
                    let mut simulation =
                        mc::Octahedron::<AllPairs<SiteKey>>::with_microstate(microstate_3d)?;
                    execute(&mut results, &mut simulation, &benchmark, name, n)?;
                }
            }
        }

        n *= 2;
        if n > n_max {
            break;
        }
    }

    if options.json {
        let mut filename_suffix = String::new();
        if let Some(suffix) = options.suffix {
            filename_suffix.push('-');
            filename_suffix.push_str(&suffix);
        }
        filename_suffix.push_str(".json");

        for (name, performance) in results {
            let performance_json = serde_json::to_string(&performance)?;
            let filename = name.to_string() + &filename_suffix;
            info!("Writing {filename}.");
            let mut file = File::create(filename)?;
            file.write_all(performance_json.as_bytes())?;
        }
    } else {
        for (name, performance) in results {
            let operations_per_second = 1.0
                / performance
                    .time_per_operation
                    .last()
                    .expect("results should contain at least one measurement");
            println!(
                "{name:20}: {operations_per_second:9.3} {}s/s",
                performance.unit
            );
        }
    }

    Ok(())
}

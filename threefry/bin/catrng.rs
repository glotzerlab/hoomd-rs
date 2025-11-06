// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! A small utility to concatenate the output of an RNG to stdout.
//!
//! To use with [``PractRand``](https://pracrand.sourceforge.net):
//! `$ catrng | RNG_test stdin -multithreaded` (Random seed from ``StdRng``)
//! `$ catrng single-seed 12345 | ...` (Single u64 seed)
//! `$ catrng single-seed 1 2 3 0 | ...` (Four u64 values as a seed)
//!
//! Note this also works with [gjrand](https://gjrand.sourceforge.net)
//! `$ catrng | ./mcp --huge`
//! This is drawn from [simd_prngs](https://github.com/TheIronBorn/simd_prngs/blob/master/src/bin/cat_rng.rs) with a few modifications for our use case.

extern crate rand;
extern crate threefry;

use std::{io, io::prelude::*};

use clap::Parser;
use rand::prelude::*;
use threefry::SFC64Rng;

/// Creates an RNG based on CLI arguments.
#[expect(clippy::print_stderr, reason = "Required.")]
fn seed_from_cli(args: &[String]) -> SFC64Rng {
    match args.len() {
        1 => {
            if let Ok(seed_u64) = args[0].parse::<u64>() {
                eprintln!("Using 1 u64 seed: {seed_u64}");
                return SFC64Rng::seed_from_u64(seed_u64);
            }
            eprintln!("Failed to parse 1 u64 input. Using entropy seed.");
        }
        4 => {
            let nums: Result<Vec<u64>, _> = args.iter().map(|s| s.parse::<u64>()).collect();

            if let Ok(n) = nums {
                eprintln!("Using 4 u64 seeds: {}, {}, {}, {}", n[0], n[1], n[2], n[3]);
                return SFC64Rng::from_state_and_counter([n[0], n[1], n[2]], n[3]);
            }
            eprintln!("Failed to parse 4 u64 inputs. Using entropy seed.");
        }
        0 => {
            // This is fine, just fall through to the default entropy seed.
        }
        _ => {
            eprintln!(
                "Expected 0, 1, or 4 arguments (as u64). Got {}. Using entropy seed.",
                args.len()
            );
        }
    }

    // Default case: Use entropy
    let seed: [u8; 32] = rand::rngs::StdRng::seed_from_u64(0).random();
    eprintln!("Using entropy seed: {seed:?}");
    SFC64Rng::from_seed(seed)
}

/// Command line options for RNG testing.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
enum Cli {
    /// Cat data from a single SFC64 seed to STDOUT.
    SingleSeed {
        /// Optional seed values (0, 1, or 4 u64s).
        #[arg(num_args(0..=4))]
        seeds: Vec<String>,
    },
    /// Interleave bytes from N RNGs with similar seeds.
    TestInterleaved {
        /// Number of RNGs to interleave.
        #[arg(short, long, default_value_t = 4)]
        n: usize,
    },
    /// Use the seed as a counter, generating one value per seed.
    SeedIncrement,
}

fn main() -> io::Result<()> {
    let cli = Cli::parse();

    let stdout = io::stdout();
    let mut writer = stdout.lock();
    let mut buf = [0; 4096];

    match cli {
        Cli::SingleSeed { seeds } => {
            let mut rng = seed_from_cli(&seeds);
            loop {
                rng.fill_bytes(&mut buf);
                writer.write_all(&buf)?;
            }
        }
        Cli::TestInterleaved { n } => {
            let mut rngs: Vec<SFC64Rng> =
                (0..n).map(|i| SFC64Rng::seed_from_u64(i as u64)).collect();
            loop {
                for (i, chunk) in buf.chunks_mut(8).enumerate() {
                    let val = rngs[i % n].next_u64();
                    chunk.copy_from_slice(&val.to_le_bytes());
                }
                writer.write_all(&buf)?;
            }
        }
        Cli::SeedIncrement => {
            let mut seed_counter = 0u64;
            loop {
                for chunk in buf.chunks_mut(8) {
                    let mut rng = SFC64Rng::seed_from_u64(seed_counter);
                    let val = rng.next_u64();
                    chunk.copy_from_slice(&val.to_le_bytes());
                    seed_counter = seed_counter.wrapping_add(1);
                }
                writer.write_all(&buf)?;
            }
        }
    }
}

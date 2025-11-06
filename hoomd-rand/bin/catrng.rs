// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! A small utility to concatenate the output of an RNG to stdout.
//!
//! To use with [``PractRand``](https://pracrand.sourceforge.net):
//! `$ catrng | RNG_test stdin -multithreaded` (Random seed from ``StdRng``)
//! `$ catrng seed-single 12345 | ...` (Single u64 seed)
//! `$ catrng seed-single 1 2 3 0 | ...` (Four u64 values as a seed)
//! `$ catrng seed-increment | ...` (Alternate u64 generation with seed changes.)
//! `$ catrng test-interleaved -n 8 | ...` (8 interleaved RNGs with similar seeds.)
//!
//! Note this also works with [gjrand](https://gjrand.sourceforge.net)
//! `$ catrng | ./mcp --tera`
//! This is drawn from [simd_prngs](https://github.com/TheIronBorn/simd_prngs/blob/master/src/bin/cat_rng.rs) with a few modifications for our use case.

extern crate hoomd_rand;
extern crate rand;

use std::{io, io::prelude::*};

use clap::Parser;
use hoomd_rand::SFC64Rng;
use rand::prelude::*;

/// Command line options for RNG testing.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
enum Cli {
    /// Cat data from a single SFC64 seed to STDOUT.
    SeedSingle {
        /// Optional seed values (0, 1, or 4 u64s).
        #[arg(num_args(0..=4))]
        seeds: Vec<u64>,
    },
    /// Interleave bytes from N RNGs with similar seeds.
    TestInterleaved {
        /// Number of RNGs to interleave.
        #[arg(short, long, default_value_t = 4)]
        n: usize,
    },
    /// Generate single items from RNGs with a fixed seed and manually incremented ctr.
    ManualCounter,
    /// Use the seed as a counter, generating one value per seed.
    SeedIncrement,
}

#[expect(clippy::print_stderr, reason = "Required.")]
fn main() -> io::Result<()> {
    let cli = Cli::parse();

    let stdout = io::stdout();
    let mut writer = stdout.lock();
    let mut buf = [0; 4096];

    match cli {
        Cli::SeedSingle { seeds } => {
            let mut rng = match seeds.len() {
                1 => {
                    eprintln!("Using 1 u64 seed: {}", seeds[0]);
                    SFC64Rng::seed_from_u64(seeds[0])
                }
                4 => {
                    eprintln!(
                        "Using 4 u64 seeds: {}, {}, {}, {}",
                        seeds[0], seeds[1], seeds[2], seeds[3]
                    );
                    SFC64Rng::from_state_and_counter([seeds[0], seeds[1], seeds[2]], seeds[3])
                }
                0 => {
                    let seed: [u8; 32] = rand::rngs::StdRng::seed_from_u64(0).random();
                    eprintln!("Using entropy seed: {seed:?}");
                    SFC64Rng::from_seed(seed)
                }
                _ => {
                    eprintln!(
                        "Expected 0, 1, or 4 arguments (as u64). Got {}. Using entropy seed.",
                        seeds.len()
                    );
                    let seed: [u8; 32] = rand::rngs::StdRng::seed_from_u64(0).random();
                    SFC64Rng::from_seed(seed)
                }
            };
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
        Cli::ManualCounter => {
            let mut counter = 0u64;

            loop {
                let mut rng = SFC64Rng::from_state_and_counter([0; 3], counter);
                rng.fill_bytes(&mut buf);
                counter = counter.wrapping_add(1);
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

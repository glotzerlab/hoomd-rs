// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! A small utility to concatenate the output of an RNG to stdout.
//!
//! To use with [``PractRand``](https://pracrand.sourceforge.net):
//! `$ catrng | RNG_test stdin -multithreaded` (Random seed from ``StdRng``)
//! `$ catrng 12345 | ...` (Single u64 seed)
//! `$ catrng 1 2 3 0 | ...` (Four u64 values as a seed)
//!
//! Note this also works with [gjrand](https://gjrand.sourceforge.net)
//! This is drawn from [simd_prngs](https://github.com/TheIronBorn/simd_prngs/blob/master/src/bin/cat_rng.rs) with a few modifications for our use case.

extern crate rand;
extern crate threefry;

use std::{io, io::prelude::*};

use std::env;

use rand::prelude::*;
use threefry::SFC64Rng;

/// Creates an RNG based on CLI arguments.
#[expect(clippy::print_stderr, reason = "Required.")]
fn get_rng() -> SFC64Rng {
    let args: Vec<String> = env::args().skip(1).collect();

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

fn main() -> io::Result<()> {
    let mut rng = get_rng();

    let mut buf = [0; 4096];
    let stdout = io::stdout();
    let mut writer = stdout.lock();

    loop {
        rng.fill_bytes(&mut buf);
        writer.write_all(&buf)?;
    }
}

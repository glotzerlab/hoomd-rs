//! A small utility to concatenate the output of an RNG to stdout.
//!
//! To use with PractRand:
//! `$ cat_rng | RNG_test stdin -multithreaded` (Entropy seed)
//! `$ cat_rng 12345 | ...` (Single u64 seed)
//! `$ cat_rng 1 2 3 0 | ...` (Four u64 values as a seed)
//!
//! This is drawn from [simd_prngs](https://github.com/TheIronBorn/simd_prngs/blob/master/src/bin/cat_rng.rs) with a few modifications for our use case.

extern crate rand;
extern crate threefry;

use std::io;
use std::io::prelude::*;

use rand::prelude::*;
use threefry::sfc::SFC64Rng;

extern crate rand;
extern crate threefry;

use std::io;
use std::io::prelude::*;
use std::env;

use rand::prelude::*;
use threefry::sfc::SFC64Rng;

/// Creates an RNG based on CLI arguments.
/// - 0 args: Uses EntropyRng.
/// - 1 arg: Uses seed_from_u64.
/// - 4 args: Uses initialize(a, b, c, counter).
/// - Other: Falls back to EntropyRng.
fn get_rng() -> SFC64Rng {
    let args: Vec<String> = env::args().skip(1).collect();

    match args.len() {
        1 => {
            if let Ok(seed_u64) = args[0].parse::<u64>() {
                eprintln!("Using 1 u64 seed: {}", seed_u64);
                return SFC64Rng::seed_from_u64(seed_u64);
            } else {
                eprintln!("Failed to parse 1 u64 input. Using entropy seed.");
            }
        }
        4 => {
            let nums: Result<Vec<u64>, _> = args.iter().map(|s| s.parse::<u64>()).collect();

            if let Ok(n) = nums {
                eprintln!("Using 4 u64 seeds: {}, {}, {}, {}", n[0], n[1], n[2], n[3]);
                return SFC64Rng::initialize(n[0], n[1], n[2], n[3]);
            } else {
                eprintln!("Failed to parse 4 u64 inputs. Using entropy seed.");
            }
        }
        0 => {
            // This is fine, just fall through to the default entropy seed.
        }
        _ => {
            eprintln!("Expected 0, 1, or 4 arguments (as u64). Got {}. Using entropy seed.", args.len());
        }
    }

    // Default case: Use entropy
    let seed: [u8; 32] = rand::rngs::EntropyRng::new().gen();
    eprintln!("Using entropy seed: {:?}", seed);
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

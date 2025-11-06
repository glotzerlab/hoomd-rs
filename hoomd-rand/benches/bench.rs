// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Benchmark code for PRNGs

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]
use chacha20::ChaCha8Rng;
use divan::{Bencher, black_box};
use hoomd_rand::{
    SFC64Rng,
    squares::{Squares64, Squares128},
    threefry2x64::ThreeFry2x64Rng,
};
use rand_core::{RngCore, SeedableRng};

fn main() {
    divan::main();
}

const SEED: u64 = 42;

/// Time to first generated value
#[divan::bench_group(sample_count = 1000)]
mod latency {
    use super::{Bencher, ChaCha8Rng, RngCore, SEED, SFC64Rng, SeedableRng, black_box};
    use hoomd_rand::{
        squares::{Squares64, Squares128},
        threefry2x64::ThreeFry2x64Rng,
    };

    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    ))]
    use hoomd_rand::AESRandRng;

    #[divan::bench]
    fn chacha8(bencher: Bencher) {
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        bencher.bench_local(|| {
            black_box(rng.next_u64());
        });
    }

    #[divan::bench]
    fn threefry2x64r13(bencher: Bencher) {
        let mut rng = ThreeFry2x64Rng::<13>::seed_from_u64(SEED);
        bencher.bench_local(|| {
            black_box(rng.next_u64());
        });
    }

    #[divan::bench]
    fn squares64(bencher: Bencher) {
        let mut rng = Squares64::seed_from_u64(SEED);
        bencher.bench_local(|| {
            black_box(rng.next_u64());
        });
    }

    #[divan::bench]
    fn squares128(bencher: Bencher) {
        let mut rng = Squares128::seed_from_u64(SEED);
        bencher.bench_local(|| {
            black_box(rng.next_u64());
        });
    }

    #[divan::bench]
    fn sfc64(bencher: Bencher) {
        let mut rng = SFC64Rng::seed_from_u64(SEED);
        bencher.bench_local(|| {
            black_box(rng.next_u64());
        });
    }

    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    ))]
    #[divan::bench]
    fn aesrand(bencher: Bencher) {
        let mut rng = AESRandRng::seed_from_u64(SEED);
        bencher.bench_local(|| {
            black_box(rng.next_u64());
        });
    }
}

/// Measure the time to generate a particular quantitity of data.
#[divan::bench_group]
mod throughput {
    use super::{Bencher, ChaCha8Rng, RngCore, SEED, SFC64Rng, SeedableRng, black_box};
    use divan::counter::BytesCount;
    use hoomd_rand::{
        squares::{Squares64, Squares128},
        threefry2x64::ThreeFry2x64Rng,
    };

    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    ))]
    use hoomd_rand::AESRandRng;

    /// 1 MiB
    const SIZE: usize = 1024 * 1024;

    #[divan::bench(counters = [BytesCount::new(SIZE)])]
    fn chacha8(bencher: Bencher) {
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let mut buffer = vec![0u8; SIZE];
        bencher.bench_local(|| {
            rng.fill_bytes(black_box(&mut buffer));
        });
    }
    #[divan::bench(counters = [BytesCount::new(SIZE)])]
    fn threefry2x64r13(bencher: Bencher) {
        let mut rng = ThreeFry2x64Rng::<13>::seed_from_u64(SEED);
        let mut buffer = vec![0u8; SIZE];
        bencher.bench_local(|| {
            rng.fill_bytes(black_box(&mut buffer));
        });
    }

    #[divan::bench(counters = [BytesCount::new(SIZE)])]
    fn squares64(bencher: Bencher) {
        let mut rng = Squares64::seed_from_u64(SEED);
        let mut buffer = vec![0u8; SIZE];
        bencher.bench_local(|| {
            rng.fill_bytes(black_box(&mut buffer));
        });
    }

    #[divan::bench(counters = [BytesCount::new(SIZE)])]
    fn squares128(bencher: Bencher) {
        let mut rng = Squares128::seed_from_u64(SEED);
        let mut buffer = vec![0u8; SIZE];
        bencher.bench_local(|| {
            rng.fill_bytes(black_box(&mut buffer));
        });
    }

    #[divan::bench(counters = [BytesCount::new(SIZE)])]
    fn sfc64(bencher: Bencher) {
        let mut rng = SFC64Rng::seed_from_u64(SEED);
        let mut buffer = vec![0u8; SIZE];
        bencher.bench_local(|| {
            rng.fill_bytes(black_box(&mut buffer));
        });
    }

    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    ))]
    #[divan::bench(counters = [BytesCount::new(SIZE)])]
    fn aesrand(bencher: Bencher) {
        let mut rng = AESRandRng::seed_from_u64(SEED);
        let mut buffer = vec![0u8; SIZE];
        bencher.bench_local(|| {
            rng.fill_bytes(black_box(&mut buffer));
        });
    }
}

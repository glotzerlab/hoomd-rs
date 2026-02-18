// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Benchmark code for PRNGs

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]
use chacha20::ChaCha8Rng;
use divan::{Bencher, black_box, counter::ItemsCount};
use hoomd_rand::SFC64;
use rand::{
    Rng,
    rand_core::{Rng, SeedableRng},
};

fn main() {
    divan::main();
}

const SEED: u64 = 42;

/// Time to first generated value
#[divan::bench_group(sample_count = 1000)]
mod latency {
    use super::{Bencher, ChaCha8Rng, Rng, SEED, SFC64, SeedableRng, black_box};
    #[cfg(feature = "extras")]
    use hoomd_rand::ThreeFry2x64Rng;

    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    ))]
    use hoomd_rand::AESRand;

    #[divan::bench]
    fn chacha8(bencher: Bencher) {
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        bencher.bench_local(|| {
            black_box(rng.next_u64());
        });
    }

    #[cfg(feature = "extras")]
    #[divan::bench]
    fn threefry2x64r13(bencher: Bencher) {
        let mut rng = ThreeFry2x64Rng::<13>::seed_from_u64(SEED);
        bencher.bench_local(|| {
            black_box(rng.next_u64());
        });
    }

    #[divan::bench]
    fn sfc64(bencher: Bencher) {
        let mut rng = SFC64::seed_from_u64(SEED);
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
        let mut rng = AESRand::seed_from_u64(SEED);
        bencher.bench_local(|| {
            black_box(rng.next_u64());
        });
    }
}

/// Measure the time to generate a particular quantitity of data.
#[divan::bench_group]
mod throughput {
    use super::{Bencher, ChaCha8Rng, Rng, SEED, SFC64, SeedableRng, black_box};
    use divan::counter::BytesCount;
    #[cfg(feature = "extras")]
    use hoomd_rand::ThreeFry2x64Rng;

    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    ))]
    use hoomd_rand::AESRand;

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
    #[cfg(feature = "extras")]
    #[divan::bench(counters = [BytesCount::new(SIZE)])]
    fn threefry2x64r13(bencher: Bencher) {
        let mut rng = ThreeFry2x64Rng::<13>::seed_from_u64(SEED);
        let mut buffer = vec![0u8; SIZE];
        bencher.bench_local(|| {
            rng.fill_bytes(black_box(&mut buffer));
        });
    }

    #[divan::bench(counters = [BytesCount::new(SIZE)])]
    fn sfc64(bencher: Bencher) {
        let mut rng = SFC64::seed_from_u64(SEED);
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
        let mut rng = AESRand::seed_from_u64(SEED);
        let mut buffer = vec![0u8; SIZE];
        bencher.bench_local(|| {
            rng.fill_bytes(black_box(&mut buffer));
        });
    }
}
const N: &[usize] = &[1, 4, 8, 16, 32];
#[divan::bench(consts = N)]
fn bench_counter<const N: usize>(bencher: Bencher) {
    bencher.counter(ItemsCount::from(N)).bench_local(|| {
        let mut rng = hoomd_rand::Counter::new(black_box(10), black_box(11), black_box(12))
            .index(black_box(13))
            .make_rng();
        black_box(core::array::from_fn::<_, N, _>(|_| rng.random::<f64>()))
    });
}

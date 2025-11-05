//! Benchmark code for PRNGs

use divan::{Bencher, black_box};
use rand_core::{RngCore, SeedableRng};
use threefry::{AESRandRng, SFC64Rng, Squares64, Squares128, ThreeFry2x64Rng};

fn main() {
    divan::main();
}

const SEED: u64 = 42;

#[divan::bench_group]
mod latency {
    use super::*;

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

    #[divan::bench]
    fn aesrand(bencher: Bencher) {
        let mut rng = AESRandRng::seed_from_u64(SEED);
        bencher.bench_local(|| {
            black_box(rng.next_u64());
        });
    }
}

/// Throughput benmchmarks
#[divan::bench_group]
mod throughput {
    use super::*;
    use divan::counter::BytesCount;

    /// 1 GiB
    const SIZE: usize = 1024 * 1024 * 1024;

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

    #[divan::bench(counters = [BytesCount::new(SIZE)])]
    fn aesrand(bencher: Bencher) {
        let mut rng = AESRandRng::seed_from_u64(SEED);
        let mut buffer = vec![0u8; SIZE];
        bencher.bench_local(|| {
            rng.fill_bytes(black_box(&mut buffer));
        });
    }
}

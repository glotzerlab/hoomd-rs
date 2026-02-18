// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::missing_docs_in_private_items,
    reason = "benches don't need public documentation"
)]

//! Benchmark Quaternion

use divan::{self, Bencher, black_box, counter::ItemsCount};
use hoomd_rand::Counter;
use rand::{Rng, RngExt, SeedableRng, rngs::StdRng};

use hoomd_vector::{Cartesian, Rotate, RotationMatrix, Versor};

fn main() {
    divan::main();
}

#[divan::bench]
fn rotate(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    let q: Versor = rng.random();

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| -> Cartesian<3> { rng.random::<Cartesian<3>>() })
        .bench_local_values(|vec| black_box(q.rotate(&vec)));
}

#[divan::bench]
fn rotate_matrix(bencher: Bencher) {
    let mut rng = StdRng::seed_from_u64(1);

    let a: Versor = rng.random();
    let matrix = RotationMatrix::from(a);

    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| -> Cartesian<3> { rng.random::<Cartesian<3>>() })
        .bench_local_values(|vec| black_box(matrix.rotate(&vec)));
}

#[divan::bench]
fn gen_random(bencher: Bencher) {
    let mut rng = Counter::new(0, 0, 0).make_rng();

    bencher
        .counter(ItemsCount::from(1_u32))
        .bench_local(|| black_box(rng.random::<Versor>()));
}

#[divan::bench_group]
mod quat_metric {
    use super::{Bencher, Counter, ItemsCount, RngExt, Versor, black_box, divan};
    #[divan::bench]
    fn arc_distance(bencher: Bencher) {
        let mut rng = Counter::new(0, 0, 0).make_rng();

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| (rng.random::<Versor>(), rng.random::<Versor>()))
            .bench_local_refs(|(l, r)| black_box(l.arc_distance(r)));
    }
    #[divan::bench]
    fn half_euclidean_norm_squared(bencher: Bencher) {
        let mut rng = Counter::new(0, 0, 0).make_rng();

        bencher
            .counter(ItemsCount::from(1_u32))
            .with_inputs(|| (rng.random::<Versor>(), rng.random::<Versor>()))
            .bench_local_refs(|(l, r)| black_box(l.half_euclidean_norm_squared(r)));
    }
}

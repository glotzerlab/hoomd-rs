// Copyright (c) 2024 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use divan::counter::ItemsCount;
use divan::{self, black_box, Bencher};
use rand::distributions::Uniform;
use rand::{thread_rng, Rng};

use hoomd_rs_vector::{vector::Cartesian, Cross, Vector};


fn main() {
    divan::main();
}

fn create_random_vector_pair<const N: usize>() -> (Cartesian<N>, Cartesian<N>) {
    (rand::random::<Cartesian<N>>(), rand::random::<Cartesian<N>>())
}

const DIMENSIONS: &[usize] = &[2, 3, 8, 16, 32, 128];

#[divan::bench(consts = DIMENSIONS)]
fn create_vecn_tryfrom_vec<const N: usize>(bencher: Bencher) {
    let mut rng = thread_rng();

    let range = Uniform::new(-100.0, 100.0);
    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(|| -> Vec<f64> { (&mut rng).sample_iter(range).take(N).collect() })
        .bench_local_values(|vec| black_box(Cartesian::<N>::try_from(vec)));
}

#[divan::bench(consts = DIMENSIONS)]
fn add_vecn<const N: usize>(bencher: Bencher) {
    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(create_random_vector_pair::<N>)
        .bench_local_values(|(a, b)| black_box(a + b));
}

#[divan::bench(consts = DIMENSIONS)]
fn sub_vecn<const N: usize>(bencher: Bencher) {
    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(create_random_vector_pair::<N>)
        .bench_local_values(|(a, b)| black_box(a - b));
}

#[divan::bench(consts = DIMENSIONS)]
fn mul_vecn<const N: usize>(bencher: Bencher) {
    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(create_random_vector_pair::<N>)
        .bench_local_values(|(a, b)| {
            black_box(a * b.coordinates[0]);
        });
}

#[divan::bench(consts = DIMENSIONS)]
fn div_vecn<const N: usize>(bencher: Bencher) {
    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(create_random_vector_pair::<N>)
        .bench_local_values(|(a, b)| black_box(a / b.coordinates[0]));
}

#[divan::bench(consts = DIMENSIONS)]
fn dot_vecn<const N: usize>(bencher: Bencher) {
    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(create_random_vector_pair::<N>)
        .bench_local_values(|(a, b)| black_box(a.dot(&b)));
}

#[divan::bench]
fn cross_vec3(bencher: Bencher) {
    bencher
        .counter(ItemsCount::from(1_u32))
        .with_inputs(create_random_vector_pair::<3>)
        .bench_local_values(|(a, b)| black_box(a.cross(&b)));
}

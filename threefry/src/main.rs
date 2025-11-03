// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! .
use rand::{RngCore, SeedableRng};
use threefry::Squares;
use threefry::ThreeFry2x64Rng;
use threefry::XSM64Rng;

fn main() {
    let mut x = ThreeFry2x64Rng::<20>::seed_from_u64(0);
    x.set_stream_from_u64(0);
    assert_eq!(x.next_u64(), 14_030_652_003_081_164_901);
    assert_eq!(x.next_u64(), 8_034_964_082_011_408_461);
    // for _ in (0..10) {
    //     println!("{}", x.next_u64());
    // }
    // (0..500_000_000).for_each(|_| {
    //     x.next_u64();
    // });
    // println!("Done");
    // let mut x = Squares::seed_from_u64(1_347_891_341_384);
    // // x.set_stream_from_u64(0);
    // (0..500_000_000).for_each(|_| {
    //     x.next_u64();
    // });
    // println!("{}", x.next_u64());

    let mut x = XSM64Rng::seed_from_u64(1_347_891_341_384);
    // x.set_stream_from_u64(0);
    (0..500_000_000).for_each(|_| {
        x.next_u64();
    });
    println!("{}", x.next_u64());
    println!("{}", x.next_u64());
    // println!("{}", x.next_u64());
}

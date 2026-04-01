// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Test derive(Position)

use assert2::check;
use hoomd_microstate::property::Position;

// Compile error
// #[derive(Position)]
// struct Tuple(f64);

// Compile error
// #[derive(Position)]
// struct Unit;

// Compile error
// #[derive(Position)]
// enum Enum {
//     A,B
// };

// Compile error
// #[derive(Position)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

// Compile Error
// #[derive(Position)]
// struct InvalidNamed {
//     positions: f64,
// }

#[derive(Position)]
struct Named {
    position: f64,
}

#[test]
fn derive_position() {
    let mut test = Named { position: 15.0 };
    check!(*test.position() == 15.0);

    *test.position_mut() = 32.0;
    check!(test.position == 32.0);
}

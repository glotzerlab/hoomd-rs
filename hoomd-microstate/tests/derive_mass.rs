// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Test derive(Mass)

use hoomd_microstate::property::Mass;

// Compile error
// #[derive(Mass)]
// struct Tuple(f64);

// Compile error
// #[derive(Mass)]
// struct Unit;

// Compile error
// #[derive(Mass)]
// enum Enum {
//     A,B
// };

// Compile error
// #[derive(Mass)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

// Compile Error
// #[derive(Mass)]
// struct InvalidNamed {
//     masss: f64,
// }

#[derive(Mass)]
struct Named {
    mass: f64,
}

#[test]
fn derive_mass() {
    let test = Named { mass: 15.0 };
    assert_eq!(test.mass(), 15.0);
}

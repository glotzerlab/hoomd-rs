// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Test derive(MomentOfInertia)

use hoomd_microstate::property::MomentOfInertia;

// Compile error
// #[derive(MomentOfInertia)]
// struct Tuple(f64);

// Compile error
// #[derive(MomentOfInertia)]
// struct Unit;

// Compile error
// #[derive(MomentOfInertia)]
// enum Enum {
//     A,B
// };

// Compile error
// #[derive(MomentOfInertia)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

// Compile Error
// #[derive(MomentOfInertia)]
// struct InvalidNamed {
//     moment_of_inertias: f64,
// }

#[derive(MomentOfInertia)]
struct Named {
    moment_of_inertia: f64,
}

#[test]
fn derive_moment_of_inertia() {
    let mut test = Named { moment_of_inertia: 15.0 };
    assert_eq!(*test.moment_of_inertia(), 15.0);

    *test.moment_of_inertia_mut() = 32.0;
    assert_eq!(test.moment_of_inertia, 32.0);
}

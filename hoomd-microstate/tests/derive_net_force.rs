// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Test derive(NetForce)

use hoomd_microstate::property::NetForce;

// Compile error
// #[derive(NetForce)]
// struct Tuple(f64);

// Compile error
// #[derive(NetForce)]
// struct Unit;

// Compile error
// #[derive(NetForce)]
// enum Enum {
//     A,B
// };

// Compile error
// #[derive(NetForce)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

// Compile Error
// #[derive(NetForce)]
// struct InvalidNamed {
//     net_forces: f64,
// }

#[derive(NetForce)]
struct Named {
    net_force: f64,
}

#[test]
fn derive_net_force() {
    let mut test = Named { net_force: 15.0 };
    assert_eq!(*test.net_force(), 15.0);

    *test.net_force_mut() = 32.0;
    assert_eq!(test.net_force, 32.0);
}

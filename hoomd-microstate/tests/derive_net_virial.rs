// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Test derive(NetVirial)

use hoomd_microstate::property::NetVirial;

// Compile error
// #[derive(NetVirial)]
// struct Tuple(f64);

// Compile error
// #[derive(NetVirial)]
// struct Unit;

// Compile error
// #[derive(NetVirial)]
// enum Enum {
//     A,B
// };

// Compile error
// #[derive(NetVirial)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

// Compile Error
// #[derive(NetVirial)]
// struct InvalidNamed {
//     net_virials: f64,
// }

#[derive(NetVirial)]
struct Named {
    net_virial: f64,
}

#[test]
fn derive_net_virial() {
    let mut test = Named { net_virial: 15.0 };
    assert_eq!(*test.net_virial(), 15.0);

    *test.net_virial_mut() = 32.0;
    assert_eq!(test.net_virial, 32.0);
}

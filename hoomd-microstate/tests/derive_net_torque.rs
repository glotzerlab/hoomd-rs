// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Test derive(NetTorque)

use hoomd_microstate::property::NetTorque;

// Compile error
// #[derive(NetTorque)]
// struct Tuple(f64);

// Compile error
// #[derive(NetTorque)]
// struct Unit;

// Compile error
// #[derive(NetTorque)]
// enum Enum {
//     A,B
// };

// Compile error
// #[derive(NetTorque)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

// Compile Error
// #[derive(NetTorque)]
// struct InvalidNamed {
//     net_torques: f64,
// }

#[derive(NetTorque)]
struct Named {
    net_torque: f64,
}

#[test]
fn derive_net_torque() {
    let mut test = Named { net_torque: 15.0 };
    assert_eq!(*test.net_torque(), 15.0);

    *test.net_torque_mut() = 32.0;
    assert_eq!(test.net_torque, 32.0);
}

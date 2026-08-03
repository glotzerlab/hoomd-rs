// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Test derive(AngularMomentum)

use hoomd_microstate::property::AngularMomentum;

// Compile error
// #[derive(AngularMomentum)]
// struct Tuple(f64);

// Compile error
// #[derive(AngularMomentum)]
// struct Unit;

// Compile error
// #[derive(AngularMomentum)]
// enum Enum {
//     A,B
// };

// Compile error
// #[derive(AngularMomentum)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

// Compile Error
// #[derive(AngularMomentum)]
// struct InvalidNamed {
//     angular_momentums: f64,
// }

#[derive(AngularMomentum)]
struct Named {
    angular_momentum: f64,
}

#[test]
fn derive_angular_momentum() {
    let mut test = Named { angular_momentum: 15.0 };
    assert_eq!(*test.angular_momentum(), 15.0);

    *test.angular_momentum_mut() = 32.0;
    assert_eq!(test.angular_momentum, 32.0);
}

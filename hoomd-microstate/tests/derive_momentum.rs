// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Test derive(Momentum)

use hoomd_microstate::property::Momentum;
use hoomd_microstate::property::Mass;
use hoomd_vector::Cartesian;

// Compile error
// #[derive(Momentum)]
// struct Tuple(f64);

// Compile error
// #[derive(Momentum)]
// struct Unit;

// Compile error
// #[derive(Momentum)]
// enum Enum {
//     A,B
// };

// Compile error
// #[derive(Momentum)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

// Compile Error
// #[derive(Momentum)]
// struct InvalidNamed {
//     momentums: f64,
// }

#[derive(Mass, Momentum)]
struct Named {
    mass: f64,
    momentum: Cartesian<2>,
}

#[test]
fn derive_momentum() {
    let mut test = Named { mass: 0.5, momentum: Cartesian::<2>::from([1.0, 1.0]) };
    assert_eq!(*test.momentum(), Cartesian::<2>::from([1.0, 1.0]));
    assert_eq!(test.velocity(), Cartesian::<2>::from([2.0, 2.0]));

    *test.momentum_mut() = Cartesian::<2>::from([2.0, 2.0]);
    assert_eq!(test.momentum, Cartesian::<2>::from([2.0, 2.0]));
    assert_eq!(test.velocity(), Cartesian::<2>::from([4.0, 4.0]));

    test.set_velocity(Cartesian::<2>::from([2.0, 2.0]));
    assert_eq!(test.momentum, Cartesian::<2>::from([1.0, 1.0]));
}

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Test derive_dynamic_point

use hoomd_derive::derive_dynamic_point;
use hoomd_vector::{Cartesian, Outer};

// Compile error
// #[derive_dynamic_point(Cartesian::<2>)]
// struct Tuple(f64);

// Compile error
// #[derive_dynamic_point(Cartesian::<2>)]
// struct Unit;

// Compile error
// #[derive_dynamic_point(Cartesian::<2>)]
// enum Enum {
//     A,B
// };

// Compile error
// #[derive_dynamic_point(Cartesian::<2>)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

// Compile error
// #[derive_dynamic_point]
// struct Named {
//     other: f64,
// }

#[derive_dynamic_point(Cartesian::<3>)]
struct MyDynamicPoint<'a> {
// struct MyBodyProperties{
    int: i32,
    float: f64,
    arr: [f64; 3],
    name: &'a str,
}

#[test]
fn test_derive_dynamic_point() {
    let mdp1 = MyDynamicPoint {
        name: "Conrad",
        ..Default::default()
    };

    assert_eq!(mdp1.position, Cartesian::<3>::default());
    assert_eq!(mdp1.mass, 1.0);
    assert_eq!(mdp1.momentum, Cartesian::<3>::default());
    assert_eq!(mdp1.net_force, Cartesian::<3>::default());
    assert_eq!(mdp1.net_virial, <Cartesian<3> as Outer>::Tensor::default());
    assert_eq!(mdp1.drag, 1.0);
    assert_eq!(mdp1.int, 0 as i32);
    assert_eq!(mdp1.float, 0.0 as f64);
    assert_eq!(mdp1.arr, [0.0, 0.0, 0.0]);
    assert_eq!(mdp1.name, "Conrad");
}

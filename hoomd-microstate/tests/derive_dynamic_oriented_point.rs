// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Test derive_dynamic_oriented_point

use hoomd_derive::derive_dynamic_oriented_point;
use hoomd_vector::{Cartesian, Outer, Angle};

// Compile error
// #[derive_dynamic_oriented_point(Cartesian::<2>, Angle)]
// struct Tuple(f64);

// Compile error
// #[derive_dynamic_oriented_point(Cartesian::<2>, Angle)]
// struct Unit;

// Compile error
// #[derive_dynamic_oriented_point(Cartesian::<2>, Angle)]
// enum Enum {
//     A,B
// };

// Compile error
// #[derive_dynamic_oriented_point(Cartesian::<2>, Angle)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

// Compile error
// #[derive_dynamic_oriented_point]
// struct Named {
//     other: f64,
// }

#[derive_dynamic_oriented_point(Cartesian::<2>, Angle)]
struct MyDynamicOrientedPoint<'a> {
// struct MyBodyProperties{
    int: i32,
    float: f64,
    arr: [f64; 3],
    heap: String,
    stack: &'a str,
}

#[test]
fn test_derive_dynamic_oriented_point() {
    let mdop1 = MyDynamicOrientedPoint {
        heap: String::from("Jimothy"),
        stack: "Raccoon",
        ..Default::default()
    };

    assert_eq!(mdop1.position, Cartesian::<2>::default());
    assert_eq!(mdop1.orientation, Angle::default());
    assert_eq!(mdop1.mass, 1.0);
    assert_eq!(mdop1.momentum, Cartesian::<2>::default());
    assert_eq!(mdop1.net_force, Cartesian::<2>::default());
    assert_eq!(mdop1.net_virial, <Cartesian<2> as Outer>::Tensor::default());
    assert_eq!(mdop1.moment_of_inertia, 1.0);
    assert_eq!(mdop1.angular_momentum, 0.0);
    assert_eq!(mdop1.net_torque, 0.0);
    assert_eq!(mdop1.drag, 1.0);
    assert_eq!(mdop1.rotational_drag, 1.0);
    assert_eq!(mdop1.int, 0 as i32);
    assert_eq!(mdop1.float, 0.0 as f64);
    assert_eq!(mdop1.arr, [0.0, 0.0, 0.0]);
    assert_eq!(mdop1.heap, String::from("Jimothy"));
    assert_eq!(mdop1.stack, "Raccoon");
}

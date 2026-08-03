// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Test derive(RotationalDrag)

use hoomd_microstate::property::RotationalDrag;

// Compile error
// #[derive(RotationalDrag)]
// struct Tuple(f64);

// Compile error
// #[derive(RotationalDrag)]
// struct Unit;

// Compile error
// #[derive(RotationalDrag)]
// enum Enum {
//     A,B
// };

// Compile error
// #[derive(RotationalDrag)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

// Compile Error
// #[derive(RotationalDrag)]
// struct InvalidNamed {
//     rotational_drags: f64,
// }

#[derive(RotationalDrag)]
struct Named {
    rotational_drag: f64,
}

#[test]
fn derive_rotational_drag() {
    let mut test = Named { rotational_drag: 15.0 };
    assert_eq!(*test.rotational_drag(), 15.0);

    *test.rotational_drag_mut() = 32.0;
    assert_eq!(test.rotational_drag, 32.0);
}

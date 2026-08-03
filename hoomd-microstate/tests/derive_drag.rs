// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Test derive(Drag)

use hoomd_microstate::property::Drag;

// Compile error
// #[derive(Drag)]
// struct Tuple(f64);

// Compile error
// #[derive(Drag)]
// struct Unit;

// Compile error
// #[derive(Drag)]
// enum Enum {
//     A,B
// };

// Compile error
// #[derive(Drag)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

// Compile Error
// #[derive(Drag)]
// struct InvalidNamed {
//     drags: f64,
// }

#[derive(Drag)]
struct Named {
    drag: f64,
}

#[test]
fn derive_drag() {
    let mut test = Named { drag: 15.0 };
    assert_eq!(*test.drag(), 15.0);

    *test.drag_mut() = 32.0;
    assert_eq!(test.drag, 32.0);
}

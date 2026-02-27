//! Test derive(Orientation)

use hoomd_microstate::property::Orientation;
use assert2::check;

// Compile error
// #[derive(Orientation)]
// struct Tuple(f64);

// Compile error
// #[derive(Orientation)]
// struct Unit;

// Compile error
// #[derive(Orientation)]
// enum Enum {
//     A,B
// };

// Compile error
// #[derive(Orientation)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

// Compile Error
// #[derive(Orientation)]
// struct InvalidNamed {
//     orientation: f64,
// }

#[derive(Orientation)]
struct Named {
    orientation: f64,
}

#[test]
fn derive_orientation() {
    let mut test = Named { orientation: 15.0 };
    check!(*test.orientation() == 15.0);

    *test.orientation_mut() = 32.0;
    check!(test.orientation == 32.0);
}

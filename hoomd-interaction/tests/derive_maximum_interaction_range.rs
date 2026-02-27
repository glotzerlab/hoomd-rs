//! Test derive(MaximumInteractionRange)

use hoomd_interaction::{
    MaximumInteractionRange, SitePairEnergy,
    pairwise::{AngularMask, Anisotropic, HardSphere},
    univariate::Boxcar,
};
use hoomd_vector::Cartesian;
use assert2::check;

// Compile error
// #[derive(MaximumInteractionRange)]
// struct Tuple(f64);

// Compile error
// #[derive(MaximumInteractionRange)]
// enum Enum {
//     A,B
// }

// Compile error
// #[derive(MaximumInteractionRange)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

// Compile Error
// #[derive(MaximumInteractionRange)]
// struct InvalidNamed {
//     positions: f64,
// }

struct One;
struct Two;
struct Three;

impl MaximumInteractionRange for One {
    fn maximum_interaction_range(&self) -> f64 {
        1.0
    }
}
impl MaximumInteractionRange for Two {
    fn maximum_interaction_range(&self) -> f64 {
        2.0
    }
}
impl MaximumInteractionRange for Three {
    fn maximum_interaction_range(&self) -> f64 {
        3.0
    }
}

#[derive(MaximumInteractionRange)]
struct Unit;

#[test]
fn unit() {
    let unit = Unit;
    check!(unit.maximum_interaction_range() == 0.0);
}

#[derive(MaximumInteractionRange)]
struct CombinedNamed {
    one: One,
    two: Two,
    three: Three,
}

#[test]
fn combined_named() {
    let combined_named = CombinedNamed { one: One, two: Two, three: Three};
    check!(combined_named.maximum_interaction_range() == 3.0);
}

#[derive(MaximumInteractionRange)]
struct CombinedUnnamed(One, Two, Three);

#[test]
fn combined_unnamed() {
    let combined_unnamed = CombinedUnnamed(One, Two, Three);
    check!(combined_unnamed.maximum_interaction_range() == 3.0);
}


#[derive(MaximumInteractionRange)]
struct SitePair {
    maximum_interaction_range: f64,
}

#[test]
fn site_pair() {
    let site_pair = SitePair { maximum_interaction_range: 5.0, };
    check!(site_pair.maximum_interaction_range() == 5.0);
}

// Test that the derive macro doesn't have issues with Anisotropic<AngularMask...
// In some contexts, it needs to be used as Anisotropic::<AngularMask..., but cargo fmt
// removes the :: when not needed.

#[allow(dead_code, reason = "The test passes if it compiles")]
#[derive(MaximumInteractionRange, SitePairEnergy)]
struct SitePairInteraction {
    hard_disk: HardSphere,
    angular_mask: Anisotropic<AngularMask<Boxcar, Cartesian<2>>>,
}

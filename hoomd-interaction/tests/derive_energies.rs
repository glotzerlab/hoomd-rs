// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Test derive(DeltaEnergyInsert)

use hoomd_interaction::{
    DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, External, TotalEnergy, external::ConstantForce,
};
use hoomd_microstate::{Body, Microstate};
use hoomd_vector::{Cartesian, Vector};

use assert2::check;

// Compile error
// #[derive(DeltaEnergyInsert)]
// enum Enum {
//     A,B
// }

// Compile error
// #[derive(DeltaEnergyInsert)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

#[derive(DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, TotalEnergy)]
struct Unit;

#[test]
fn unit() -> anyhow::Result<()> {
    let mut microstate = Microstate::new();
    microstate.extend_bodies([
        Body::point(Cartesian::from([1.0, 0.0])),
        Body::point(Cartesian::from([0.0, 2.0])),
    ])?;

    let unit = Unit;
    check!(unit.total_energy(&microstate) == 0.0);

    let new_body = Body::point(Cartesian::from([1.0, 4.0]));
    check!(unit.delta_energy_one(&microstate, 0, &new_body) == 0.0);
    check!(unit.delta_energy_insert(&microstate, &new_body) == 0.0);
    check!(unit.delta_energy_remove(&microstate, 0) == 0.0);

    Ok(())
}

#[derive(DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, TotalEnergy)]
struct CombinedNamed {
    one: External<ConstantForce<Cartesian<2>>>,
    two: External<ConstantForce<Cartesian<2>>>,
    three: External<ConstantForce<Cartesian<2>>>,
}

#[test]
fn combined_named() -> anyhow::Result<()> {
    let mut microstate = Microstate::new();
    microstate.extend_bodies([
        Body::point(Cartesian::from([3.0, 0.0])),
        Body::point(Cartesian::from([0.0, 1.0])),
    ])?;

    let one = External(ConstantForce {
        force: [-1.0, 0.0].into(),
        r_0: Cartesian::default(),
    });
    let two = External(ConstantForce {
        force: [0.0, 2.0].into(),
        r_0: Cartesian::default(),
    });
    let three = External(ConstantForce {
        force: [-3.0, 0.0].into(),
        r_0: Cartesian::default(),
    });
    check!(one.total_energy(&microstate) == 3.0);
    check!(two.total_energy(&microstate) == -2.0);
    check!(three.total_energy(&microstate) == 9.0);

    let combined_named = CombinedNamed { one, two, three };
    check!(combined_named.total_energy(&microstate) == 10.0);

    let new_body = Body::point(Cartesian::from([2.0, 1.0]));
    check!(combined_named.delta_energy_one(&microstate, 0, &new_body) == -6.0);
    check!(combined_named.delta_energy_insert(&microstate, &new_body) == 6.0);
    check!(combined_named.delta_energy_remove(&microstate, 1) == 2.0);

    Ok(())
}

#[derive(DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, TotalEnergy)]
struct CombinedUnnamed(
    External<ConstantForce<Cartesian<2>>>,
    External<ConstantForce<Cartesian<2>>>,
    External<ConstantForce<Cartesian<2>>>,
);

#[test]
fn combined_unnamed() -> anyhow::Result<()> {
    let mut microstate = Microstate::new();
    microstate.extend_bodies([
        Body::point(Cartesian::from([3.0, 0.0])),
        Body::point(Cartesian::from([0.0, 1.0])),
    ])?;

    let one = External(ConstantForce {
        force: [-1.0, 0.0].into(),
        r_0: Cartesian::default(),
    });
    let two = External(ConstantForce {
        force: [0.0, 2.0].into(),
        r_0: Cartesian::default(),
    });
    let three = External(ConstantForce {
        force: [-3.0, 0.0].into(),
        r_0: Cartesian::default(),
    });
    check!(one.total_energy(&microstate) == 3.0);
    check!(two.total_energy(&microstate) == -2.0);
    check!(three.total_energy(&microstate) == 9.0);

    let combined_unnamed = CombinedUnnamed(one, two, three);
    check!(combined_unnamed.total_energy(&microstate) == 10.0);

    let new_body = Body::point(Cartesian::from([2.0, 1.0]));
    check!(combined_unnamed.delta_energy_one(&microstate, 0, &new_body) == -6.0);
    check!(combined_unnamed.delta_energy_insert(&microstate, &new_body) == 6.0);
    check!(combined_unnamed.delta_energy_remove(&microstate, 1) == 2.0);

    Ok(())
}

#[derive(DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, TotalEnergy)]
struct CombinedNamedGeneric<V: Vector, E> where
E: Clone,
{
    one: External<ConstantForce<V>>,
    two: External<ConstantForce<V>>,
    three: E,
}

#[test]
fn combined_named_generic() -> anyhow::Result<()> {
    let mut microstate = Microstate::new();
    microstate.extend_bodies([
        Body::point(Cartesian::from([3.0, 0.0])),
        Body::point(Cartesian::from([0.0, 1.0])),
    ])?;

    let one = External(ConstantForce {
        force: [-1.0, 0.0].into(),
        r_0: Cartesian::default(),
    });
    let two = External(ConstantForce {
        force: [0.0, 2.0].into(),
        r_0: Cartesian::default(),
    });
    let three = External(ConstantForce {
        force: [-3.0, 0.0].into(),
        r_0: Cartesian::default(),
    });
    check!(one.total_energy(&microstate) == 3.0);
    check!(two.total_energy(&microstate) == -2.0);
    check!(three.total_energy(&microstate) == 9.0);

    let combined_named_generic = CombinedNamedGeneric { one, two, three };
    check!(combined_named_generic.total_energy(&microstate) == 10.0);

    let new_body = Body::point(Cartesian::from([2.0, 1.0]));
    check!(combined_named_generic.delta_energy_one(&microstate, 0, &new_body) == -6.0);
    check!(combined_named_generic.delta_energy_insert(&microstate, &new_body) == 6.0);
    check!(combined_named_generic.delta_energy_remove(&microstate, 1) == 2.0);

    Ok(())
}

// Check that no syntax errors are created when there is no trailing comma.
#[expect(dead_code, reason = "The implementation is tested above.")]
#[derive(DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, TotalEnergy)]
struct CombinedNamedGenericNoComma<V: Vector, E> where
E: Clone
{
    one: External<ConstantForce<V>>,
    two: External<ConstantForce<V>>,
    three: E,
}

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Test derive(NetSiteForceAndVirial)

use hoomd_interaction::{
    External, NetSiteForceAndVirial, external::ConstantForce
};
use hoomd_microstate::{Body, Microstate};
use hoomd_vector::{Cartesian, Vector};

use assert2::check;

// Compile error
// #[derive(NetSiteForceAndVirial)]
// enum Enum {
//     A,B
// }

// Compile error
// #[derive(NetSiteForceAndVirial)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

#[derive(NetSiteForceAndVirial)]
struct Unit;

#[test]
fn unit() -> anyhow::Result<()> {
    let mut microstate = Microstate::new();
    microstate.extend_bodies([
        Body::point(Cartesian::from([1.0, 0.0])),
        Body::point(Cartesian::from([0.0, 2.0])),
    ])?;

    let unit = Unit;

    let (force, virial) = unit.net_site_force_and_virial(&microstate, 0);
    check!(force == [0.0, 0.0].into());

    let (force, virial) = unit.net_site_force_and_virial(&microstate, 1);
    check!(force == [0.0, 0.0].into());

    Ok(())
}

#[derive(NetSiteForceAndVirial)]
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

    let combined_named = CombinedNamed { one, two, three };

    let (force, virial) = combined_named.net_site_force_and_virial(&microstate, 0);
    check!(force == [-4.0, 2.0].into());

    let (force, virial) = combined_named.net_site_force_and_virial(&microstate, 1);
    check!(force == [-4.0, 2.0].into());

    Ok(())
}

#[derive(NetSiteForceAndVirial)]
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

    let combined_unnamed = CombinedUnnamed(one, two, three);

    let (force, virial) = combined_unnamed.net_site_force_and_virial(&microstate, 0);
    check!(force == [-4.0, 2.0].into());

    let (force, virial) = combined_unnamed.net_site_force_and_virial(&microstate, 1);
    check!(force == [-4.0, 2.0].into());

    Ok(())
}

#[derive(NetSiteForceAndVirial)]
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

    let combined_named_generic = CombinedNamedGeneric { one, two, three };

    let (force, virial) = combined_named_generic.net_site_force_and_virial(&microstate, 0);
    check!(force == [-4.0, 2.0].into());

    let (force, virial) = combined_named_generic.net_site_force_and_virial(&microstate, 1);
    check!(force == [-4.0, 2.0].into());

    Ok(())
}

// Check that no syntax errors are created when there is no trailing comma.
#[expect(dead_code, reason = "The implementation is tested above.")]
#[derive(NetSiteForceAndVirial)]
struct CombinedNamedGenericNoComma<V: Vector, E> where
E: Clone
{
    one: External<ConstantForce<V>>,
    two: External<ConstantForce<V>>,
    three: E,
}

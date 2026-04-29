// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Test derive(NetSiteForce)

use hoomd_interaction::{
    External, NetSiteForce, external::ConstantForce
};
use hoomd_microstate::{Body, Microstate};
use hoomd_vector::{Cartesian, Vector};

use assert2::check;

// Compile error
// #[derive(NetSiteForce)]
// enum Enum {
//     A,B
// }

// Compile error
// #[derive(NetSiteForce)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

#[derive(NetSiteForce)]
struct Unit;

#[test]
fn unit() -> anyhow::Result<()> {
    let mut microstate = Microstate::new();
    microstate.extend_bodies([
        Body::point(Cartesian::from([1.0, 0.0])),
        Body::point(Cartesian::from([0.0, 2.0])),
    ])?;

    let unit = Unit;
    check!(unit.net_site_force(&microstate, 0) == [0.0, 0.0].into());
    check!(unit.net_site_force(&microstate, 1) == [0.0, 0.0].into());

    Ok(())
}

#[derive(NetSiteForce)]
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
    check!(combined_named.net_site_force(&microstate, 0) == [-4.0, 2.0].into());
    check!(combined_named.net_site_force(&microstate, 1) == [-4.0, 2.0].into());

    Ok(())
}

#[derive(NetSiteForce)]
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
    check!(combined_unnamed.net_site_force(&microstate, 0) == [-4.0, 2.0].into());
    check!(combined_unnamed.net_site_force(&microstate, 1) == [-4.0, 2.0].into());

    Ok(())
}

#[derive(NetSiteForce)]
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
    check!(combined_named_generic.net_site_force(&microstate, 0) == [-4.0, 2.0].into());
    check!(combined_named_generic.net_site_force(&microstate, 1) == [-4.0, 2.0].into());

    Ok(())
}

// Check that no syntax errors are created when there is no trailing comma.
#[expect(dead_code, reason = "The implementation is tested above.")]
#[derive(NetSiteForce)]
struct CombinedNamedGenericNoComma<V: Vector, E> where
E: Clone
{
    one: External<ConstantForce<V>>,
    two: External<ConstantForce<V>>,
    three: E,
}

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Test derive(NetSiteForceAndTorque)

use hoomd_interaction::{
    External, NetSiteForceAndTorque, external::{ConstantForce, ConstantTorque}
};
use hoomd_microstate::{Body, Microstate};
use hoomd_vector::{Cartesian, Vector};

use assert2::check;

// Compile error
// #[derive(NetSiteForceAndTorque)]
// enum Enum {
//     A,B
// }

// Compile error
// #[derive(NetSiteForceAndTorque)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

#[derive(NetSiteForceAndTorque)]
struct Unit;

#[test]
fn unit_2d() -> anyhow::Result<()> {
    let mut microstate = Microstate::new();
    microstate.extend_bodies([
        Body::point(Cartesian::from([1.0, 0.0])),
        Body::point(Cartesian::from([0.0, 2.0])),
    ])?;

    let unit = Unit;
    let (force, torque) = unit.net_site_force_and_torque(&microstate, 0);
    check!(force == [0.0, 0.0].into());
    check!(torque == 0.0);

    let (force, torque) = unit.net_site_force_and_torque(&microstate, 1);
    check!(force == [0.0, 0.0].into());
    check!(torque == 0.0);

    Ok(())
}

#[test]
fn unit_3d() -> anyhow::Result<()> {
    let mut microstate = Microstate::new();
    microstate.extend_bodies([
        Body::point(Cartesian::from([1.0, 0.0, 0.0])),
        Body::point(Cartesian::from([0.0, 2.0, -1.0])),
    ])?;

    let unit = Unit;
    let (force, torque) = unit.net_site_force_and_torque(&microstate, 0);
    check!(force == [0.0, 0.0, 0.0].into());
    check!(torque == [0.0, 0.0, 0.0].into());

    let (force, torque) = unit.net_site_force_and_torque(&microstate, 1);
    check!(force == [0.0, 0.0, 0.0].into());
    check!(torque == [0.0, 0.0, 0.0].into());

    Ok(())
}

#[derive(NetSiteForceAndTorque)]
struct CombinedNamed {
    one: External<ConstantForce<Cartesian<2>>>,
    two: External<ConstantForce<Cartesian<2>>>,
    three: External<ConstantTorque<Cartesian<2>>>,
    four: External<ConstantTorque<Cartesian<2>>>,
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
    let three = External(ConstantTorque {
        torque: 4.0,
    });
    let four = External(ConstantTorque {
        torque: -6.0,
    });

    let combined_named = CombinedNamed { one, two, three, four };

    let (force, torque) = combined_named.net_site_force_and_torque(&microstate, 0);
    check!(force == [-1.0, 2.0].into());
    check!(torque == -2.0);

    let (force, torque) = combined_named.net_site_force_and_torque(&microstate, 1);
    check!(force == [-1.0, 2.0].into());
    check!(torque == -2.0);

    Ok(())
}

#[derive(NetSiteForceAndTorque)]
struct CombinedUnnamed(
    External<ConstantForce<Cartesian<3>>>,
    External<ConstantForce<Cartesian<3>>>,
    External<ConstantTorque<Cartesian<3>>>,
    External<ConstantTorque<Cartesian<3>>>,
);

#[test]
fn combined_unnamed() -> anyhow::Result<()> {
    let mut microstate = Microstate::new();
    microstate.extend_bodies([
        Body::point(Cartesian::from([3.0, 0.0, 0.0])),
        Body::point(Cartesian::from([0.0, 1.0, 0.0])),
    ])?;

    let one = External(ConstantForce {
        force: [-1.0, 0.0, 2.5].into(),
        r_0: Cartesian::default(),
    });
    let two = External(ConstantForce {
        force: [0.0, 2.0, 4.5].into(),
        r_0: Cartesian::default(),
    });
    let three = External(ConstantTorque {
        torque: [-3.0, 6.0, -2.0].into(),
    });
    let four = External(ConstantTorque {
        torque: [4.0, -3.0, -3.0].into(),
    });

    let combined_unnamed = CombinedUnnamed(one, two, three, four);

    let (force, torque) = combined_unnamed.net_site_force_and_torque(&microstate, 0);
    check!(force == [-1.0, 2.0, 7.0].into());
    check!(torque == [1.0, 3.0, -5.0].into());

    let (force, torque) = combined_unnamed.net_site_force_and_torque(&microstate, 1);
    check!(force == [-1.0, 2.0, 7.0].into());
    check!(torque == [1.0, 3.0, -5.0].into());

    Ok(())
}

#[derive(NetSiteForceAndTorque)]
struct CombinedNamedGeneric<V: Vector, E> where
E: Clone,
{
    one: External<ConstantForce<V>>,
    two: External<ConstantForce<V>>,
    three: E,
    four: E,
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
    let three = External(ConstantTorque {
        torque: 4.0,
    });
    let four = External(ConstantTorque {
        torque: -6.0,
    });

    let combined_named = CombinedNamedGeneric { one, two, three, four };

    let (force, torque) = combined_named.net_site_force_and_torque(&microstate, 0);
    check!(force == [-1.0, 2.0].into());
    check!(torque == -2.0);

    let (force, torque) = combined_named.net_site_force_and_torque(&microstate, 1);
    check!(force == [-1.0, 2.0].into());
    check!(torque == -2.0);

    Ok(())
}

// Check that no syntax errors are created when there is no trailing comma.
#[expect(dead_code, reason = "The implementation is tested above.")]
#[derive(NetSiteForceAndTorque)]
struct CombinedNamedGenericNoComma<V: Vector, E> where
E: Clone
{
    one: External<ConstantForce<V>>,
    two: External<ConstantForce<V>>,
    three: E,
    four: E,
}

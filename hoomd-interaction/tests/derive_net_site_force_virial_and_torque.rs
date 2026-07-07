// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Test derive(NetSiteForceVirialAndTorque)

use hoomd_interaction::{
    External, NetSiteForceVirialAndTorque,
    external::{ConstantForce, ConstantTorque},
};
use hoomd_linear_algebra::matrix::Matrix;
use hoomd_microstate::{Body, Microstate};
use hoomd_vector::{Cartesian, Vector};

use assert2::check;

// Compile error
// #[derive(NetSiteForceVirialAndTorque)]
// enum Enum {
//     A,B
// }

// Compile error
// #[derive(NetSiteForceVirialAndTorque)]
// union Union {
//     f1: u32,
//     f2: f32,
// }

#[derive(NetSiteForceVirialAndTorque)]
struct Unit;

#[test]
fn unit_2d() -> anyhow::Result<()> {
    let mut microstate = Microstate::new();
    microstate.extend_bodies([
        Body::point(Cartesian::from([1.0, 0.0])),
        Body::point(Cartesian::from([0.0, 2.0])),
    ])?;

    let unit = Unit;
    let (force, virial, torque) = unit.net_site_force_virial_and_torque(&microstate, 0);
    check!(force == [0.0, 0.0].into());
    check!(torque == 0.0);
    check!(
        virial
            == Matrix {
                rows: [[0.0, 0.0], [0.0, 0.0]]
            }
    );

    let (force, virial, torque) = unit.net_site_force_virial_and_torque(&microstate, 1);
    check!(force == [0.0, 0.0].into());
    check!(torque == 0.0);
    check!(
        virial
            == Matrix {
                rows: [[0.0, 0.0], [0.0, 0.0]]
            }
    );

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
    let (force, virial, torque) = unit.net_site_force_virial_and_torque(&microstate, 0);
    check!(force == [0.0, 0.0, 0.0].into());
    check!(torque == [0.0, 0.0, 0.0].into());
    check!(
        virial
            == Matrix {
                rows: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            }
    );

    let (force, virial, torque) = unit.net_site_force_virial_and_torque(&microstate, 1);
    check!(force == [0.0, 0.0, 0.0].into());
    check!(torque == [0.0, 0.0, 0.0].into());
    check!(
        virial
            == Matrix {
                rows: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            }
    );

    Ok(())
}

#[derive(NetSiteForceVirialAndTorque)]
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
    let three = External(ConstantTorque { torque: 4.0 });
    let four = External(ConstantTorque { torque: -6.0 });

    let combined_named = CombinedNamed {
        one,
        two,
        three,
        four,
    };

    let (force, virial, torque) = combined_named.net_site_force_virial_and_torque(&microstate, 0);
    check!(force == [-1.0, 2.0].into());
    check!(torque == -2.0);
    check!(
        virial
            == Matrix {
                rows: [[-3.0, 0.0], [6.0, 0.0]]
            }
    );

    let (force, virial, torque) = combined_named.net_site_force_virial_and_torque(&microstate, 1);
    check!(force == [-1.0, 2.0].into());
    check!(torque == -2.0);
    check!(
        virial
            == Matrix {
                rows: [[0.0, -1.0], [0.0, 2.0]]
            }
    );

    Ok(())
}

#[derive(NetSiteForceVirialAndTorque)]
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

    let (force, virial, torque) = combined_unnamed.net_site_force_virial_and_torque(&microstate, 0);
    check!(force == [-1.0, 2.0, 7.0].into());
    check!(torque == [1.0, 3.0, -5.0].into());
    check!(
        virial
            == Matrix {
                rows: [[-3.0, 0.0, 0.0], [6.0, 0.0, 0.0], [21.0, 0.0, 0.0],]
            }
    );

    let (force, virial, torque) = combined_unnamed.net_site_force_virial_and_torque(&microstate, 1);
    check!(force == [-1.0, 2.0, 7.0].into());
    check!(torque == [1.0, 3.0, -5.0].into());
    check!(
        virial
            == Matrix {
                rows: [[0.0, -1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 7.0, 0.0],]
            }
    );

    Ok(())
}

#[derive(NetSiteForceVirialAndTorque)]
struct CombinedNamedGeneric<V: Vector, E>
where
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
    let three = External(ConstantTorque { torque: 4.0 });
    let four = External(ConstantTorque { torque: -6.0 });

    let combined_named = CombinedNamedGeneric {
        one,
        two,
        three,
        four,
    };

    let (force, virial, torque) = combined_named.net_site_force_virial_and_torque(&microstate, 0);
    check!(force == [-1.0, 2.0].into());
    check!(torque == -2.0);
    check!(
        virial
            == Matrix {
                rows: [[-3.0, 0.0], [6.0, 0.0]]
            }
    );

    let (force, virial, torque) = combined_named.net_site_force_virial_and_torque(&microstate, 1);
    check!(force == [-1.0, 2.0].into());
    check!(torque == -2.0);
    check!(
        virial
            == Matrix {
                rows: [[0.0, -1.0], [0.0, 2.0]]
            }
    );

    Ok(())
}

// Check that no syntax errors are created when there is no trailing comma.
#[expect(dead_code, reason = "The implementation is tested above.")]
#[derive(NetSiteForceVirialAndTorque)]
struct CombinedNamedGenericNoComma<V: Vector, E>
where
    E: Clone,
{
    one: External<ConstantForce<V>>,
    two: External<ConstantForce<V>>,
    three: E,
    four: E,
}

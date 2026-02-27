//! Test derive(DeltaEnergyInsert)

use hoomd_interaction::{
    DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, TotalEnergy, External, external::Linear,
};
use hoomd_microstate::{Body, Microstate};
use hoomd_vector::Cartesian;

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
    one: External<Linear<Cartesian<2>>>,
    two: External<Linear<Cartesian<2>>>,
    three: External<Linear<Cartesian<2>>>,
}

#[test]
fn combined_named() -> anyhow::Result<()> {
    let mut microstate = Microstate::new();
    microstate.extend_bodies([
        Body::point(Cartesian::from([3.0, 0.0])),
        Body::point(Cartesian::from([0.0, 1.0])),
    ])?;
    
    let one = External(Linear {
        alpha: 1.0,
        plane_origin: Cartesian::default(),
        plane_normal: [1.0, 0.0].try_into()?,
    });
    let two = External(Linear {
        alpha: 2.0,
        plane_origin: Cartesian::default(),
        plane_normal: [0.0, -1.0].try_into()?,
    });
    let three = External(Linear {
        alpha: 3.0,
        plane_origin: Cartesian::default(),
        plane_normal: [1.0, 0.0].try_into()?,
    });
    check!(one.total_energy(&microstate) == 3.0);
    check!(two.total_energy(&microstate) == -2.0);
    check!(three.total_energy(&microstate) == 9.0);

    let combined_named = CombinedNamed { one, two, three};
    check!(combined_named.total_energy(&microstate) == 10.0);
   
    let new_body = Body::point(Cartesian::from([2.0, 1.0]));
    check!(combined_named.delta_energy_one(&microstate, 0, &new_body) == -6.0);
    check!(combined_named.delta_energy_insert(&microstate, &new_body) == 6.0);
    check!(combined_named.delta_energy_remove(&microstate, 1) == 2.0);

    Ok(())
}

#[derive(DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, TotalEnergy)]
struct CombinedUnnamed(External<Linear<Cartesian<2>>>, External<Linear<Cartesian<2>>>, External<Linear<Cartesian<2>>>);

#[test]
fn combined_unnamed() -> anyhow::Result<()> {
    let mut microstate = Microstate::new();
    microstate.extend_bodies([
        Body::point(Cartesian::from([3.0, 0.0])),
        Body::point(Cartesian::from([0.0, 1.0])),
    ])?;
    
    let one = External(Linear {
        alpha: 1.0,
        plane_origin: Cartesian::default(),
        plane_normal: [1.0, 0.0].try_into()?,
    });
    let two = External(Linear {
        alpha: 2.0,
        plane_origin: Cartesian::default(),
        plane_normal: [0.0, -1.0].try_into()?,
    });
    let three = External(Linear {
        alpha: 3.0,
        plane_origin: Cartesian::default(),
        plane_normal: [1.0, 0.0].try_into()?,
    });
    check!(one.total_energy(&microstate) == 3.0);
    check!(two.total_energy(&microstate) == -2.0);
    check!(three.total_energy(&microstate) == 9.0);

    let combined_unnamed = CombinedUnnamed ( one, two, three);
    check!(combined_unnamed.total_energy(&microstate) == 10.0);
   
    let new_body = Body::point(Cartesian::from([2.0, 1.0]));
    check!(combined_unnamed.delta_energy_one(&microstate, 0, &new_body) == -6.0);
    check!(combined_unnamed.delta_energy_insert(&microstate, &new_body) == 6.0);
    check!(combined_unnamed.delta_energy_remove(&microstate, 1) == 2.0);

    Ok(())
}

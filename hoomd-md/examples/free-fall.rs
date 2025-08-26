// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Simple example of a falling body with MD.
*/

use hoomd_vector::Cartesian;
use hoomd_microstate::{Microstate, Body, property::{Point, DynamicsPoint, Position}};
use hoomd_interaction::{Single, external::Linear};
use hoomd_md::{ConstantVolume, TranslationalMotion, thermostat::NoThermostat};

// Question: Why do I have to import TranslationalMotion and Position?

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut microstate = Microstate::new();
    let body = Body {
        properties: DynamicsPoint {
            position: Cartesian::from([0.0, 1.0]),
            velocity: Cartesian::from([0.0, 0.0]),
            acceleration: Cartesian::from([0.0, 0.0]),
            mass: 1.0
        },
        sites: vec![Point::default()]
    };

    microstate.add_body(body)?;

    let force = Single(Linear {
        alpha: 2.0,
        plane_origin: [0.0, 1.0].into(),
        plane_normal: [0.0, 1.0].try_into()?,
    });

    let (kT, dt) = (1.0, 0.1);
    let thermostat = NoThermostat;

    let integrator = ConstantVolume { dt, kT, thermostat };

    for _ in 0..10 {
        integrator.integrate_translation(
            &mut microstate,
            &force
        );

        println!("Position of body 0: {}", microstate.bodies()[0].item.properties.position);
        microstate.increment_step();
    }

    Ok(())
}
// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Simple example of two bodies attracted to each other.
*/

use hoomd_vector::Cartesian;
use hoomd_microstate::{Microstate, Body, property::{Point, DynamicsPoint}};
use hoomd_interaction::{pairwise::LennardJones, CutoffPair};
use hoomd_md::{ConstantVolume, TranslationalMotion, thermostat::NoThermostat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create system
    let mut microstate = Microstate::new();
    let body1 = Body {
        properties: DynamicsPoint {
            position: Cartesian::from([0.0, 0.0]),
            velocity: Cartesian::from([0.0, 0.0]),
            acceleration: Cartesian::from([0.0, 0.0]),
            mass: 1.0
        },
        sites: vec![Point::default()]
    };

    let body2 = Body {
        properties: DynamicsPoint {
            position: Cartesian::from([0.0, 1.0]),
            velocity: Cartesian::from([0.0, 0.0]),
            acceleration: Cartesian::from([0.0, 0.0]),
            mass: 1.0
        },
        sites: vec![Point::default()]
    };

    microstate.add_body(body1)?;
    microstate.add_body(body2)?;

    // Model interactions
    let force = CutoffPair {
        r_cut: 1000.0,
        evaluator: LennardJones::<12,6> {
            epsilon: 1.0,
            sigma: 2.0
        }
    };

    // Create integrator
    let (kT, dt) = (1.0, 0.1);
    let thermostat = NoThermostat;
    let integrator = ConstantVolume { dt, kT, thermostat };

    // Simulation loop
    for _ in 0..5 {
        integrator.integrate_translation(
            &mut microstate,
            &force
        );

        println!("Position of body 0: {}", microstate.bodies()[0].item.properties.position);
        println!("Position of body 1: {}", microstate.bodies()[1].item.properties.position);

        microstate.increment_step();
    }

    Ok(())
}
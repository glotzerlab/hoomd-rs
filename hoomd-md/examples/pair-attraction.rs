// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Simple example of two bodies attracted to each other.
*/

use hoomd_vector::Cartesian;
use hoomd_microstate::{Microstate, Body, property::{Point, DynamicsPoint}};
use hoomd_interaction::{pairwise::{Isotropic, LennardJones}, rigid::Rigid, CutoffPair};
use hoomd_md::{ConstantVolume, TranslationalMotion, thermostat::NoThermostat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a microstate with two bodies, each with a single site
    let mut microstate = Microstate::new();
    let body1 = Body {
        properties: DynamicsPoint {
            position: Cartesian::from([0.0, 0.0]),
            momentum: Cartesian::from([0.0, 0.0]),
            net_force: Cartesian::from([0.0, 0.0]),
            mass: 1.0
        },
        sites: vec![Point::default()]
    };

    let body2 = Body {
        properties: DynamicsPoint {
            position: Cartesian::from([0.0, 2.5]),
            momentum: Cartesian::from([0.0, 0.0]),
            net_force: Cartesian::from([0.0, 0.0]),
            mass: 1.0
        },
        sites: vec![Point::default()]
    };

    microstate.add_body(body1)?;
    microstate.add_body(body2)?;


    // Model interactions (in this case, a pairwise Lennard-Jones)
    let force = Rigid(CutoffPair {
        r_cut: 6.0,
        evaluator: Isotropic(LennardJones::<12,6> {
            epsilon: 1.0,
            sigma: 1.0
        })
    });

    // Create an NVE macrostate
    struct Isoenergy {};
    
    let macrostate = Isoenergy{};

    // Create a constant-volume integrator
    let dt = 0.1;
    let mut integrator = ConstantVolume::new(dt);

    // Constant V integration requires a thermostat, even if it does nothing
    let mut thermostat = NoThermostat;

    // Simulation loop
    for timestep in 0..10 {
        integrator.integrate_translation_step_one(
            &mut microstate,
            &force,
            &mut thermostat,
            &macrostate,
        );

        integrator.integrate_translation_step_two(
            &mut microstate,
            &force,
            &mut thermostat,
            &macrostate,
        );

        println!("=============== {} ===============", timestep);
        println!("Position of body 0: {}", microstate.bodies()[0].item.properties.position);
        println!("Position of body 1: {}", microstate.bodies()[1].item.properties.position);

        microstate.increment_step();
    }

    Ok(())
}
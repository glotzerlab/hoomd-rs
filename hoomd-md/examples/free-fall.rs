// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Simple example of a falling body with MD.
*/

use hoomd_simulation::macrostate::{Isoentropic, Isochoric};
use hoomd_vector::Cartesian;
use hoomd_microstate::{Microstate, Body, property::{Point, DynamicsPoint, Position}};
use hoomd_interaction::{external::Linear, pairwise::Isotropic, Single};
use hoomd_md::{ConstantVolume, TranslationalMotion, thermostat::NoThermostat};

// Question: Why do I have to import TranslationalMotion and Position?

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize a particle.
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

    // Define my set macrostat.
    struct IsoentropicMacrostat {}
    impl Isoentropic for IsoentropicMacrostat {}
    impl Isochoric for IsoentropicMacrostat{}

    let macrostate = IsoentropicMacrostat{};

    // Add gravity
    let force = Single(Linear {
        alpha: -2.0,
        plane_origin: [0.0, 1.0].into(),
        plane_normal: [0.0, 1.0].try_into()?,
    });

    let (kT, dt) = (1.0, 0.1);
    let mut thermostat = NoThermostat;

    let mut integrator = ConstantVolume::new(dt);

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
        
        println!("==============={}===============", timestep);
        println!("Position of body 0: {}", microstate.bodies()[0].item.properties.position);
        println!("Kinetic energy of body 0: {}", integrator.get_kinetic_energy());
        microstate.increment_step();
    }

    Ok(())
}
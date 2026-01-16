// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Simple example of a falling body with MD.
*/

// use hoomd_simulation::macrostate::{Isoenergy};
use hoomd_interaction::{External, external::ConstantForce, rigid::Rigid};
use hoomd_md::{ConstantVolume, ForceUpdate, TranslationalMotion, thermostat::NoThermostat};
use hoomd_microstate::{
    Body, Microstate,
    property::{DynamicsPoint, Point},
};
use hoomd_vector::Cartesian;

// Question: Why do I have to import TranslationalMotion and Position?

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a microstate with a single body containing a single site
    let mut microstate = Microstate::new();
    let body = Body {
        properties: DynamicsPoint {
            position: Cartesian::from([0.0, 1.0]),
            momentum: Cartesian::from([0.0, 0.0]),
            net_force: Cartesian::from([0.0, 0.0]),
            mass: 1.0,
        },
        sites: vec![Point::default()],
    };

    microstate.add_body(body)?;

    // Model interactions (in this case, just gravity)
    let force = Rigid(External(ConstantForce {
        alpha: -2.0,
        plane_origin: [0.0, 1.0].into(),
        plane_normal: [0.0, 1.0].try_into()?,
    }));

    // Create an NVE macrostate
    struct Isoenergy {};

    let macrostate = Isoenergy {};

    // Create a constant-volume integrator
    let dt = 0.1;
    let mut integrator = ConstantVolume::new(dt);

    // Constant V integration requires a thermostat, even if it does nothing
    let mut thermostat = NoThermostat;

    // Simulation loop
    for timestep in 0..10 {
        // Evolve the system forward using the integrator
        integrator.integrate_translation_step_one(
            &mut microstate,
            &mut thermostat,
            &macrostate,
        );

        integrator
            .update_force(&mut microstate, &force);

        integrator.integrate_translation_step_two(
            &mut microstate,
            &mut thermostat,
            &macrostate,
        );

        println!("=============== {} ===============", timestep);
        println!(
            "Position of body 0: {}",
            microstate.bodies()[0].item.properties.position
        );
        println!(
            "Kinetic energy of body 0: {}",
            integrator.get_translational_kinetic_energy()
        );
        microstate.increment_step();
    }

    Ok(())
}

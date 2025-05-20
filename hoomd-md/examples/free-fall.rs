#![allow(clippy::print_stdout, reason = "Demonstration purposes")]

/*! This is an example
*/

use hoomd_microstate::property::Position;
use hoomd_microstate::{Body, Microstate};
use hoomd_vector::Cartesian;
use hoomd_md::{thermostat, ConstantVolume};
use hoomd_md::thermostat::NoThermostat;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut microstate = Microstate::new();
    microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])))?;

    // What's needed for an integrator?
    let dt = 0.01;
    let kT = 1.0;
    let thermostat = NoThermostat;

    // let integrator = ConstantVolume {};

    for _ in 0..10 {
        // integrator.integrate(&mut microstate);
        println!("{}", microstate.bodies()[0].item.properties.position());
        microstate.increment_step();
    }

    Ok(())
}

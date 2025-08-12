#![allow(clippy::print_stdout, reason = "Demonstration purposes")]

/*! This is an example
*/

use hoomd_interaction;
use hoomd_microstate::property::Position;
use hoomd_microstate::{Body, Microstate};
use hoomd_vector::Cartesian;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut microstate = Microstate::new();
    microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])))?;

    let kt = 1.0;
    let hamiltonian = hoomd_interaction::Single(
        hoomd_interaction::external::Linear {
            alpha: 2.0,
            plane_origin: [0.0, 0.0].into(),
            plane_normal: [0.0, 1.0].try_into()?,
        }
    );
    let d = 0.1;

    let translate = hoomd_mc::Translate {
        maximum_distance: d.try_into()?,
    };
    let translate_sweep = hoomd_mc::Sweep { local: translate };

    for _ in 0..100_000 {
        translate_sweep.apply(&mut microstate, &hamiltonian, &kt);
        println!("{}", microstate.bodies()[0].item.properties.position());
        microstate.increment_step();
    }

    Ok(())
}
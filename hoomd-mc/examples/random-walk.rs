#![allow(clippy::print_stdout, reason = "Demonstration purposes")]

/*! This is an example
*/

use hoomd_mc::{Sweep, Translate, Trial, Zero};
use hoomd_microstate::property::Position;
use hoomd_microstate::{Body, Microstate};
use hoomd_vector::{Cartesian, PositiveReal};

fn main() {
    let mut microstate = Microstate::new();
    microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])));

    let kt = 1.0;
    let hamiltonian = Zero;

    let translate = Translate::new(PositiveReal::new(0.1).expect("positive real"));
    let translate_sweep = Sweep::new(translate);

    for _ in 0..100_000 {
        translate_sweep.apply(&mut microstate, &hamiltonian, &kt);
        println!("{}", microstate.bodies()[0].item.properties.position());
        microstate.increment_step();
    }
}

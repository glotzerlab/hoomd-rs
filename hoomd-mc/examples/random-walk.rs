/*! This is an example
*/

use hoomd_microstate::{Body, Microstate};
use hoomd_microstate::property::Position;
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_vector::Cartesian;


fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = Microstate::new();
microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])));

let translate = Translate::with_maximum_distance(0.1);
let translate_sweep = Sweep::new(translate);

for _ in 0..100_000 {
    translate_sweep.apply(&mut microstate);
    println!("{}", microstate.bodies()[0].item.properties.position());
    microstate.increment_step();
}


Ok(())
}

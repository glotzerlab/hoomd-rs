// ANCHOR: all
// ANCHOR: use
use hoomd_interaction::Zero;
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_microstate::{Body, MicrostateBuilder};
use hoomd_simulation::macrostate::Isothermal;
use hoomd_vector::Cartesian;
// ANCHOR_END: use

// ANCHOR: main
fn main() -> anyhow::Result<()> {
    // ANCHOR_END: main
    // ANCHOR: microstate
    let mut microstate = MicrostateBuilder::new()
        .bodies([Body::point(Cartesian::from([0.0, 0.0]))])
        .try_build()?;
    // ANCHOR_END: microstate

    // ANCHOR: local_trial
    let translate = Translate::with_maximum_distance(0.15.try_into()?);
    // ANCHOR_END: local_trial

    // ANCHOR: sweep
    let translate_sweep = Sweep(translate);
    // ANCHOR_END: sweep

    // ANCHOR: hamiltonian
    let hamiltonian = Zero;
    // ANCHOR_END: hamiltonian

    // ANCHOR: macrostate
    let macrostate = Isothermal { temperature: 1.0 };
    // ANCHOR_END: macrostate

    // ANCHOR: steps
    for _ in 0..100 {
        // ANCHOR_END: steps
        // ANCHOR: apply
        translate_sweep.apply(&mut microstate, &hamiltonian, &macrostate);
        // ANCHOR_END: apply
        // ANCHOR: increment
        microstate.increment_step();
        // ANCHOR_END: increment

        // ANCHOR: print
        println!("{}", microstate.sites()[0].properties.position);
    }
    // ANCHOR_END: print

    // ANCHOR: end_main
    Ok(())
}
// ANCHOR_END: end_main
// ANCHOR_END: all

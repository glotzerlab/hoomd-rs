// ANCHOR: all
// ANCHOR: use
use rand::{Rng, seq::IndexedRandom};
use std::iter;

use hoomd_geometry::IsPointInside;
use hoomd_interaction::Zero;
use hoomd_mc::{LocalTrial, Sweep, Trial};
use hoomd_microstate::{
    Body, Microstate, MicrostateBuilder, boundary::Closed, property::Point,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_vector::{Cartesian, Vector};
// ANCHOR_END: use

// ANCHOR: boundary_struct
/// Closed circular boundary condition.
struct Circle {
    radius: f64,
}
// ANCHOR_END: boundary_struct

// ANCHOR: boundary_impl
impl IsPointInside<Cartesian<2>> for Circle {
    fn is_point_inside(&self, point: &Cartesian<2>) -> bool {
        point.distance(&[0.0, 0.0].into()) < self.radius
    }
}
// ANCHOR_END: boundary_impl

// ANCHOR: local_trial_struct
/// Take fixed steps left, right, down, or up.
struct Discrete;
// ANCHOR_END: local_trial_struct

// ANCHOR: local_trial_impl
impl LocalTrial<Point<Cartesian<2>>> for Discrete {
    fn propose<R: Rng>(
        &self,
        rng: &mut R,
        body_properties: Point<Cartesian<2>>,
    ) -> Point<Cartesian<2>> {
        // ANCHOR_END: local_trial_impl
        // ANCHOR: local_trial_steps
        let steps = [
            [0.0, -1.0].into(),
            [0.0, 1.0].into(),
            [-1.0, 0.0].into(),
            [1.0, 0.0].into(),
        ];
        // ANCHOR_END: local_trial_steps

        // ANCHOR: local_trial_mut
        let mut trial = body_properties;
        trial.position += *steps
            .choose(rng)
            .expect("steps should have at least 1 element");
        trial
    }
}
// ANCHOR_END: local_trial_mut

// Remove the cfg_attr(...) line when using this code outside the hoomd-rs/examples directory.
#[cfg_attr(feature = "bevy", derive(Resource))]
// ANCHOR: simulation_struct
struct CustomRandomWalk {
    /// Positions of all the bodies in the simulation.
    microstate:
        Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Closed<Circle>>,
    /// How sites interact with other sites and fields.
    hamiltonian: Zero,
    /// Trial moves to apply.
    translate_sweep: Sweep<Discrete>,
    /// Temperature set point.
    macrostate: Isothermal,
}
// ANCHOR_END: simulation_struct

// ANCHOR: simulation_new
impl CustomRandomWalk {
    /// Construct a new random walk simulation.
    fn new() -> anyhow::Result<CustomRandomWalk> {
        // ANCHOR_END: simulation_new
        // ANCHOR: parameters
        let n = 1000;
        let radius = 50.0;
        let macrostate = Isothermal { temperature: 1.0 };
        // ANCHOR_END: parameters

        // ANCHOR: microstate
        let circle = Circle { radius };

        let microstate = MicrostateBuilder::with_boundary(Closed(circle))
            .bodies(iter::repeat_n(Body::point(Cartesian::default()), n))
            .try_build()?;
        // ANCHOR_END: microstate

        // ANCHOR: sweep
        let translate_sweep = Sweep(Discrete);
        // ANCHOR_END: sweep

        // ANCHOR: hamiltonian
        let hamiltonian = Zero;
        // ANCHOR_END: hamiltonian

        // ANCHOR: initialize_struct
        Ok(CustomRandomWalk {
            microstate,
            hamiltonian,
            translate_sweep,
            macrostate,
        })
    }
}
// ANCHOR_END: initialize_struct

// ANCHOR: impl_simulation
impl Simulation for CustomRandomWalk {
    // ANCHOR_END: impl_simulation
    // ANCHOR: advance
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        self.translate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
        );
        self.microstate.increment_step();
        Ok(())
    }
    // ANCHOR_END: advance

    // ANCHOR: step
    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}
// ANCHOR_END: step

// Remove the cfg(not(...)) line when using this code outside the hoomd-rs/examples directory.
#[cfg(not(feature = "bevy"))]
// ANCHOR: main
fn main() -> anyhow::Result<()> {
    let mut simulation = CustomRandomWalk::new()?;
    // TODO: Write GSD file.

    for _ in 0..100_000 {
        simulation.advance()?;
    }

    Ok(())
}
// ANCHOR_END: main
// ANCHOR_END: all

#[cfg(feature = "bevy")]
mod custom_random_walk_interactive;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
#[cfg(feature = "bevy")]
use custom_random_walk_interactive::main;

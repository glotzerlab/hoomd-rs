// ANCHOR: all
// ANCHOR: use
use hoomd_geometry::shape::Rectangle;
use hoomd_interaction::{
    CutoffPair, External, TotalEnergy,
    external::Linear,
    pairwise::{Boxcar, Isotropic},
};
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_microstate::{
    Body, Microstate, MicrostateBuilder, boundary::Closed, property::Point,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_vector::Cartesian;
// ANCHOR_END: use

// Remove the cfg_attr(...) line when using this code outside the hoomd-rs/examples directory.
#[cfg_attr(feature = "bevy", derive(Resource))]
// ANCHOR: simulation_struct
struct Fill {
    /// Positions of all the bodies in the simulation.
    microstate:
        Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Closed<Rectangle>>,
    /// How sites interact with other sites and fields.
    hamiltonian: (
        External<Linear<Cartesian<2>>>,
        CutoffPair<Isotropic<Boxcar>>,
    ),
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate<Cartesian<2>>>,
    /// Temperature set point.
    macrostate: Isothermal,
}
// ANCHOR_END: simulation_struct

// ANCHOR: simulation_new
impl Fill {
    /// Construct a new fill simulation.
    fn new() -> anyhow::Result<Fill> {
        // ANCHOR_END: simulation_new
        // ANCHOR: parameters
        let box_length = 30.0;
        let maximum_distance = 0.15;
        let alpha = 10.0;
        let epsilon = 1000.0;
        let sigma = 1.0;
        let macrostate = Isothermal { temperature: 1.0 };
        // ANCHOR_END: parameters

        // ANCHOR: microstate
        let square = Rectangle::with_equal_edges(box_length.try_into()?);
        let microstate =
            MicrostateBuilder::with_boundary(Closed(square)).try_build()?;
        // ANCHOR_END: microstate

        // ANCHOR: external
        let linear = External(Linear {
            alpha,
            plane_origin: Cartesian::default(),
            plane_normal: [0.0, 1.0].try_into()?,
        });
        // ANCHOR_END: external

        // ANCHOR: pair
        let boxcar = Boxcar {
            epsilon,
            left: 0.0,
            right: sigma,
        };
        let isotropic = Isotropic(boxcar);
        let cutoff_pair = CutoffPair {
            r_cut: sigma,
            evaluator: isotropic,
        };
        // ANCHOR_END: pair

        // ANCHOR: hamiltonian
        let hamiltonian = (linear, cutoff_pair);
        // ANCHOR_END: hamiltonian

        // ANCHOR: sweep
        let translate =
            Translate::with_maximum_distance(maximum_distance.try_into()?);
        let translate_sweep = Sweep(translate);
        // ANCHOR_END: sweep

        // ANCHOR: initialize_struct
        Ok(Fill {
            microstate,
            hamiltonian,
            translate_sweep,
            macrostate,
        })
    }
}
// ANCHOR_END: initialize_struct

// ANCHOR: impl_simulation
impl Simulation for Fill {
    // ANCHOR_END: impl_simulation
    // ANCHOR: advance
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        // ANCHOR_END: advance
        // ANCHOR: add
        let boundary = self.microstate.boundary();
        let y = boundary.0.edge_lengths[1].get() / 2.0 - 0.5;
        if self.microstate.step() % 100 == 0 {
            self.microstate.add_body(Body::point([0.0, y].into()))?;
        }
        // ANCHOR_END: add

        // ANCHOR: apply
        self.translate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
        );
        self.microstate.increment_step();
        // ANCHOR_END: apply

        // ANCHOR: reset
        if self.hamiltonian.1.total_energy(&self.microstate) > 20_000.0 {
            self.microstate.clear();
        }

        Ok(())
    }
    // ANCHOR_END: reset

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
    let mut simulation = Fill::new()?;
    // TODO: Write GSD file.

    for _ in 0..100_000 {
        simulation.advance()?;
    }

    Ok(())
}
// ANCHOR_END: main
// ANCHOR_END: all

#[cfg(feature = "bevy")]
mod applying_interactions_interactive;
#[cfg(feature = "bevy")]
use applying_interactions_interactive::main;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;

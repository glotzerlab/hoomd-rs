// ANCHOR: all
// ANCHOR: use
use anyhow::{Context, anyhow};

use hoomd_geometry::{
    Volume,
    shape::{Circle, Rectangle},
};
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff,
    pairwise::{HardSphere, Isotropic},
    univariate::{Expanded, OverlapPenalty},
};
use hoomd_mc::{QuickCompress, Sweep, Translate, Trial};
use hoomd_microstate::{
    Body, Microstate, SiteKey, boundary::Periodic, property::Point,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::Cartesian;
// ANCHOR_END: use

// ANCHOR: type_aliases
type PositionVector = Cartesian<2>;
type BodyProperties = Point<PositionVector>;
type SiteProperties = Point<PositionVector>;
// ANCHOR_END: type_aliases

#[cfg_attr(feature = "bevy", derive(Resource))]
// ANCHOR: simulation_struct
struct HardDiskSelfAssembly {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<
        BodyProperties,
        SiteProperties,
        VecCell<SiteKey, 2>,
        Periodic<Rectangle>,
    >,
    /// How sites interact with other sites and fields.
    hamiltonian: PairwiseCutoff<HardSphere>,
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate<PositionVector>>,
    /// Temperature set point.
    macrostate: Isothermal,
    /// Quick compress algorithm
    quick_compress: QuickCompress<Periodic<Rectangle>>,
    /// How sites interact during compression.
    overlap_penalty_hamiltonian:
        PairwiseCutoff<Isotropic<Expanded<OverlapPenalty>>>,
    /// The current phase of the simulation.
    phase: Phase,
}
// ANCHOR_END: simulation_struct

// ANCHOR: phase
enum Phase {
    Compress,
    Equilibrate,
}
// ANCHOR_END: phase

// ANCHOR: simulation_new
impl HardDiskSelfAssembly {
    /// Construct a new hard disk self-assembly simulation.
    fn new() -> anyhow::Result<HardDiskSelfAssembly> {
        // ANCHOR_END: simulation_new
        // ANCHOR: parameters
        let initial_packing_fraction = 0.4;
        let target_packing_fraction = 1.0;
        let n_disks = 64_usize.pow(2);
        let maximum_distance = 0.07;
        let sigma = 1.0;
        let macrostate = Isothermal { temperature: 1.0 };
        // ANCHOR_END: parameters

        // ANCHOR: hamiltonian
        let hamiltonian = PairwiseCutoff(HardSphere { diameter: sigma });
        // ANCHOR_END: hamiltonian

        // ANCHOR: periodic
        let circle = Circle {
            radius: (sigma / 2.0).try_into()?,
        };
        let initial_box_volume =
            n_disks as f64 * circle.volume() / initial_packing_fraction;
        let initial_box_edge_length = initial_box_volume.sqrt();
        let square =
            Rectangle::with_equal_edges(initial_box_edge_length.try_into()?);
        let periodic_square =
            Periodic::new(hamiltonian.0.maximum_interaction_range(), square)?;
        // ANCHOR_END: periodic

        // ANCHOR: microstate
        let vec_cell = VecCell::builder()
            .nominal_search_radius(sigma.try_into()?)
            .build();
        let mut microstate = Microstate::builder()
            .boundary(periodic_square)
            .spatial_data(vec_cell)
            .try_build()?;
        // ANCHOR_END: microstate

        // ANCHOR: place_disks
        let n_on_side_f64 = (n_disks as f64).sqrt().ceil();
        let a = initial_box_edge_length / n_on_side_f64;
        let n_on_side = n_on_side_f64 as usize;
        for j in 0..n_on_side {
            let y = -initial_box_edge_length / 2.0 + j as f64 * a;
            for i in 0..n_on_side {
                let x = -initial_box_edge_length / 2.0 + i as f64 * a;
                if microstate.bodies().len() < n_disks {
                    microstate
                        .add_body(Body::point(Cartesian::from([x, y])))?;
                }
            }
        }
        // ANCHOR_END: place_disks

        // ANCHOR: trial_moves
        let translate =
            Translate::with_maximum_distance(maximum_distance.try_into()?);
        let translate_sweep = Sweep(translate);
        // ANCHOR_END: trial_moves

        // ANCHOR: quick_compress
        let target_box_volume =
            n_disks as f64 * circle.volume() / target_packing_fraction;
        let quick_compress =
            QuickCompress::with_target_volume(target_box_volume.try_into()?);
        // ANCHOR_END: quick_compress

        // ANCHOR: compress_hamiltonian
        let overlap_penalty = Isotropic {
            interaction: Expanded {
                delta: sigma,
                f: OverlapPenalty::default(),
            },
            r_cut: sigma,
        };

        let overlap_penalty_hamiltonian = PairwiseCutoff(overlap_penalty);
        // ANCHOR_END: compress_hamiltonian

        // ANCHOR: struct_initialize
        Ok(HardDiskSelfAssembly {
            microstate,
            overlap_penalty_hamiltonian,
            hamiltonian,
            translate_sweep,
            quick_compress,
            macrostate,
            phase: Phase::Compress,
        })
    }
}
// ANCHOR_END: struct_initialize

// ANCHOR: impl_simulation
impl Simulation for HardDiskSelfAssembly {
    // ANCHOR_END: impl_simulation
    // ANCHOR: advance
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        match self.phase {
            Phase::Compress => self.apply().context("failed to compress")?,
            Phase::Equilibrate => self.equilibrate(),
        }

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

// ANCHOR: inherent_simulation
impl HardDiskSelfAssembly {
    // ANCHOR_END: inherent_simulation
    // ANCHOR: compress
    fn compress(&mut self) -> anyhow::Result<()> {
        // ANCHOR_END: compress
        // ANCHOR: apply_quick_compress
        self.quick_compress.apply(
            &mut self.microstate,
            &self.overlap_penalty_hamiltonian,
            |_| true,
        );
        // ANCHOR_END: apply_quick_compress

        // ANCHOR: compress_trial_moves
        self.translate_sweep.apply(
            &mut self.microstate,
            &self.overlap_penalty_hamiltonian,
            &Isothermal { temperature: 1.0 },
        );
        // ANCHOR_END: compress_trial_moves

        // ANCHOR: state_transition
        if self.quick_compress.is_complete() {
            self.phase = Phase::Equilibrate;
            println!(
                "Compression complete at step {}.",
                self.microstate.step()
            );
        }
        // ANCHOR_END: state_transition

        // ANCHOR: failed
        if self.step() >= 10_000 {
            let current = self.microstate.boundary().volume();
            let target = self.quick_compress.target_volume();
            let step = self.microstate.step();
            return Err(anyhow!(
                "Achieved volume {current} after {step} steps. The target was {target}."
            ));
        }

        Ok(())
    }
    // ANCHOR_END: failed

    // ANCHOR: equilibrate
    fn equilibrate(&mut self) {
        self.translate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
        );
    }
}
// ANCHOR_END: equilibrate

// Remove the cfg(not(...)) line when using this code outside the hoomd-rs/examples directory.
#[cfg(not(feature = "bevy"))]
// ANCHOR: main
fn main() -> anyhow::Result<()> {
    let mut simulation = HardDiskSelfAssembly::new()?;
    // TODO: Write GSD file.

    for _ in 0..10_000 {
        simulation.advance()?;
    }

    Ok(())
}
// ANCHOR_END: main
// ANCHOR_END: all

#[cfg(feature = "bevy")]
mod hard_disk_self_assembly_interactive;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
#[cfg(feature = "bevy")]
use hard_disk_self_assembly_interactive::main;

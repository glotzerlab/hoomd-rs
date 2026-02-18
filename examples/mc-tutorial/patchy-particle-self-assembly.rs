// ANCHOR: all
// ANCHOR: use
use anyhow::{Context, anyhow};
use parquet_derive::ParquetRecordWriter;

use hoomd_geometry::{
    Volume,
    shape::{Circle, Rectangle},
};
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff,
    pairwise::{
        AngularMask, Anisotropic, HardSphere, Isotropic, angular_mask::Patch,
    },
    univariate::{Boxcar, Expanded, OverlapPenalty},
};
use hoomd_mc::{
    QuickCompress, QuickInsert, Rotate, Sweep, Translate, Trial, Tune,
    UniformIn,
};
use hoomd_microstate::{
    Microstate, SiteKey, boundary::Periodic, property::OrientedPoint,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::{Angle, Cartesian};
// #[cfg(not(feature = "bevy"))]
use hoomd_interaction::TotalEnergy;
use thiserror::Error;
#[cfg(not(feature = "bevy"))]
use hoomd_utility::data::ParquetLogger;

use parquet::{
    file::{properties::WriterProperties, writer::SerializedFileWriter},
    record::RecordWriter,
};
use std::{fs::File, io, path::Path, sync::Arc};
// ANCHOR_END: use

// ANCHOR: type_aliases
type PositionVector = Cartesian<2>;
type Orientation = Angle;
type BodyProperties = OrientedPoint<PositionVector, Orientation>;
type SiteProperties = OrientedPoint<PositionVector, Orientation>;
// ANCHOR_END: type_aliases

// ANCHOR: simulation_new
impl PatchyParticleSelfAssembly {
    /// Construct a new hard disk self-assembly simulation.
    fn new() -> anyhow::Result<PatchyParticleSelfAssembly> {
        // ANCHOR_END: simulation_new
        // ANCHOR: parameters
        let initial_packing_fraction = 0.3;
        let target_packing_fraction = 0.3;
        let n_disks = 512;
        let maximum_distance = 0.07;
        let maximum_rotation = 0.04;
        let sigma = 1.0;
        let patch_interaction_range = 1.12;
        let patch_half_angle = 37.0_f64.to_radians();
        let patch_energy = -5.8;
        let macrostate = Isothermal { temperature: 1.0 };
        // ANCHOR_END: parameters

        // ANCHOR: hard_disk
        let hard_disk = HardSphere { diameter: sigma };
        // ANCHOR_END: hard_disk

        // ANCHOR: patch
        let boxcar = Boxcar {
            epsilon: patch_energy,
            left: 0.0,
            right: patch_interaction_range,
        };
        let masks = [
            Patch {
                director: [0.0, 1.0].try_into()?,
                cos_delta: patch_half_angle.cos(),
            },
            Patch {
                director: [0.0, -1.0].try_into()?,
                cos_delta: patch_half_angle.cos(),
            },
        ];
        let angular_mask = Anisotropic {
            interaction: AngularMask::new(boxcar, masks),
            r_cut: patch_interaction_range,
        };
        // ANCHOR_END: patch

        // ANCHOR: hamiltonian
        let hamiltonian = PairwiseCutoff((hard_disk, angular_mask));
        // ANCHOR_END: hamiltonian

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

        // ANCHOR: remainder
        let circle = Circle {
            radius: (sigma / 2.0).try_into()?,
        };
        let initial_box_volume =
            n_disks as f64 * circle.volume() / initial_packing_fraction;
        let initial_box_edge_length = initial_box_volume.sqrt();
        let square =
            Rectangle::with_equal_edges(initial_box_edge_length.try_into()?);
        let periodic_square =
            Periodic::new(hamiltonian.maximum_interaction_range(), square)?;

        let vec_cell = VecCell::builder()
            .nominal_search_radius(
                hamiltonian.maximum_interaction_range().try_into()?,
            )
            .build();
        let microstate = Microstate::builder()
            .boundary(periodic_square)
            .spatial_data(vec_cell)
            .try_build()?;

        let translate =
            Translate::with_maximum_distance(maximum_distance.try_into()?);
        let translate_sweep = Sweep(translate);

        let rotate =
            Rotate::with_maximum_rotation(maximum_rotation.try_into()?);
        let rotate_sweep = Sweep(rotate);

        let distribution = UniformIn {
            boundary: microstate.boundary().clone(),
            template_sites: vec![SiteProperties::default()],
        };
        let quick_insert = QuickInsert::new(distribution, n_disks);

        let target_box_volume =
            n_disks as f64 * circle.volume() / target_packing_fraction;
        let quick_compress =
            QuickCompress::with_target_volume(target_box_volume.try_into()?);

        Ok(PatchyParticleSelfAssembly {
            microstate,
            overlap_penalty_hamiltonian,
            hamiltonian,
            translate_sweep,
            rotate_sweep,
            quick_insert,
            quick_compress,
            macrostate,
            phase: Phase::Initialize,
        })
    }
}

#[cfg_attr(feature = "bevy", derive(Resource))]
struct PatchyParticleSelfAssembly {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<
        BodyProperties,
        SiteProperties,
        VecCell<SiteKey, 2>,
        Periodic<Rectangle>,
    >,
    /// How sites interact with other sites and fields.
    hamiltonian: PairwiseCutoff<(
        HardSphere,
        Anisotropic<AngularMask<Boxcar, PositionVector>>,
    )>,
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate<PositionVector>>,
    /// Trial moves to apply.
    rotate_sweep: Sweep<Rotate<Orientation>>,
    /// Temperature set point.
    macrostate: Isothermal,
    /// Quick insert algorithm.
    quick_insert: QuickInsert<UniformIn<BodyProperties, Periodic<Rectangle>>>,
    /// Quick compress algorithm
    quick_compress: QuickCompress<Periodic<Rectangle>>,
    /// How sites interact during compression.
    overlap_penalty_hamiltonian:
        PairwiseCutoff<Isotropic<Expanded<OverlapPenalty>>>,
    /// The current phase of the simulation.
    phase: Phase,
}

enum Phase {
    Initialize,
    Equilibrate,
}

impl Simulation for PatchyParticleSelfAssembly {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        match self.phase {
            Phase::Initialize => {
                self.initialize().context("failed to initialize")?
            }
            Phase::Equilibrate => self.equilibrate(),
        }

        self.microstate.increment_step();

        Ok(())
    }

    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl PatchyParticleSelfAssembly {
    fn initialize(&mut self) -> anyhow::Result<()> {
        if self.quick_insert.is_complete() {
            self.quick_compress.apply(
                &mut self.microstate,
                &self.overlap_penalty_hamiltonian,
                |_| true,
            );
        } else {
            self.quick_insert
                .apply(&mut self.microstate, &self.overlap_penalty_hamiltonian);
        }

        self.translate_sweep.apply(
            &mut self.microstate,
            &self.overlap_penalty_hamiltonian,
            &Isothermal { temperature: 1.0 },
        );

        self.rotate_sweep.apply(
            &mut self.microstate,
            &self.overlap_penalty_hamiltonian,
            &Isothermal { temperature: 1.0 },
        );

        if self.quick_compress.is_complete() {
            self.translate_sweep.tune_default(
                &self.microstate,
                &self.hamiltonian,
                &self.macrostate,
            );
            self.rotate_sweep.tune_default(
                &self.microstate,
                &self.hamiltonian,
                &self.macrostate,
            );

            self.phase = Phase::Equilibrate;
            println!(
                "Initialization complete at step {}.",
                self.microstate.step()
            );
        }

        if self.step() >= 20_000 {
            let n = self.microstate.bodies().len();
            let target_n = self.quick_insert.target();
            let volume = self.microstate.boundary().volume();
            let target_volume = self.quick_compress.target_volume();
            return Err(anyhow!(
                "inserted {n}/{target_n} bodies and compressed to {volume} / {target_volume}"
            ));
        }

        Ok(())
    }

    fn equilibrate(&mut self) {
        self.translate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
        );

        self.rotate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
        );
    }
}
// ANCHOR_END: remainder

// ANCHOR: log_record
/// A single entry in the log.
#[derive(ParquetRecordWriter)]
pub struct LogRecord {
    /// The simulation step.
    step: u64,

    /// Total system potential energy.
    potential_energy: f64,
}
// ANCHOR_END: log_record

/// Enumerate possible sources of error when writing log files.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum Error {
    /// Encountered an IO error.
    #[error("I/O error")]
    IO(#[from] io::Error),

    /// Encountered an IO error.
    #[error("Parquet error")]
    Parquet(#[from] parquet::errors::ParquetError),
}

// impl RecordWriter<LogRecord> for Vec<LogRecord> {
//     fn write_to_row_group<W: std::io::Write + Send>(
//         &self,
//         row_group_writer: &mut parquet::file::writer::SerializedRowGroupWriter<W>,
//     ) -> Result<(), parquet::errors::ParquetError> {
//         todo!()
//     }

//     fn schema(&self) -> Result<parquet::schema::types::TypePtr, parquet::errors::ParquetError> {
//         todo!()
//     }
// }


// Remove the cfg(not(...)) line when using this code outside the hoomd-rs/examples directory.
#[cfg(not(feature = "bevy"))]
// ANCHOR: main
fn main() -> anyhow::Result<()> {
// ANCHOR_END: main
    // ANCHOR: log_open
    let mut parquet_logger = ParquetLogger::<LogRecord>::create("patchy-particle-self-assembly.parquet")?;
    // ANCHOR_END: log_open

    // ANCHOR: run_simulation
    let mut simulation = PatchyParticleSelfAssembly::new()?;
    // TODO: Write GSD file.

    for _ in 0..2001 {
        simulation.advance()?;
        // ANCHOR_END: run_simulation

        // ANCHOR: write_log
        if simulation.step().is_multiple_of(1000) {
            parquet_logger.log(LogRecord {
                step: simulation.microstate.step(),
                potential_energy: simulation.hamiltonian.total_energy(&simulation.microstate),
            })?;
        }
    }
    // ANCHOR_END: write_log

    // ANCHOR: exit
    Ok(())
}
// ANCHOR_END: exit
// ANCHOR_END: all

#[cfg(feature = "bevy")]
mod patchy_particle_self_assembly_interactive;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
#[cfg(feature = "bevy")]
use patchy_particle_self_assembly_interactive::main;

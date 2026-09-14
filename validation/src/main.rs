//! Validate HOOMD-rs against HOOMD-blue.

use core::panic;
use std::path::Path;

mod simulation;
mod workspace;
use hoomd_interaction::TotalEnergy;
use hoomd_md::{RotationalKineticEnergy, TranslationalKineticEnergy};
use hoomd_microstate::AppendMicrostate;
use hoomd_workspace::Entry;
use hoomd_utility::data::ParquetLogger;

use crate::workspace::{BodyVariants, MethodVariants, StatePoint, ThermostatVariants};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

// Workspace constants
const NDIMS: [usize; 1] = [3];
const PARTICLE_TYPES: [BodyVariants; 2] = [BodyVariants::Sphere, BodyVariants::Dumbbell];
const METHODS: [MethodVariants; 2] = [MethodVariants::ConstantVolume, MethodVariants::Langevin];
const THERMOSTATS: [ThermostatVariants; 3] = [
    ThermostatVariants::NoThermostat,
    ThermostatVariants::Bussi,
    ThermostatVariants::MTTK,
];

// Procedure constants
const DT: f64 = 0.001;
const TAU: f64 = 0.1;
const LOG_PERIOD: usize = 30;
const DURATION: usize = 30_000;

const SAVE_GSD: bool = false;
const GSD_NAME: &str = "trajectory.gsd";

const LOG_NAME: &str = "log.parquet";

// System constants
const KT: f64 = 1.5;
const PARTICLES_PER_SIDE: usize = 10;

const DENSITY_SPHERE_3D: f64 = 0.6269137133228043;
const DENSITY_SPHERE_2D: f64 = 0.6269137133228043;

const DENSITY_DUMBBELL_3D: f64 = 0.5;
const DENSITY_DUMBBELL_2D: f64 = 0.5;

const EPSILON: f64 = 1.0;
const SIGMA: f64 = 1.0;
const R_CUT: f64 = 4.0;
const R_ON: f64 = 3.2;

fn main() -> anyhow::Result<()> {
    // Create the workspace
    workspace::make_workspace()?;

    // Get a list of entry ids in the workspace
    let ids = workspace::identifiers();

    // Progress bars
    let multi_pb = MultiProgress::new();
    let style = ProgressStyle::with_template(
        "[{elapsed_precise} / {duration_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}"
    )
    .unwrap()
    .progress_chars("##-");

    let pb_outer = multi_pb.add(ProgressBar::new(ids.len() as u64));
    pb_outer.set_style(style.clone());

    let pb_inner = multi_pb.add(ProgressBar::new(DURATION as u64));
    pb_inner.set_style(style.clone());

    // For every id...
    for id in ids {
        // get the state point...
        let sp: StatePoint = hoomd_workspace::state_point(Path::new(&id))?
            .ok_or(anyhow::anyhow!("state point not found"))?;

        pb_outer.set_message(
            format!("{}D, {}, {}, {}", sp.ndims, sp.particle_type, sp.method, sp.thermostat)
        );

        // (skipping state points that are already finished)
        if workspace::is_finished(&sp)? {
            continue
        }

        match sp.ndims {
            // 2D Systems
            2 => {
                // create the params objects...
                let system = simulation::two::SystemParams::from_state_point(&sp);
                let procedure = simulation::two::ProcedureParams::from_state_point(&sp);

                // Create the logging and trajectory writing stuff
                let gsd_path = sp.path()?.join(&GSD_NAME);
                let mut gsd_file = hoomd_gsd::hoomd::HoomdGsdFile::create(Path::new(&gsd_path))?;

                let log_path = sp.path()?.join(&LOG_NAME);
                let mut logger = ParquetLogger::<simulation::LogRecord>::create(log_path)?;
                
                // create the microstate...
                let mut microstate = simulation::two::make_microstate(&system)?;

                // create the method...
                let mut method = simulation::two::make_method(&system, &procedure);

                // save initial state...
                let mut ke_trans = microstate.translational_kinetic_energy().0;
                let mut ke_rot = microstate.rotational_kinetic_energy().0;

                gsd_file.append_microstate(&microstate)?
                    .log_scalar("kinetic_energy", ke_trans + ke_rot)?;

                // run the simulation...
                for t in 0..DURATION {
                    method.integrate(&mut microstate, &system.macrostate(), &system.interaction_model());

                    // writing to gsd every period
                    if t.is_multiple_of(LOG_PERIOD) {
                        ke_trans = microstate.translational_kinetic_energy().0;
                        ke_rot = microstate.rotational_kinetic_energy().0;

                        // sanity check
                        if let BodyVariants::Sphere = system.particle_type {
                            assert_eq!(
                                ke_rot,
                                0.0,
                                "Rot KE should be zero for body {}, method {}",
                                system.particle_type,
                                procedure.method
                            );
                        } 

                        if SAVE_GSD {
                            gsd_file.append_microstate(&microstate)?
                                .log_scalar("kinetic_energy", ke_trans + ke_rot)?
                                .log_scalar("potential_energy", system.interaction_model().total_energy(&microstate))?;
                        }

                        logger.log(simulation::LogRecord {
                            step: microstate.step(),
                            potential_energy: system.interaction_model().total_energy(&microstate),
                            kinetic_energy: ke_trans + ke_rot
                        })?;
                    }
                    microstate.increment_step();
                
                    pb_inner.inc(1);
                }
            },

            // 3D systems
            3 => {
                // create the params objects...
                let system = simulation::three::SystemParams::from_state_point(&sp);
                let procedure = simulation::three::ProcedureParams::from_state_point(&sp);

                // Create the logging and trajectory writing stuff
                let gsd_path = sp.path()?.join(&GSD_NAME);
                let mut gsd_file = hoomd_gsd::hoomd::HoomdGsdFile::create(Path::new(&gsd_path))?;

                let log_path = sp.path()?.join(&LOG_NAME);
                let mut logger = ParquetLogger::<simulation::LogRecord>::create(log_path)?;
                
                // create the microstate...
                let mut microstate = simulation::three::make_microstate(&system)?;

                // create the method...
                let mut method = simulation::three::make_method(&system, &procedure);

                // save initial state...
                let mut ke_trans = microstate.translational_kinetic_energy().0;
                let mut ke_rot = microstate.rotational_kinetic_energy().0;

                gsd_file.append_microstate(&microstate)?
                    .log_scalar("kinetic_energy", ke_trans + ke_rot)?;

                // run the simulation...
                for t in 0..DURATION {
                    method.integrate(&mut microstate, &system.macrostate(), &system.interaction_model());

                    // writing to gsd every period
                    if t.is_multiple_of(LOG_PERIOD) {
                        ke_trans = microstate.translational_kinetic_energy().0;
                        ke_rot = microstate.rotational_kinetic_energy().0;

                        // sanity check
                        if let BodyVariants::Sphere = system.particle_type {
                            assert_eq!(
                                ke_rot,
                                0.0,
                                "Rot KE should be zero for body {}, method {}",
                                system.particle_type,
                                procedure.method
                            );
                        } 

                        if SAVE_GSD {
                            gsd_file.append_microstate(&microstate)?
                                .log_scalar("kinetic_energy", ke_trans + ke_rot)?
                                .log_scalar("potential_energy", system.interaction_model().total_energy(&microstate))?;
                        }

                        logger.log(simulation::LogRecord {
                            step: microstate.step(),
                            potential_energy: system.interaction_model().total_energy(&microstate),
                            kinetic_energy: ke_trans + ke_rot
                        })?;
                    }
                    microstate.increment_step();
                    
                    pb_inner.inc(1);
                }
            },
            _ => panic!("ndims must be 2 or 3!")
        }

        pb_outer.inc(1);

        pb_inner.reset()

    }

    multi_pb.clear().unwrap();

    Ok(())
}

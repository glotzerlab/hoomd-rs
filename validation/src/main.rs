//! Validate HOOMD-rs against HOOMD-blue.

use std::{path::{Path}};

mod simulation;
mod workspace;
use crate::{
    simulation::{
        ProcedureParams,
        SystemParams
    },
    workspace::{
        MethodVariants,
        BodyVariants,
        StatePoint,
        ThermostatVariants
    }
};

// fn parse_toml(path: &str) -> anyhow::Result<Table> {
//     let toml_str = fs::read_to_string(path)?;
//     let table = toml_str.parse::<Table>()?;
//     Ok(table)
// }

// // Ingest config
// let config_path = "C:\\Users\\Joseph\\computer\\Github\\hoomd-rs\\validation\\src\\config.toml";
// let config = parse_toml(config_path)?;
// println!("{}", config);


// Workspace constants
const NDIMS: [usize; 2] = [2, 3];
const PARTICLE_TYPES: [BodyVariants; 2] = [
    BodyVariants::Sphere,
    BodyVariants::Dumbbell,
];
const METHODS: [MethodVariants; 2] = [
    MethodVariants::ConstantVolume,
    MethodVariants::Langevin,
];
const THERMOSTATS: [ThermostatVariants; 3] = [
    ThermostatVariants::NoThermostat,
    ThermostatVariants::Bussi,
    ThermostatVariants::MTTK,
];

// Procedure constants
const DT: f64 = 0.001;
const TAU: f64 = 0.1;
const GSD_PERIOD: usize = 30;
const SIM_DURATION: usize = 30_000;
const NLIST_BUFFER: f64 = 0.4;

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
// const R_ON: f64 = 3.2;
// const MODE: &str = "xplor"; // TODO


fn main() -> anyhow::Result<()> {
    // Create the workspace
    workspace::make_workspace()?;

    // Get a list of entry ids in the workspace
    let ids = workspace::identifiers();

    // For every id...
    for id in ids {
        // get the state point...
        let sp: StatePoint = hoomd_workspace::state_point(Path::new(&id))?
            .ok_or(anyhow::anyhow!("state point not found"))?;
        
        // create the params objects...
        let system = SystemParams::from_state_point(&sp);
        let procedure = ProcedureParams::from_state_point(&sp);

        // Create the GSD writer stuff

        // create the microstate...

        // run the simulation...
        for _ in 0..procedure.sim_duration {
            procedure.method.integrate(
                microstate,
                system.macrostate(),
                system.interaction_model(),
            )

            // GSD write
        }


    }

    Ok(())
}

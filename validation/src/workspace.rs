use serde::{Deserialize, Serialize};
// use toml::Table;
use itertools::iproduct;
// use hoomd_workspace::Entry;


/// The names of variants for the body type.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum BodyVariants {
    #[serde(rename = "sphere")]
    Sphere,
    #[serde(rename = "dumbbell")]
    Dumbbell,
}

/// The names of variants for the method type.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum MethodVariants {
    #[serde(rename = "constant-volume")]
    ConstantVolume,
    #[serde(rename = "langevin")]
    Langevin,
}

/// The names of variants for the thermostat type.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum ThermostatVariants {
    #[serde(rename = "no-thermostat")]
    NoThermostat,
    #[serde(rename = "bussi")]
    Bussi,
    #[serde(rename = "mttk")]
    MTTK,
}

/// The thermostat.
#[derive(Debug, Serialize, Deserialize)]
pub struct StatePoint {
    // System
    pub ndims: usize,
    pub particle_type: BodyVariants,
    pub particles_per_side: usize,

    // Procedure
    pub method: MethodVariants,
    pub thermostat: ThermostatVariants,
    pub gsd_period: usize,
    pub sim_duration: usize,
    pub hoomd: String,
}

/// Create the workspace and populate it with entries.
pub fn make_workspace() -> anyhow::Result<()> {
    let params = iproduct!(
        crate::NDIMS,
        crate::PARTICLE_TYPES,
        crate::METHODS,
        crate::THERMOSTATS
    );

    for (ndims, particle_type, method, thermostat) in params {
        let sp = StatePoint {
            ndims,
            particle_type,
            particles_per_side: crate::PARTICLES_PER_SIDE,
            method,
            thermostat,
            gsd_period: crate::GSD_PERIOD,
            sim_duration: crate::SIM_DURATION,
            hoomd: String::from("rs"),
        };

        hoomd_workspace::add(&sp)?;
    }

    Ok(())
}

/// Return a vector of identifiers in the workspace.
pub fn identifiers() -> Vec<String> {
    let ids: Vec<String> = std::fs::read_dir("./workspace")
        .unwrap()
        .into_iter()
        .filter(|entry| entry.as_ref().unwrap().file_type().unwrap().is_dir())
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();

    ids
}

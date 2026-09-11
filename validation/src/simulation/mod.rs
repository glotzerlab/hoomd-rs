//! Stuff for simulations that is independent of the number of dimensions.
use hoomd_derive::SitePairEnergy;
use hoomd_interaction::{
    MaximumInteractionRange,
    PairwiseCutoff,
    Rigid,
    pairwise::Isotropic,
    univariate::{LennardJones, Xplor},
};
use hoomd_md::Thermostat;
use hoomd_simulation::macrostate::Isothermal;
use parquet_derive::ParquetRecordWriter;
use rand::Rng;
use strum_macros::VariantNames;

pub mod two;
pub mod three;

/// Interaction constants
pub const SPHERE_LJ: Rigid<PairwiseCutoff<Isotropic<Xplor<LennardJones<12, 6>>>>> =
    Rigid(PairwiseCutoff(Isotropic {
        interaction: Xplor {
            f: LennardJones {
                epsilon: crate::EPSILON,
                sigma: crate::SIGMA,
            },
            r_cut: crate::R_CUT,
            r_smooth: crate::R_ON,
        },
        r_cut: crate::R_CUT,
    }));

pub const DUMBBELL_LJ: Rigid<PairwiseCutoff<DumbbellInteraction>> =
    Rigid(PairwiseCutoff(DumbbellInteraction {
        aa: Isotropic {
            interaction: Xplor {
                f: LennardJones {
                    epsilon: crate::EPSILON,
                    sigma: crate::SIGMA / 5.0,
                },
                r_cut: crate::R_CUT,
                r_smooth: crate::R_ON,
            },
            r_cut: crate::R_CUT,
        },
        bb: Isotropic {
            interaction: Xplor {
                f: LennardJones {
                    epsilon: crate::EPSILON / 5.0,
                    sigma: crate::SIGMA,
                },
                r_cut: crate::R_CUT,
                r_smooth: crate::R_ON,
            },
            r_cut: crate::R_CUT,
        },
        ab: Isotropic {
            interaction: Xplor {
                f: LennardJones {
                    epsilon: crate::EPSILON / 5.0,
                    sigma: crate::SIGMA / 5.0,
                },
                r_cut: crate::R_CUT,
                r_smooth: crate::R_ON,
            },
            r_cut: crate::R_CUT,
        },
    }));

/// The site types.
#[derive(Clone, Copy, Default, PartialEq, VariantNames)]
pub enum SiteVariants {
    #[default]
    A,
    B,
}

/// The interaction types.
#[derive(MaximumInteractionRange, SitePairEnergy)]
pub struct DumbbellInteraction {
    aa: Isotropic<Xplor<LennardJones<12, 6>>>,
    bb: Isotropic<Xplor<LennardJones<12, 6>>>,
    ab: Isotropic<Xplor<LennardJones<12, 6>>>,
}

pub enum InteractionModels {
    Sphere(Rigid<PairwiseCutoff<Isotropic<Xplor<LennardJones<12, 6>>>>>),
    Dumbbell(Rigid<PairwiseCutoff<DumbbellInteraction>>),
}


/// The log record type.
#[derive(ParquetRecordWriter)]
pub struct LogRecord {
    pub step: u64,
    pub potential_energy: f64,
    pub kinetic_energy: f64,
}

/// The macrostate type.
type Macrostate = Isothermal;

/// The thermostat types.
#[derive(Debug, Clone)]
pub enum Thermostats {
    NoThermostat(hoomd_md::thermostat::NoThermostat),
    Bussi(hoomd_md::thermostat::Bussi),
    MTTK(hoomd_md::thermostat::MartynaTuckermanTobiasKlein),
}

impl Thermostat<Macrostate> for Thermostats {
    fn integrate_half_step_one<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        macrostate: &Macrostate,
        delta_t: f64,
        kinetic_energy: f64,
        degrees_of_freedom: usize,
    ) -> f64 {
        match self {
            Self::NoThermostat(inner_thermostat) => inner_thermostat.integrate_half_step_one(
                rng,
                macrostate,
                delta_t,
                kinetic_energy,
                degrees_of_freedom,
            ),
            Self::Bussi(inner_thermostat) => inner_thermostat.integrate_half_step_one(
                rng,
                macrostate,
                delta_t,
                kinetic_energy,
                degrees_of_freedom,
            ),
            Self::MTTK(inner_thermostat) => inner_thermostat.integrate_half_step_one(
                rng,
                macrostate,
                delta_t,
                kinetic_energy,
                degrees_of_freedom,
            ),
        }
    }

    fn integrate_half_step_two<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        macrostate: &Macrostate,
        delta_t: f64,
        kinetic_energy: f64,
        degrees_of_freedom: usize,
    ) -> f64 {
        match self {
            Self::NoThermostat(inner_thermostat) => inner_thermostat.integrate_half_step_two(
                rng,
                macrostate,
                delta_t,
                kinetic_energy,
                degrees_of_freedom,
            ),
            Self::Bussi(inner_thermostat) => inner_thermostat.integrate_half_step_two(
                rng,
                macrostate,
                delta_t,
                kinetic_energy,
                degrees_of_freedom,
            ),
            Self::MTTK(inner_thermostat) => inner_thermostat.integrate_half_step_two(
                rng,
                macrostate,
                delta_t,
                kinetic_energy,
                degrees_of_freedom,
            ),
        }
    }
}

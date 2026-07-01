// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Benchmark Lennard Jones molecular dynamics simulations.

use std::{
    fmt,
    fs::{self, File},
    io::{self, Write},
};

use anyhow::Context;
use hoomd_geometry::shape::Hypercuboid;
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff, Rigid, pairwise::Isotropic, univariate::{self, Expanded, OverlapPenalty}
};
use hoomd_mc::{Sweep, Translate};
use hoomd_md::{ThermalizeMomentum, TranslationalMotion, ZeroCenterAngularMomentum, ZeroCenterMomentum, method::ConstantVolume, thermostat::Bussi};
use hoomd_microstate::{
    Microstate, SiteKey, boundary::{GenerateGhosts, Periodic}, property::{DynamicPoint, Point}
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::{IndexFromPosition, PointUpdate, PointsNearBall, WithSearchRadius};
use hoomd_vector::Cartesian;
use log::{debug, trace};
use serde::{Deserialize, Serialize};

use crate::{Effort, place::place_single_site_point_bodies};

/// Relax configurations this many steps before tuning move sizes.
const RELAX_STEPS: usize = 1_000;

/// The Lennard Jones simulation.
#[derive(Serialize, Deserialize)]
pub struct LennardJones<const D: usize, X> {
    /// Simulation microstate
    microstate: Microstate<DynamicPoint<Cartesian<D>>, Point<Cartesian<D>>, X, Periodic<Hypercuboid<D>>>,

    /// Translate moves (serial)
    translate_sweep: Sweep<Translate<Cartesian<D>>>,

    /// MD integration
    constant_volume: ConstantVolume<Bussi>,

    /// Lennard Jones interaction.
    rigid: Rigid<PairwiseCutoff<Isotropic<univariate::LennardJones>>>,

    /// Temperature set point
    macrostate: Isothermal,

    /// Track steps advanced during the benchmark period.
    steps: usize,
}

impl<const D: usize, X> Effort for LennardJones<D, X> {
    #[inline]
    fn units() -> String {
        "steps per second".to_string()
    }

    #[inline]
    fn effort(&self) -> f64 {
        self.steps as f64
    }
}

impl<const D: usize, X> Simulation for LennardJones<D, X>
where
    X: PointsNearBall<Cartesian<D>, SiteKey>
        + PointUpdate<Cartesian<D>, SiteKey>
        + Sync
        + IndexFromPosition<Cartesian<D>>,
    Periodic<Hypercuboid<D>>: GenerateGhosts<Point<Cartesian<D>>>,
{
    #[inline]
    fn advance(&mut self) -> anyhow::Result<()> {
        if self.microstate.step().is_multiple_of(300) {
            self.microstate.sort_sites();
        }

        self.constant_volume.integrate_translation(
            &mut self.microstate,
            &self.macrostate,
            &self.rigid);

        self.microstate.increment_step();
        self.steps += 1;

        Ok(())
    }

    #[inline]
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl<const D: usize, X> fmt::Display for LennardJones<D, X>
where
    X: fmt::Display,
{
    #[inline]
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl<const D: usize, X> LennardJones<D, X>
where
    X: PointsNearBall<Cartesian<D>, SiteKey>
        + PointUpdate<Cartesian<D>, SiteKey>
        + WithSearchRadius
        + Clone
        + for<'a> Deserialize<'a>
        + Serialize
        + Sync
        + IndexFromPosition<Cartesian<D>>,
    Periodic<Hypercuboid<D>>: GenerateGhosts<Point<Cartesian<D>>>,
{
    /// Construct a new Lennard Jones simulation
    ///
    /// # Errors
    /// Returns an error when the microstate cannot be constructed.
    #[inline]
    pub fn new(n: usize) -> anyhow::Result<Self> {
        let delta_t = 0.005;
        let temperature = 1.2;
        let macrostate = Isothermal { temperature };
        let maximum_interaction_range = 2.5;

        let number_density = 1.0;
        let cache_filename = format!("md_{D}d_lennard_jones_{number_density}_{n}.postcard");

        match fs::read(&cache_filename) {
            Ok(bytes) => {
                debug!("Reading cache '{cache_filename}'.");

                let mut result: Self = postcard::from_bytes(&bytes)
                    .with_context(|| format!("Could not read {cache_filename}"))?;
                result.microstate.sort_sites();
                return Ok(result);
            }
            Err(error) => match error.kind() {
                io::ErrorKind::NotFound => (),
                _ => return Err(error).with_context(|| format!("Could not read {cache_filename}")),
            },
        }

        let hamiltonian = PairwiseCutoff(Isotropic {
            interaction: univariate::LennardJones {
                epsilon: 1.0,
                sigma: 1.0,
            },
            r_cut: maximum_interaction_range,
        });
        let rigid = Rigid(hamiltonian);

        let translate = Translate::with_maximum_distance(0.1.try_into()?);
        let translate_sweep = Sweep(translate.clone());

        let overlap_penalty = Isotropic {
            interaction: Expanded {
                delta: 1.0,
                f: OverlapPenalty::default(),
            },
            r_cut: 1.0,
        };

        let insert_hamiltonian = PairwiseCutoff(overlap_penalty);

        let mut microstate = place_single_site_point_bodies(
            n,
            number_density,
            rigid.maximum_interaction_range(),
            &insert_hamiltonian,
        )?;
            
        microstate.thermalize_momentum(temperature);
        microstate.zero_center_momentum();

        let thermostat = Bussi::new(0.0);
        let constant_volume = ConstantVolume::builder(delta_t)
            .thermostat(thermostat)
            .build();

        let mut simulation = Self {
            microstate,
            translate_sweep,
            constant_volume,
            rigid,
            macrostate,
            steps: 0,
        };

        debug!("Relaxing configuration...");

        for i in 0..RELAX_STEPS {
            simulation.advance()?;
            if (i + 1).is_multiple_of(100) {
                trace!("{:.1}%", ((i + 1) as f64 / RELAX_STEPS as f64) * 100.0);
            }
        }

        simulation.microstate.sort_sites();

        simulation.steps = 0;

        let out_bytes: Vec<u8> = postcard::to_stdvec(&simulation)?;
        let mut file = File::create(cache_filename)?;
        file.write_all(&out_bytes)?;

        Ok(simulation)
    }
}

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Benchmark Lennard Jones Monte Carlo simulations.

use std::{
    fmt,
    fs::{self, File},
    io::{self, Write},
};

use anyhow::Context;
use hoomd_geometry::{
    Volume,
    shape::{Hypercuboid, Hypersphere},
};
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff,
    pairwise::Isotropic,
    univariate::{self, Expanded, OverlapPenalty},
};
use hoomd_mc::{
    Count, HypercuboidCheckerboard, ParallelSweep, Sweep, Translate, Trial, Tune, TuneOptions,
};
use hoomd_microstate::{
    Microstate, SiteKey,
    boundary::{GenerateGhosts, Periodic},
    property::Point,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::{IndexFromPosition, PointUpdate, PointsNearBall, WithSearchRadius};
use hoomd_vector::Cartesian;
use log::{debug, info, trace};
use serde::{Deserialize, Serialize};

use crate::{Effort, place::place_single_site_point_bodies};

/// Relax configurations this many steps before tuning move sizes.
const RELAX_STEPS: usize = 1_000;

/// The Lennard Jones simulation.
#[derive(Serialize, Deserialize)]
pub struct LennardJones<const D: usize, X> {
    /// Simulation microstate
    microstate: Microstate<Point<Cartesian<D>>, Point<Cartesian<D>>, X, Periodic<Hypercuboid<D>>>,

    /// Translate moves (serial)
    translate_sweep: Sweep<Translate<Cartesian<D>>>,

    /// Translate moves (parallel)
    parallel_translate_sweep: ParallelSweep<
        Translate<Cartesian<D>>,
        HypercuboidCheckerboard<D>,
        Point<Cartesian<D>>,
        Point<Cartesian<D>>,
    >,

    /// Lennard Jones interaction.
    hamiltonian: PairwiseCutoff<Isotropic<univariate::LennardJones>>,

    /// Temperature set point
    macrostate: Isothermal,

    /// Track moves accepted during the benchmark period.
    count: Count,

    /// Set to true to use parallel translate moves.
    parallel: bool,
}

impl<const D: usize, X> Effort for LennardJones<D, X> {
    #[inline]
    fn units() -> String {
        "sweep".to_string()
    }

    #[inline]
    fn effort(&self) -> f64 {
        self.count.total() as f64 / self.microstate.bodies().len() as f64
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

        if self.parallel {
            self.count += self.parallel_translate_sweep.apply(
                &mut self.microstate,
                &self.hamiltonian,
                &self.macrostate,
            );
        } else {
            self.count += self.translate_sweep.apply(
                &mut self.microstate,
                &self.hamiltonian,
                &self.macrostate,
            );
        }

        self.microstate.increment_step();

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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.microstate.fmt(f)?;
        write!(
            f,
            "\nTranslate acceptance: {}",
            self.count
                .acceptance_ratio()
                .expect("there should be some trial moves")
        )
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
    pub fn new(n: usize, parallel: bool) -> anyhow::Result<Self> {
        let macrostate = Isothermal { temperature: 1.0 };
        let maximum_interaction_range = 2.5;

        let packing_fraction = 0.50;
        let sphere = Hypersphere::<D>::with_radius(0.5.try_into()?);
        let number_density = packing_fraction / sphere.volume();
        let cache_filename = format!("mc_{D}d_lennard_jones_{packing_fraction}_{n}.postcard");

        match fs::read(&cache_filename) {
            Ok(bytes) => {
                debug!("Reading cache '{cache_filename}'.");

                let mut result: Self = postcard::from_bytes(&bytes)
                    .with_context(|| format!("Could not read {cache_filename}"))?;
                // The cache may have been generated with a different value of parallel.
                result.parallel = parallel;
                result.microstate.sort_sites();
                return Ok(result);
            }
            Err(error) => match error.kind() {
                io::ErrorKind::NotFound => (),
                _ => return Err(error).with_context(|| format!("Could not read {cache_filename}")),
            },
        }

        let maximum_distance = match D {
            2 => 0.63,
            3 => 0.19,
            _ => 0.1,
        };

        let hamiltonian = PairwiseCutoff(Isotropic {
            interaction: univariate::LennardJones {
                epsilon: 1.0,
                sigma: 1.0,
            },
            r_cut: maximum_interaction_range,
        });

        let translate = Translate::with_maximum_distance(maximum_distance.try_into()?);
        let translate_sweep = Sweep(translate.clone());
        let parallel_translate_sweep = ParallelSweep::new(
            hamiltonian.maximum_interaction_range().try_into()?,
            translate,
        );

        let overlap_penalty = Isotropic {
            interaction: Expanded {
                delta: 1.0,
                f: OverlapPenalty::default(),
            },
            r_cut: 1.0,
        };

        let insert_hamiltonian = PairwiseCutoff(overlap_penalty);

        let microstate = place_single_site_point_bodies(
            n,
            number_density,
            hamiltonian.maximum_interaction_range(),
            &insert_hamiltonian,
        )?;

        let mut simulation = Self {
            microstate,
            translate_sweep,
            parallel_translate_sweep,
            hamiltonian,
            macrostate,
            count: Count::default(),
            parallel,
        };

        debug!("Relaxing configuration...");

        for i in 0..RELAX_STEPS {
            simulation.advance()?;
            if (i + 1).is_multiple_of(100) {
                trace!("{:.1}%", ((i + 1) as f64 / RELAX_STEPS as f64) * 100.0);
            }
        }

        simulation.microstate.sort_sites();

        // Move sizes are fixed above for comparison with HOOMD-blue. Uncomment this code
        // when there is a need to retune the move sizes.

        // simulation.translate_sweep.tune_default(&simulation.microstate, &simulation.hamiltonian, &simulation.macrostate);
        // *simulation.parallel_translate_sweep
        //     .local_trial_mut()
        //     .maximum_distance_mut() = *simulation.translate_sweep.0.maximum_distance();

        info!(
            "Translation move size: {}",
            simulation.translate_sweep.0.maximum_distance()
        );
        simulation.count = Count::default();

        let out_bytes: Vec<u8> = postcard::to_stdvec(&simulation)?;
        let mut file = File::create(cache_filename)?;
        file.write_all(&out_bytes)?;

        Ok(simulation)
    }
}

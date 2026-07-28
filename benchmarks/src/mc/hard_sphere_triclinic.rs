// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Benchmark hard sphere Monte Carlo simulations.

use std::{
    fmt,
    fs::{self, File},
    io::{self, Write},
};

use anyhow::Context;
use hoomd_geometry::{
    Volume,
    shape::{Triclinic, Hypersphere},
};
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff,
    pairwise::{HardSphere, Isotropic},
    univariate::{Expanded, OverlapPenalty},
};
use hoomd_mc::{Count, Sweep, Translate, Trial};
use hoomd_microstate::{
    Microstate, SiteKey,
    boundary::Periodic,
    property::Point,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::Cartesian;
use log::{debug, info, trace};
use serde::{Deserialize, Serialize};

use crate::{Effort, place::place_single_site_point_bodies};

/// Relax configurations this many steps before tuning move sizes.
const RELAX_STEPS: usize = 1_000;

/// The hard sphere simulation.
#[derive(Serialize, Deserialize)]
pub struct HardSphereTriclinicSim {
    /// Simulation microstate
    microstate: Microstate<Point<Cartesian<3>>, Point<Cartesian<3>>, VecCell<SiteKey, 3>, Periodic<Triclinic>>,

    /// Translate moves (serial)
    translate_sweep: Sweep<Translate<Cartesian<3>>>,

    /// Hard sphere interaction.
    hamiltonian: PairwiseCutoff<HardSphere>,

    /// Temperature set point.
    macrostate: Isothermal,

    /// Track moves attempted during the benchmark period.
    count: Count,
}

impl Effort for HardSphereTriclinicSim {
    #[inline]
    fn units() -> String {
        "sweep".to_string()
    }

    #[inline]
    fn effort(&self) -> f64 {
        self.count.total() as f64 / self.microstate.bodies().len() as f64
    }
}

impl Simulation for HardSphereTriclinicSim
{
    #[inline]
    fn advance(&mut self) -> anyhow::Result<()> {
        if self.microstate.step().is_multiple_of(300) {
            self.microstate.sort_sites();
        }

        self.count += self.translate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
        );
        self.microstate.increment_step();

        Ok(())
    }

    #[inline]
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl fmt::Display for HardSphereTriclinicSim
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

impl HardSphereTriclinicSim
{
    /// Construct a new hard sphere simulation
    ///
    /// # Errors
    /// Returns an error when the microstate cannot be constructed.
    #[inline]
    pub fn new(n: usize, _parallel: bool) -> anyhow::Result<Self> {
        let sigma = 1.0;
        let packing_fraction = 0.50;
        let sphere = Hypersphere::<3>::with_radius(0.5.try_into()?);
        let number_density = packing_fraction / sphere.volume();
        let cache_filename = format!("mc_3d_sphere_triclinic_{packing_fraction}_{n}.postcard");

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

        let maximum_distance = 0.13;

        let hamiltonian = PairwiseCutoff(HardSphere { diameter: sigma });

        let translate = Translate::with_maximum_distance(maximum_distance.try_into()?);
        let translate_sweep = Sweep(translate.clone());

        let overlap_penalty = Isotropic {
            interaction: Expanded {
                delta: sigma,
                f: OverlapPenalty::default(),
            },
            r_cut: sigma,
        };

        let overlap_penalty_hamiltonian = PairwiseCutoff(overlap_penalty);

        let microstate_hypercuboid: Microstate<_, _, VecCell<SiteKey, 3>, _> = place_single_site_point_bodies(
            n,
            number_density,
            hamiltonian.0.maximum_interaction_range(),
            &overlap_penalty_hamiltonian,
        )?;

        let triclinic = Triclinic::cube(microstate_hypercuboid.boundary().shape().edge_lengths[0]);
        let periodic_triclinic = Periodic::new(hamiltonian.maximum_interaction_range(), triclinic)?; 
        let vec_cell = VecCell::builder()
            .nominal_search_radius(
                hamiltonian.maximum_interaction_range().try_into()?,
            )
            .build();
        let microstate = Microstate::builder()
            .boundary(periodic_triclinic)
            .spatial_data(vec_cell)
            .bodies(microstate_hypercuboid.bodies().iter().map(|b| b.item.clone()))
            .try_build()?;

        let mut simulation = Self {
            microstate,
            translate_sweep,
            hamiltonian,
            macrostate: Isothermal { temperature: 1.0 },
            count: Count::default(),
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

        // simulation.translate_sweep.tune_default(&simulation.microstate, &simulation.hamiltonian, &Isothermal { temperature: 1.0 });
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

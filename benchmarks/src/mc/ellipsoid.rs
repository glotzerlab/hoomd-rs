// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Benchmark hard ellipsoid Monte Carlo simulations.

use anyhow::Context;
use log::debug;
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    fs::{self, File},
    io::{self, Write},
};

use hoomd_geometry::{
    Convex, Volume,
    shape::{Hypercuboid, Hyperellipsoid},
};
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff,
    pairwise::{Anisotropic, ApproximateShapeOverlap, HardShape},
    univariate::OverlapPenalty,
};
use hoomd_mc::{
    Count, HypercuboidCheckerboard, ParallelSweep, Rotate, Sweep, Translate, Trial, Tune,
};
use hoomd_microstate::{
    Microstate, SiteKey,
    boundary::{GenerateGhosts, Periodic},
    property::OrientedPoint,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::{PointUpdate, PointsNearBall, WithSearchRadius};
use hoomd_vector::{Cartesian, Versor};

use crate::{Effort, place::place_single_site_orientable_bodies};

/// The hard octahedra simulation.
#[derive(Serialize, Deserialize)]
pub struct EllipsoidSim<X> {
    /// Simulation microstate
    microstate: Microstate<
        OrientedPoint<Cartesian<3>, Versor>,
        OrientedPoint<Cartesian<3>, Versor>,
        X,
        Periodic<Hypercuboid<3>>,
    >,

    /// Translate moves (serial)
    translate_sweep: Sweep<Translate<Cartesian<3>>>,

    /// Rotate moves (serial)
    rotate_sweep: Sweep<Rotate<Versor>>,

    /// Translate moves (parallel)
    parallel_translate_sweep: ParallelSweep<
        Translate<Cartesian<3>>,
        HypercuboidCheckerboard<3>,
        OrientedPoint<Cartesian<3>, Versor>,
        OrientedPoint<Cartesian<3>, Versor>,
    >,

    /// Rotate moves (parallel)
    parallel_rotate_sweep: ParallelSweep<
        Rotate<Versor>,
        HypercuboidCheckerboard<3>,
        OrientedPoint<Cartesian<3>, Versor>,
        OrientedPoint<Cartesian<3>, Versor>,
    >,

    /// Hard octahedra interaction.
    hamiltonian: PairwiseCutoff<HardShape<Convex<Hyperellipsoid<3>>>>,

    /// Temperature set point.
    macrostate: Isothermal,

    /// Track translate moves attempted during the benchmark period.
    translate_count: Count,

    /// Track rotate moves attempted during the benchmark period.
    rotate_count: Count,

    /// Set to true to use the parallel translate moves.
    parallel: bool,
}

impl<X> Effort for EllipsoidSim<X> {
    #[inline]
    fn units() -> String {
        "sweep".to_string()
    }

    #[inline]
    fn effort(&self) -> f64 {
        let complete_count = self.translate_count + self.rotate_count;
        complete_count.total() as f64 / self.microstate.bodies().len() as f64
    }
}

impl<X> Simulation for EllipsoidSim<X>
where
    X: PointsNearBall<Cartesian<3>, SiteKey> + PointUpdate<Cartesian<3>, SiteKey> + Sync,
    Periodic<Hypercuboid<3>>: GenerateGhosts<OrientedPoint<Cartesian<3>, Versor>>,
{
    #[inline]
    fn advance(&mut self) -> anyhow::Result<()> {
        if self.parallel {
            self.translate_count += self.parallel_translate_sweep.apply(
                &mut self.microstate,
                &self.hamiltonian,
                &self.macrostate,
            );
            self.rotate_count += self.parallel_rotate_sweep.apply(
                &mut self.microstate,
                &self.hamiltonian,
                &self.macrostate,
            );
        } else {
            self.translate_count += self.translate_sweep.apply(
                &mut self.microstate,
                &self.hamiltonian,
                &self.macrostate,
            );
            self.rotate_count +=
                self.rotate_sweep
                    .apply(&mut self.microstate, &self.hamiltonian, &self.macrostate);
        }
        self.microstate.increment_step();

        Ok(())
    }

    #[inline]
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl<X> fmt::Display for EllipsoidSim<X>
where
    X: fmt::Display,
{
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.microstate.fmt(f)?;
        write!(
            f,
            "\nTranslate acceptance: {}",
            self.translate_count
                .acceptance_ratio()
                .expect("there should be some trial moves")
        )?;
        write!(
            f,
            "\nRotate acceptance: {}",
            self.rotate_count
                .acceptance_ratio()
                .expect("there should be some trial moves")
        )
    }
}

impl<X> EllipsoidSim<X>
where
    X: PointsNearBall<Cartesian<3>, SiteKey>
        + PointUpdate<Cartesian<3>, SiteKey>
        + WithSearchRadius
        + Clone
        + for<'a> Deserialize<'a>
        + Serialize,
    Periodic<Hypercuboid<3>>: GenerateGhosts<OrientedPoint<Cartesian<3>, Versor>>,
{
    /// Construct a new hard octahedra simulation
    ///
    /// # Errors
    /// Returns an error when the microstate cannot be constructed.
    #[inline]
    pub fn new(n: usize, parallel: bool) -> anyhow::Result<Self> {
        let macrostate = Isothermal { temperature: 1.0 };
        let packing_fraction = 0.5;
        let ellipsoid =
            Hyperellipsoid::with_semi_axes([2.5.try_into()?, 0.5.try_into()?, 0.5.try_into()?]);
        let number_density = packing_fraction / ellipsoid.volume();
        let cache_filename = format!("mc_3d_ellipsoid_{packing_fraction}_{n}.postcard");

        match fs::read(&cache_filename) {
            Ok(bytes) => {
                debug!("Reading cache '{cache_filename}'.");

                let mut result: Self = postcard::from_bytes(&bytes)
                    .with_context(|| format!("Could not read {cache_filename}"))?;
                // The cache may have been generated with a different value of parallel.
                result.parallel = parallel;
                return Ok(result);
            }
            Err(error) => match error.kind() {
                io::ErrorKind::NotFound => (),
                _ => return Err(error).with_context(|| format!("Could not read {cache_filename}")),
            },
        }

        let hamiltonian = PairwiseCutoff(HardShape(Convex(ellipsoid.clone())));

        let translate = Translate::with_maximum_distance(0.05.try_into()?);
        let mut translate_sweep = Sweep(translate.clone());
        let mut parallel_translate_sweep = ParallelSweep::new(
            hamiltonian.maximum_interaction_range().try_into()?,
            translate,
        );

        let rotate = Rotate::with_maximum_rotation((0.03).try_into()?);
        let mut rotate_sweep = Sweep(rotate.clone());
        let mut parallel_rotate_sweep =
            ParallelSweep::new(hamiltonian.maximum_interaction_range().try_into()?, rotate);

        let approximate_shape_overlap = Anisotropic {
            interaction: ApproximateShapeOverlap::new(
                Convex(ellipsoid),
                OverlapPenalty::default(),
                0.01.try_into()?,
            ),
            r_cut: hamiltonian.maximum_interaction_range(),
        };
        let overlap_penalty_hamiltonian = PairwiseCutoff(approximate_shape_overlap);

        let microstate = place_single_site_orientable_bodies(
            n,
            number_density,
            hamiltonian.maximum_interaction_range(),
            &overlap_penalty_hamiltonian,
        )?;

        translate_sweep.tune_default(&microstate, &hamiltonian, &Isothermal { temperature: 1.0 });
        *parallel_translate_sweep
            .local_trial_mut()
            .maximum_distance_mut() = *translate_sweep.0.maximum_distance();

        rotate_sweep.tune_default(&microstate, &hamiltonian, &Isothermal { temperature: 1.0 });
        *parallel_rotate_sweep
            .local_trial_mut()
            .maximum_rotation_mut() = *rotate_sweep.0.maximum_rotation();

        let simulation = Self {
            microstate,
            translate_sweep,
            rotate_sweep,
            parallel_translate_sweep,
            parallel_rotate_sweep,
            hamiltonian,
            macrostate,
            translate_count: Count::default(),
            rotate_count: Count::default(),
            parallel,
        };

        let out_bytes: Vec<u8> = postcard::to_stdvec(&simulation)?;
        let mut file = File::create(cache_filename)?;
        file.write_all(&out_bytes)?;

        Ok(simulation)
    }
}

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Benchmark hard polygon Monte Carlo simulations.

use std::{fmt, fs::{self, File}, io::{self, Write}};

use anyhow::Context;
use hoomd_geometry::{
    Convex,
    shape::{ConvexPolygon, Hypercuboid},
};
use hoomd_interaction::{MaximumInteractionRange, PairwiseCutoff, pairwise::{Anisotropic, ApproximateShapeOverlap, HardShape}, univariate::OverlapPenalty};
use hoomd_mc::{Count, HypercuboidCheckerboard, ParallelSweep, Rotate, Sweep, Translate, Trial, Tune};
use hoomd_microstate::{
    Microstate, SiteKey,
    boundary::{GenerateGhosts, Periodic},
    property::OrientedPoint,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::{PointUpdate, PointsNearBall, WithSearchRadius};
use hoomd_vector::{Angle, Cartesian};
use log::debug;
use serde::{Deserialize, Serialize};

use crate::{Effort, place::place_single_site_orientable_bodies};

/// The hard polygon simulation.
#[derive(Serialize, Deserialize)]
pub struct RegularPolygon<X> {
    /// Simulation microstate
    microstate: Microstate<
        OrientedPoint<Cartesian<2>, Angle>,
        OrientedPoint<Cartesian<2>, Angle>,
        X,
        Periodic<Hypercuboid<2>>,
    >,

    /// Translate moves (serial)
    translate_sweep: Sweep<Translate<Cartesian<2>>>,

    /// Translate moves (parallel)
    parallel_translate_sweep: ParallelSweep<
        Translate<Cartesian<2>>,
        HypercuboidCheckerboard<2>,
        OrientedPoint<Cartesian<2>, Angle>,
        OrientedPoint<Cartesian<2>, Angle>,
    >,

    /// Rotate moves (serial)
    rotate_sweep: Sweep<Rotate<Angle>>,

    /// Rotate moves (parallel)
    parallel_rotate_sweep: ParallelSweep<
        Rotate<Angle>,
        HypercuboidCheckerboard<2>,
        OrientedPoint<Cartesian<2>, Angle>,
        OrientedPoint<Cartesian<2>, Angle>,
    >,

    /// Hard polygon interaction.
    hamiltonian: PairwiseCutoff<HardShape<Convex<ConvexPolygon>>>,

    /// Temperature set point.
    macrostate: Isothermal,

    /// Track translate moves attempted during the benchmark period.
    translate_count: Count,

    /// Track rotate moves attempted during the benchmark period.
    rotate_count: Count,

    /// Set to true to use the parallel translate moves.
    parallel: bool,
}

impl<X> Effort for RegularPolygon<X> {
    #[inline]
    fn units() -> String {
        "sweep".to_string()
    }

    #[inline]
    fn effort(&self) -> f64 {
        (self.translate_count.total() + self.rotate_count.total()) as f64
            / self.microstate.bodies().len() as f64
    }
}

impl<X> Simulation for RegularPolygon<X>
where
    X: PointsNearBall<Cartesian<2>, SiteKey> + PointUpdate<Cartesian<2>, SiteKey> + Sync,
    Periodic<Hypercuboid<2>>: GenerateGhosts<OrientedPoint<Cartesian<2>, Angle>>,
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

impl<X> fmt::Display for RegularPolygon<X>
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

impl<X> RegularPolygon<X>
where
    X: PointsNearBall<Cartesian<2>, SiteKey>
        + PointUpdate<Cartesian<2>, SiteKey>
        + WithSearchRadius        + Clone
        + for<'a> Deserialize<'a>
        + Serialize,

    Periodic<Hypercuboid<2>>: GenerateGhosts<OrientedPoint<Cartesian<2>, Angle>>,
{
    /// Construct a new polygon simulation
    ///
    /// # Errors
    /// Returns an error when the microstate cannot be constructed.
    #[inline]
    pub fn new(
        n: usize,
        parallel: bool,
    ) -> anyhow::Result<Self> {
        let macrostate = Isothermal { temperature: 1.0 };
        let initial_maximum_rotation = 0.5;
        let packing_fraction = 0.60;
        let hexagon_area = 3.0 * 3.0f64.sqrt() / 2.0 * 0.5 * 0.5;
        let number_density = packing_fraction / hexagon_area;
        let cache_filename = format!("mc_2d_hexagon_{packing_fraction}_{n}.postcard");

        match fs::read(&cache_filename) {
            Ok(bytes) => {
                debug!("Reading cache '{cache_filename}'.");

                let mut result: Self = postcard::from_bytes(&bytes).with_context(|| format!("Could not read {cache_filename}"))?;
                // The cache may have been generated with a different value of parallel.
                result.parallel = parallel;
                return Ok(result)
            }
            Err(error) => match error.kind() {
                io::ErrorKind::NotFound => (),
                _ => return Err(error).with_context(|| format!("Could not read {cache_filename}")),
            },
        }

        let hexagon = ConvexPolygon::regular(6);

        let hamiltonian = PairwiseCutoff(HardShape(Convex(hexagon.clone())));

        let translate = Translate::with_maximum_distance(0.2.try_into()?);
        let mut translate_sweep = Sweep(translate.clone());
        let mut parallel_translate_sweep = ParallelSweep::new(hamiltonian.0.maximum_interaction_range().try_into()?, translate);

        let rotate = Rotate::with_maximum_rotation(initial_maximum_rotation.try_into()?);
        let mut rotate_sweep = Sweep(rotate.clone());
        let mut parallel_rotate_sweep = ParallelSweep::new(hamiltonian.0.maximum_interaction_range().try_into()?, rotate);

        let approximate_shape_overlap = Anisotropic {
            interaction: ApproximateShapeOverlap::new(
                Convex(hexagon),
                OverlapPenalty::default(),
                0.01.try_into()?,
            ),
            r_cut: hamiltonian.0.maximum_interaction_range(),
        };
        let overlap_penalty_hamiltonian =
            PairwiseCutoff(approximate_shape_overlap);

        let microstate = place_single_site_orientable_bodies(n, number_density, hamiltonian.0.maximum_interaction_range(), &overlap_penalty_hamiltonian)?;

        translate_sweep.tune_default(&microstate, &hamiltonian, &Isothermal { temperature: 1.0 });
        *parallel_translate_sweep.local_trial_mut().maximum_distance_mut() = *translate_sweep.0.maximum_distance();

        rotate_sweep.tune_default(&microstate, &hamiltonian, &Isothermal { temperature: 1.0 });
        *parallel_rotate_sweep.local_trial_mut().maximum_rotation_mut() = *rotate_sweep.0.maximum_rotation();

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

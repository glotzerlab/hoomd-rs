// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::fmt;

use hoomd_geometry::shape::Hypercuboid;
use hoomd_interaction::{PairwiseCutoff, pairwise::HardSphere};
use hoomd_mc::{HypercuboidCheckerboard, Count, ParallelSweep, Sweep, Translate, Trial};
use hoomd_microstate::{
    Body, Microstate, SiteKey,
    boundary::{GenerateGhosts, Periodic},
    property::{Point, Position},
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::{PointUpdate, PointsNearBall, WithSearchRadius};
use hoomd_vector::Cartesian;

use crate::Effort;

pub struct HardSphereSim<const D: usize, X> {
    microstate: Microstate<Point<Cartesian<D>>, Point<Cartesian<D>>, X, Periodic<Hypercuboid<D>>>,
    translate_sweep: Sweep<Translate<Cartesian<D>>>,
    parallel_translate_sweep: ParallelSweep<Translate<Cartesian<D>>, HypercuboidCheckerboard<D>, Point<Cartesian<D>>, Point<Cartesian<D>>>,
    hamiltonian: PairwiseCutoff<HardSphere>,
    macrostate: Isothermal,
    count: Count,
    parallel: bool,
}

impl<const D: usize, X> Effort for HardSphereSim<D, X> {
    fn units() -> String {
        "sweep".to_string()
    }

    fn effort(&self) -> f64 {
        self.count.total() as f64 / self.microstate.bodies().len() as f64
    }
}

impl<const D: usize, X> Simulation for HardSphereSim<D, X>
where
    X: PointsNearBall<Cartesian<D>, SiteKey> + PointUpdate<Cartesian<D>, SiteKey> + Sync,
    Periodic<Hypercuboid<D>>: GenerateGhosts<Point<Cartesian<D>>>,
{
    fn advance(&mut self) -> anyhow::Result<()> {
        if self.parallel {
            self.count += self.parallel_translate_sweep
                .apply(&mut self.microstate, &self.hamiltonian, &self.macrostate);
        } else {
            self.count += self.translate_sweep
                .apply(&mut self.microstate, &self.hamiltonian, &self.macrostate);
        }
        self.microstate.increment_step();

        Ok(())
    }

    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl<const D: usize, X> fmt::Display for HardSphereSim<D, X>
where
    X: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.microstate.fmt(f)?;
        write!(f, "\nTranslate acceptance: {}", self.count.acceptance_ratio().expect("there should be some trial moves"))
    }
}

impl<const D: usize, X> HardSphereSim<D, X>
where
    X: PointsNearBall<Cartesian<D>, SiteKey>
        + PointUpdate<Cartesian<D>, SiteKey>
        + WithSearchRadius,
    Periodic<Hypercuboid<D>>: GenerateGhosts<Point<Cartesian<D>>>,
{
    pub fn new<B, S, X2>(
        microstate: &Microstate<B, S, X2, Periodic<Hypercuboid<D>>>,
        parallel: bool,
    ) -> anyhow::Result<Self>
    where
        B: Position<Position = Cartesian<D>>,
    {
        let sigma = 1.0;

        let translate = Translate::with_maximum_distance((sigma * 0.24).try_into()?);
        let translate_sweep = Sweep(translate.clone());
        let parallel_translate_sweep = ParallelSweep::new(sigma.try_into()?, translate.clone());

        let hamiltonian = PairwiseCutoff(HardSphere { diameter: sigma });

        let cell_list = X::with_search_radius(sigma.try_into()?);
        let microstate = Microstate::builder()
            .spatial_data(cell_list)
            .boundary(microstate.boundary().clone())
            .bodies(microstate.bodies().iter().map(|b| Body {
                properties: Point::<Cartesian<D>>::new(*b.item.properties.position()),
                sites: vec![Point::<Cartesian<D>>::default()],
            }))
            .try_build()?;

        Ok(Self {
            microstate,
            translate_sweep,
            parallel_translate_sweep,
            hamiltonian,
            macrostate: Isothermal { temperature: 1.0 },
            count: Count::default(),
            parallel,
        })
    }
}

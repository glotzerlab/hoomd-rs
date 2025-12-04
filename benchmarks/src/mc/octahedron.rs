// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Benchmark hard octahedra Monte Carlo simulations.

use std::fmt;

use hoomd_geometry::{
    Convex,
    shape::{ConvexPolyhedron, Hypercuboid},
};
use hoomd_interaction::{PairwiseCutoff, pairwise::HardShape};
use hoomd_mc::{Count, HypercuboidCheckerboard, ParallelSweep, Sweep, Translate, Trial};
use hoomd_microstate::{
    Microstate, SiteKey,
    boundary::{GenerateGhosts, Periodic},
    property::OrientedPoint,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::{PointUpdate, PointsNearBall, WithSearchRadius};
use hoomd_vector::{Cartesian, Versor};

use crate::Effort;

/// The hard octahedra simulation.
pub struct Octahedron<X> {
    /// Simulation microstate
    microstate: Microstate<
        OrientedPoint<Cartesian<3>, Versor>,
        OrientedPoint<Cartesian<3>, Versor>,
        X,
        Periodic<Hypercuboid<3>>,
    >,

    /// Translate moves (serial)
    translate_sweep: Sweep<Translate<Cartesian<3>>>,

    /// Translate moves (parallel)
    parallel_translate_sweep: ParallelSweep<
        Translate<Cartesian<3>>,
        HypercuboidCheckerboard<3>,
        OrientedPoint<Cartesian<3>, Versor>,
        OrientedPoint<Cartesian<3>, Versor>,
    >,

    // TODO: add rotate moves.

    /// Hard octahedra interaction.
    hamiltonian: PairwiseCutoff<HardShape<Convex<ConvexPolyhedron>>>,

    /// Temperature set point.
    macrostate: Isothermal,

    /// Track moves attempted during the benchmark period.
    count: Count,

    /// Set to true to use the parallel translate moves.
    parallel: bool,
}

impl<X> Effort for Octahedron<X> {
    #[inline]
    fn units() -> String {
        "sweep".to_string()
    }

    #[inline]
    fn effort(&self) -> f64 {
        self.count.total() as f64 / self.microstate.bodies().len() as f64
    }
}

impl<X> Simulation for Octahedron<X>
where
    X: PointsNearBall<Cartesian<3>, SiteKey> + PointUpdate<Cartesian<3>, SiteKey> + Sync,
    Periodic<Hypercuboid<3>>: GenerateGhosts<OrientedPoint<Cartesian<3>, Versor>>,
{
    #[inline]
    fn advance(&mut self) -> anyhow::Result<()> {
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

        // TODO: Rotate moves

        Ok(())
    }

    #[inline]
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl<X> fmt::Display for Octahedron<X>
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

impl<X> Octahedron<X>
where
    X: PointsNearBall<Cartesian<3>, SiteKey>
        + PointUpdate<Cartesian<3>, SiteKey>
        + WithSearchRadius,
    Periodic<Hypercuboid<3>>: GenerateGhosts<OrientedPoint<Cartesian<3>, Versor>>,
{
    /// Construct a new hard octahedra simulation
    ///
    /// # Errors
    /// Returns an error when the microstate cannot be constructed.
    #[inline]
    pub fn new<X2>(
        microstate: &Microstate<
            OrientedPoint<Cartesian<3>, Versor>,
            OrientedPoint<Cartesian<3>, Versor>,
            X2,
            Periodic<Hypercuboid<3>>,
        >,
        parallel: bool,
    ) -> anyhow::Result<Self> {
        let sigma = 1.0;

        let translate = Translate::with_maximum_distance((sigma * 0.1).try_into()?);
        let translate_sweep = Sweep(translate.clone());
        let parallel_translate_sweep = ParallelSweep::new(sigma.try_into()?, translate);

        let octahedron = ConvexPolyhedron::with_vertices(vec![
            [-0.5, 0.0, 0.0].into(),
            [0.5, 0.0, 0.0].into(),
            [0.0, -0.5, 0.0].into(),
            [0.0, 0.5, 0.0].into(),
            [0.0, 0.0, -0.5].into(),
            [0.0, 0.0, 0.5].into(),
        ])?;
        let hamiltonian = PairwiseCutoff(HardShape(Convex(octahedron)));

        let cell_list = X::with_search_radius(sigma.try_into()?);
        let microstate = Microstate::builder()
            .spatial_data(cell_list)
            .boundary(microstate.boundary().clone())
            .bodies(microstate.bodies().iter().map(|b| b.item.clone()))
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

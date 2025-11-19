// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::fmt;

use hoomd_geometry::{
    Convex,
    shape::{ConvexPolygon, Hypercuboid},
};
use hoomd_interaction::{PairwiseCutoff, pairwise::HardShape};
use hoomd_mc::{checkerboard::HypercuboidCheckerboard, Count, ParallelSweep, Rotate, Sweep, Translate, Trial};
use hoomd_microstate::{
    Microstate, SiteKey,
    boundary::{GenerateGhosts, Periodic},
    property::OrientedPoint,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::{PointUpdate, PointsNearBall, WithSearchRadius};
use hoomd_vector::{Angle, Cartesian};

use crate::Effort;

pub struct RegularPolygon<X> {
    microstate: Microstate<
        OrientedPoint<Cartesian<2>, Angle>,
        OrientedPoint<Cartesian<2>, Angle>,
        X,
        Periodic<Hypercuboid<2>>,
    >,
    translate_sweep: Sweep<Translate<Cartesian<2>>>,
    parallel_translate_sweep: ParallelSweep<Translate<Cartesian<2>>, HypercuboidCheckerboard<2>, OrientedPoint<Cartesian<2>, Angle>, OrientedPoint<Cartesian<2>, Angle>>,
    rotate_sweep: Sweep<Rotate<Angle>>,
    parallel_rotate_sweep: ParallelSweep<Rotate<Angle>, HypercuboidCheckerboard<2>,OrientedPoint<Cartesian<2>, Angle>,OrientedPoint<Cartesian<2>, Angle>>,
    hamiltonian: PairwiseCutoff<HardShape<Convex<ConvexPolygon>>>,
    macrostate: Isothermal,
    translate_count: Count,
    rotate_count: Count,
    parallel: bool,
}

impl<X> Effort for RegularPolygon<X> {
    fn units() -> String {
        "sweep".to_string()
    }

    fn effort(&self) -> f64 {
        (self.translate_count.total() + self.rotate_count.total()) as f64 / self.microstate.bodies().len() as f64
    }
}

impl<X> Simulation for RegularPolygon<X>
where
    X: PointsNearBall<Cartesian<2>, SiteKey> + PointUpdate<Cartesian<2>, SiteKey> + Sync,
    Periodic<Hypercuboid<2>>: GenerateGhosts<OrientedPoint<Cartesian<2>, Angle>>,
{
    fn advance(&mut self) -> anyhow::Result<()> {
        if self.parallel {
            self.translate_count += self.parallel_translate_sweep
                .apply(&mut self.microstate, &self.hamiltonian, &self.macrostate);
            self.rotate_count += self.parallel_rotate_sweep
                .apply(&mut self.microstate, &self.hamiltonian, &self.macrostate);
        } else {
            self.translate_count += self.translate_sweep
                .apply(&mut self.microstate, &self.hamiltonian, &self.macrostate);
            self.rotate_count += self.rotate_sweep
                .apply(&mut self.microstate, &self.hamiltonian, &self.macrostate);
        }

        self.microstate.increment_step();

        Ok(())
    }

    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl<X> fmt::Display for RegularPolygon<X>
where
    X: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.microstate.fmt(f)?;
        write!(f, "\nTranslate acceptance: {}", self.translate_count.acceptance_ratio().expect("there should be some trial moves"))?;
        write!(f, "\nRotate acceptance: {}", self.rotate_count.acceptance_ratio().expect("there should be some trial moves"))
    }
}

impl<X> RegularPolygon<X>
where
    X: PointsNearBall<Cartesian<2>, SiteKey>
        + PointUpdate<Cartesian<2>, SiteKey>
        + WithSearchRadius,
    Periodic<Hypercuboid<2>>: GenerateGhosts<OrientedPoint<Cartesian<2>, Angle>>,
{
    pub fn new<X2>(
        microstate: &Microstate<
            OrientedPoint<Cartesian<2>, Angle>,
            OrientedPoint<Cartesian<2>, Angle>,
            X2,
            Periodic<Hypercuboid<2>>,
        >,
        parallel: bool,
    ) -> anyhow::Result<Self> {
        let sigma = 1.0;
        let maximum_rotation = 0.5;

        let translate = Translate::with_maximum_distance((sigma * 0.6).try_into()?);
        let translate_sweep = Sweep(translate.clone());
        let parallel_translate_sweep = ParallelSweep::new(sigma.try_into()?, translate);

        let rotate =
            Rotate::with_maximum_rotation(maximum_rotation.try_into()?);
        let rotate_sweep = Sweep(rotate.clone());
        let parallel_rotate_sweep = ParallelSweep::new(sigma.try_into()?, rotate);

        let big_hexagon = ConvexPolygon::regular(6);
        let hexagon =
            ConvexPolygon::with_vertices(big_hexagon.vertices().iter().map(|v| *v / 2.0))?;

        let hamiltonian = PairwiseCutoff(HardShape(Convex(hexagon)));

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
            rotate_sweep,
            parallel_rotate_sweep,
            hamiltonian,
            macrostate: Isothermal { temperature: 1.0 },
            translate_count: Count::default(),
            rotate_count: Count::default(),
            parallel,
        })
    }
}

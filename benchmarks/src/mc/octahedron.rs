// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::fmt;

use hoomd_geometry::{
    Convex,
    shape::{ConvexPolyhedron, Hypercuboid},
};
use hoomd_interaction::{PairwiseCutoff, pairwise::HardShape};
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_microstate::{
    Microstate, SiteKey,
    boundary::{GenerateGhosts, Periodic},
    property::OrientedPoint,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::{PointUpdate, PointsNearBall, WithSearchRadius};
use hoomd_vector::{Cartesian, Versor};

pub struct Octahedron<X> {
    microstate: Microstate<
        OrientedPoint<Cartesian<3>, Versor>,
        OrientedPoint<Cartesian<3>, Versor>,
        X,
        Periodic<Hypercuboid<3>>,
    >,
    translate_sweep: Sweep<Translate<Cartesian<3>>>,
    hamiltonian: PairwiseCutoff<HardShape<Convex<ConvexPolyhedron>>>,
    macrostate: Isothermal,
}

impl<X> Simulation for Octahedron<X>
where
    X: PointsNearBall<Cartesian<3>, SiteKey> + PointUpdate<Cartesian<3>, SiteKey>,
    Periodic<Hypercuboid<3>>: GenerateGhosts<OrientedPoint<Cartesian<3>, Versor>>,
{
    fn advance(&mut self) -> anyhow::Result<()> {
        self.translate_sweep
            .apply(&mut self.microstate, &self.hamiltonian, &self.macrostate);
        self.microstate.increment_step();

        // TODO: Rotate moves

        Ok(())
    }

    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl<X> fmt::Display for Octahedron<X>
where
    X: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.microstate.fmt(f)
    }
}

impl<X> Octahedron<X>
where
    X: PointsNearBall<Cartesian<3>, SiteKey>
        + PointUpdate<Cartesian<3>, SiteKey>
        + WithSearchRadius,
    Periodic<Hypercuboid<3>>: GenerateGhosts<OrientedPoint<Cartesian<3>, Versor>>,
{
    pub fn with_microstate<X2>(
        microstate: &Microstate<
            OrientedPoint<Cartesian<3>, Versor>,
            OrientedPoint<Cartesian<3>, Versor>,
            X2,
            Periodic<Hypercuboid<3>>,
        >,
    ) -> anyhow::Result<Self> {
        let sigma = 1.0;

        let translate = Translate::with_maximum_distance((sigma * 0.1).try_into()?);
        let translate_sweep = Sweep(translate);

        let octahedron = ConvexPolyhedron::with_vertices(vec![
            [-0.5, 0.0, 0.0].into(),
            [0.5, 0.0, 0.0].into(),
            [0.0, -0.5, 0.0].into(),
            [0.0, 0.5, 0.0].into(),
            [0.0, 0.0, -0.5].into(),
            [0.0, 0.0, 0.5].into(),
        ])?;
        let hamiltonian = PairwiseCutoff (
            HardShape(Convex(octahedron)),
        );

        let cell_list = X::with_search_radius(sigma.try_into()?);
        let microstate = Microstate::builder()
            .spatial_data(cell_list)
            .boundary(microstate.boundary().clone())
            .bodies(microstate.bodies().iter().map(|b| b.item.clone()))
            .try_build()?;

        Ok(Self {
            microstate,
            translate_sweep,
            hamiltonian,
            macrostate: Isothermal { temperature: 1.0 },
        })
    }
}

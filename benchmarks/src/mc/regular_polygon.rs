// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::fmt;

use hoomd_geometry::{
    Convex,
    shape::{ConvexPolygon, Hypercuboid},
};
use hoomd_interaction::{CutoffPairOverlap, pairwise::HardShape};
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_microstate::{
    Microstate, SiteKey,
    boundary::{GenerateGhosts, Periodic},
    property::OrientedPoint,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::{PointUpdate, PointsInBall, WithSearchRadius};
use hoomd_vector::{Angle, Cartesian};

pub struct RegularPolygon<X> {
    microstate: Microstate<
        OrientedPoint<Cartesian<2>, Angle>,
        OrientedPoint<Cartesian<2>, Angle>,
        X,
        Periodic<Hypercuboid<2>>,
    >,
    translate_sweep: Sweep<Translate<Cartesian<2>>>,
    hamiltonian: CutoffPairOverlap<HardShape<Convex<ConvexPolygon>>>,
    macrostate: Isothermal,
}

impl<X> Simulation for RegularPolygon<X>
where
    X: PointsInBall<Cartesian<2>, SiteKey> + PointUpdate<Cartesian<2>, SiteKey>,
    Periodic<Hypercuboid<2>>: GenerateGhosts<OrientedPoint<Cartesian<2>, Angle>>,
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

impl<X> fmt::Display for RegularPolygon<X> where
    X: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.microstate.fmt(f)
    }
}

impl<X> RegularPolygon<X>
where
    X: PointsInBall<Cartesian<2>, SiteKey> + PointUpdate<Cartesian<2>, SiteKey> + WithSearchRadius,
    Periodic<Hypercuboid<2>>: GenerateGhosts<OrientedPoint<Cartesian<2>, Angle>>,
{
    pub fn with_microstate<X2>(
        microstate: &Microstate<
            OrientedPoint<Cartesian<2>, Angle>,
            OrientedPoint<Cartesian<2>, Angle>,
            X2,
            Periodic<Hypercuboid<2>>,
        >,
    ) -> anyhow::Result<Self> {
        let sigma = 1.0;

        let translate = Translate::with_maximum_distance((sigma * 0.1).try_into()?);
        let translate_sweep = Sweep(translate);

        let big_hexagon = ConvexPolygon::regular(6);
        let hexagon =
            ConvexPolygon::with_vertices(big_hexagon.vertices().iter().map(|v| *v / 2.0))?;

        let hamiltonian = CutoffPairOverlap {
            r_cut: sigma,
            evaluator: HardShape(Convex(hexagon)),
        };

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

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use hoomd_spatial::{PointUpdate, PointsInBall, VecCell};
use hoomd_vector::{Cartesian, Angle};
use hoomd_simulation::{macrostate::Isothermal, Simulation};
use hoomd_microstate::{boundary::{GenerateGhosts, Periodic}, property::OrientedPoint, Microstate, MicrostateBuilder, SiteKey};
use hoomd_geometry::{shape::Hypercuboid, Convex};
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_interaction::{CutoffPairOverlap, pairwise::HardShape};
use hoomd_geometry::shape::ConvexPolygon;

pub struct RegularPolygon<X> {
    microstate: Microstate<OrientedPoint<Cartesian<2>, Angle>, OrientedPoint<Cartesian<2>, Angle>, X, Periodic<Hypercuboid<2>>>,
    translate_sweep: Sweep<Translate<Cartesian<2>>>,
    hamiltonian: CutoffPairOverlap<HardShape<Convex<ConvexPolygon>>>,
    macrostate: Isothermal,
}

impl<X> Simulation for RegularPolygon<X> where
X: PointsInBall<Cartesian<2>, SiteKey> + PointUpdate<Cartesian<2>, SiteKey>,
Periodic<Hypercuboid<2>>: GenerateGhosts<OrientedPoint<Cartesian<2>, Angle>>,
{
    fn advance(&mut self) -> anyhow::Result<()> {
        self.translate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
        );
        self.microstate.increment_step();

        // TODO: Rotate moves

        Ok(())
    }

    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl RegularPolygon<VecCell<SiteKey, 2>> where
Periodic<Hypercuboid<2>>: GenerateGhosts<OrientedPoint<Cartesian<2>, Angle>>,
{
    pub fn with_microstate<X>(microstate: &Microstate<OrientedPoint<Cartesian<2>, Angle>, OrientedPoint<Cartesian<2>, Angle>, X, Periodic<Hypercuboid<2>>>) -> anyhow::Result<Self> {
        let sigma = 1.0;

        let translate = Translate::with_maximum_distance((sigma * 0.1).try_into()?);
        let translate_sweep = Sweep(translate);

        let big_hexagon = ConvexPolygon::regular(6);
        let hexagon = ConvexPolygon::with_vertices(big_hexagon.vertices().iter().map(|v| *v / 2.0))?;

        let hamiltonian = CutoffPairOverlap {
            r_cut: sigma,
            evaluator: HardShape(Convex(hexagon)),
        };    
    
        let cell_list = VecCell::new(sigma, 1);
        let microstate = MicrostateBuilder::with_spatial_data_and_boundary(cell_list, microstate.boundary().clone())
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

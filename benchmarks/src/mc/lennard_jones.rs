// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use hoomd_spatial::{PointUpdate, PointsInBall, VecCell};
use hoomd_vector::Cartesian;
use hoomd_simulation::{macrostate::Isothermal, Simulation};
use hoomd_microstate::{boundary::{GenerateGhosts, Periodic}, property::Point, Microstate, MicrostateBuilder, SiteKey};
use hoomd_geometry::shape::Hypercuboid;
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_interaction::{pairwise::{self, Isotropic}, CutoffPair};

struct HardSphereWell {
    maximum_interaction_range: f64
}

pub struct LennardJones<const D: usize, X> {
    microstate: Microstate<Point<Cartesian<D>>, Point<Cartesian<D>>, X, Periodic<Hypercuboid<D>>>,
    translate_sweep: Sweep<Translate<Cartesian<D>>>,
    hamiltonian: CutoffPair<Isotropic<pairwise::LennardJones>>,
    macrostate: Isothermal,
}

impl<const D: usize, X> Simulation for LennardJones<D, X> where
X: PointsInBall<Cartesian<D>, SiteKey> + PointUpdate<Cartesian<D>, SiteKey>,
Periodic<Hypercuboid<D>>: GenerateGhosts<Point<Cartesian<D>>>,
{
    fn advance(&mut self) -> anyhow::Result<()> {
        self.translate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
        );
        self.microstate.increment_step();

        Ok(())
    }

    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl<const D: usize> LennardJones<D, VecCell<SiteKey, D>> where
Periodic<Hypercuboid<D>>: GenerateGhosts<Point<Cartesian<D>>>,
{
    pub fn with_microstate<X>(microstate: &Microstate<Point<Cartesian<D>>, Point<Cartesian<D>>, X, Periodic<Hypercuboid<D>>>) -> anyhow::Result<Self> {
        let maximum_interaction_range = 2.5;

        let translate = Translate::with_maximum_distance(0.18.try_into()?);
        let translate_sweep = Sweep(translate);

        let hamiltonian = CutoffPair {
            r_cut: maximum_interaction_range,
            evaluator: Isotropic(pairwise::LennardJones { epsilon: 1.0, sigma: 1.0 }),
        };    
    
        let cell_list = VecCell::new(maximum_interaction_range, 1);
        let boundary = Periodic::new(maximum_interaction_range,
            microstate.boundary().shape().clone())?;
        let microstate = MicrostateBuilder::with_spatial_data_and_boundary(cell_list, boundary)
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

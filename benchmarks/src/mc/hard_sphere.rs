// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use hoomd_spatial::{PointUpdate, PointsInBall, WithSearchRadius};
use hoomd_vector::Cartesian;
use hoomd_simulation::{macrostate::Isothermal, Simulation};
use hoomd_microstate::{boundary::{GenerateGhosts, Periodic}, property::{Point, Position}, Body, Microstate, SiteKey};
use hoomd_geometry::shape::Hypercuboid;
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_interaction::{CutoffPairOverlap, pairwise::AlwaysTrue};

pub struct HardSphere<const D: usize, X> {
    microstate: Microstate<Point<Cartesian<D>>, Point<Cartesian<D>>, X, Periodic<Hypercuboid<D>>>,
    translate_sweep: Sweep<Translate<Cartesian<D>>>,
    hamiltonian: CutoffPairOverlap<AlwaysTrue>,
    macrostate: Isothermal,
}

impl<const D: usize, X> Simulation for HardSphere<D, X> where
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

impl<const D: usize, X> HardSphere<D, X> where
X: PointsInBall<Cartesian<D>, SiteKey> + PointUpdate<Cartesian<D>, SiteKey> + WithSearchRadius,
Periodic<Hypercuboid<D>>: GenerateGhosts<Point<Cartesian<D>>>,
{
    pub fn with_microstate<B, S, X2>(microstate: &Microstate<B, S, X2, Periodic<Hypercuboid<D>>>) -> anyhow::Result<Self>
    where B: Position<Position = Cartesian<D>>,
    {
        let sigma = 1.0;

        let translate = Translate::with_maximum_distance((sigma * 0.1).try_into()?);
        let translate_sweep = Sweep(translate);

        let hamiltonian = CutoffPairOverlap {
            r_cut: sigma,
            evaluator: AlwaysTrue,
        };    
    
        let cell_list = X::with_search_radius(sigma.try_into()?);
        let microstate = Microstate::builder().spatial_data(cell_list).boundary(microstate.boundary().clone())
            .bodies(microstate.bodies().iter().map(|b|
                Body { properties: Point::<Cartesian<D>>::new(*b.item.properties.position()),
                    sites: vec![Point::<Cartesian<D>>::default()],
                }
                ))
            .try_build()?;

        Ok(Self {
            microstate,
            translate_sweep,
            hamiltonian,
            macrostate: Isothermal { temperature: 1.0 },
        })
    }
}

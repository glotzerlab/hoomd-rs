// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::fmt;

use hoomd_geometry::shape::Hypercuboid;
use hoomd_interaction::{
    CutoffPair,
    pairwise::{self, Isotropic},
};
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_microstate::{
    Body, Microstate, SiteKey,
    boundary::{GenerateGhosts, Periodic},
    property::{Point, Position},
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::{PointUpdate, PointsNearBall, WithSearchRadius};
use hoomd_vector::Cartesian;

pub struct LennardJones<const D: usize, X> {
    microstate: Microstate<Point<Cartesian<D>>, Point<Cartesian<D>>, X, Periodic<Hypercuboid<D>>>,
    translate_sweep: Sweep<Translate<Cartesian<D>>>,
    hamiltonian: CutoffPair<Isotropic<pairwise::LennardJones>>,
    macrostate: Isothermal,
}

impl<const D: usize, X> Simulation for LennardJones<D, X>
where
    X: PointsNearBall<Cartesian<D>, SiteKey> + PointUpdate<Cartesian<D>, SiteKey>,
    Periodic<Hypercuboid<D>>: GenerateGhosts<Point<Cartesian<D>>>,
{
    fn advance(&mut self) -> anyhow::Result<()> {
        self.translate_sweep
            .apply(&mut self.microstate, &self.hamiltonian, &self.macrostate);
        self.microstate.increment_step();

        Ok(())
    }

    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl<const D: usize, X> fmt::Display for LennardJones<D, X>
where
    X: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.microstate.fmt(f)
    }
}

impl<const D: usize, X> LennardJones<D, X>
where
    X: PointsNearBall<Cartesian<D>, SiteKey>
        + PointUpdate<Cartesian<D>, SiteKey>
        + WithSearchRadius,
    Periodic<Hypercuboid<D>>: GenerateGhosts<Point<Cartesian<D>>>,
{
    pub fn with_microstate<B, S, X2>(
        microstate: &Microstate<B, S, X2, Periodic<Hypercuboid<D>>>,
    ) -> anyhow::Result<Self>
    where
        B: Position<Position = Cartesian<D>>,
    {
        let maximum_interaction_range = 2.5;

        let translate = Translate::with_maximum_distance(0.18.try_into()?);
        let translate_sweep = Sweep(translate);

        let hamiltonian = CutoffPair {
            r_cut: maximum_interaction_range,
            evaluator: Isotropic(pairwise::LennardJones {
                epsilon: 1.0,
                sigma: 1.0,
            }),
        };

        let cell_list = X::with_search_radius(maximum_interaction_range.try_into()?);
        let boundary = Periodic::new(
            maximum_interaction_range,
            microstate.boundary().shape().clone(),
        )?;
        let microstate = Microstate::builder()
            .spatial_data(cell_list)
            .boundary(boundary)
            .bodies(microstate.bodies().iter().map(|b| Body {
                properties: Point::<Cartesian<D>>::new(*b.item.properties.position()),
                sites: vec![Point::<Cartesian<D>>::default()],
            }))
            .try_build()?;

        Ok(Self {
            microstate,
            translate_sweep,
            hamiltonian,
            macrostate: Isothermal { temperature: 1.0 },
        })
    }
}

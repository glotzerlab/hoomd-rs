// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use hoomd_spatial::{PointUpdate, PointsInBall, VecCell};
use hoomd_vector::{Cartesian, Versor};
use hoomd_simulation::{macrostate::Isothermal, Simulation};
use hoomd_microstate::{boundary::{GenerateGhosts, Periodic}, property::{OrientedPoint, Point}, Body, Microstate, MicrostateBuilder, SiteKey};
use hoomd_geometry::shape::Hypercuboid;
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_interaction::{pairwise::{WeeksChandlerAnderson, Isotropic}, CutoffPair};

pub struct WcaUnion<X> {
    microstate: Microstate<OrientedPoint<Cartesian<3>, Versor>, Point<Cartesian<3>>, X, Periodic<Hypercuboid<3>>>,
    translate_sweep: Sweep<Translate<Cartesian<3>>>,
    hamiltonian: CutoffPair<Isotropic<WeeksChandlerAnderson>>,
    macrostate: Isothermal,
}

impl<X> Simulation for WcaUnion<X> where
X: PointsInBall<Cartesian<3>, SiteKey> + PointUpdate<Cartesian<3>, SiteKey>,
Periodic<Hypercuboid<3>>: GenerateGhosts<Point<Cartesian<3>>>,
{
    fn advance(&mut self) -> anyhow::Result<()> {
        self.translate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
        );
        self.microstate.increment_step();

        // TODO: Rotation moves.
    
        Ok(())
    }

    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl WcaUnion<VecCell<SiteKey, 3>> where
Periodic<Hypercuboid<3>>: GenerateGhosts<Point<Cartesian<3>>>,
{
    pub fn with_microstate<X>(microstate: &Microstate<OrientedPoint<Cartesian<3>, Versor>, Point<Cartesian<3>>, X, Periodic<Hypercuboid<3>>>) -> anyhow::Result<Self> {
        let maximum_interaction_range = 0.1 * 2.0f64.powf(1.0/6.0);

        let translate = Translate::with_maximum_distance(0.3.try_into()?);
        let translate_sweep = Sweep(translate);

        let hamiltonian = CutoffPair {
            r_cut: maximum_interaction_range,
            evaluator: Isotropic(WeeksChandlerAnderson { epsilon: 1.0, sigma: 1.0 }),
        };
    
        let cell_list = VecCell::new(maximum_interaction_range, 1);
        let boundary = Periodic::new(maximum_interaction_range,
            microstate.boundary().shape().clone())?;

        let template_sites = vec![
            Point::new([0.0, 0.0, -0.5].into()),
            Point::new([0.0, 0.0, -0.16666667].into()),
            Point::new([0.0, 0.0, 0.16666667].into()),
            Point::new([0.0, 0.0, 0.5].into()),
            Point::new([0.0, -0.5, 0.0].into()),
            Point::new([0.0, -0.16666667, 0.0].into()),
            Point::new([0.0, 0.16666667, 0.0].into()),
            Point::new([0.0, 0.5, 0.0].into()),
            Point::new([-0.5, 0.0, 0.0 ].into()),
            Point::new([-0.16666667, 0.0, 0.0 ].into()),
            Point::new([0.16666667, 0.0, 0.0 ].into()),
            Point::new([0.5, 0.0, 0.0 ].into()),
        ];
            
        let microstate = MicrostateBuilder::with_spatial_data_and_boundary(cell_list, boundary)
            .bodies(microstate.bodies().iter().map(|b|
                Body {
                    properties: b.item.properties,
                    sites: template_sites.clone(),
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

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Benchmark Lennard Jones Monte Carlo simulations.

use std::fmt;

use hoomd_geometry::shape::EightEight;
use hoomd_interaction::{PairwiseCutoff, pairwise::Isotropic, univariate};
use hoomd_manifold::Hyperbolic;
use hoomd_mc::{Count, Sweep, Translate, Trial};
use hoomd_microstate::{
    Body, Microstate, SiteKey,
    boundary::{GenerateGhosts, Periodic},
    property::{Point, Position},
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::{PointUpdate, PointsNearBall, WithSearchRadius};

use crate::Effort;

/// The Lennard Jones simulation.
pub struct HyperbolicLennardJones<X> {
    /// Simulation microstate
    microstate: Microstate<Point<Hyperbolic<3>>, Point<Hyperbolic<3>>, X, Periodic<EightEight>>,

    /// Translate moves (serial)
    translate_sweep: Sweep<Translate<Point<Hyperbolic<3>>>>,

    /// Lennard Jones interaction.
    hamiltonian: PairwiseCutoff<Isotropic<univariate::LennardJonesGauss>>,

    /// Temperature set point
    macrostate: Isothermal,

    /// Track moves accepted during the benchmark period.
    count: Count,
}

impl<X> Effort for HyperbolicLennardJones<X> {
    #[inline]
    fn units() -> String {
        "sweep".to_string()
    }

    #[inline]
    fn effort(&self) -> f64 {
        self.count.total() as f64 / self.microstate.bodies().len() as f64
    }
}

impl<X> Simulation for HyperbolicLennardJones<X>
where
    X: PointsNearBall<Hyperbolic<3>, SiteKey> + PointUpdate<Hyperbolic<3>, SiteKey> + Sync,
    Periodic<EightEight>: GenerateGhosts<Point<Hyperbolic<3>>>,
{
    #[inline]
    fn advance(&mut self) -> anyhow::Result<()> {
        self.count += self.translate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
        );
        

        self.microstate.increment_step();

        Ok(())
    }

    #[inline]
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl<X> fmt::Display for HyperbolicLennardJones<X>
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

impl<X> HyperbolicLennardJones<X>
where
    X: PointsNearBall<Hyperbolic<3>, SiteKey>
        + PointUpdate<Hyperbolic<3>, SiteKey>
        + WithSearchRadius,
    Periodic<EightEight>: GenerateGhosts<Point<Hyperbolic<3>>>,
{
    /// Construct a new Lennard Jones simulation
    ///
    /// # Errors
    /// Returns an error when the microstate cannot be constructed.
    #[inline]
    pub fn new<B, S, X2>(
        microstate: &Microstate<B, S, X2, Periodic<EightEight>>,
    ) -> anyhow::Result<Self>
    where
        B: Position<Position = Hyperbolic<3>>,
    {
        let maximum_interaction_range = 0.5;

        let translate = Translate::with_maximum_distance(0.01.try_into()?);
        let translate_sweep = Sweep(translate.clone());

        let hamiltonian = PairwiseCutoff(Isotropic {
            interaction: univariate::LennardJonesGauss {
                epsilon: 1.8,
                sigma_squared: 0.02,
                r_0: 1.52,
                scale: 0.1,
            },
            r_cut: maximum_interaction_range,
        });

        let cell_list = X::with_search_radius(maximum_interaction_range.try_into()?);
        let boundary = Periodic::new(
            maximum_interaction_range,
            microstate.boundary().shape().clone(),
        )?;
        let microstate = Microstate::builder()
            .spatial_data(cell_list)
            .boundary(boundary)
            .bodies(microstate.bodies().iter().map(|b| Body {
                properties: Point::<Hyperbolic<3>>::new(*b.item.properties.position()),
                sites: vec![Point::<Hyperbolic<3>>::default()],
            }))
            .try_build()?;

        Ok(Self {
            microstate,
            translate_sweep,
            hamiltonian,
            macrostate: Isothermal { temperature: 1.0 },
            count: Count::default(),
        })
    }
}
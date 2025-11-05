// ANCHOR: all
// ANCHOR: use
use rand::{Rng, seq::IndexedRandom};
use std::f64::consts::PI;

use hoomd_geometry::shape::Rectangle;
use hoomd_interaction::{
    PairwiseCutoff, External, TotalEnergy,
    external::Linear,
    pairwise::{Boxcar, Isotropic},
};
use hoomd_mc::{LocalTrial, Sweep, Trial};
use hoomd_microstate::{
    Body, Microstate, SiteKey,
    boundary::Closed,
    property::{OrientedPoint, Point},
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::{Angle, Cartesian};
// ANCHOR_END: use

// ANCHOR: type_aliases
type PositionVector = Cartesian<2>;
type BodyProperties = OrientedPoint<PositionVector, Angle>;
type SiteProperties = Point<PositionVector>;
// ANCHOR_END: type_aliases

// ANCHOR: local_trial
/// Take fixed steps left, right, down, up, rotate left, or rotate right.
struct DiscreteRotateOrTranslate;

impl LocalTrial<BodyProperties> for DiscreteRotateOrTranslate {
    fn propose<R: Rng>(
        &self,
        rng: &mut R,
        body_properties: BodyProperties,
    ) -> BodyProperties {
        // ANCHOR_END: local_trial
        // ANCHOR: local_trial_steps
        let translate_steps = [
            [0.0, -1.0].into(),
            [0.0, 1.0].into(),
            [-1.0, 0.0].into(),
            [1.0, 0.0].into(),
        ];
        let rotate_steps = [-PI / 2.0, PI / 2.0];
        // ANCHOR_END: local_trial_steps

        // ANCHOR: local_trial_mut
        let mut trial = body_properties;
        if rng.random_bool(0.9) {
            trial.position += *translate_steps
                .choose(rng)
                .expect("translate_steps should have at least 1 element");
        } else {
            trial.orientation.theta += *rotate_steps
                .choose(rng)
                .expect("rotate_steps should have at least 1 element");
        }
        trial
    }
}
// ANCHOR_END: local_trial_mut

// Remove the cfg_attr(...) line when using this code outside the hoomd-rs/examples directory.
#[cfg_attr(feature = "bevy", derive(Resource))]
// ANCHOR: simulation_struct
struct Tetronimoes {
    /// Positions and orientations of all the bodies in the simulation.
    microstate: Microstate<
        BodyProperties,
        SiteProperties,
        VecCell<SiteKey, 2>,
        Closed<Rectangle>,
    >,
    /// How sites interact with other sites and fields.
    hamiltonian: (
        External<Linear<PositionVector>>,
        PairwiseCutoff<Isotropic<Boxcar>>,
    ),
    /// Trial moves to apply.
    sweep: Sweep<DiscreteRotateOrTranslate>,
    /// Temperature set point.
    macrostate: Isothermal,
    /// Tetronimo shapes.
    template_sites: Vec<Vec<Point<PositionVector>>>,
}
// ANCHOR_END: simulation_struct

// ANCHOR: simulation_new
impl Tetronimoes {
    /// Construct a new tetronimo simulation.
    fn new() -> anyhow::Result<Tetronimoes> {
        // ANCHOR_END: simulation_new
        // ANCHOR: parameters
        let box_height = 30.0;
        let macrostate = Isothermal { temperature: 1.0 };
        let alpha = 1.0;
        let epsilon = 1000.0;
        let sigma = 1.0;
        // ANCHOR_END: parameters

        // ANCHOR: microstate
        let vec_cell = VecCell::builder()
            .nominal_search_radius(sigma.try_into()?)
            .build();
        let square = Rectangle::with_equal_edges(box_height.try_into()?);
        let microstate = Microstate::builder()
            .spatial_data(vec_cell)
            .boundary(Closed(square))
            .try_build()?;
        // ANCHOR_END: microstate

        // ANCHOR: hamiltonian
        let linear = External(Linear {
            alpha,
            plane_origin: Cartesian::default(),
            plane_normal: [0.0, 1.0].try_into()?,
        });

        let boxcar = Boxcar {
            epsilon,
            left: 0.0,
            right: sigma,
        };
        let isotropic = Isotropic(boxcar);
        let cutoff_pair = PairwiseCutoff {
            r_cut: sigma,
            evaluator: isotropic,
        };

        let hamiltonian = (linear, cutoff_pair);
        // ANCHOR_END: hamiltonian

        // ANCHOR: trial_moves
        let sweep = Sweep(DiscreteRotateOrTranslate);
        // ANCHOR_END: trial_moves

        // ANCHOR: template_sites
        let template_sites = vec![
            // square
            vec![
                Point::new([-0.5, -0.5].into()),
                Point::new([0.5, -0.5].into()),
                Point::new([0.5, 0.5].into()),
                Point::new([-0.5, 0.5].into()),
            ],
            // line
            vec![
                Point::new([-1.5, 0.5].into()),
                Point::new([-0.5, 0.5].into()),
                Point::new([0.5, 0.5].into()),
                Point::new([1.5, 0.5].into()),
            ],
            // T
            vec![
                Point::new([-1.5, -0.5].into()),
                Point::new([-0.5, -0.5].into()),
                Point::new([0.5, -0.5].into()),
                Point::new([-0.5, 0.5].into()),
            ],
            // L1
            vec![
                Point::new([-1.5, -0.5].into()),
                Point::new([-0.5, -0.5].into()),
                Point::new([0.5, -0.5].into()),
                Point::new([0.5, 0.5].into()),
            ],
            // L2
            vec![
                Point::new([-1.5, 0.5].into()),
                Point::new([-0.5, 0.5].into()),
                Point::new([0.5, 0.5].into()),
                Point::new([0.5, -0.5].into()),
            ],
        ];
        // ANCHOR_END: template_sites

        // ANCHOR: struct_initialize
        Ok(Tetronimoes {
            microstate,
            hamiltonian,
            sweep,
            macrostate,
            template_sites,
        })
    }
}
// ANCHOR_END: struct_initialize

// ANCHOR: impl_simulation
impl Simulation for Tetronimoes {
    // ANCHOR_END: impl_simulation
    // ANCHOR: advance
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        // ANCHOR_END: advance
        // ANCHOR: add
        if self.microstate.step().is_multiple_of(100) {
            let mut rng = self.microstate.counter().make_rng();
            let sites = self
                .template_sites
                .choose(&mut rng)
                .expect("template_sites should have at least 1 element")
                .clone();

            let properties = OrientedPoint {
                position: [
                    0.0,
                    self.microstate.boundary().0.edge_lengths[1].get() / 2.0
                        - 2.0,
                ]
                .into(),
                orientation: Angle::from(0.0),
            };

            self.microstate.add_body(Body { sites, properties })?;
            self.microstate.increment_substep();
        }
        // ANCHOR_END: add

        // ANCHOR: apply
        self.sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
        );
        self.microstate.increment_step();
        // ANCHOR_END: apply

        // ANCHOR: reset
        if self.hamiltonian.1.total_energy(&self.microstate) > 20_000.0 {
            self.microstate.clear();
        }

        Ok(())
    }
    // ANCHOR_END: reset

    // ANCHOR: step
    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}
// ANCHOR_END: step

// Remove the cfg(not(...)) line when using this code outside the hoomd-rs/examples directory.
#[cfg(not(feature = "bevy"))]
// ANCHOR: main
fn main() -> anyhow::Result<()> {
    let mut simulation = Tetronimoes::new()?;
    // TODO: Write GSD file.

    for _ in 0..20_000 {
        simulation.advance()?;
    }

    Ok(())
}
// ANCHOR_END: main
// ANCHOR_END: all

#[cfg(feature = "bevy")]
mod tetronimoes_interactive;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
#[cfg(feature = "bevy")]
use tetronimoes_interactive::main;

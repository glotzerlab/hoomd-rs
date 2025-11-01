// ANCHOR: all
// ANCHOR: use
use anyhow::{Context, anyhow};

use hoomd_geometry::shape::{Ellipse, Rectangle};
use hoomd_interaction::{
    CutoffPair, CutoffPairOverlap,
    pairwise::{
        Anisotropic, ApproximateShapeOverlap, HardShape, OverlapPenalty,
    },
};
use hoomd_mc::{QuickInsert, Rotate, Sweep, Translate, Trial, UniformIn};
use hoomd_microstate::{
    Microstate, SiteKey, boundary::Periodic, property::OrientedPoint,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::{self, Angle, Cartesian};
// ANCHOR_END: use

// ANCHOR: type_aliases
type PositionVector = Cartesian<2>;
type Orientation = Angle;
type BodyProperties = OrientedPoint<PositionVector, Orientation>;
type SiteProperties = OrientedPoint<PositionVector, Orientation>;
// ANCHOR_END: type_aliases

#[cfg_attr(feature = "bevy", derive(Resource))]
// ANCHOR: simulation_struct
struct HardEllipseSelfAssembly {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<
        BodyProperties,
        SiteProperties,
        VecCell<SiteKey, 2>,
        Periodic<Rectangle>,
    >,
    /// How sites interact with other sites and fields.
    hamiltonian: CutoffPairOverlap<HardShape<Ellipse>>,
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate<PositionVector>>,
    /// Trial moves to apply.
    rotate_sweep: Sweep<Rotate<Orientation>>,
    /// Temperature set point.
    macrostate: Isothermal,
    /// Quick insert
    quick_insert: QuickInsert<UniformIn<BodyProperties, Periodic<Rectangle>>>,
    /// How sites interact when inserted.
    insert_hamiltonian: CutoffPair<
        Anisotropic<ApproximateShapeOverlap<OverlapPenalty, Ellipse>>,
    >,
    /// The current phase of the simulation.
    phase: Phase,
}
// ANCHOR_END: simulation_struct

// ANCHOR: phase
enum Phase {
    Initialize,
    Equilibrate,
}
// ANCHOR_END: phase

// ANCHOR: simulation_new
impl HardEllipseSelfAssembly {
    /// Construct a new hard ellipsoid self-assembly simulation.
    fn new() -> anyhow::Result<HardEllipseSelfAssembly> {
        // ANCHOR_END: simulation_new
        // ANCHOR: parameters
        let box_height = 14.0;
        let n_bodies = 820;
        let maximum_distance = 0.05;
        let maximum_rotation = 0.1;
        let sigma = 1.0;
        let aspect = 5.0;
        let macrostate = Isothermal { temperature: 1.0 };
        assert!(aspect >= 1.0);
        // ANCHOR_END: parameters

        // ANCHOR: hamiltonian
        let ellipse = Ellipse {
            semi_axes: [
                (sigma / 2.0).try_into()?,
                (sigma / aspect / 2.0).try_into()?,
            ],
        };
        let hamiltonian = CutoffPairOverlap {
            r_cut: sigma,
            evaluator: HardShape(ellipse.clone()),
        };
        // ANCHOR_END: hamiltonian

        // ANCHOR: periodic
        let square = Rectangle::with_equal_edges(box_height.try_into()?);
        let periodic_square = Periodic::new(sigma, square)?;
        // ANCHOR_END: periodic

        // ANCHOR: microstate
        let cell_list = VecCell::builder()
            .nominal_search_radius(sigma.try_into()?)
            .build();
        let microstate = Microstate::builder()
            .spatial_data(cell_list)
            .boundary(periodic_square)
            .try_build()?;
        // ANCHOR_END: microstate

        // ANCHOR: trial_moves
        let translate =
            Translate::with_maximum_distance(maximum_distance.try_into()?);
        let translate_sweep = Sweep(translate);

        let rotate =
            Rotate::with_maximum_rotation(maximum_rotation.try_into()?);
        let rotate_sweep = Sweep(rotate);
        // ANCHOR_END: trial_moves

        // ANCHOR: quick_insert
        let distribution = UniformIn {
            boundary: microstate.boundary().clone(),
            template_sites: vec![SiteProperties::default()],
        };
        let quick_insert = QuickInsert::new(distribution, n_bodies);
        // ANCHOR_END: quick_insert

        // ANCHOR: insert_hamiltonian
        let approximate_shape_overlap =
            Anisotropic(ApproximateShapeOverlap::new(
                ellipse,
                OverlapPenalty::default(),
                0.01.try_into()?,
            ));

        let insert_hamiltonian = CutoffPair {
            r_cut: sigma,
            evaluator: approximate_shape_overlap,
        };
        // ANCHOR_END: insert_hamiltonian

        // ANCHOR: struct_initialize
        Ok(HardEllipseSelfAssembly {
            microstate,
            insert_hamiltonian,
            hamiltonian,
            translate_sweep,
            rotate_sweep,
            quick_insert,
            macrostate,
            phase: Phase::Initialize,
        })
    }
}
// ANCHOR_END: struct_initialize

// ANCHOR: impl_simulation
impl Simulation for HardEllipseSelfAssembly {
    // ANCHOR_END: impl_simulation
    // ANCHOR: advance
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        match self.phase {
            Phase::Initialize => {
                self.initialize().context("failed to initialize")?
            }
            Phase::Equilibrate => self.equilibrate(),
        }

        self.microstate.increment_step();

        Ok(())
    }
    // ANCHOR_END: advance

    // ANCHOR: step
    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}
// ANCHOR_END: step

// ANCHOR: inherent_simulation
impl HardEllipseSelfAssembly {
    // ANCHOR_END: inherent_simulation
    // ANCHOR: initialize
    fn initialize(&mut self) -> anyhow::Result<()> {
        // ANCHOR_END: initialize
        // ANCHOR: apply_quick_insert
        self.quick_insert
            .apply(&mut self.microstate, &self.insert_hamiltonian);
        // ANCHOR_END: apply_quick_insert

        // ANCHOR: initialize_trial_moves
        self.translate_sweep.apply(
            &mut self.microstate,
            &self.insert_hamiltonian,
            &Isothermal { temperature: 1.0 },
        );

        self.rotate_sweep.apply(
            &mut self.microstate,
            &self.insert_hamiltonian,
            &Isothermal { temperature: 1.0 },
        );
        // ANCHOR_END: initialize_trial_moves

        // ANCHOR: state_transition
        if self.quick_insert.is_complete() {
            self.phase = Phase::Equilibrate;
            println!(
                "Initialization complete at step {}.",
                self.microstate.step()
            );
        }
        // ANCHOR_END: state_transition

        // ANCHOR: failed
        if self.step() >= 10_000 {
            let n = self.microstate.bodies().len();
            let target = self.quick_insert.target();
            let step = self.microstate.step();
            return Err(anyhow!(
                "{n} of {target} bodies inserted after {step} steps"
            ));
        }

        Ok(())
    }
    // ANCHOR_END: failed

    // ANCHOR: equilibrate
    fn equilibrate(&mut self) {
        self.translate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
        );

        self.rotate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
        );
    }
}
// ANCHOR_END: equilibrate

// Remove the cfg(not(...)) line when using this code outside the hoomd-rs/examples directory.
#[cfg(not(feature = "bevy"))]
// ANCHOR: main
fn main() -> anyhow::Result<()> {
    let mut simulation = HardEllipseSelfAssembly::new()?;
    // TODO: Write GSD file.

    for _ in 0..20_000 {
        simulation.advance()?;
    }

    Ok(())
}
// ANCHOR_END: main
// ANCHOR_END: all

#[cfg(feature = "bevy")]
mod hard_particle_self_assembly_interactive;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
#[cfg(feature = "bevy")]
use hard_particle_self_assembly_interactive::main;

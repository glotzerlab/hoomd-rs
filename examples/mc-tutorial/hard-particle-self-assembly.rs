// ANCHOR: use
use hoomd_geometry::shape::{Cuboid, Ellipse};
use hoomd_interaction::{
    CutoffPair, CutoffPairOverlap,
    pairwise::{
        Anisotropic, ApproximateShapeOverlap, HardShape, OverlapPenalty,
    },
};
use hoomd_mc::{QuickInsert, Rotate, Sweep, Translate, Trial, UniformIn};
use hoomd_microstate::{
    Microstate, MicrostateBuilder, boundary::Periodic, property::OrientedPoint,
};
use hoomd_simulation::Simulation;
use hoomd_vector::{self, Angle, Cartesian};
// ANCHOR_END: use

// ANCHOR: type_aliases
type PositionVector = Cartesian<2>;
type BodyProperties = OrientedPoint<PositionVector, Angle>;
type SiteProperties = OrientedPoint<PositionVector, Angle>;
// ANCHOR_END: type_aliases

// ANCHOR: phase
enum Phase {
    Initialize,
    Equilibrate,
}
// ANCHOR_END: phase

#[cfg_attr(feature = "bevy", derive(Resource))]
// ANCHOR: simulation_struct
struct HardEllipseSelfAssembly {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<BodyProperties, SiteProperties, Periodic<Cuboid<2>>>,
    /// How sites interact when inserted.
    insert_hamiltonian: CutoffPair<
        Anisotropic<ApproximateShapeOverlap<OverlapPenalty, Ellipse>>,
    >,
    /// How sites interact with other sites and fields.
    hamiltonian: CutoffPairOverlap<HardShape<Ellipse>>,
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate>,
    /// Trial moves to apply.
    rotate_sweep: Sweep<Rotate>,
    /// Quick insert
    quick_insert: QuickInsert<UniformIn<BodyProperties, Periodic<Cuboid<2>>>>,
    /// Temperature set point.
    kt: f64,
    phase: Phase,
}
// ANCHOR_END: simulation_struct

// ANCHOR: simulation_new
impl HardEllipseSelfAssembly {
    /// Construct a new fill simulation.
    fn new() -> anyhow::Result<HardEllipseSelfAssembly> {
        // ANCHOR_END: simulation_new
        // ANCHOR: parameters
        let box_height = 14.0;
        let n_bodies = 820;
        let maximum_distance = 0.05;
        let maximum_rotation = 0.1;
        let sigma = 1.0;
        let aspect = 5.0;
        let kt = 1.0;
        assert!(aspect >= 1.0);
        // ANCHOR_END: parameters

        // ANCHOR: hamiltonian
        let ellipse = Ellipse {
            axes: [(sigma/2.0).try_into()?, (sigma / aspect / 2.0).try_into()?],
        };
        let hamiltonian = CutoffPairOverlap {
            r_cut: sigma,
            evaluator: HardShape(ellipse),
        };
        // ANCHOR_END: hamiltonian

        // ANCHOR: periodic
        let square = Cuboid::with_equal_edges(box_height.try_into()?);
        let periodic_square = Periodic::new(sigma, square)?;
        // ANCHOR_END: periodic

        // ANCHOR: microstate
        let microstate =
            MicrostateBuilder::with_boundary(periodic_square)
                .try_build()?;
        // ANCHOR_END: microstate

        // ANCHOR: trial_moves
        let translate = Translate {
            maximum_distance: maximum_distance.try_into()?,
        };
        let translate_sweep = Sweep(translate);

        let rotate = Rotate {
            maximum_rotation: maximum_rotation.try_into()?,
        };
        let rotate_sweep = Sweep(rotate);
        // ANCHOR_END: trial_moves

        // ANCHOR: quick_insert
        let distribution = UniformIn {
            boundary: *microstate.boundary(),
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
            kt,
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
            Phase::Initialize => self.initialize(),
            Phase::Equilibrate => self.equilibrate(),
            }
        
        self.microstate.increment_step();

        Ok(())
    }
    // ANCHOR_END: advance

    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}
// ANCHOR_END: impl_simulation

// ANCHOR: inherent_simulation
impl HardEllipseSelfAssembly {
// ANCHOR_END: inherent_simulation
    // ANCHOR: initialize
    fn initialize(&mut self) {
        self.quick_insert.apply(
            &mut self.microstate,
            &self.insert_hamiltonian,
            &self.translate_sweep,
            &1.0,
        );

        self.rotate_sweep.apply(
            &mut self.microstate,
            &self.insert_hamiltonian,
            &1.0,
        );

        if self.quick_insert.is_complete() {
            self.phase = Phase::Equilibrate;
            println!("Initialization complete at step {}.", self.microstate.step());
        }
    }
    // ANCHOR_END: initialize

    // ANCHOR: equilibrate
    fn equilibrate(&mut self) {
        self.translate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.kt,
        );

        self.rotate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &1.0,
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

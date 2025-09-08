// ANCHOR: use
use hoomd_geometry::{
    shape::{Cuboid, Ellipse},
};
use hoomd_interaction::{
    CutoffPair, CutoffPairOverlap,
    pairwise::{Anisotropic, ApproximateShapeOverlap, HardShape, OverlapPenalty},
};
use hoomd_mc::{QuickInsert, Rotate, Sweep, Translate, Trial, UniformIn};
use hoomd_microstate::{
    Microstate, MicrostateBuilder, boundary::Periodic, property::OrientedPoint,
};
use hoomd_vector::{self, Angle, Cartesian};
use hoomd_bevy::Simulation;
// ANCHOR_END: use

// ANCHOR: type_aliases
type PositionVector = Cartesian<2>;
type BodyProperties = OrientedPoint<PositionVector, Angle>;
type SiteProperties = OrientedPoint<PositionVector, Angle>;
// ANCHOR_END: type_aliases

// ANCHOR: phase
enum Phase {
    Initialization,
    Equilibration,
}
// ANCHOR_END: phase

// ANCHOR: simulation_new
impl HardEllipseSelfAssembly {
    /// Construct a new fill simulation.
    fn new() -> anyhow::Result<HardEllipseSelfAssembly> {
        // ANCHOR: parameters
        let box_height = 14.0;
        let kt = 1.0;
        let d = 0.05;
        let a = 0.1;
        let sigma = 1.0;
        // ANCHOR_END: parameters

        // ANCHOR: microstate
        let square = Cuboid::with_equal_edges(box_height.try_into()?);
        let microstate =
            MicrostateBuilder::with_boundary(Periodic::new(sigma, square)?)
                .try_build()?;
        // ANCHOR_END: microstate

        // ANCHOR: hamiltonian
        let ellipse = Ellipse {
            axes: [0.5.try_into()?, (0.5 / 5.0).try_into()?],
        };
        let hamiltonian = CutoffPairOverlap {
            r_cut: sigma,
            evaluator: HardShape(ellipse.clone()),
        };
        // ANCHOR_END: hamiltonian

        // ANCHOR: trial_moves
        let translate = Translate {
            maximum_distance: d.try_into()?,
        };
        let translate_sweep = Sweep(translate);

        let rotate = Rotate {
            maximum_rotation: a.try_into()?,
        };
        let rotate_sweep = Sweep(rotate);
        // ANCHOR_END: trial_moves

        // ANCHOR: quick_insert
        let distribution = UniformIn {
            boundary: *microstate.boundary(),
            template_sites: vec![SiteProperties::default()],
        };
        let quick_insert = QuickInsert::new(distribution, 820);
        // ANCHOR_END: quick_insert

        // ANCHOR: insert_hamiltonian
        let approximate_shape_overlap = Anisotropic(ApproximateShapeOverlap::new(
            ellipse,
            OverlapPenalty::default(),
            0.01.try_into()?));
            
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
            phase: Phase::Initialization,
        })
        // ANCHOR_END: struct_initialize
    }
}
// ANCHOR_END: simulation_new

// ANCHOR: impl_simulation
impl Simulation for HardEllipseSelfAssembly {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        let n = self.microstate.sites().len();

        match self.phase {
            Phase::Initialization => {
                self.quick_insert.apply(
                    &mut self.microstate,
                    &self.insert_hamiltonian,
                    &self.translate_sweep,
                    &1.0,
                );

                if self.quick_insert.is_complete() {
                    self.phase = Phase::Equilibration;
                    println!("{}: Complete", self.microstate.step());
                }
            }
            Phase::Equilibration => {
                self.translate_sweep.apply(
                    &mut self.microstate,
                    &self.hamiltonian,
                    &self.kt,
                );
            }
        }

        let n_new = self.microstate.sites().len();
        if n_new != n {
            println!("{}: {n_new}", self.microstate.step());
        }

        self.rotate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.kt,
        );

        self.microstate.increment_step();

        Ok(())
    }

    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}
// ANCHOR_END: impl_simulation


#[cfg_attr(feature = "bevy", derive(Resource))]
// ANCHOR: simulation_struct
struct HardEllipseSelfAssembly {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<
        BodyProperties,
        SiteProperties,
        Periodic<Cuboid<2>>,
    >,
    /// How sites interact when inserted.
    insert_hamiltonian: CutoffPair<Anisotropic<ApproximateShapeOverlap<OverlapPenalty, Ellipse>>>,
    /// How sites interact with other sites and fields.
    hamiltonian: CutoffPairOverlap<HardShape<Ellipse>>,
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate>,
    /// Trial moves to apply.
    rotate_sweep: Sweep<Rotate>,
    /// Quick insert
    quick_insert: QuickInsert<
        UniformIn<BodyProperties, Periodic<Cuboid<2>>>,
    >,
    /// Temperature set point.
    kt: f64,
    phase: Phase,
}
// ANCHOR_END: simulation_struct

#[cfg(feature = "bevy")]
mod hard_particle_self_assembly_interactive;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
#[cfg(feature = "bevy")]
use hard_particle_self_assembly_interactive::main;

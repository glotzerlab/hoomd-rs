// ANCHOR: all
// ANCHOR: use
use anyhow::{Context, anyhow};

use hoomd_geometry::{
    Convex, Volume,
    shape::{ConvexPolyhedron, Cuboid},
};
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff,
    pairwise::{Anisotropic, ApproximateShapeOverlap, HardShape},
    univariate::OverlapPenalty,
};
use hoomd_mc::{
    QuickCompress, QuickInsert, Rotate, Sweep, Translate, Trial, Tune,
    UniformIn,
};
use hoomd_microstate::{
    Microstate, SiteKey, boundary::Periodic, property::OrientedPoint,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::{self, Cartesian, Versor};
// ANCHOR_END: use

// ANCHOR: type_aliases
type PositionVector = Cartesian<3>;
type Orientation = Versor;
type BodyProperties = OrientedPoint<PositionVector, Orientation>;
type SiteProperties = OrientedPoint<PositionVector, Orientation>;
// ANCHOR_END: type_aliases

// ANCHOR: simulation_new
impl HardTetrahedronSelfAssembly {
    /// Construct a new hard tetrahedron self-assembly simulation.
    fn new() -> anyhow::Result<HardTetrahedronSelfAssembly> {
        // ANCHOR_END: simulation_new
        // ANCHOR: parameters
        let initial_packing_fraction = 0.3;
        let target_packing_fraction = 0.50;
        let n_bodies = 256;
        let maximum_distance = 0.04;
        let maximum_rotation = 0.04;
        let macrostate = Isothermal { temperature: 1.0 };
        // ANCHOR_END: parameters

        // ANCHOR: hamiltonian
        let a = 1.0_f64;
        let h = 6.0_f64.sqrt() / 3.0 * a;
        let tetrahedron_volume = 1.0 / 12.0 * 2.0_f64.sqrt() * a.powi(3);

        let tetrahedron = ConvexPolyhedron::with_vertices(vec![
            [3.0_f64.sqrt() / 3.0 * a, 0.0, -h / 4.0].into(),
            [-3.0_f64.sqrt() / 6.0 * a, 0.5 * a, -h / 4.0].into(),
            [-3.0_f64.sqrt() / 6.0 * a, -0.5 * a, -h / 4.0].into(),
            [0.0, 0.0, 3.0 * h / 4.0].into(),
        ])?;
        let hamiltonian =
            PairwiseCutoff(HardShape(Convex(tetrahedron.clone())));
        // ANCHOR_END: hamiltonian

        // ANCHOR: remainder
        let initial_box_volume =
            n_bodies as f64 * tetrahedron_volume / initial_packing_fraction;
        let initial_box_edge_length = initial_box_volume.cbrt();
        let cube =
            Cuboid::with_equal_edges(initial_box_edge_length.try_into()?);
        let periodic_cube =
            Periodic::new(hamiltonian.0.maximum_interaction_range(), cube)?;

        let vec_cell = VecCell::builder()
            .nominal_search_radius(
                hamiltonian.0.maximum_interaction_range().try_into()?,
            )
            .build();
        let microstate = Microstate::builder()
            .boundary(periodic_cube)
            .spatial_data(vec_cell)
            .try_build()?;

        let translate =
            Translate::with_maximum_distance(maximum_distance.try_into()?);
        let translate_sweep = Sweep(translate);

        let rotate =
            Rotate::with_maximum_rotation(maximum_rotation.try_into()?);
        let rotate_sweep = Sweep(rotate);

        let distribution = UniformIn {
            boundary: microstate.boundary().clone(),
            template_sites: vec![SiteProperties::default()],
        };
        let quick_insert = QuickInsert::new(distribution, n_bodies);

        let target_box_volume =
            n_bodies as f64 * tetrahedron_volume / target_packing_fraction;
        let quick_compress =
            QuickCompress::with_target_volume(target_box_volume.try_into()?);

        let approximate_shape_overlap = Anisotropic {
            interaction: ApproximateShapeOverlap::new(
                Convex(tetrahedron),
                OverlapPenalty::default(),
                0.01.try_into()?,
            ),
            r_cut: hamiltonian.0.maximum_interaction_range(),
        };

        let overlap_penalty_hamiltonian =
            PairwiseCutoff(approximate_shape_overlap);

        Ok(HardTetrahedronSelfAssembly {
            microstate,
            overlap_penalty_hamiltonian,
            hamiltonian,
            translate_sweep,
            rotate_sweep,
            quick_compress,
            quick_insert,
            macrostate,
            phase: Phase::Initialize,
        })
    }
}

#[cfg_attr(feature = "bevy", derive(Resource))]
struct HardTetrahedronSelfAssembly {
    /// Positions and orientations of all the bodies in the simulation.
    microstate: Microstate<
        BodyProperties,
        SiteProperties,
        VecCell<SiteKey, 3>,
        Periodic<Cuboid>,
    >,
    /// How sites interact with other sites and fields.
    hamiltonian: PairwiseCutoff<HardShape<Convex<ConvexPolyhedron>>>,
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate<PositionVector>>,
    /// Trial moves to apply.
    rotate_sweep: Sweep<Rotate<Orientation>>,
    /// Temperature set point.
    macrostate: Isothermal,
    /// Quick compress algorithm.
    quick_compress: QuickCompress<Periodic<Cuboid>>,
    /// Quick insert algorithm.
    quick_insert: QuickInsert<UniformIn<BodyProperties, Periodic<Cuboid>>>,
    /// How sites interact when inserted and compressed.
    overlap_penalty_hamiltonian: PairwiseCutoff<
        Anisotropic<
            ApproximateShapeOverlap<OverlapPenalty, Convex<ConvexPolyhedron>>,
        >,
    >,
    /// The current phase of the simulation.
    phase: Phase,
}

enum Phase {
    Initialize,
    Equilibrate,
}

impl Simulation for HardTetrahedronSelfAssembly {
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

    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl HardTetrahedronSelfAssembly {
    fn initialize(&mut self) -> anyhow::Result<()> {
        if self.quick_insert.is_complete() {
            self.quick_compress.apply(
                &mut self.microstate,
                &self.overlap_penalty_hamiltonian,
                |_| true,
            );
        } else {
            self.quick_insert
                .apply(&mut self.microstate, &self.overlap_penalty_hamiltonian);
        }

        self.translate_sweep.apply(
            &mut self.microstate,
            &self.overlap_penalty_hamiltonian,
            &Isothermal { temperature: 1.0 },
        );

        self.rotate_sweep.apply(
            &mut self.microstate,
            &self.overlap_penalty_hamiltonian,
            &Isothermal { temperature: 1.0 },
        );

        if self.quick_compress.is_complete() {
            self.translate_sweep.tune_default(
                &self.microstate,
                &self.hamiltonian,
                &self.macrostate,
            );
            self.rotate_sweep.tune_default(
                &self.microstate,
                &self.hamiltonian,
                &self.macrostate,
            );

            self.phase = Phase::Equilibrate;
            println!(
                "Initialization complete at step {}.",
                self.microstate.step()
            );
        }

        if self.step() >= 20_000 {
            let n = self.microstate.bodies().len();
            let target_n = self.quick_insert.target();
            let volume = self.microstate.boundary().volume();
            let target_volume = self.quick_compress.target_volume();
            return Err(anyhow!(
                "inserted {n}/{target_n} bodies and compressed to {volume} / {target_volume}"
            ));
        }

        Ok(())
    }

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
// ANCHOR_END: remainder

// Remove the cfg(not(...)) line when using this code outside the hoomd-rs/examples directory.
#[cfg(not(feature = "bevy"))]
// ANCHOR: main
fn main() -> anyhow::Result<()> {
    let mut simulation = HardTetrahedronSelfAssembly::new()?;
    // TODO: Write GSD file.

    for _ in 0..40_000 {
        simulation.advance()?;
    }

    Ok(())
}
// ANCHOR_END: main
// ANCHOR_END: all

#[cfg(feature = "bevy")]
mod hard_tetrahedron_self_assembly_interactive;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
#[cfg(feature = "bevy")]
use hard_tetrahedron_self_assembly_interactive::main;

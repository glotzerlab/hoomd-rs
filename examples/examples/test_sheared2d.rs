use anyhow::{Context, anyhow};

use hoomd_geometry::{
    Convex, Volume,
    shape::{ConvexPolygon, Rectangle, Rhomboid},
};
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff, TotalEnergy,
    pairwise::{Anisotropic, ApproximateShapeOverlap, HardShape},
    univariate::OverlapPenalty,
};
use hoomd_mc::{
    QuickCompress, QuickInsert, Rotate, Sweep, Translate, Trial, Tune,
    TuneOptions, UniformIn,
};
use hoomd_microstate::{
    Microstate, SiteKey, boundary::Periodic, property::OrientedPoint,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::{self, Angle, Cartesian};

type PositionVector = Cartesian<2>;
type Orientation = Angle;
type BodyProperties = OrientedPoint<PositionVector, Orientation>;
type SiteProperties = OrientedPoint<PositionVector, Orientation>;

struct HardRhomboidSelfAssembly {
    /// Positions and orientations of all the bodies in the simulation.
    microstate: Microstate<
        BodyProperties,
        SiteProperties,
        VecCell<SiteKey, 2>,
        Periodic<Rhomboid>,
    >,
    /// How sites interact with other sites and fields.
    hamiltonian: PairwiseCutoff<HardShape<Convex<Rhomboid>>>,
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate<PositionVector>>,
    /// Trial moves to apply.
    rotate_sweep: Sweep<Rotate<Orientation>>,
    /// Temperature set point.
    macrostate: Isothermal,
    /// Quick compress algorithm.
    quick_compress: QuickCompress<Periodic<Rhomboid>>,
    /// Quick insert algorithm.
    quick_insert: QuickInsert<UniformIn<SiteProperties, Periodic<Rhomboid>>>,
    /// How sites interact when inserted and compressed.
    overlap_penalty_hamiltonian: PairwiseCutoff<
        Anisotropic<ApproximateShapeOverlap<OverlapPenalty, Rhomboid>>,
    >,
    /// The current phase of the simulation.
    phase: Phase,
}

enum Phase {
    Initialize,
    Equilibrate,
}

impl HardRhomboidSelfAssembly {
    /// Construct a new hard rhomboid self-assembly simulation.
    fn new() -> anyhow::Result<HardRhomboidSelfAssembly> {
        let initial_packing_fraction = 0.01;
        let target_packing_fraction = 0.8;
        let n_bodies = 512;
        let maximum_distance = 0.07;
        let maximum_rotation = 0.3;
        let sigma = 1.5;
        let aspect = 5.0;
        let macrostate = Isothermal { temperature: 1.0 };
        assert!(aspect >= 1.0);

        let rhomboid = Rhomboid::from_box_vector([
            1.0.try_into()?,
            3.0_f64.sqrt() / 2.0,
            3.0_f64.sqrt() / 3.0,
        ]);
        let hamiltonian = PairwiseCutoff(HardShape(Convex(rhomboid.clone())));

        let initial_box_volume =
            n_bodies as f64 * rhomboid.volume() / initial_packing_fraction;
        let initial_box_edge_length = initial_box_volume.sqrt();
        let sheared_box = Rhomboid::from_box_vector([
            initial_box_edge_length.try_into()?,
            initial_box_edge_length * 3.0_f64.sqrt() / 2.0,
            3.0_f64.sqrt() / 3.0,
        ]);
        // let sheared_box =
        //     Rectangle::with_equal_edges(initial_box_edge_length.try_into()?);
        let periodic_sheared = Periodic::new(
            hamiltonian.maximum_interaction_range(),
            sheared_box,
        )?;

        let vec_cell = VecCell::builder()
            .nominal_search_radius(
                hamiltonian.maximum_interaction_range().try_into()?,
            )
            .build();
        let microstate = Microstate::builder()
            .boundary(periodic_sheared)
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
            n_bodies as f64 * rhomboid.volume() / target_packing_fraction;
        let quick_compress =
            QuickCompress::with_target_volume(target_box_volume.try_into()?);

        let approximate_shape_overlap = Anisotropic {
            interaction: ApproximateShapeOverlap::new(
                rhomboid,
                OverlapPenalty::default(),
                0.01.try_into()?,
            ),
            r_cut: hamiltonian.maximum_interaction_range().try_into()?,
        };

        let overlap_penalty_hamiltonian =
            PairwiseCutoff(approximate_shape_overlap);

        Ok(HardRhomboidSelfAssembly {
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

impl Simulation for HardRhomboidSelfAssembly {
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

impl HardRhomboidSelfAssembly {
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
            self.translate_sweep.tune_with_options(
                &self.microstate,
                &self.hamiltonian,
                &self.macrostate,
                &TuneOptions::default(),
            );
            self.rotate_sweep.tune_with_options(
                &self.microstate,
                &self.hamiltonian,
                &self.macrostate,
                &TuneOptions::default(),
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

    /// Get the total energy of the system.
    fn energy(&self) -> f64 {
        self.hamiltonian.total_energy(&self.microstate)
    }
}

// Remove the cfg(not(...)) line when using this code outside the hoomd-rs/examples directory.
#[cfg(not(feature = "bevy"))]
fn main() -> anyhow::Result<()> {
    use hoomd_gsd::hoomd::HoomdGsdFile;
    use hoomd_microstate::AppendMicrostate;

    let mut simulation = HardRhomboidSelfAssembly::new()?;
    let mut hoomd_gsd_file =
        HoomdGsdFile::create("hard-rhomboid-self-assembly.gsd")?;

    for _ in 0..100_000 {
        simulation.advance()?;

        if simulation.step().is_multiple_of(1000) {
            let energy = simulation.energy();
            println!("Step {}: energy = {}", simulation.step(), energy);
            hoomd_gsd_file
                .append_microstate(&simulation.microstate)?
                .end()?;
        }
    }
    println!("Simulation_Finished");
    Ok(())
}

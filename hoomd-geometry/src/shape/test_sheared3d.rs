use anyhow::{Context, anyhow};

use hoomd_geometry::{
    Convex, Volume,
    shape::{ConvexPolyhedron, Triclinic},
};
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff, TotalEnergy,
    pairwise::{Anisotropic, ApproximateShapeOverlap, HardShape},
    univariate::OverlapPenalty,
};
use hoomd_mc::{
    QuickCompress, QuickInsert, Rotate, Sweep, Translate, Trial, Tune, TuneOptions, UniformIn,
};
use hoomd_microstate::{Microstate, SiteKey, boundary::Periodic, property::OrientedPoint};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::{self, Cartesian, Versor};

type PositionVector = Cartesian<3>;
type Orientation = Versor;
type BodyProperties = OrientedPoint<PositionVector, Orientation>;
type SiteProperties = OrientedPoint<PositionVector, Orientation>;

#[cfg_attr(feature = "bevy", derive(Resource))]
struct HardTriclinicSelfAssembly {
    /// Positions and orientations of all the bodies in the simulation.
    microstate:
        Microstate<BodyProperties, SiteProperties, VecCell<SiteKey, 3>, Periodic<Triclinic>>,
    /// How sites interact with other sites and fields.
    hamiltonian: PairwiseCutoff<HardShape<Convex<ConvexPolyhedron>>>,
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate<PositionVector>>,
    /// Trial moves to apply.
    rotate_sweep: Sweep<Rotate<Orientation>>,
    /// Temperature set point.
    macrostate: Isothermal,
    /// Quick compress algorithm.
    quick_compress: QuickCompress<Periodic<Triclinic>>,
    /// Quick insert algorithm.
    quick_insert: QuickInsert<UniformIn<SiteProperties, Periodic<Triclinic>>>,
    /// How sites interact when inserted and compressed.
    overlap_penalty_hamiltonian: PairwiseCutoff<
        Anisotropic<ApproximateShapeOverlap<OverlapPenalty, Convex<ConvexPolyhedron>>>,
    >,
    /// The current phase of the simulation.
    phase: Phase,
}

enum Phase {
    Initialize,
    Equilibrate,
}

impl HardTriclinicSelfAssembly {
    /// Construct a new hard triclinic self-assembly simulation.
    ///
    /// The particle is a parallelepiped (triclinic polyhedron) whose shape
    /// mirrors the simulation box, with edge lengths [1, √3/2, √3/3] and a
    /// shared xy tilt factor. This is the 3D analogue of the 2D rhomboid
    /// simulation.
    fn new() -> anyhow::Result<HardTriclinicSelfAssembly> {
        let initial_packing_fraction = 0.01;
        let target_packing_fraction = 0.5;
        let n_bodies = 512;
        let maximum_distance = 0.07;
        let maximum_rotation = 0.3;
        let sigma = 1.0;
        let macrostate = Isothermal { temperature: 1.0 };

        // Build a triclinic particle whose box vector matches the 2D rhomboid:
        //   Lx=1,  Ly=√3/2,  Lz=√3/3,  xy=√3/3,  xz=0,  yz=0
        // This gives a 3D parallelepiped with the same shear in the xy-plane.
        let particle_box_vector = [
            1.0,
            3.0_f64.sqrt() / 2.0,
            3.0_f64.sqrt() / 3.0,
            3.0_f64.sqrt() / 3.0, // xy tilt (same as 2D rhomboid)
            0.0,                  // xz tilt
            0.0,                  // yz tilt
        ];
        let particle_shape = Triclinic::from_box_vector(particle_box_vector);

        // Derive the 8 vertices of the triclinic parallelepiped and build a
        // convex polyhedron for overlap detection.
        let edges = particle_shape.get_edge_vectors();
        let [a1, a2, a3] = edges;
        let half_a1: Cartesian<3> = (a1 * 0.5).into();
        let half_a2: Cartesian<3> = (a2 * 0.5).into();
        let half_a3: Cartesian<3> = (a3 * 0.5).into();

        // All 8 corners: ±½a₁ ± ½a₂ ± ½a₃
        let vertices: Vec<Cartesian<3>> = [
            [-1.0_f64, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        .iter()
        .map(|&[s1, s2, s3]| {
            Cartesian::from([
                s1 * half_a1[0] + s2 * half_a2[0] + s3 * half_a3[0],
                s1 * half_a1[1] + s2 * half_a2[1] + s3 * half_a3[1],
                s1 * half_a1[2] + s2 * half_a2[2] + s3 * half_a3[2],
            ])
        })
        .collect();

        let particle = ConvexPolyhedron::with_vertices(vertices)?;
        let hamiltonian = PairwiseCutoff(HardShape(Convex(particle.clone())));

        // Build the initial simulation box as a scaled triclinic box with the
        // same tilt factors as the particle, so the particles tile perfectly
        // under compression.
        let initial_box_volume =
            n_bodies as f64 * particle_shape.volume() / initial_packing_fraction;
        let initial_box_edge_length = initial_box_volume.cbrt();
        let scale = initial_box_edge_length / particle_box_vector[0]; // Lx=1
        let initial_box = Triclinic::from_box_vector([
            particle_box_vector[0] * scale,
            particle_box_vector[1] * scale,
            particle_box_vector[2] * scale,
            particle_box_vector[3], // preserve tilt factors
            particle_box_vector[4],
            particle_box_vector[5],
        ]);

        let periodic_triclinic =
            Periodic::new(hamiltonian.maximum_interaction_range(), initial_box)?;

        let vec_cell = VecCell::builder()
            .nominal_search_radius(hamiltonian.maximum_interaction_range().try_into()?)
            .build();
        let microstate = Microstate::builder()
            .boundary(periodic_triclinic)
            .spatial_data(vec_cell)
            .try_build()?;

        let translate = Translate::with_maximum_distance(maximum_distance.try_into()?);
        let translate_sweep = Sweep(translate);

        let rotate = Rotate::with_maximum_rotation(maximum_rotation.try_into()?);
        let rotate_sweep = Sweep(rotate);

        let distribution = UniformIn {
            boundary: microstate.boundary().clone(),
            template_sites: vec![SiteProperties::default()],
        };
        let quick_insert = QuickInsert::new(distribution, n_bodies);

        let target_box_volume = n_bodies as f64 * particle_shape.volume() / target_packing_fraction;
        let quick_compress = QuickCompress::with_target_volume(target_box_volume.try_into()?);

        let approximate_shape_overlap = Anisotropic {
            interaction: ApproximateShapeOverlap::new(
                Convex(particle),
                OverlapPenalty::default(),
                0.1.try_into()?,
            ),
            r_cut: sigma,
        };

        let overlap_penalty_hamiltonian = PairwiseCutoff(approximate_shape_overlap);

        Ok(HardTriclinicSelfAssembly {
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

impl Simulation for HardTriclinicSelfAssembly {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        match self.phase {
            Phase::Initialize => self.initialize().context("failed to initialize")?,
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

impl HardTriclinicSelfAssembly {
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
        self.translate_sweep
            .apply(&mut self.microstate, &self.hamiltonian, &self.macrostate);

        self.rotate_sweep
            .apply(&mut self.microstate, &self.hamiltonian, &self.macrostate);
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

    let mut simulation = HardTriclinicSelfAssembly::new()?;
    let mut hoomd_gsd_file = HoomdGsdFile::create("hard-triclinic-self-assembly.gsd")?;

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

use anyhow::{Context, anyhow};

use hoomd_geometry::{
    Convex, Volume,
    shape::{ConvexPolyhedron, Parallelepiped},
};
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff,
    pairwise::{Anisotropic, ApproximateShapeOverlap, HardShape},
    univariate::OverlapPenalty,
};
use hoomd_mc::{
    QuickCompress, QuickInsert, Rotate, Sweep, Translate, Trial, Tune,
    TuneOptions, UniformIn,
};
use hoomd_microstate::{
    Body, Microstate, SiteKey, boundary::Periodic, property::OrientedPoint,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::{self, Cartesian, Versor};

type PositionVector = Cartesian<3>;
type Orientation = Versor;
type BodyProperties = OrientedPoint<PositionVector, Orientation>;
type SiteProperties = OrientedPoint<PositionVector, Orientation>;

#[cfg_attr(feature = "bevy", derive(Resource))]
struct HardParallelepipedSelfAssembly {
    /// Positions and orientations of all the bodies in the simulation.
    microstate: Microstate<
        BodyProperties,
        SiteProperties,
        VecCell<SiteKey, 3>,
        Periodic<Parallelepiped>,
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
    quick_compress: QuickCompress<Periodic<Parallelepiped>>,
    /// Quick insert algorithm.
    quick_insert: QuickInsert<UniformIn<SiteProperties, Periodic<Parallelepiped>>>,
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

impl HardParallelepipedSelfAssembly {
    /// Construct a new hard parallelepiped self-assembly simulation.
    ///
    /// The particle is a parallelepiped (triclinic polyhedron) whose shape
    /// mirrors the simulation box, with edge lengths [1, √3/2, √3/3] and a
    /// shared xy tilt factor. This is the 3D analogue of the 2D rhomboid
    /// simulation.
    ///
    /// Unlike `test_sheared3d`, the simulation box is represented as a
    /// [`Parallelepiped`] (= `Hyperparallelepiped<3>`) defined by three
    /// explicit edge vectors rather than the HOOMD triclinic [lx, ly, lz, xy,
    /// xz, yz] parametrisation. This removes the `|xy/Ly| ≤ 0.5` tilt
    /// constraint and lets us build the supercell directly from the natural
    /// lattice vectors **a₁**, **a₂**, **a₃** of the particle.
    ///
    /// Particles are initialised on a 10×4×6 crystal lattice at packing
    /// fraction 0.5 (zero overlaps), then QuickCompress drives the system to
    /// the target packing fraction 0.6.
    fn new() -> anyhow::Result<HardParallelepipedSelfAssembly> {
        let target_packing_fraction = 0.6;
        let n_bodies = 240;
        let maximum_distance = 0.07;
        let maximum_rotation = 0.3;
        let macrostate = Isothermal { temperature: 1.0 };

        // Crystal grid dimensions: 10 × 4 × 6 = 240 particles.
        // n_x/n_y = 2.5 ensures d_x ≈ n_x/(n_y·xy_p) ≈ 4.3 >> 2·r_cut at target η.
        // n_z = 6 ensures d_z = n_z·lz_p·s ≈ 4.1 > 2·r_cut at target η=0.6.
        let n_x: usize = 10;
        let n_y: usize = 4;
        let n_z: usize = 6;

        // Particle lattice vectors in HOOMD upper-triangular form:
        //   a₁ = [lx_p, 0,    0   ]  (= [1,      0,       0     ])
        //   a₂ = [xy_p, ly_p, 0   ]  (= [√3/3,   √3/2,    0     ])
        //   a₃ = [0,    0,    lz_p]  (= [0,       0,       √3/3  ])
        let lx_p = 1.0_f64;
        let ly_p = 3.0_f64.sqrt() / 2.0;
        let lz_p = 3.0_f64.sqrt() / 3.0;
        let xy_p = 3.0_f64.sqrt() / 3.0;

        // Compute particle volume from the determinant of its edge-vector matrix:
        //   V_particle = lx_p · ly_p · lz_p  (a₁ × a₂ × a₃, already upper triangular)
        let particle_volume = lx_p * ly_p * lz_p;

        // 8 corners of the particle parallelepiped: ±½a₁ ± ½a₂ ± ½a₃
        //   ½a₁ = [0.5,       0,       0      ]
        //   ½a₂ = [√3/6,      √3/4,    0      ] ≈ [0.288675, 0.433013, 0]
        //   ½a₃ = [0,         0,       √3/6   ] ≈ [0,        0,        0.288675]
        let vertices: Vec<Cartesian<3>> = vec![
            Cartesian::from([-0.788675, -0.433013, -0.288675]),
            Cartesian::from([ 0.211325, -0.433013, -0.288675]),
            Cartesian::from([-0.211325,  0.433013, -0.288675]),
            Cartesian::from([ 0.788675,  0.433013, -0.288675]),
            Cartesian::from([-0.788675, -0.433013,  0.288675]),
            Cartesian::from([ 0.211325, -0.433013,  0.288675]),
            Cartesian::from([-0.211325,  0.433013,  0.288675]),
            Cartesian::from([ 0.788675,  0.433013,  0.288675]),
        ];

        let particle = ConvexPolyhedron::with_vertices(vertices)?;
        let hamiltonian = PairwiseCutoff(HardShape(Convex(particle.clone())));

        // Build the supercell from scaled lattice vectors directly — no
        // tilt-factor constraint applies to Parallelepiped.
        //   A₁ = n_x · s · a₁
        //   A₂ = n_y · s · a₂   (a₂ = [xy_p, ly_p, 0])
        //   A₃ = n_z · s · a₃
        let initial_packing_fraction = 0.50_f64;
        let s = initial_packing_fraction.recip().cbrt(); // 2^(1/3) ≈ 1.2599

        let mut initial_box = Parallelepiped::new([
            Cartesian::from([n_x as f64 * s * lx_p, 0.0, 0.0]),
            Cartesian::from([n_y as f64 * s * xy_p, n_y as f64 * s * ly_p, 0.0]),
            Cartesian::from([0.0, 0.0, n_z as f64 * s * lz_p]),
        ]);
        initial_box.calc_qr();

        let periodic_parallelepiped = Periodic::new(
            hamiltonian.maximum_interaction_range(),
            initial_box,
        )?;

        let vec_cell = VecCell::builder()
            .nominal_search_radius(
                hamiltonian.maximum_interaction_range().try_into()?,
            )
            .build();
        let mut microstate = Microstate::builder()
            .boundary(periodic_parallelepiped)
            .spatial_data(vec_cell)
            .try_build()?;

        // Place all particles on the crystal lattice.  Particle (ix, iy, iz)
        // sits at the centre of its scaled unit cell:
        //
        //   r = s · (fx·a₁ + fy·a₂ + fz·a₃)
        //     = [s·(fx·lx_p + fy·xy_p),  s·fy·ly_p,  s·fz·lz_p]
        //
        // Using the natural lattice vectors means particles tile the supercell
        // without any remapping — every fractional coordinate lies in (−0.5, 0.5].
        for iz in 0..n_z {
            for iy in 0..n_y {
                for ix in 0..n_x {
                    let fx = ix as f64 - (n_x as f64 - 1.0) / 2.0;
                    let fy = iy as f64 - (n_y as f64 - 1.0) / 2.0;
                    let fz = iz as f64 - (n_z as f64 - 1.0) / 2.0;

                    let rx = s * (fx * lx_p + fy * xy_p);
                    let ry = s * (fy * ly_p);
                    let rz = s * (fz * lz_p);

                    microstate.add_body(Body {
                        properties: OrientedPoint {
                            position: Cartesian::from([rx, ry, rz]),
                            orientation: Versor::default(),
                        },
                        sites: vec![OrientedPoint {
                            position: Cartesian::from([0.0_f64, 0.0, 0.0]),
                            orientation: Versor::default(),
                        }],
                    })?;
                }
            }
        }

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
        let quick_insert = QuickInsert::new(distribution, 0);

        let target_box_volume =
            n_bodies as f64 * particle_volume / target_packing_fraction;
        let mut quick_compress =
            QuickCompress::with_target_volume(target_box_volume.try_into()?);
        *quick_compress.maximum_delta_mut() = 0.01.try_into()?;
        *quick_compress.maximum_energy_per_site_mut() = 25.0;

        let approximate_shape_overlap = Anisotropic {
            interaction: ApproximateShapeOverlap::new(
                Convex(particle),
                OverlapPenalty::scaled_default(10.0),
                0.01.try_into()?,
            ),
            r_cut: hamiltonian.maximum_interaction_range().try_into()?,
        };

        let overlap_penalty_hamiltonian =
            PairwiseCutoff(approximate_shape_overlap);

        Ok(HardParallelepipedSelfAssembly {
            microstate,
            hamiltonian,
            overlap_penalty_hamiltonian,
            translate_sweep,
            rotate_sweep,
            quick_compress,
            quick_insert,
            macrostate,
            phase: Phase::Initialize,
        })
    }
}

impl Simulation for HardParallelepipedSelfAssembly {
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

    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl HardParallelepipedSelfAssembly {
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

#[cfg(not(feature = "bevy"))]
fn main() -> anyhow::Result<()> {
    use hoomd_gsd::hoomd::HoomdGsdFile;
    use hoomd_microstate::AppendMicrostate;

    let mut simulation = HardParallelepipedSelfAssembly::new()?;
    let mut hoomd_gsd_file =
        HoomdGsdFile::create("hard-parallelepiped-self-assembly.gsd")?;

    for _ in 0..20_000 {
        simulation.advance()?;

        if simulation.step().is_multiple_of(1000) {
            hoomd_gsd_file
                .append_microstate(&simulation.microstate)?
                .end()?;
        }
    }
    println!("Simulation_Finished");
    Ok(())
}

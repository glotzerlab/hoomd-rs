use anyhow::{Context, anyhow};

use hoomd_geometry::{
    Convex, Volume,
    shape::{ConvexPolyhedron, Triclinic},
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
struct HardTriclinicSelfAssembly {
    /// Positions and orientations of all the bodies in the simulation.
    microstate: Microstate<
        BodyProperties,
        SiteProperties,
        VecCell<SiteKey, 3>,
        Periodic<Triclinic>,
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
    quick_compress: QuickCompress<Periodic<Triclinic>>,
    /// Quick insert algorithm.
    quick_insert: QuickInsert<UniformIn<SiteProperties, Periodic<Triclinic>>>,
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

impl HardTriclinicSelfAssembly {
    /// Construct a new hard triclinic self-assembly simulation.
    ///
    /// The particle is a parallelepiped (triclinic polyhedron) whose shape
    /// mirrors the simulation box, with edge lengths [1, √3/2, √3/3] and a
    /// shared xy tilt factor. This is the 3D analogue of the 2D rhomboid
    /// simulation.
    /// Construct a new hard triclinic self-assembly simulation.
    ///
    /// Particles are initialized on a 5×5×8 crystal lattice at packing
    /// fraction 0.5 (well-separated, zero overlaps), then QuickCompress
    /// drives the system to the target packing fraction 0.6. Starting from
    /// a crystal avoids kinetic jamming that would otherwise stall random
    /// insertion at high packing fractions.
    fn new() -> anyhow::Result<HardTriclinicSelfAssembly> {
        let target_packing_fraction = 0.6;
        let n_bodies = 240;
        let maximum_distance = 0.07;
        let maximum_rotation = 0.3;
        let macrostate = Isothermal { temperature: 1.0 };

        // Crystal grid dimensions: 10 × 4 × 6 = 240 particles.
        // n_x/n_y = 2.5 ensures d_x ≈ n_x/(n_y·|bxy_p|) ≈ 5.92 >> 2·r_cut.
        // n_z = 6 ensures d_z at target η=0.6 is 6·lz_p·s/2 ≈ 2.06 > r_cut.
        let n_x: usize = 10;
        let n_y: usize = 4;
        let n_z: usize = 6;

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
        let [lx_p, ly_p, lz_p, xy_p, xz_p, yz_p] = particle_box_vector;
        let particle_shape = Triclinic::from_box_vector(particle_box_vector);

        // 8 corners of the parallelepiped: ±½a₁ ± ½a₂ ± ½a₃
        // With a₁=[1,0,0], a₂=[xy,Ly,0]=[√3/3,√3/2,0], a₃=[0,0,√3/3]:
        //   ½a₁ = [0.5,       0,       0      ]
        //   ½a₂ = [√3/6,      √3/4,    0      ] ≈ [0.288675, 0.433013, 0]
        //   ½a₃ = [0,         0,       √3/6   ] ≈ [0,        0,        0.288675]
        // x-coords are ±0.5 ± √3/6 ≈ {±0.788675, ±0.211325}.
        let vertices: Vec<Cartesian<3>> = vec![
            Cartesian::from([-0.788675, -0.433013, -0.288675]),
            Cartesian::from([0.211325, -0.433013, -0.288675]),
            Cartesian::from([-0.211325, 0.433013, -0.288675]),
            Cartesian::from([0.788675, 0.433013, -0.288675]),
            Cartesian::from([-0.788675, -0.433013, 0.288675]),
            Cartesian::from([0.211325, -0.433013, 0.288675]),
            Cartesian::from([-0.211325, 0.433013, 0.288675]),
            Cartesian::from([0.788675, 0.433013, 0.288675]),
        ];

        let particle = ConvexPolyhedron::with_vertices(vertices)?;
        let hamiltonian = PairwiseCutoff(HardShape(Convex(particle.clone())));

        // Build the initial simulation box as the supercell that exactly holds
        // n_x × n_y × n_z crystal cells at packing fraction 0.5.
        //
        // HOOMD requires |xy/Ly| ≤ 0.5 for the minimum-image convention, but
        // the particle's own tilt ratio is xy_p/Ly_p = (√3/3)/(√3/2) = 2/3 > 0.5.
        // Using a₂ directly as the y-box-vector would exceed this limit.
        //
        // Solution: use b₂ = a₂ − a₁ as the y-periodic vector instead.
        //   b₂ = [xy_p − Lx_p, Ly_p, 0]  →  |b₂.x / b₂.y| ≈ 0.488 < 0.5  ✓
        //
        // The 5 × 5 × 8 crystal supercell with {A₁ = n_x·s·a₁, A₂ = n_y·s·b₂, A₃}
        // contains exactly 200 distinct lattice sites (particle n1,n2 maps to
        // original site (n1−n2, n2)), covers the same volume, and satisfies the
        // tilt constraint.
        let initial_packing_fraction = 0.50_f64;
        let s = initial_packing_fraction.recip().cbrt(); // 2^(1/3) ≈ 1.2599
        // b₂.x = xy_p − Lx_p  (the x-component of a₂ − a₁)
        let bxy_p = xy_p - lx_p; // ≈ −0.4226
        let initial_box = Triclinic::from_box_vector([
            n_x as f64 * s * lx_p,
            n_y as f64 * s * ly_p,
            n_z as f64 * s * lz_p,
            n_y as f64 * s * bxy_p, // xy_box = n_y · s · b₂.x
            xz_p,
            yz_p,
        ]);

        let periodic_triclinic = Periodic::new(
            hamiltonian.maximum_interaction_range(),
            initial_box,
        )?;

        let vec_cell = VecCell::builder()
            .nominal_search_radius(
                hamiltonian.maximum_interaction_range().try_into()?,
            )
            .build();
        let mut microstate = Microstate::builder()
            .boundary(periodic_triclinic)
            .spatial_data(vec_cell)
            .try_build()?;

        // Place all particles on a perfect crystal lattice.  Each particle
        // (ix, iy, iz) sits at the centre of its scaled unit cell:
        //
        //   r = s · [(ix − (n_x−1)/2)·a₁ + (iy − (n_y−1)/2)·a₂ + (iz − (n_z−1)/2)·a₃]
        //
        // All fractional coordinates are in (−0.5, 0.5] so every body lies
        // within the initial box.  At packing fraction 0.5 the inter-particle
        // gap equals (2^(1/3) − 1) ≈ 0.26 particle-edge-lengths, so there are
        // no overlaps and total_energy = 0 from the start.
        for iz in 0..n_z {
            for iy in 0..n_y {
                for ix in 0..n_x {
                    let fx = ix as f64 - (n_x as f64 - 1.0) / 2.0;
                    let fy = iy as f64 - (n_y as f64 - 1.0) / 2.0;
                    let fz = iz as f64 - (n_z as f64 - 1.0) / 2.0;

                    // Cartesian position: r = s · (fx·a₁ + fy·b₂ + fz·a₃)
                    // where b₂ = a₂ − a₁, so b₂.x = xy_p − Lx_p = bxy_p.
                    let rx = s * (fx * lx_p + fy * bxy_p);
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

        // QuickInsert with target=0: all particles are already placed on the
        // crystal, so this completes trivially on the first apply() call.
        let distribution = UniformIn {
            boundary: microstate.boundary().clone(),
            template_sites: vec![SiteProperties::default()],
        };
        let quick_insert = QuickInsert::new(distribution, 0);

        let target_box_volume =
            n_bodies as f64 * particle_shape.volume() / target_packing_fraction;
        let mut quick_compress =
            QuickCompress::with_target_volume(target_box_volume.try_into()?);
        // Use conservative parameters: the library's hard-coded defaults
        // (max_delta=0.05, max_energy_per_site=1000) are too aggressive and
        // create dozens of simultaneous overlaps that are hard to resolve.
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

        Ok(HardTriclinicSelfAssembly {
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

impl Simulation for HardTriclinicSelfAssembly {
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

    // Get the total energy of the system.
    // fn energy(&self) -> f64 {
    //     self.hamiltonian.total_energy(&self.microstate)
    // }
}

// Remove the cfg(not(...)) line when using this code outside the hoomd-rs/examples directory.
#[cfg(not(feature = "bevy"))]
fn main() -> anyhow::Result<()> {
    use hoomd_gsd::hoomd::HoomdGsdFile;
    use hoomd_microstate::AppendMicrostate;

    let mut simulation = HardTriclinicSelfAssembly::new()?;
    let mut hoomd_gsd_file =
        HoomdGsdFile::create("hard-triclinic-self-assembly.gsd")?;

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

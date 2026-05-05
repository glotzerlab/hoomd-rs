// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.
//! TODO
use hoomd_chimes::{builder::ChimesBuilder, potential::ChimesTwobPotential};
use hoomd_geometry::shape::{Hypercuboid,};
use hoomd_interaction::{
    PairwiseCutoff, pairwise::Isotropic,
};
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_microstate::{
    Body, Microstate, SiteKey,
    boundary::Periodic,
    property::{Point},
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::Cartesian;

use anyhow::Context;
use bevy::prelude::*;

// ANCHOR_END: use

const NCOEFF: usize = 12;

// Remove the cfg_attr(...) line when using this code outside the hoomd-rs/examples directory.
#[derive(Resource)]
// ANCHOR: simulation_struct
struct Fill {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<
        Point<Cartesian<3>>,
        Point<Cartesian<3>>,
        VecCell<SiteKey, 3>,
        Periodic<Hypercuboid<3>>,
    >,
    /// How sites interact with other sites and fields.
    hamiltonian: PairwiseCutoff<Isotropic<ChimesTwobPotential<12>>>,
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate<Cartesian<3>>>,
    /// Temperature set point.
    macrostate: Isothermal,
}
// ANCHOR_END: simulation_struct

// ANCHOR: simulation_new
impl Fill {
    /// Construct a new fill simulation.
    fn new() -> anyhow::Result<Fill> {
        // ANCHOR_END: simulation_new
        // ANCHOR: parameters

        let kT = 9.94;
        let n: f64 = 8.0;
        let box_length = 4202.1_f64.cbrt();

        let macrostate = Isothermal { temperature: kT };
        let maximum_distance = 0.3;

        // ChIMES model
        let mut file_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        file_path.push("test-data");
        file_path.push("C-twobody.txt");
        let params = ChimesBuilder::<NCOEFF, 0>::parse(
            file_path.to_str().expect("Path contains invalid symbols"),
        )
            .expect("Failed to parse parameter file");
        let pairwise_cutoff = params
            .get_twob_chimes_potential(0)
            .expect("Error assemling ChIMES potential");

        // hamiltonian
        let hamiltonian = pairwise_cutoff;

        let cube = Hypercuboid::<3>::with_equal_edges(box_length.try_into()?);
        let boundary = Periodic::new(6.0, cube)?;
        // spatial_data (neighbor list)
        let vec_cell = VecCell::builder()
            .nominal_search_radius(params.pair_data[0].r_out.try_into()?) // get chimes outer cutoff
            .build();

        // microstate
        let mut microstate = Microstate::builder()
            .spatial_data(vec_cell)
            .boundary(boundary)
            .try_build()?;

        let space = box_length / n;
        assert!(
            space > 1.0,
            "Density too high to initialize on cubic lattice'!"
        );

        for i in 0..n as u32 {
            for j in 0..n as u32 {
                for k in 0..n as u32 {
                    let x = space * f64::from(i + 1) - ((1.0 + n) * space / 2.0);
                    let y = space * f64::from(j + 1) - ((1.0 + n) * space / 2.0);
                    let z = space * f64::from(k + 1) - ((1.0 + n) * space / 2.0);
                    microstate.add_body(Body {
                        properties: Point {
                            position: Cartesian::from([x, y, z]),
                        },
                        sites: vec![Point::default()],
                    })?;
                }
            }
        }

        // sweep
        let translate =
            Translate::<Cartesian<3>>::with_maximum_distance(maximum_distance.try_into()?);
        let translate_sweep = Sweep(translate);

        Ok(Fill {
            microstate,
            hamiltonian,
            translate_sweep,
            macrostate,
        })
    }
}
// ANCHOR_END: initialize_struct

// ANCHOR: impl_simulation
impl Simulation for Fill {
    // ANCHOR_END: impl_simulation
    // ANCHOR: advance
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        // ANCHOR_END: advance
        // ANCHOR: add
        // ANCHOR_END: add

        // ANCHOR: apply
        self.translate_sweep
            .apply(&mut self.microstate, &self.hamiltonian, &self.macrostate);
        self.microstate.increment_step();
        // ANCHOR_END: apply

        Ok(())
    }
    // ANCHOR_END: reset

    // ANCHOR: step
    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

fn main() -> anyhow::Result<()> {
    use hoomd_gsd::hoomd::HoomdGsdFile;
    use hoomd_microstate::AppendMicrostate;

    let mut simulation = Fill::new().context("failed to setup simulation")?;
    let mut hoomd_gsd_file = HoomdGsdFile::create("carbon-5000K.gsd")?;

    for _ in 0..1_000_000 {
        let _ = simulation.advance();

        if simulation.step().is_multiple_of(5_000) {
            println!("dump frame at step {:?}", simulation.step());
            hoomd_gsd_file
                .append_microstate(&simulation.microstate)?
                .end()?;
        }
    }

    Ok(())
}

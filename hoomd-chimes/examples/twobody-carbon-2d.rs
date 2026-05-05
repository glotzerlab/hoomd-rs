// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.
//! TODO
use hoomd_chimes::{builder::ChimesBuilder, potential::ChimesTwobPotential};
use hoomd_geometry::shape::Rectangle;
use hoomd_interaction::{
    External, PairwiseCutoff, TotalEnergy, external::Linear, pairwise::Isotropic,
};
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_microstate::{Body, Microstate, SiteKey, boundary::Closed, property::Point};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::Cartesian;

use anyhow::Context;
use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, InitialCamera, Settings,
    representation::{
        RectangularBoundary,
        disk::{self, Disk},
    },
};
// ANCHOR_END: use

struct A {}
const NCOEFF: usize = 12;

// Remove the cfg_attr(...) line when using this code outside the hoomd-rs/examples directory.
#[derive(Resource)]
// ANCHOR: simulation_struct
struct Fill {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<
        Point<Cartesian<2>>,
        Point<Cartesian<2>>,
        VecCell<SiteKey, 2>,
        Closed<Rectangle>,
    >,
    /// How sites interact with other sites and fields.
    hamiltonian: (
        External<Linear<Cartesian<2>>>,
        PairwiseCutoff<Isotropic<ChimesTwobPotential<12>>>,
    ),
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate<Cartesian<2>>>,
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
        let box_length = 60.0;
        let macrostate = Isothermal { temperature: 2.5 };
        let maximum_distance = 0.15;

        // ChIMES model
        let params = ChimesBuilder::<NCOEFF, 0>::parse(
            "/Users/alexlee/Documents/git_repo/hoomd-rs/hoomd-chimes/test-data/C-twobody.txt",
        )
        .expect("Failed to parse parameter file");
        let pairwise_cutoff = params
            .get_twob_chimes_potential(0)
            .expect("Error assemling ChIMES potential");

        // hamiltonian
        let linear = External(Linear {
            alpha: 10.0,
            plane_origin: Cartesian::default(),
            plane_normal: [0.0, 1.0].try_into()?,
        });
        let hamiltonian = (linear, pairwise_cutoff);

        // sweep
        let translate = Translate::with_maximum_distance(maximum_distance.try_into()?);
        let translate_sweep = Sweep(translate);

        // boundary
        let square = Rectangle::with_equal_edges(box_length.try_into()?);

        // spatial_data (neighbor list)
        let vec_cell = VecCell::builder()
            .nominal_search_radius(params.pair_data[0].r_out.try_into()?) // get chimes outer cutoff
            .build();

        // microstate
        let microstate = Microstate::builder()
            .spatial_data(vec_cell)
            .boundary(Closed(square))
            .try_build()?;

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
        let boundary = self.microstate.boundary();
        let y = boundary.0.edge_lengths[1].get() / 2.0 - 0.5;
        if self.microstate.step().is_multiple_of(100) {
            self.microstate.add_body(Body::point([0.0, y].into()))?;
        }
        // ANCHOR_END: add

        // ANCHOR: apply
        self.translate_sweep
            .apply(&mut self.microstate, &self.hamiltonian, &self.macrostate);
        self.microstate.increment_step();
        // ANCHOR_END: apply

        // ANCHOR: reset
        if self.hamiltonian.0.total_energy(&self.microstate) > 20_0000.0 {
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

fn main() -> anyhow::Result<()> {
    let mut simulation = Fill::new().context("failed to setup simulation")?;
    let l = simulation.microstate.boundary().0.edge_lengths[0].get() as f32;
    let hoomd_bevy_plugin = HoomdBevyPlugin {
        initial_settings: Settings {
            camera: InitialCamera::Orthographic2d(l + 1.0),
            ..default()
        },
        simulation,
    };

    let mut app = App::new();
    hoomd_bevy::add_default_plugins(&mut app);
    app.add_plugins(EguiPlugin::default());
    hoomd_bevy_plugin.build(&mut app);
    app.add_systems(
        Startup,
        (|| disk::MaterialParameters::default()).pipe(Disk::<A>::setup),
    );
    app.add_systems(
        Startup,
        (move || RectangularBoundary {
            width: l,
            height: l,
            ..default()
        })
        .pipe(RectangularBoundary::setup),
    );
    app.add_systems(
        Update,
        (
            // move_swimmer,
            sync_simulation
                .run_if(resource_changed::<Fill>)
                .after(AdvanceSet),
        )
            .chain(),
    );

    app.run();

    Ok(())
}

/// Copy the current positions of simulation particles to bevy entities.
fn sync_simulation(
    mut commands: Commands,
    disk_representation: Res<disk::Representation<A>>,
    query: Query<(Entity, &mut Transform), With<Disk<A>>>,
    simulation: Res<Fill>,
) {
    let sites = simulation.microstate.sites();
    Disk::sync(
        &mut commands,
        disk_representation,
        query,
        sites.iter().map(|site| {
            (
                Vec3::new(
                    site.properties.position[0] as f32,
                    site.properties.position[1] as f32,
                    0.0,
                ),
                1.0f32,
            )
        }),
    );
}

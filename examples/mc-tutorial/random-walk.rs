use hoomd_mc::{Sweep, Translate, Trial, Zero};
use hoomd_microstate::{Body, Microstate, MicrostateBuilder, property::Point};
use hoomd_vector::Cartesian;

use std::iter;

use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, Settings, Simulation,
    representation::{Disk, DiskAssets, DiskMaterial},
};

use anyhow::Context;
use bevy::prelude::*;

#[derive(Resource)]
// ANCHOR: simulation_struct
struct RandomWalk {
    microstate: Microstate<Point<Cartesian<2>>>,
    hamiltonian: Zero,
    translate_sweep: Sweep<Translate>,
    kt: f64,
}
// ANCHOR_END: simulation_struct

impl RandomWalk {
    /// Construct a new random walk simulation
    fn new() -> anyhow::Result<RandomWalk> {
        let kt = 1.0;
        let d = 0.15;
        let n = 100;

        let microstate = MicrostateBuilder::new()
        .bodies(iter::repeat_n(Body::point(Cartesian::default()), n))
        .try_build()?;

        let hamiltonian = Zero;

        let translate = Translate {
            maximum_distance: d.try_into()?,
        };
        let translate_sweep = Sweep(translate);

        Ok(RandomWalk {
            microstate,
            hamiltonian,
            translate_sweep,
            kt,
        })
    }
}

impl Simulation for RandomWalk {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        self.translate_sweep
            .apply(&mut self.microstate, &self.hamiltonian, &self.kt);
        self.microstate.increment_step();
        Ok(())
    }

    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

/// Mark the disk representation type.
struct A;

fn main() -> anyhow::Result<()> {
    let simulation = RandomWalk::new().context("failed to setup simulation")?;
    let hoomd_bevy_plugin = HoomdBevyPlugin {
        initial_settings: Settings {
            viewport_height: 30.0,
            ..default()
        },
        simulation,
    };

    let mut app = App::new();
    app.add_plugins(DefaultPlugins);
    hoomd_bevy_plugin.build(&mut app);
    app.add_systems(Startup, (|| DiskMaterial::default()).pipe(Disk::<A>::setup));
    app.add_systems(
        Update,
        sync_simulation
            .run_if(resource_changed::<RandomWalk>)
            .after(AdvanceSet),
    );

    app.run();

    Ok(())
}

/// Copy the current positions of simulation particles to bevy entities.
fn sync_simulation(
    mut commands: Commands,
    disk_assets: Res<DiskAssets<A>>,
    query: Query<(Entity, &mut Transform), With<Disk<A>>>,
    simulation: Res<RandomWalk>,
) {
    let sites = simulation.microstate.sites();
    Disk::sync(
        &mut commands,
        disk_assets,
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

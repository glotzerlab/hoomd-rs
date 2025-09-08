// ANCHOR: use
use std::iter;

use hoomd_interaction::Zero;
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_microstate::{Body, Microstate, MicrostateBuilder, property::Point};
use hoomd_vector::Cartesian;
// ANCHOR_END: use

use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, InitialCamera, Settings, Simulation,
    representation::disk::{self, Disk},
};

use anyhow::Context;
use bevy::prelude::*;

#[derive(Resource)]
// ANCHOR: simulation_struct
/// The simulation model.
struct RandomWalk {
    microstate: Microstate<Point<Cartesian<2>>>,
    hamiltonian: Zero,
    translate_sweep: Sweep<Translate>,
    kt: f64,
}
// ANCHOR_END: simulation_struct

// ANCHOR: simulation_new
// ANCHOR: new_fn
impl RandomWalk {
    /// Construct a new random walk simulation.
    fn new() -> anyhow::Result<RandomWalk> {
        // ANCHOR_END: new_fn
        // ANCHOR: params
        let kt = 1.0;
        let d = 0.15;
        let n = 100;
        // ANCHOR_END: params

        // ANCHOR: microstate
        let microstate = MicrostateBuilder::new()
            .bodies(iter::repeat_n(Body::point(Cartesian::default()), n))
            .try_build()?;
        // ANCHOR_END: microstate

        // ANCHOR: local_trial
        let translate = Translate {
            maximum_distance: d.try_into()?,
        };
        // ANCHOR_END: local_trial
        // ANCHOR: sweep
        let translate_sweep = Sweep(translate);
        // ANCHOR_END: sweep

        // ANCHOR: hamiltonian
        let hamiltonian = Zero;
        // ANCHOR_END: hamiltonian

        // ANCHOR: return
        Ok(RandomWalk {
            microstate,
            hamiltonian,
            translate_sweep,
            kt,
        })
        // ANCHOR_END: return
    }
}
// ANCHOR_END: simulation_new

// ANCHOR: impl_simulation
impl Simulation for RandomWalk {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        // ANCHOR: apply
        self.translate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.kt,
        );
        // ANCHOR_END: apply
        // ANCHOR: increment_step
        self.microstate.increment_step();
        // ANCHOR_END: increment_step
        Ok(())
    }

    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}
// ANCHOR_END: impl_simulation

/// Mark the disk representation type.
struct A;

fn main() -> anyhow::Result<()> {
    let simulation = RandomWalk::new().context("failed to setup simulation")?;
    let hoomd_bevy_plugin = HoomdBevyPlugin {
        initial_settings: Settings {
            camera: InitialCamera::Orthographic2d(110.0),
            ..default()
        },
        simulation,
    };

    let mut app = App::new();
    hoomd_bevy::add_default_plugins(&mut app);
    hoomd_bevy_plugin.build(&mut app);
    app.add_systems(
        Startup,
        (|| disk::MaterialParameters::default()).pipe(Disk::<A>::setup),
    );
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
    disk_representation: Res<disk::Representation<A>>,
    query: Query<(Entity, &mut Transform), With<Disk<A>>>,
    simulation: Res<RandomWalk>,
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

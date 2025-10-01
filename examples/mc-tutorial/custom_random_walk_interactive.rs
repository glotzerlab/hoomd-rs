use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, InitialCamera, Settings,
    representation::disk::{self, Disk},
};

use anyhow::Context;
use bevy::prelude::*;
use bevy_egui::EguiPlugin;

use super::CustomRandomWalk;

/// Mark the disk representation type.
struct A;

pub(crate) fn main() -> anyhow::Result<()> {
    let simulation =
        CustomRandomWalk::new().context("failed to setup simulation")?;

    let viewport_height =
        (simulation.microstate.boundary().0.radius * 2.0 + 10.0) as f32;
    let hoomd_bevy_plugin = HoomdBevyPlugin {
        initial_settings: Settings {
            camera: InitialCamera::Orthographic2d(viewport_height),
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
        Update,
        sync_simulation
            .run_if(resource_changed::<CustomRandomWalk>)
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
    simulation: Res<CustomRandomWalk>,
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

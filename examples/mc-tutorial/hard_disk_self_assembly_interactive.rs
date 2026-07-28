use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, InitialCamera, MUTED_COLOR, Settings, representation::disk,
};

use anyhow::Context;
use bevy::prelude::*;
use bevy_egui::EguiPlugin;

use super::HardDiskSelfAssembly;

/// Mark the ellipse representation type.
struct A;

/// Mark the ghost representation type.
struct Ghost;

pub(crate) fn main() -> anyhow::Result<()> {
    let simulation =
        HardDiskSelfAssembly::new().context("failed to setup simulation")?;
    let l =
        simulation.microstate.boundary().shape().extents[1].get() as f32;
    let hoomd_bevy_plugin = HoomdBevyPlugin {
        initial_settings: Settings {
            camera: InitialCamera::Orthographic2d(l + 2.0),
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
        (|| disk::MaterialParameters::default()).pipe(disk::Disk::<A>::setup),
    );
    app.add_systems(
        Startup,
        (|| disk::MaterialParameters {
            background_color: MUTED_COLOR.into(),
            ..default()
        })
        .pipe(disk::Disk::<Ghost>::setup),
    );
    app.add_systems(
        Update,
        (sync_sites, sync_ghosts)
            .run_if(resource_changed::<HardDiskSelfAssembly>)
            .after(AdvanceSet),
    );

    app.run();

    Ok(())
}

/// Copy the current positions of simulation sites to bevy entities.
fn sync_sites(
    mut commands: Commands,
    site_representation: Res<disk::Representation<A>>,
    site_query: Query<(Entity, &mut Transform), With<disk::Disk<A>>>,
    simulation: Res<HardDiskSelfAssembly>,
) {
    let sites = simulation.microstate.sites();
    disk::Disk::sync(
        &mut commands,
        site_representation,
        site_query,
        sites.iter().map(|site| {
            (
                Vec3::new(
                    site.properties.position[0] as f32,
                    site.properties.position[1] as f32,
                    0.0,
                ),
                1.0_f32,
            )
        }),
    );
}

/// Copy the current positions of simulation ghosts to bevy entities.
fn sync_ghosts(
    mut commands: Commands,
    ghost_representation: Res<disk::Representation<Ghost>>,
    ghost_query: Query<(Entity, &mut Transform), With<disk::Disk<Ghost>>>,
    simulation: Res<HardDiskSelfAssembly>,
) {
    let ghosts = simulation.microstate.ghosts();
    disk::Disk::sync(
        &mut commands,
        ghost_representation,
        ghost_query,
        ghosts.iter().map(|site| {
            (
                Vec3::new(
                    site.properties.position[0] as f32,
                    site.properties.position[1] as f32,
                    0.0,
                ),
                1.0_f32,
            )
        }),
    );
}

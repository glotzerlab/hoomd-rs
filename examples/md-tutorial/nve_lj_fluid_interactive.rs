use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, InitialCamera, PRIMARY_COLOR_3D, Settings,
    representation::surface_mesh,
};

use anyhow::Context;
use bevy::prelude::*;
use bevy_egui::EguiPlugin;

use super::LJFluid;

/// Mark the tetrahedron representation type.
struct A;

pub(crate) fn main() -> anyhow::Result<()> {
    let simulation = LJFluid::new()
        .context("failed to setup simulation")?;
    let l =
        simulation.microstate.boundary().shape().edge_lengths[1].get() as f32;
    let hoomd_bevy_plugin = HoomdBevyPlugin {
        initial_settings: Settings {
            camera: InitialCamera::Orthographic3d(l + 1.0),
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
        Update,
        (sync_sites,)
            .run_if(resource_changed::<LJFluid>)
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
    simulation: Res<LJFluid>,
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
                    site.properties.position[2] as f32,
                ),
                1.0_f32,
            )
        }),
    );
}
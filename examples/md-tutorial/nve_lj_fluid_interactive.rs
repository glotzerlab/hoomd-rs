use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, InitialCamera, PRIMARY_COLOR_3D, Settings,
    representation::disk,     representation::surface_mesh,
};

use anyhow::Context;
use bevy::prelude::*;
use bevy_egui::EguiPlugin;

use super::LJFluid;

/// Mark the disk representation type.
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

    let sphere_mesh = Sphere { radius: 0.5 };
    let sphere_material = StandardMaterial::from(PRIMARY_COLOR_3D);

    app.add_systems(
        Startup,
        (move || {
            (
                sphere_mesh.mesh().build(),
                sphere_material.clone(),
            )
        })
        .pipe(surface_mesh::SurfaceMesh::<A>::setup),
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
    site_representation: Res<surface_mesh::Representation<A>>,
    site_query: Query<
        (Entity, &mut Transform),
        With<surface_mesh::SurfaceMesh<A>>,
    >,
    simulation: Res<LJFluid>,
) {
    let sites = simulation.microstate.sites();
    surface_mesh::SurfaceMesh::sync(
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
                Quat::default(),
            )
        }),
    );
}
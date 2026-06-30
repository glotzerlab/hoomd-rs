use hoomd_bevy::{
    AdvanceSet, HIGHLIGHT_COLOR, HoomdBevyPlugin, InitialCamera, PRIMARY_COLOR_3D, Settings, representation::surface_mesh
};

use anyhow::Context;
use bevy::prelude::*;
use bevy_egui::EguiPlugin;

use super::{PatchyBody3D, SiteType};

/// Mark the type A sphere representation.
struct A;
/// Mark the type B sphere representation.
struct B;

pub(crate) fn main() -> anyhow::Result<()> {
    let simulation = PatchyBody3D::new()
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
        Startup,
        (move || {
            (
                sphere_mesh.mesh().build(),
                StandardMaterial::from(HIGHLIGHT_COLOR),
            )
        })
        .pipe(surface_mesh::SurfaceMesh::<B>::setup),
    );

    app.add_systems(
        Update,
        (sync_a_sites, sync_b_sites)
            .run_if(resource_changed::<PatchyBody3D>)
            .after(AdvanceSet),
    );

    app.run();

    Ok(())
}

/// Copy the current positions of simulation sites to bevy entities.
fn sync_a_sites(
    mut commands: Commands,
    site_representation: Res<surface_mesh::Representation<A>>,
    site_query: Query<
        (Entity, &mut Transform),
        With<surface_mesh::SurfaceMesh<A>>,
    >,
    simulation: Res<PatchyBody3D>,
) {
    let sites = simulation.microstate.sites();
    surface_mesh::SurfaceMesh::sync(
        &mut commands,
        site_representation,
        site_query,
        sites.iter()
        .filter(|s| s.properties.site_type == SiteType::A)
        .map(|site| {
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

/// Copy the current positions of simulation sites to bevy entities.
fn sync_b_sites(
    mut commands: Commands,
    site_representation: Res<surface_mesh::Representation<B>>,
    site_query: Query<
        (Entity, &mut Transform),
        With<surface_mesh::SurfaceMesh<B>>,
    >,
    simulation: Res<PatchyBody3D>,
) {
    let sites = simulation.microstate.sites();
    surface_mesh::SurfaceMesh::sync(
        &mut commands,
        site_representation,
        site_query,
        sites.iter()
        .filter(|s| s.properties.site_type == SiteType::B)
        .map(|site| {
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

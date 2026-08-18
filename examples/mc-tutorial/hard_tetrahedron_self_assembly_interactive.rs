use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, InitialCamera, PRIMARY_COLOR_3D, Settings,
    representation::surface_mesh,
};

use anyhow::Context;
use bevy::prelude::*;
use bevy_egui::EguiPlugin;

use super::HardTetrahedronSelfAssembly;

/// Mark the tetrahedron representation type.
struct A;

pub(crate) fn main() -> anyhow::Result<()> {
    let simulation = HardTetrahedronSelfAssembly::new()
        .context("failed to setup simulation")?;
    let vertices = simulation.hamiltonian.0.0.0.vertices().to_vec();

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

    let tetrahedron_mesh = Tetrahedron {
        vertices: [
            Vec3::new(
                vertices[0][0] as f32,
                vertices[0][1] as f32,
                vertices[0][2] as f32,
            ),
            Vec3::new(
                vertices[1][0] as f32,
                vertices[1][1] as f32,
                vertices[1][2] as f32,
            ),
            Vec3::new(
                vertices[2][0] as f32,
                vertices[2][1] as f32,
                vertices[2][2] as f32,
            ),
            Vec3::new(
                vertices[3][0] as f32,
                vertices[3][1] as f32,
                vertices[3][2] as f32,
            ),
        ],
    };
    let tetrahedron_material = StandardMaterial {
                    base_color: PRIMARY_COLOR_3D,
                    perceptual_roughness: 0.2,
                    ..default()
                };

    app.add_systems(
        Startup,
        (move || {
            (
                tetrahedron_mesh.mesh().build(),
                tetrahedron_material.clone(),
            )
        })
        .pipe(surface_mesh::SurfaceMesh::<A>::setup),
    );

    app.add_systems(
        Update,
        (sync_sites,)
            .run_if(resource_changed::<HardTetrahedronSelfAssembly>)
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
    simulation: Res<HardTetrahedronSelfAssembly>,
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
                Quat::from_xyzw(
                    site.properties.orientation.get().vector[0] as f32,
                    site.properties.orientation.get().vector[1] as f32,
                    site.properties.orientation.get().vector[2] as f32,
                    site.properties.orientation.get().scalar as f32,
                ),
            )
        }),
    );
}

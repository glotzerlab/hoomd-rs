use std::f64::consts::PI;

use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, InitialCamera, PRIMARY_COLOR_3D, Settings,
    representation::surface_mesh,
};

use anyhow::Context;
use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use hoomd_vector::{Rotation, Versor};

use super::Quasi2dCapsuleSelfAssembly;

/// Mark the tetrahedron representation type.
struct A;

pub(crate) fn main() -> anyhow::Result<()> {
    let simulation = Quasi2dCapsuleSelfAssembly::new()
        .context("failed to setup simulation")?;

    let l =
        simulation.microstate.boundary().0.shape().edge_lengths[1].get() as f32;
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

    let capsule = Capsule3d {
        radius: 1.0,
        half_length: 2.5,
    };
    let capsule_material = StandardMaterial {
        base_color: PRIMARY_COLOR_3D,
        perceptual_roughness: 0.2,
        ..default()
    };

    app.add_systems(
        Startup,
        (move || (capsule.mesh().build(), capsule_material.clone()))
            .pipe(surface_mesh::SurfaceMesh::<A>::setup),
    );

    app.add_systems(
        Update,
        (sync_sites,)
            .run_if(resource_changed::<Quasi2dCapsuleSelfAssembly>)
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
    simulation: Res<Quasi2dCapsuleSelfAssembly>,
) {
    let sites = simulation.microstate.sites();

    surface_mesh::SurfaceMesh::sync(
        &mut commands,
        site_representation,
        site_query,
        sites.iter().map(|site| {
            let orientation = site.properties.orientation;
            let rotation = Versor::from_axis_angle(
                [1.0, 0.0, 0.0]
                    .try_into()
                    .expect("hard-coded vector should be non-zero length"),
                PI / 2.0,
            );
            let orientation = orientation.combine(&rotation);

            (
                Vec3::new(
                    site.properties.position[0] as f32,
                    site.properties.position[1] as f32,
                    site.properties.position[2] as f32,
                ),
                Quat::from_xyzw(
                    orientation.get().vector[0] as f32,
                    orientation.get().vector[1] as f32,
                    orientation.get().vector[2] as f32,
                    orientation.get().scalar as f32,
                ),
            )
        }),
    );
}

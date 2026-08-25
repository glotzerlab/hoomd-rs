use hoomd_bevy::{
    AdvanceSet, HIGHLIGHT_COLOR, HoomdBevyPlugin, InitialCamera, MUTED_COLOR,
    PRIMARY_COLOR, Settings, representation::plane_mesh,
};

use anyhow::Context;
use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use hoomd_geometry::shape::ConvexSurfaceMesh2d;

use crate::SiteType;

use super::BinaryHardShapes;

/// Mark the square representation type.
struct A;

/// Mark the triangle representation type.
struct B;

/// Mark the ghost square representation type.
struct GhostA;

/// Mark the ghost triangle representation type.
struct GhostB;

pub(crate) fn main() -> anyhow::Result<()> {
    let simulation =
        BinaryHardShapes::new().context("failed to setup simulation")?;

    let shape_a = simulation.hamiltonian.0.shape_a.clone();
    let shape_b = simulation.hamiltonian.0.shape_b.clone();

    let l =
        simulation.microstate.boundary().shape().edge_lengths[1].get() as f32;
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

    let bevy_mesh_a = ConvexPolygon::new(
        shape_a
            .vertices()
            .iter()
            .map(|v| Vec2::new(v[0] as f32, v[1] as f32)),
    )?;
    let bevy_mesh_a2 = bevy_mesh_a.clone();
    let bevy_mesh_b = ConvexPolygon::new(
        shape_b
            .vertices()
            .iter()
            .map(|v| Vec2::new(v[0] as f32, v[1] as f32)),
    )?;
    let bevy_mesh_b2 = bevy_mesh_b.clone();

    app.add_systems(
        Startup,
        (move || {
            (
                bevy_mesh_a.mesh().build(),
                ColorMaterial::from(PRIMARY_COLOR),
            )
        })
        .pipe(plane_mesh::PlaneMesh::<A>::setup),
    );
    app.add_systems(
        Startup,
        (move || {
            (
                bevy_mesh_a2.mesh().build(),
                ColorMaterial::from(MUTED_COLOR),
            )
        })
        .pipe(plane_mesh::PlaneMesh::<GhostA>::setup),
    );
    app.add_systems(
        Startup,
        (move || {
            (
                bevy_mesh_b.mesh().build(),
                ColorMaterial::from(HIGHLIGHT_COLOR),
            )
        })
        .pipe(plane_mesh::PlaneMesh::<B>::setup),
    );
    app.add_systems(
        Startup,
        (move || {
            (
                bevy_mesh_b2.mesh().build(),
                ColorMaterial::from(MUTED_COLOR),
            )
        })
        .pipe(plane_mesh::PlaneMesh::<GhostB>::setup),
    );
    app.add_systems(
        Update,
        (sync_sites_a, sync_ghosts_a, sync_sites_b, sync_ghosts_b)
            .run_if(resource_changed::<BinaryHardShapes>)
            .after(AdvanceSet),
    );

    app.run();

    Ok(())
}

/// Copy the current positions of simulation sites (type A) to bevy entities.
fn sync_sites_a(
    mut commands: Commands,
    site_representation: Res<plane_mesh::Representation<A>>,
    site_query: Query<(Entity, &mut Transform), With<plane_mesh::PlaneMesh<A>>>,
    simulation: Res<BinaryHardShapes>,
) {
    let sites = simulation.microstate.sites();
    plane_mesh::PlaneMesh::sync(
        &mut commands,
        site_representation,
        site_query,
        sites
            .iter()
            .filter(|site| site.properties.site_type == SiteType::A)
            .map(|site| {
                (
                    Vec3::new(
                        site.properties.position[0] as f32,
                        site.properties.position[1] as f32,
                        0.0,
                    ),
                    site.properties.orientation.theta as f32,
                )
            }),
    );
}

/// Copy the current positions of simulation ghosts (type A) to bevy entities.
fn sync_ghosts_a(
    mut commands: Commands,
    ghost_representation: Res<plane_mesh::Representation<GhostA>>,
    ghost_query: Query<
        (Entity, &mut Transform),
        With<plane_mesh::PlaneMesh<GhostA>>,
    >,
    simulation: Res<BinaryHardShapes>,
) {
    let ghosts = simulation.microstate.ghosts();
    plane_mesh::PlaneMesh::sync(
        &mut commands,
        ghost_representation,
        ghost_query,
        ghosts
            .iter()
            .filter(|site| site.properties.site_type == SiteType::A)
            .map(|site| {
                (
                    Vec3::new(
                        site.properties.position[0] as f32,
                        site.properties.position[1] as f32,
                        0.0,
                    ),
                    site.properties.orientation.theta as f32,
                )
            }),
    );
}

/// Copy the current positions of simulation sites (type A) to bevy entities.
fn sync_sites_b(
    mut commands: Commands,
    site_representation: Res<plane_mesh::Representation<B>>,
    site_query: Query<(Entity, &mut Transform), With<plane_mesh::PlaneMesh<B>>>,
    simulation: Res<BinaryHardShapes>,
) {
    let sites = simulation.microstate.sites();
    plane_mesh::PlaneMesh::sync(
        &mut commands,
        site_representation,
        site_query,
        sites
            .iter()
            .filter(|site| site.properties.site_type == SiteType::B)
            .map(|site| {
                (
                    Vec3::new(
                        site.properties.position[0] as f32,
                        site.properties.position[1] as f32,
                        0.0,
                    ),
                    site.properties.orientation.theta as f32,
                )
            }),
    );
}

/// Copy the current positions of simulation ghosts (type B) to bevy entities.
fn sync_ghosts_b(
    mut commands: Commands,
    ghost_representation: Res<plane_mesh::Representation<GhostB>>,
    ghost_query: Query<
        (Entity, &mut Transform),
        With<plane_mesh::PlaneMesh<GhostB>>,
    >,
    simulation: Res<BinaryHardShapes>,
) {
    let ghosts = simulation.microstate.ghosts();
    plane_mesh::PlaneMesh::sync(
        &mut commands,
        ghost_representation,
        ghost_query,
        ghosts
            .iter()
            .filter(|site| site.properties.site_type == SiteType::B)
            .map(|site| {
                (
                    Vec3::new(
                        site.properties.position[0] as f32,
                        site.properties.position[1] as f32,
                        0.0,
                    ),
                    site.properties.orientation.theta as f32,
                )
            }),
    );
}

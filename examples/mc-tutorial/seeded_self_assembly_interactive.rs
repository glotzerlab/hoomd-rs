use std::f64::consts::PI;

use hoomd_bevy::{
    AdvanceSet, HIGHLIGHT_COLOR, HoomdBevyPlugin, InitialCamera, MUTED_COLOR,
    Settings,
    representation::{RectangularBoundary, disk, plane_mesh},
};

use anyhow::Context;
use bevy::prelude::*;
use bevy_egui::EguiPlugin;

use super::SeededSelfAssembly;

/// Mark the circle representation type.
struct A;

/// Mark the top patch representation type.
struct Top;

/// Mark the bottom patch representation type.
struct Bottom;

/// Mark the ghost representation type.
struct Ghost;

pub(crate) fn main() -> anyhow::Result<()> {
    let simulation = SeededSelfAssembly::new()
        .context("failed to setup simulation")?;

    let inner_radius =
        (simulation.hamiltonian.0.hard_disk.diameter / 2.0) as f32;
    let outer_radius = (simulation
        .hamiltonian
        .0
        .angular_mask
        .interaction
        .isotropic
        .right
        / 2.0) as f32;
    let angle = (simulation.hamiltonian.0.angular_mask.interaction.masks_i[0]
        .cos_delta
        .acos()
        * 2.0) as f32;

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

    let ring_mesh = Ring::new(
        CircularSector::new(outer_radius, angle),
        CircularSector::new(inner_radius, angle),
    );
    let ring_material_top = ColorMaterial::from(HIGHLIGHT_COLOR);
    let ring_material_bottom = ColorMaterial::from(HIGHLIGHT_COLOR);

    app.add_systems(
        Startup,
        (move || (ring_mesh.clone().mesh().build(), ring_material_top.clone()))
            .pipe(plane_mesh::PlaneMesh::<Top>::setup),
    );
    app.add_systems(
        Startup,
        (move || (ring_mesh.mesh().build(), ring_material_bottom.clone()))
            .pipe(plane_mesh::PlaneMesh::<Bottom>::setup),
    );

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
        Startup,
        (move || RectangularBoundary {
            width: l,
            height: l,
            ..default()
        })
        .pipe(RectangularBoundary::setup),
    );
    app.add_systems(
        Update,
        (
            sync_sites,
            sync_rings_top,
            sync_rings_bottom,
            sync_ghosts,
            sync_boundary,
        )
            .run_if(resource_changed::<SeededSelfAssembly>)
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
    simulation: Res<SeededSelfAssembly>,
) {
    let sites = simulation.microstate.sites();
    disk::Disk::sync(
        &mut commands,
        site_representation,
        site_query,
        sites.iter().filter(|site| site.body_tag >= simulation.seed_size).map(|site| {
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

/// Place rings to highlight the interaction regions (top).
fn sync_rings_top(
    mut commands: Commands,
    site_representation: Res<plane_mesh::Representation<Top>>,
    site_query: Query<
        (Entity, &mut Transform),
        With<plane_mesh::PlaneMesh<Top>>,
    >,
    simulation: Res<SeededSelfAssembly>,
) {
    let sites = simulation.microstate.sites();
    let ghosts = simulation.microstate.ghosts();
    plane_mesh::PlaneMesh::sync(
        &mut commands,
        site_representation,
        site_query,
        sites.iter().chain(ghosts).map(|site| {
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

/// Place rings to highlight the interaction regions (bottom).
fn sync_rings_bottom(
    mut commands: Commands,
    site_representation: Res<plane_mesh::Representation<Bottom>>,
    site_query: Query<
        (Entity, &mut Transform),
        With<plane_mesh::PlaneMesh<Bottom>>,
    >,
    simulation: Res<SeededSelfAssembly>,
) {
    let sites = simulation.microstate.sites();
    let ghosts = simulation.microstate.ghosts();
    plane_mesh::PlaneMesh::sync(
        &mut commands,
        site_representation,
        site_query,
        sites.iter().chain(ghosts).map(|site| {
            (
                Vec3::new(
                    site.properties.position[0] as f32,
                    site.properties.position[1] as f32,
                    0.0,
                ),
                (site.properties.orientation.theta + PI) as f32,
            )
        }),
    );
}

/// Copy the current positions of simulation ghosts to bevy entities.
fn sync_ghosts(
    mut commands: Commands,
    ghost_representation: Res<disk::Representation<Ghost>>,
    ghost_query: Query<(Entity, &mut Transform), With<disk::Disk<Ghost>>>,
    simulation: Res<SeededSelfAssembly>,
) {
    let sites = simulation.microstate.sites();
    let ghosts = simulation.microstate.ghosts();
    disk::Disk::sync(
        &mut commands,
        ghost_representation,
        ghost_query,
        ghosts.iter().chain(sites.iter().filter(|site| site.body_tag < simulation.seed_size)).map(|site| {
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

/// Draw the simulation boundary at its current size.
fn sync_boundary(
    entity_rectangle: Single<(Entity, &RectangularBoundary)>,
    children: Query<&Children>,
    transforms: Query<&mut Transform>,
    simulation: Res<SeededSelfAssembly>,
) {
    let l =
        simulation.microstate.boundary().shape().edge_lengths[1].get() as f32;
    RectangularBoundary::sync(entity_rectangle, children, transforms, l, l);
}

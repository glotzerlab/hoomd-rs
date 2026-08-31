use hoomd_bevy::{
    AdvanceSet, HIGHLIGHT_COLOR, HoomdBevyPlugin, InitialCamera, MUTED_COLOR,
    PRIMARY_COLOR, Settings,
    representation::{disk, plane_mesh},
};

use anyhow::Context;
use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use hoomd_geometry::shape::ConvexSurfaceMesh2d;

use crate::SiteType;

use super::MultiSitePatchyShape;

/// Mark the hexagon representation type.
struct A;
/// Mark the patch representation type.
struct P;

/// Mark the ghost hexagon representation type.
struct GhostA;
/// Mark the ghost patch representation type.
struct GhostP;

pub(crate) fn main() -> anyhow::Result<()> {
    let simulation =
        MultiSitePatchyShape::new().context("failed to setup simulation")?;
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

    let regular_hexagon = hoomd_geometry::shape::ConvexPolygon::regular(6);
    let hoomd_mesh = ConvexSurfaceMesh2d::try_from(regular_hexagon)?;
    let bevy_mesh = ConvexPolygon::new(
        hoomd_mesh
            .vertices()
            .iter()
            .map(|v| Vec2::new(v[0] as f32, v[1] as f32)),
    )?;
    let bevy_mesh2 = bevy_mesh.clone();

    app.add_systems(
        Startup,
        (move || {
            (bevy_mesh.mesh().build(), ColorMaterial::from(PRIMARY_COLOR))
        })
        .pipe(plane_mesh::PlaneMesh::<A>::setup),
    );
    app.add_systems(
        Startup,
        (move || (bevy_mesh2.mesh().build(), ColorMaterial::from(MUTED_COLOR)))
            .pipe(plane_mesh::PlaneMesh::<GhostA>::setup),
    );
    app.add_systems(
        Startup,
        (|| disk::MaterialParameters {
            background_color: HIGHLIGHT_COLOR.into(),
            ..default()
        })
        .pipe(disk::Disk::<P>::setup),
    );
    app.add_systems(
        Startup,
        (|| disk::MaterialParameters {
            background_color: MUTED_COLOR.into(),
            ..default()
        })
        .pipe(disk::Disk::<GhostP>::setup),
    );
    app.add_systems(
        Update,
        (sync_sites, sync_ghosts, sync_patch_sites, sync_patch_ghosts)
            .run_if(resource_changed::<MultiSitePatchyShape>)
            .after(AdvanceSet),
    );

    app.run();

    Ok(())
}

/// Copy the current positions of simulation sites to bevy entities.
fn sync_sites(
    mut commands: Commands,
    site_representation: Res<plane_mesh::Representation<A>>,
    site_query: Query<(Entity, &mut Transform), With<plane_mesh::PlaneMesh<A>>>,
    simulation: Res<MultiSitePatchyShape>,
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

/// Copy the current positions of simulation ghosts to bevy entities.
fn sync_ghosts(
    mut commands: Commands,
    ghost_representation: Res<plane_mesh::Representation<GhostA>>,
    ghost_query: Query<
        (Entity, &mut Transform),
        With<plane_mesh::PlaneMesh<GhostA>>,
    >,
    simulation: Res<MultiSitePatchyShape>,
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

/// Copy the current positions of simulation sites to bevy entities.
fn sync_patch_sites(
    mut commands: Commands,
    site_representation: Res<disk::Representation<P>>,
    site_query: Query<(Entity, &mut Transform), With<disk::Disk<P>>>,
    simulation: Res<MultiSitePatchyShape>,
) {
    let sites = simulation.microstate.sites();
    disk::Disk::sync(
        &mut commands,
        site_representation,
        site_query,
        sites
            .iter()
            .filter(|site| site.properties.site_type == SiteType::P)
            .map(|site| {
                (
                    Vec3::new(
                        site.properties.position[0] as f32,
                        site.properties.position[1] as f32,
                        -0.1,
                    ),
                    0.3_f32,
                )
            }),
    );
}

/// Copy the current positions of simulation ghosts to bevy entities.
fn sync_patch_ghosts(
    mut commands: Commands,
    ghost_representation: Res<disk::Representation<GhostP>>,
    ghost_query: Query<(Entity, &mut Transform), With<disk::Disk<GhostP>>>,
    simulation: Res<MultiSitePatchyShape>,
) {
    let ghosts = simulation.microstate.ghosts();
    disk::Disk::sync(
        &mut commands,
        ghost_representation,
        ghost_query,
        ghosts
            .iter()
            .filter(|site| site.properties.site_type == SiteType::P)
            .map(|site| {
                (
                    Vec3::new(
                        site.properties.position[0] as f32,
                        site.properties.position[1] as f32,
                        -0.1,
                    ),
                    0.3_f32,
                )
            }),
    );
}

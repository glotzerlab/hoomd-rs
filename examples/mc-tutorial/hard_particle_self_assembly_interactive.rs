use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, InitialCamera, MUTED_COLOR, Settings,
    representation::RectangularBoundary, representation::ellipse,
};

use anyhow::Context;
use bevy::prelude::*;
use bevy_egui::EguiPlugin;

use super::HardEllipseSelfAssembly;

/// Mark the ellipse representation type.
struct A;

/// Mark the ghost representation type.
struct Ghost;

pub(crate) fn main() -> anyhow::Result<()> {
    let simulation =
        HardEllipseSelfAssembly::new().context("failed to setup simulation")?;
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
    app.add_systems(
        Startup,
        (|| ellipse::MaterialParameters {
            outline_width: 0.025,
            ..default()
        })
        .pipe(ellipse::Ellipse::<A>::setup),
    );
    app.add_systems(
        Startup,
        (|| ellipse::MaterialParameters {
            outline_width: 0.025,
            background_color: MUTED_COLOR.into(),
            ..default()
        })
        .pipe(ellipse::Ellipse::<Ghost>::setup),
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
        (sync_sites, sync_ghosts)
            .run_if(resource_changed::<HardEllipseSelfAssembly>)
            .after(AdvanceSet),
    );

    app.run();

    Ok(())
}

/// Copy the current positions of simulation sites to bevy entities.
fn sync_sites(
    mut commands: Commands,
    site_representation: Res<ellipse::Representation<A>>,
    site_query: Query<(Entity, &mut Transform), With<ellipse::Ellipse<A>>>,
    simulation: Res<HardEllipseSelfAssembly>,
) {
    let sites = simulation.microstate.sites();
    ellipse::Ellipse::sync(
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
                site.properties.orientation.theta as f32,
                (simulation.hamiltonian.0.0.semi_axes()[0].get() * 2.0) as f32,
                (simulation.hamiltonian.0.0.semi_axes()[1].get() * 2.0) as f32,
            )
        }),
    );
}

/// Copy the current positions of simulation ghosts to bevy entities.
fn sync_ghosts(
    mut commands: Commands,
    ghost_representation: Res<ellipse::Representation<Ghost>>,
    ghost_query: Query<(Entity, &mut Transform), With<ellipse::Ellipse<Ghost>>>,
    simulation: Res<HardEllipseSelfAssembly>,
) {
    let ghosts = simulation.microstate.ghosts();
    ellipse::Ellipse::sync(
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
                site.properties.orientation.theta as f32,
                (simulation.hamiltonian.0.0.semi_axes()[0].get() * 2.0) as f32,
                (simulation.hamiltonian.0.0.semi_axes()[1].get() * 2.0) as f32,
            )
        }),
    );
}

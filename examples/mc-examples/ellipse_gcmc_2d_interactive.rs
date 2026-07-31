use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, InitialCamera, MUTED_COLOR, ParametersWindowState, Settings, representation::{RectangularBoundary, ellipse}
};

use anyhow::Context;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

use super::HardEllipseGCMC;

/// Mark the ellipse representation type.
struct A;

/// Mark the ghost representation type.
struct Ghost;

pub(crate) fn main() -> anyhow::Result<()> {
    let simulation =
        HardEllipseGCMC::new().context("failed to setup simulation")?;
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
        (sync_sites, sync_ghosts, sync_boundary)
            .run_if(resource_changed::<HardEllipseGCMC>)
            .after(AdvanceSet),
    );
    app.add_systems(EguiPrimaryContextPass, ui_system);

    app.run();

    Ok(())
}

fn ui_system(
    mut simulation: ResMut<HardEllipseGCMC>,
    mut contexts: EguiContexts,
    mut parameters_window_state: ResMut<ParametersWindowState>,
) -> Result {
    let window = egui::Window::new("")
        .id(egui::Id::new("Parameters"))
        .auto_sized()
        .open(&mut parameters_window_state.0)
        .collapsible(false);

    window.show(contexts.ctx_mut()?, |ui| {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.add(
                    egui::Slider::new(
                        &mut simulation.macrostate.fugacity,
                        0.0..=100.0,
                    )
                    .text("fugacity")
                    .update_while_editing(false),
                );
            });
            ui.label(format!("N: {}", simulation.microstate.bodies().len()));
         });
    });

    Ok(())
}

/// Copy the current positions of simulation sites to bevy entities.
fn sync_sites(
    mut commands: Commands,
    site_representation: Res<ellipse::Representation<A>>,
    site_query: Query<(Entity, &mut Transform), With<ellipse::Ellipse<A>>>,
    simulation: Res<HardEllipseGCMC>,
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
    simulation: Res<HardEllipseGCMC>,
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

/// Draw the simulation boundary at its current size.
fn sync_boundary(
    entity_rectangle: Single<(Entity, &RectangularBoundary)>,
    children: Query<&Children>,
    transforms: Query<&mut Transform>,
    simulation: Res<HardEllipseGCMC>,
) {
    let l =
        simulation.microstate.boundary().shape().edge_lengths[1].get() as f32;
    RectangularBoundary::sync(entity_rectangle, children, transforms, l, l);
}

use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, InitialCamera, ParametersWindowState,
    Settings,
    representation::RectangularBoundary,
    representation::disk::{self, Disk},
};

use anyhow::Context;
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

use super::Fill;

/// Mark the disk representation type.
struct A;

pub(crate) fn main() -> anyhow::Result<()> {
    let simulation = Fill::new().context("failed to setup simulation")?;
    let l = simulation.microstate.boundary().0.edge_lengths[1].get() as f32;
    let hoomd_bevy_plugin = HoomdBevyPlugin {
        initial_settings: Settings {
            camera: InitialCamera::Orthographic2d(l + 1.0),
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
        (|| disk::MaterialParameters::default()).pipe(Disk::<A>::setup),
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
        sync_simulation
            .run_if(resource_changed::<Fill>)
            .after(AdvanceSet),
    );
    app.add_systems(EguiPrimaryContextPass, ui_system);

    app.run();

    Ok(())
}

fn ui_system(
    mut simulation: ResMut<Fill>,
    mut contexts: EguiContexts,
    mut parameters_window_state: ResMut<ParametersWindowState>,
) -> Result {
    let window = egui::Window::new("")
        .id(egui::Id::new("Parameters"))
        .resizable([false, false])
        .open(&mut parameters_window_state.0)
        .collapsible(false);

    window.show(contexts.ctx_mut()?, |ui| {
        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(
                    &mut simulation.hamiltonian.0.0.alpha,
                    0.0..=20.0,
                )
                .text("alpha")
                .vertical()
                .update_while_editing(false),
            );
        });
    });

    Ok(())
}

/// Copy the current positions of simulation particles to bevy entities.
fn sync_simulation(
    mut commands: Commands,
    disk_representation: Res<disk::Representation<A>>,
    query: Query<(Entity, &mut Transform), With<Disk<A>>>,
    simulation: Res<Fill>,
) {
    let sites = simulation.microstate.sites();
    Disk::sync(
        &mut commands,
        disk_representation,
        query,
        sites.iter().map(|site| {
            (
                Vec3::new(
                    site.properties.position[0] as f32,
                    site.properties.position[1] as f32,
                    0.0,
                ),
                1.0f32,
            )
        }),
    );
}

use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, InitialCamera, ParametersWindowState,
    Settings,
    representation::RectangularBoundary,
    representation::disk::{self, Disk},
};

use anyhow::Context;
use bevy::prelude::*;
use bevy::render::storage::ShaderStorageBuffer;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use std::iter;

use super::Tetronimoes;

/// Mark the disk representation type.
struct A;

pub(crate) fn main() -> anyhow::Result<()> {
    let simulation =
        Tetronimoes::new().context("failed to setup simulation")?;
    let l = simulation.microstate.boundary().0.edge_lengths[1].get() as f32;
    let hoomd_bevy_plugin = HoomdBevyPlugin {
        initial_settings: Settings {
            sps_limit: 64.0,
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
        (
            (|| disk::MaterialParameters::default()).pipe(Disk::<A>::setup),
            setup_colors,
        )
            .chain(),
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
            .run_if(resource_changed::<Tetronimoes>)
            .after(AdvanceSet),
    );
    app.add_systems(EguiPrimaryContextPass, ui_system);

    app.run();

    Ok(())
}

fn ui_system(
    mut simulation: ResMut<Tetronimoes>,
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
                    &mut simulation.hamiltonian.linear.0.alpha,
                    0.0..=2.0,
                )
                .text("alpha")
                .vertical()
                .update_while_editing(false),
            );
        });
    });

    Ok(())
}

/// Set the tetronimo colors.
fn setup_colors(
    disk_representation: ResMut<disk::Representation<A>>,
    mut materials: ResMut<Assets<disk::Material>>,
    buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    let material = materials
        .get_mut(disk_representation.material())
        .expect("Disk::setup should have added the material");

    let color_wheel = (0..360 * 4)
        .step_by(39)
        .map(|i| Color::oklch(0.75, 0.1246, (i % 360) as f32));
    let linear_color_wheel = color_wheel.map(LinearRgba::from);
    let duplicate = linear_color_wheel.flat_map(|v| iter::repeat_n(v, 4));
    let duplicate: Vec<_> = duplicate.collect();
    material.set_background_colors(buffers, &duplicate);
}

/// Copy the current positions of simulation particles to bevy entities.
fn sync_simulation(
    mut commands: Commands,
    disk_representation: Res<disk::Representation<A>>,
    query: Query<(Entity, &mut Transform), With<Disk<A>>>,
    simulation: Res<Tetronimoes>,
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
                1.0_f32,
            )
        }),
    );
}

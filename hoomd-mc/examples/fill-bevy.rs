#![allow(clippy::print_stdout, reason = "Demonstration purposes")]

/*! This is an example
*/

use hoomd_interaction::{
    CutoffPair, Single,
    external::Linear,
    pairwise::{Boxcar, Isotropic},
};
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_microstate::{Body, Microstate, MicrostateBuilder, boundary::Square, property::Point};
use hoomd_vector::Cartesian;

use anyhow::Context;
use bevy::{prelude::*, window::PresentMode};
use bevy_diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin, Diagnostic, Diagnostics, DiagnosticPath, RegisterDiagnostic};
use std::time::{Duration, Instant};

const FRAME_BUDGET: Duration = Duration::from_millis(30);
const SPS_LIMIT: f32 = 100.0;
const SPS: DiagnosticPath = DiagnosticPath::const_new("sps");

fn main() -> anyhow::Result<()> {
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "fill".into(),
                    present_mode: PresentMode::AutoVsync,
                    ..default()
                }),
                ..default()
            }),
            FrameTimeDiagnosticsPlugin::default(),
    ))
    .insert_resource(ClearColor(Color::oklch(0.3, 0.0, 0.0)))
    .register_diagnostic(Diagnostic::new(SPS))
    .insert_resource(setup_simulation().context("failed to setup simulation")?)
    .add_systems(Startup, (setup_scene, setup_debug_text))
    .add_systems(Update, (step_simulation_system, sync_simulation).chain())
    .add_systems(Update, (keyboard_input, update_debug_text).chain())
    ;
    
    app.run();

    Ok(())
}

/// The HOOMD simulation
#[derive(Resource)]
struct Simulation {
    /// The simulation box vertical extent (in simulation units).
    box_height: f64,
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Square>,
    /// How sites interact with other sites and fields.
    hamiltonian: (CutoffPair<Isotropic<Boxcar>>, Single<Linear<Cartesian<2>>>),
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate>,
    /// Temperature set point.
    kt: f64
}

/// Set up the hoomd simulation
fn setup_simulation() -> anyhow::Result<Simulation> {
    let box_height = 10.0;
    let kt = 1.0;
    let d = 0.15;

    let microstate = MicrostateBuilder::with_boundary(Square {
        l: box_height.try_into()?,
    })
    .try_build()?;

    let boxcar = Boxcar {
        epsilon: 1000.0,
        left: 0.0,
        right: 1.0,
    };
    let evaluator = Isotropic(boxcar);
    let cutoff_pair = CutoffPair {
        r_cut: 1.0,
        evaluator,
    };

    let linear = Single(Linear {
        alpha: 10.0,
        plane_origin: Cartesian::default(),
        plane_normal: [0.0, 1.0].try_into()?,
    });

    let hamiltonian = (cutoff_pair, linear);

    let translate = Translate {
        maximum_distance: d.try_into()?,
    };
    let translate_sweep = Sweep { local: translate };

    Ok(Simulation { box_height, microstate, hamiltonian, translate_sweep, kt })
}

/// Advance the simulation forward one step.
fn step_simulation(simulation: &mut Simulation) -> anyhow::Result<()> {

    if simulation.microstate.step() % 100 == 0 {
        simulation.microstate.add_body(Body::point([0.0, simulation.box_height / 2.0 - 0.5].into()))?;
    }

    simulation.translate_sweep.apply(&mut simulation.microstate, &simulation.hamiltonian, &simulation.kt);
    simulation.microstate.increment_step();
    Ok(())
    }

/// Bevy system that advances the simulation forward one step.
fn step_simulation_system(
    mut diagnostics: Diagnostics,
    mut exit: EventWriter<AppExit>,
    simulation: ResMut<Simulation>,
    time: Res<Time>,
    mut accumulated_steps: Local<f32>) {

    // Determine the maximum number of steps that we can take in this update.
    // Accumulate fractional steps over time and remove whole steps from the
    // accumulated amount. This allows for steps per second limits that are
    // less than the monitor's refresh rate.
    let max_steps = SPS_LIMIT * time.delta_secs();
    *accumulated_steps += max_steps.fract();

    let mut max_steps = max_steps.floor() as u64;
    if *accumulated_steps > 1.0 {
        max_steps += accumulated_steps.trunc() as u64;
        *accumulated_steps = accumulated_steps.fract();
    }
    
    let simulation = simulation.into_inner();
    let step_time = Instant::now();
    let mut steps = 0;
    while step_time.elapsed() < FRAME_BUDGET && steps < max_steps{
        let result = step_simulation(simulation).with_context(|| format!("failed at step: {}", simulation.microstate.step()));
        if let Err(error) = result {
            error!("{error:?}");
            exit.write(AppExit::Error(1.try_into().expect("1 is non-zero")));
            break;
            }
        steps += 1;
        }

    diagnostics.add_measurement(&SPS, || steps as f64 / time.delta_secs_f64());
    }
    
/// Assets that represent a Disk in the scene.
#[derive(Resource)]
struct Disk {
    /// The disk's mesh.
    mesh: Handle<Mesh>,
    /// The disk's color.
    color: Handle<ColorMaterial>,
}

/// Mark entities as sites.
#[derive(Component)]
struct Site;

/// Set up the bevy scene.
fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    simulation: Res<Simulation>,
) {
    let projection = Projection::Orthographic(OrthographicProjection {
       scaling_mode: bevy::render::camera::ScalingMode::FixedVertical { viewport_height: simulation.box_height as f32 },
       ..OrthographicProjection::default_2d()
    });

    commands.spawn((Camera2d,
                    projection)
                );

    let mesh = meshes.add(Circle::new(0.5));
    let color = materials.add(Color::oklch(0.64, 0.14, 256.71));
    commands.insert_resource(Disk { mesh, color });
}

/// Mark debug text
#[derive(Component)]
struct DebugText;

/// Add debug text nodes.
fn setup_debug_text(mut commands: Commands) {
    commands.spawn((
            Text::default(),
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(12.0),
                left: Val::Px(12.0),
                ..default()
            },
            Visibility::Hidden,
            DebugText,
        children![
            TextSpan::new("FPS:\n"),
            TextSpan::new("SPS:\n"),
            TextSpan::new("Step:\n"),
            ],
        ));
}

/// Populate values in the debug text.
fn update_debug_text(
    diagnostic: Res<DiagnosticsStore>,
    debug_text: bevy::ecs::prelude::Single<Entity, With<DebugText>>,
    mut writer: TextUiWriter,
    time: Res<Time>,
    mut time_since_rerender: Local<Duration>,
    simulation: Res<Simulation>,
) {
    *time_since_rerender += time.delta();

    if *time_since_rerender >= Duration::from_millis(100) {
        *time_since_rerender = Duration::ZERO;

        let debug_text = *debug_text;
        if let Some(fps) = diagnostic.get(&FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(value) = fps.smoothed() {
                *writer.text(debug_text, 1) = format!(" FPS: {value:.2}\n");
            }
        }
        if let Some(sps) = diagnostic.get(&SPS) {
            if let Some(value) = sps.smoothed() {
                *writer.text(debug_text, 2) = format!(" SPS: {value:.2}\n");
            }
        }
        *writer.text(debug_text, 3) = format!("Step: {}\n", simulation.microstate.step());
    }
}

/// Copy the current positions of simulation particles to bevy entities.
fn sync_simulation(
    mut commands: Commands,
    disk: Res<Disk>,
    simulation: Res<Simulation>,
    mut query: Query<&mut Transform, With<Site>>) {

    let sites = simulation.microstate.sites();
    let mut n_entities = 0;
    
    for (site_index, mut transform) in &mut query.into_iter().enumerate() {
        let position = sites[site_index].properties.position;
        transform.translation = Vec3 { x: position[0] as f32, y: position[1] as f32, z: 0.0 };
        n_entities += 1;
    }

    for site in &sites[n_entities..] {
    commands.spawn((
        Mesh2d(disk.mesh.clone()),
        MeshMaterial2d(disk.color.clone()),
        Transform::from_xyz(
            site.properties.position[0] as f32,
            site.properties.position[1] as f32,
            0.0,
        ),
        Site,
    ));    
    }
}

/// Implement keyboard commands for common operations.
fn keyboard_input(
    keys: Res<ButtonInput<KeyCode>>,
    mut exit: EventWriter<AppExit>,
    mut debug_text: bevy::ecs::prelude::Single<&mut Visibility, With<DebugText>>,
) {
    if keys.just_pressed(KeyCode::Space) {
    }
    if keys.just_pressed(KeyCode::KeyQ) {
        exit.write(AppExit::Success);
    }
    if keys.just_pressed(KeyCode::F5) {
        debug_text.toggle_visible_hidden();
    }
}

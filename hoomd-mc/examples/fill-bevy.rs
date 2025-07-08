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
use bevy::{prelude::*, window::PresentMode, time::common_conditions::once_after_delay, render::view::window::screenshot::{save_to_disk, Screenshot}};
use bevy_diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin, Diagnostic, Diagnostics, DiagnosticPath, RegisterDiagnostic};
use std::time::{Duration, Instant};

// TODO: derive frame budget from refresh rate and value from 0 to 1.
const FRAME_BUDGET: Duration = Duration::from_millis(30);
const SPS_LIMIT: f32 = 100.0;
const SPS: DiagnosticPath = DiagnosticPath::const_new("sps");
const HELP_OVERLAY_ZINDEX: i32 = i32::MAX - 32;

// TODO: const background color
// TODO: const margin

#[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
enum PauseState {
    #[default]
    Paused,
    Running,
}


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
    // Goes in plugin
    .insert_resource(ClearColor(Color::oklch(0.3, 0.0, 0.0)))
    .register_diagnostic(Diagnostic::new(SPS))
    .insert_resource(setup_simulation().context("failed to setup simulation")?)
    .insert_state(PauseState::Running)
    .add_systems(Startup, setup_scene)
    .add_systems(Startup, (setup_overlay, setup_debug_text, add_pause_text, add_help_text, add_help_reminder).chain())
    .add_systems(Update, remove_help_reminder.run_if(once_after_delay(Duration::from_secs(3))))
    // TODO: expose step_simulation as a named set so that callers can use it in an after schedule
    .add_systems(Update, step_simulation_system.run_if(in_state(PauseState::Running)))
    .add_systems(Update, (keyboard_overlay, update_debug_text).chain())
    .add_systems(Update, (keyboard_pause, keyboard_help, keyboard_simulation, keyboard_screenshot, keyboard_quit))
    // Goes in the example code (sync_simulation is highly simulation-specific)
    // TODO: Implement helper methods to make sync_simulation easier to write.
    .add_systems(Update, sync_simulation.run_if(resource_changed::<Simulation>).after(step_simulation_system))
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

/// The overlay UI root node.
#[derive(Component)]
struct OverlayRoot;

fn setup_overlay(mut commands: Commands) {
commands.spawn((
            Node {
                top: Val::Px(0.0),
                left: Val::Px(0.0),
                width: Val::Vw(100.0),
                height: Val::Vh(100.0),
                ..default()
            },
            Visibility::Visible,
            OverlayRoot,
            ));
}

/// Mark debug text
#[derive(Component)]
struct DebugText;

/// Add debug text nodes.
fn setup_debug_text(mut commands: Commands, overlay_root: bevy::ecs::prelude::Single<Entity, With<OverlayRoot>>,) {
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
        ChildOf(*overlay_root),
        ));
}

/// Mark paused text
#[derive(Component)]
struct PauseText;

/// Add paused text node.
fn add_pause_text(mut commands: Commands,
    overlay_root: bevy::ecs::prelude::Single<Entity, With<OverlayRoot>>,
) {
    commands.spawn((
            Text::new("paused..."),
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(12.0),
                left: Val::Px(12.0),
                ..default()
            },
            PauseText,
            Visibility::Hidden,
            ChildOf(*overlay_root),
        ));
}

/// Mark the border containing the help text.
#[derive(Component)]
struct HelpTextContainer;

/// Mark the help text itself.
#[derive(Component)]
struct HelpText;

// TODO: Make HelpText public so that examples can add lines to it.
/// Add helpd text node.
fn add_help_text(mut commands: Commands,
    overlay_root: bevy::ecs::prelude::Single<Entity, With<OverlayRoot>>,
) {
    let text = (Text::new("q       : Quit.
<space> : Pause the simulation.
<right> : Advance one step (while paused).
shift-F1: Show/hide the user interface.
F5      : Show/hide debugging information.
F12     : Take a screenshot (screenshot.png).
?       : Show/hide this help text."),
            BackgroundColor(Color::oklch(0.2, 0.0, 0.0)),
            HelpText,
            );

    commands.spawn((
            Node {
                align_items: AlignItems::Center,
                position_type: PositionType::Absolute,
                bottom: Val::Px(12.0),
                right: Val::Px(12.0),
                margin: UiRect::all(Val::Px(0.0)),
                border: UiRect::all(Val::Px(12.0)),
                justify_content: JustifyContent::Center,                
                ..default()
            },
            HelpTextContainer,
            ChildOf(*overlay_root),
            Outline {
                width: Val::Px(6.),
                offset: Val::Px(0.),
                color: Color::WHITE,
            },
            BorderRadius::px(12.0, 12.0, 12.0, 12.0),
            BackgroundColor(Color::oklch(0.2, 0.0, 0.0)),
            BorderColor(Color::oklch(0.2, 0.0, 0.0)),
            Visibility::Hidden,
            GlobalZIndex(HELP_OVERLAY_ZINDEX),
            children![text],
        ));
}

/// Mark paused text
#[derive(Component)]
struct HelpReminder;

/// Add help reminder node.
fn add_help_reminder(mut commands: Commands,
    overlay_root: bevy::ecs::prelude::Single<Entity, With<OverlayRoot>>,
) {
    commands.spawn((
            Text::new("Press ? to show the help screen."),
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(12.0),
                right: Val::Px(12.0),
                ..default()
            },
            HelpReminder,
            ChildOf(*overlay_root),
            GlobalZIndex(HELP_OVERLAY_ZINDEX-1),
        ));
}

/// Remove paused text node.
fn remove_help_reminder(mut commands: Commands,
    overlay_root: bevy::ecs::prelude::Single<Entity, (With<OverlayRoot>, Without<HelpReminder>)>,
    help_reminder: bevy::ecs::prelude::Single<Entity, (With<HelpReminder>, Without<OverlayRoot>)>,
) {
    commands.entity(*overlay_root).remove_children(&[*help_reminder]);
    commands.entity(*help_reminder).despawn();
}

/// Populate values in the debug text.
fn update_debug_text(
    diagnostic: Res<DiagnosticsStore>,
    debug_text: bevy::ecs::prelude::Single<(Entity, &Visibility), With<DebugText>>,
    mut writer: TextUiWriter,
    time: Res<Time>,
    mut time_since_rerender: Local<Duration>,
    simulation: Res<Simulation>,
) {
    *time_since_rerender += time.delta();
    let (debug_text, visibility) = *debug_text;

    if visibility == Visibility::Hidden {
        return;
    }

    if *time_since_rerender >= Duration::from_millis(100) {
        *time_since_rerender = Duration::ZERO;

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

/// Keyboard control to pause/unpause the simulation.
fn keyboard_pause(
    keys: Res<ButtonInput<KeyCode>>,
    mut pause_text: bevy::ecs::prelude::Single<&mut Visibility, With<PauseText>>,
    pause_state: Res<State<PauseState>>,
    mut next_pause_state: ResMut<NextState<PauseState>>,
) {
    if keys.just_pressed(KeyCode::Space) {
        debug!("Toggle pause state.");
        pause_text.toggle_inherited_hidden();
        match pause_state.get() {
            PauseState::Paused => next_pause_state.set(PauseState::Running),
            PauseState::Running => next_pause_state.set(PauseState::Paused),
        }    
    }
}

/// Keyboard control to show the help screen.
fn keyboard_help(
    keys: Res<ButtonInput<KeyCode>>,
    mut help_text_container: bevy::ecs::prelude::Single<&mut Visibility, With<HelpTextContainer>>,)
    {
    if keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) && keys.just_pressed(KeyCode::Slash) {        
        debug!("Show/hide help text.");
        help_text_container.toggle_inherited_hidden();
        }    
    }

/// Keyboard control to hide the whole UI.
fn keyboard_overlay(
    keys: Res<ButtonInput<KeyCode>>,
    mut overlay_root: bevy::ecs::prelude::Single<&mut Visibility, (With<OverlayRoot>, Without<DebugText>)>,
    mut debug_text: bevy::ecs::prelude::Single<&mut Visibility, (With<DebugText>, Without<OverlayRoot>)>,)
    {
    if keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) && keys.just_pressed(KeyCode::F1) {
        debug!("Show/hide UI.");
        overlay_root.toggle_visible_hidden();
    }
    if keys.just_pressed(KeyCode::F5) && **overlay_root == Visibility::Visible {
        debug!("Show/hide debug overlay.");
        debug_text.toggle_inherited_hidden();
    }
    }

/// Keyboard bindings to control the simulation.
fn keyboard_simulation(
    mut exit: EventWriter<AppExit>,
    keys: Res<ButtonInput<KeyCode>>,
    pause_state: Res<State<PauseState>>,
    simulation: ResMut<Simulation>,) {

    if keys.just_pressed(KeyCode::ArrowRight) && *pause_state.get() == PauseState::Paused {
        let simulation = simulation.into_inner();
        let result = step_simulation(simulation).with_context(|| format!("failed at step: {}", simulation.microstate.step()));
        if let Err(error) = result {
            error!("{error:?}");
            exit.write(AppExit::Error(1.try_into().expect("1 is non-zero")));
            }
    }
}

/// Keyboard command to quit.
fn keyboard_quit(
    mut exit: EventWriter<AppExit>,
    keys: Res<ButtonInput<KeyCode>>,
) {
    if keys.just_pressed(KeyCode::KeyQ) {
        debug!("Quitting...");
        exit.write(AppExit::Success);
    }
}

/// Implement keyboard commands for common operations.
fn keyboard_screenshot(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
) {  
    if keys.just_pressed(KeyCode::F12) {
    commands.spawn(Screenshot::primary_window())
      .observe(save_to_disk("screenshot.png"));
    }
}

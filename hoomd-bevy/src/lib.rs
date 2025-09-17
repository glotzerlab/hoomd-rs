// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![allow(
    clippy::exhaustive_enums,
    reason = "States are intentionally non-exhaustive."
)]
#![allow(
    clippy::missing_inline_in_public_items,
    reason = "hoomd-bevy code is not intended to be inlined."
)]
#![allow(
    clippy::needless_pass_by_value,
    reason = "Bevy requires that args are passed by value."
)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "Bevy operates with f32 values."
)]

/*! Connect *hoomd-rs* simulations with the Bevy game engine.

Use [`HoomdBevyPlugin`] to create visual, interactive simulations. Add the
plugin to a Bevy `App` and it will step the simulation up to a configurable
limit number of steps per second. To display geometry on the screen, add one
more more `setup` methods from [`representation`] to the `Startup` schedule.
Then add a `sync` method to the `Update` schedule that synchronizes the entire
microstate (using the helper methods from [`representation`]).

# Examples

See any one of the many *hoomd-rs* examples that use [`HoomdBevyPlugin`].

# Embedded assets.

`hoomd-bevy` provides the following assets:

* `embedded://hoomd_bevy/logo.png` - The HOOMD logo (512 x 512).

# Feature flags

* `doc-example` Make examples suitable for display in a web browser.
* `webgpu` Compile for the WebGPU platform when building for the wasm32 target.
*/

use std::ops::Range;

use anyhow::Context;
use bevy::{
    asset::embedded_asset,
    input::{
        common_conditions::{input_just_released, input_pressed},
        mouse::MouseWheel,
    },
    prelude::*,
    time::common_conditions::once_after_delay,
    window::PrimaryWindow,
};
#[cfg(not(target_arch = "wasm32"))]
use bevy::{
    render::view::window::screenshot::{Screenshot, save_to_disk},
    time::common_conditions::on_timer,
};
use bevy_diagnostic::{
    Diagnostic, DiagnosticPath, Diagnostics, DiagnosticsStore, FrameTimeDiagnosticsPlugin,
    RegisterDiagnostic,
};
#[cfg(not(target_arch = "wasm32"))]
use bevy_winit::WinitWindows;
use web_time::{Duration, Instant};

use hoomd_simulation::Simulation;

pub mod representation;

/// The default color for the primary representation.
pub const PRIMARY_COLOR: Color = Color::srgb(249.0 / 255.0, 203.0 / 255.0, 136.0 / 255.0);

/// The default color for a muted representation.
pub const MUTED_COLOR: Color = Color::srgb(0.75, 0.75, 0.75);

/// The default color for the boundary representation.
pub const BOUNDARY_COLOR: Color = Color::srgb(0.0, 0.0, 0.0);

/// Camera zoom speed multiplier
const CAMERA_ZOOM_SPEED: f32 = 50.0;

/** Interface *hoomd-rs* simulations with the Bevy game engine.

[`HoomdBevyPlugin`] is used by all the *hoomd-rs* examples that create
interactive graphical displays of simulations. Specifically, it implements:

* Camera controls (2D and 3D separately).
* Simulation step and frame pacing, with a limited number of steps per second.
* Pause and advance by single step controls.
* A help screen describing common controls (examples can add lines if needed).
* Key bindings to hide the UI and take screenshots.
* A menu to control common settings (steps per second limit, camera speed, etc.)

The caller must:
* Provide type that implements [`Simulation`].
* Add a `sync` `Update` system that populates (and removes) entities for
  rendering. See [`representation`] for helper code.

The caller may optionally:
* Add UI to the upper right corner of the screen.
* Implement custom controls.
* Add lines to the [`HelpText`] entity.

To keep individual example scripts short and understandable, `hoomd-bevy` should
implement as much common code as possible.

# Examples

See any one of the many *hoomd-rs* examples that use [`HoomdBevyPlugin`].

[`Simulation`]: hoomd_simulation::Simulation
*/
pub struct HoomdBevyPlugin<S> {
    /// Configuration to use at application start (may be changed later).
    pub initial_settings: Settings,
    /// The simulation to advance and display interactively.
    pub simulation: S,
}

/// Indicate if the simulation should update in real time.
#[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum PauseState {
    /// Prevent automatic simulation advance.
    #[default]
    Paused,
    /// Automatically advance the simulation.
    Running,
}

/// Indicate what menu is displayed (if any)
#[derive(States, Default, Debug, Clone, PartialEq, Eq, Hash)]
pub enum MenuState {
    /// No menu is open.
    #[default]
    None,
    /// The settings menu is open.
    Settings,
}

/// Configure the initial camera view and set how the camera will be controlled.
#[derive(Clone)]
pub enum InitialCamera {
    /** Two dimensional top down camera showing the xy plane.

    The single field sets the height of the visible area. The width is set
    automatically based on the window dimensions.

    Controls:
    * Left click and drag to pan.
    * Scroll to zoom.
    */
    Orthographic2d(f32),
}

/// Store parameters that influence how the simulation is executed.
#[derive(Resource)]
pub struct Settings {
    /// Maximum fraction (0.0 to 1.0) of the frame time to use advancing the simulation.
    pub frame_budget_fraction: f32,

    /// Maximum number of steps per second to advance the simulation.
    pub sps_limit: f32,

    /// Initial camera.
    pub camera: InitialCamera,

    /// Clamp the orthographic camera's scale to this range.
    pub zoom_range: Range<f32>,

    /// Camera sensitivity.
    pub camera_sensitivity: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            frame_budget_fraction: 0.8,
            sps_limit: 2048.0,
            camera: InitialCamera::Orthographic2d(10.0),
            zoom_range: 0.1..10.0,
            camera_sensitivity: 0.5,
        }
    }
}

/// Total time allow to advance simulation per frame.
#[derive(Resource)]
struct FrameBudget(Duration);

/// Settings used by the camera controls.
#[derive(Debug, Default, Resource)]
pub struct CameraControl2d {
    /// Coordinates clicked in the world frame.
    world_position: Vec2,

    /// Track whether the user is dragging the view.
    dragging: bool,
}

/// The overlay UI root node.
#[derive(Component)]
pub struct OverlayRoot;

/// Mark debug text.
#[derive(Component)]
struct DebugText;

/// Mark paused text.
#[derive(Component)]
struct PauseText;

/// Mark the border containing the help text.
#[derive(Component)]
struct HelpTextContainer;

/** Mark the help text entity.

[`HoomdBevyPlugin`] populates the help text with instructions for common
controls. Callers may add lines to the text node to show example-specific
information when ? is pressed.
*/
#[derive(Component)]
pub struct HelpText;

/// Mark help reminder text.
#[derive(Component)]
struct HelpReminder;

/// Mark the logo.
#[derive(Component)]
struct Logo;

/// Mark the SPS limit text.
#[derive(Component)]
struct SPSLimitText;

/// Mark the frame budget text.
#[derive(Component)]
struct FrameBudgetText;

/// Mark the menu root.
#[derive(Component)]
struct MenuRoot;

/** Systems that run to advance the simulation.

Callers should use this to execute the sync step after the simulation is advanced:
`app.add_systems(Update, sync_simulation.run_if(resource_changed::<MySimulation>).after(AdvanceSet));`
*/
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdvanceSet;

/** Systems that always run to process input.

Callers can optionally add input handling systems to this set. It is processed
after [`AdvanceSet`] to reduce the latency between input and result.
*/
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct AlwaysInputSet;

/** Systems that run to process input only when there is no menu displayed.
 */
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct NoMenuInputSet;

/** Systems that run to process input in the settings menu.
 */
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SettingsMenuInputSet;

impl<Sim> HoomdBevyPlugin<Sim>
where
    Sim: Resource + Simulation,
{
    /// Bevy diagnostic that counts the number of steps executed per second.
    pub const SPS: DiagnosticPath = DiagnosticPath::const_new("sps");

    /** Z index at which the help text is displayed.

    Use this should you ever need to display an overlay above the help screen.
    */
    pub const HELP_OVERLAY_ZINDEX: i32 = i32::MAX - 32;

    /// Clear the window to this color before rendering each frame.
    pub const CLEAR: Color = Color::oklch(0.32, 0.0, 0.0);

    /// Display this color in the background of UI elements.
    pub const UI_BACKGROUND: Color = Color::oklch(0.2, 0.0, 0.0);

    /// Display this color on UI outlines.
    pub const UI_OUTLINE: Color = Color::WHITE;

    /// Round the UI to this radius.
    pub const UI_ROUNDING: f32 = 12.0;

    /// Offset the interface from the edge of the screen.
    pub const UI_OFFSET: f32 = 12.0;

    /// Bevy system that advances the simulation forward one step.
    fn step_simulation(
        mut diagnostics: Diagnostics,
        mut exit: EventWriter<AppExit>,
        simulation: ResMut<Sim>,
        time: Res<Time>,
        mut accumulated_steps: Local<f32>,
        settings: Res<Settings>,
        frame_budget: ResMut<FrameBudget>,
    ) {
        // Determine the maximum number of steps that we can take in this update.
        // Accumulate fractional steps over time and remove whole steps from the
        // accumulated amount. This allows for steps per second limits that are
        // less than the monitor's refresh rate.
        let max_steps = settings.sps_limit * time.delta_secs();
        *accumulated_steps += max_steps.fract();

        let mut max_steps = max_steps.floor() as i64;
        if *accumulated_steps > 1.0 {
            max_steps += accumulated_steps.trunc() as i64;
            *accumulated_steps = accumulated_steps.fract();
        }

        let simulation = simulation.into_inner();
        let step_time = Instant::now();
        let mut steps = 0;
        while step_time.elapsed() < frame_budget.0 && steps < max_steps {
            let result = simulation
                .advance()
                .with_context(|| format!("failed at step: {}", simulation.step()));
            if let Err(error) = result {
                error!("{error:?}");
                exit.write(AppExit::Error(1.try_into().expect("1 is non-zero")));
                break;
            }
            steps += 1;
        }

        diagnostics.add_measurement(&Self::SPS, || steps as f64 / time.delta_secs_f64());
    }

    /// Create the full screen UI text overlay node.
    fn setup_overlay(mut commands: Commands, mut ui_scale: ResMut<UiScale>) {
        commands.spawn((
            Node {
                top: Val::Px(0.0),
                left: Val::Px(0.0),
                width: Val::Vw(100.0),
                height: Val::Vh(100.0),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            Visibility::Visible,
            OverlayRoot,
        ));

        ui_scale.0 = if cfg!(feature = "doc-example") {
            0.5
        } else {
            1.0
        };
    }

    /// Add debug text nodes.
    fn setup_debug_text(mut commands: Commands, overlay_root: Single<Entity, With<OverlayRoot>>) {
        commands.spawn((
            Text::default(),
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(Self::UI_OFFSET),
                left: Val::Px(Self::UI_OFFSET),
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

    /// Add paused text node.
    fn add_pause_text(mut commands: Commands, overlay_root: Single<Entity, With<OverlayRoot>>) {
        commands.spawn((
            Text::new("paused..."),
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(Self::UI_OFFSET),
                left: Val::Px(Self::UI_OFFSET),
                ..default()
            },
            PauseText,
            Visibility::Hidden,
            GlobalZIndex(Self::HELP_OVERLAY_ZINDEX),
            ChildOf(*overlay_root),
        ));
    }

    /// Add the help text UI node.
    fn add_help_text(mut commands: Commands, overlay_root: Single<Entity, With<OverlayRoot>>) {
        let mut help_text = String::new();

        #[cfg(not(target_arch = "wasm32"))]
        help_text.push_str("q       : Quit.\n");

        help_text.push_str(
            "=       : Reset the camera.
<space> : Pause the simulation.
<right>: Advance one step (while paused).
shift-F1: Show/hide the user interface.
F5      : Show/hide debugging information.
",
        );

        #[cfg(not(target_arch = "wasm32"))]
        help_text.push_str("F12     : Take a screenshot (screenshot.png).\n");

        help_text.push_str(
            "<esc>   : Open/close the settings menu.
?       : Show/hide this help text.",
        );

        let text = (
            Text::new(help_text),
            BackgroundColor(Self::UI_BACKGROUND),
            HelpText,
        );

        commands.spawn((
            Node {
                align_items: AlignItems::Center,
                position_type: PositionType::Absolute,
                bottom: Val::Px(Self::UI_OFFSET),
                right: Val::Px(Self::UI_OFFSET),
                margin: UiRect::all(Val::Px(0.0)),
                border: UiRect::all(Val::Px(Self::UI_ROUNDING)),
                justify_content: JustifyContent::Center,
                ..default()
            },
            HelpTextContainer,
            ChildOf(*overlay_root),
            Outline {
                width: Val::Px(Self::UI_ROUNDING / 2.0),
                offset: Val::Px(0.),
                color: Self::UI_OUTLINE,
            },
            BorderRadius::px(
                Self::UI_ROUNDING,
                Self::UI_ROUNDING,
                Self::UI_ROUNDING,
                Self::UI_ROUNDING,
            ),
            BackgroundColor(Self::UI_BACKGROUND),
            BorderColor(Self::UI_BACKGROUND),
            Visibility::Hidden,
            GlobalZIndex(Self::HELP_OVERLAY_ZINDEX),
            children![text],
        ));
    }

    /// Add help reminder node.
    fn add_help_reminder(mut commands: Commands, overlay_root: Single<Entity, With<OverlayRoot>>) {
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
            GlobalZIndex(Self::HELP_OVERLAY_ZINDEX - 1),
        ));
    }

    /// Remove the help reminder text.
    fn remove_help_reminder(
        mut commands: Commands,
        overlay_root: Single<Entity, (With<OverlayRoot>, Without<HelpReminder>)>,
        help_reminder: Single<Entity, (With<HelpReminder>, Without<OverlayRoot>)>,
    ) {
        commands
            .entity(*overlay_root)
            .remove_children(&[*help_reminder]);
        commands.entity(*help_reminder).despawn();
    }

    /// Add help reminder node.
    fn add_logo(mut commands: Commands, server: Res<AssetServer>) {
        commands.spawn((
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(12.0),
                left: Val::Px(12.0),
                width: Val::Px(64.0),
                height: Val::Px(64.0),
                ..default()
            },
            ImageNode {
                image: server.load("embedded://hoomd_bevy/logo.png"),
                ..default()
            },
            Logo,
            GlobalZIndex(Self::HELP_OVERLAY_ZINDEX - 1),
        ));
    }

    /// Remove the help reminder text.
    fn remove_logo(mut commands: Commands, logo: Single<Entity, With<Logo>>) {
        commands.entity(*logo).despawn();
    }

    /// Populate values in the debug text.
    fn update_debug_text(
        diagnostic: Res<DiagnosticsStore>,
        debug_text: Single<(Entity, &Visibility), With<DebugText>>,
        mut writer: TextUiWriter,
        time: Res<Time>,
        mut time_since_rerender: Local<Duration>,
        simulation: Res<Sim>,
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
            if let Some(sps) = diagnostic.get(&Self::SPS) {
                if let Some(value) = sps.smoothed() {
                    *writer.text(debug_text, 2) = format!(" SPS: {value:.2}\n");
                }
            }
            *writer.text(debug_text, 3) = format!("Step: {}\n", simulation.step());
        }
    }

    /// Keyboard control to pause/unpause the simulation.
    fn keyboard_pause(
        keys: Res<ButtonInput<KeyCode>>,
        mut pause_text: Single<&mut Visibility, With<PauseText>>,
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
        mut help_text_container: Single<&mut Visibility, With<HelpTextContainer>>,
    ) {
        if keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight])
            && keys.just_pressed(KeyCode::Slash)
        {
            debug!("Show/hide help text.");
            help_text_container.toggle_inherited_hidden();
        }
    }

    /// Keyboard control to show the menu.
    fn keyboard_menu(
        keys: Res<ButtonInput<KeyCode>>,
        mut menu_root: Single<&mut Visibility, With<MenuRoot>>,
        menu_state: Res<State<MenuState>>,
        mut next_menu_state: ResMut<NextState<MenuState>>,
    ) {
        if keys.just_pressed(KeyCode::Escape) {
            debug!("Show/hide the menu.");
            menu_root.toggle_inherited_hidden();
            match menu_state.get() {
                MenuState::Settings => next_menu_state.set(MenuState::None),
                MenuState::None => next_menu_state.set(MenuState::Settings),
            }
        }
    }

    /// Keyboard control to hide the whole UI.
    fn keyboard_overlay(
        keys: Res<ButtonInput<KeyCode>>,
        mut overlay_root: Single<&mut Visibility, (With<OverlayRoot>, Without<DebugText>)>,
        mut debug_text: Single<&mut Visibility, (With<DebugText>, Without<OverlayRoot>)>,
    ) {
        if keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight])
            && keys.just_pressed(KeyCode::F1)
        {
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
        simulation: ResMut<Sim>,
    ) {
        if keys.just_pressed(KeyCode::ArrowRight) && *pause_state.get() == PauseState::Paused {
            let simulation = simulation.into_inner();
            let result = simulation
                .advance()
                .with_context(|| format!("failed at step: {}", simulation.step()));
            if let Err(error) = result {
                error!("{error:?}");
                exit.write(AppExit::Error(1.try_into().expect("1 is non-zero")));
            }
        }
    }

    /// Keyboard command to quit.
    #[cfg(not(target_arch = "wasm32"))]
    fn keyboard_quit(mut exit: EventWriter<AppExit>, keys: Res<ButtonInput<KeyCode>>) {
        #[cfg(not(target_arch = "wasm32"))]
        if keys.just_pressed(KeyCode::KeyQ) {
            debug!("Quitting...");
            exit.write(AppExit::Success);
        }
    }

    /// Implement keyboard commands for common operations.
    #[cfg(not(target_arch = "wasm32"))]
    fn keyboard_screenshot(mut commands: Commands, keys: Res<ButtonInput<KeyCode>>) {
        if keys.just_pressed(KeyCode::F12) {
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk("screenshot.png"));
        }
    }

    /** Set the time budgeted to advancing the simulation each frame.

    Derive this time from the current monitor refresh rate and the
    `frame_budget_fraction` settings.
    */
    #[cfg(not(target_arch = "wasm32"))]
    fn set_frame_budget(
        winit: NonSend<WinitWindows>,
        windows: Query<Entity, With<Window>>,
        settings: Res<Settings>,
        mut frame_budget: ResMut<FrameBudget>,
    ) {
        // adapted from: https://github.com/aevyrie/bevy_framepace/blob/main/src/lib.rs
        let new_frame_budget = match Self::detect_frame_time(winit, windows.iter()) {
            Some(frame_time) => {
                Duration::from_secs_f32(frame_time.as_secs_f32() * settings.frame_budget_fraction)
            }
            None => return,
        };

        if new_frame_budget != frame_budget.0 {
            frame_budget.0 = new_frame_budget;
            debug!("New simulation frame budget: {:?}", frame_budget.0);
        }
    }

    /// Detect the minimum frame time for all windows.
    #[cfg(not(target_arch = "wasm32"))]
    fn detect_frame_time(
        winit: NonSend<WinitWindows>,
        windows: impl Iterator<Item = Entity>,
    ) -> Option<Duration> {
        let best_framerate = {
            f64::from(
                windows
                    .filter_map(|e| winit.get_window(e))
                    .filter_map(|w| w.current_monitor())
                    .filter_map(|monitor| monitor.refresh_rate_millihertz())
                    .min()?,
            ) / 1000.0
                - 0.5
        };

        let best_frame_time = Duration::from_secs_f64(1.0 / best_framerate);
        Some(best_frame_time)
    }

    /// Set up the options menu.
    fn setup_options(
        mut commands: Commands,
        overlay_root: Single<Entity, With<OverlayRoot>>,
        settings: Res<Settings>,
    ) {
        let sps = (
            Node::default(),
            children![(
                Text("Steps per second limit (-/=):   ".into()),
                children![(TextSpan(format!("{}", settings.sps_limit)), SPSLimitText)]
            )],
        );
        let frame_budget_fraction = (
            Node::default(),
            children![(
                Text("Simulation time fraction ([/]): ".into()),
                children![(
                    TextSpan(format!("{}", settings.frame_budget_fraction)),
                    FrameBudgetText
                )]
            )],
        );

        commands.spawn((
            Node {
                align_items: AlignItems::FlexStart,
                justify_content: JustifyContent::Center,
                margin: UiRect::all(Val::Px(0.0)),
                border: UiRect::all(Val::Px(Self::UI_ROUNDING)),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            Outline {
                width: Val::Px(Self::UI_ROUNDING / 2.0),
                offset: Val::Px(0.),
                color: Self::UI_OUTLINE,
            },
            BorderRadius::px(
                Self::UI_ROUNDING,
                Self::UI_ROUNDING,
                Self::UI_ROUNDING,
                Self::UI_ROUNDING,
            ),
            BackgroundColor(Self::UI_BACKGROUND),
            BorderColor(Self::UI_BACKGROUND),
            ChildOf(*overlay_root),
            Visibility::Hidden,
            MenuRoot,
            children![sps, frame_budget_fraction],
        ));
    }

    /// Handle the increase/decrease SPS buttons.
    fn keyboard_sps(
        keys: Res<ButtonInput<KeyCode>>,
        mut text: Single<&mut TextSpan, With<SPSLimitText>>,
        mut settings: ResMut<Settings>,
    ) {
        if keys.just_pressed(KeyCode::Minus) {
            settings.sps_limit /= 2.0;
            text.0 = format!("{}", settings.sps_limit);
        }
        if keys.just_pressed(KeyCode::Equal) {
            settings.sps_limit *= 2.0;
            text.0 = format!("{}", settings.sps_limit);
        }
    }

    /// Handle the increase/decrease frame budget buttons.
    fn keyboard_frame_budget(
        keys: Res<ButtonInput<KeyCode>>,
        mut text: Single<&mut TextSpan, With<FrameBudgetText>>,
        mut settings: ResMut<Settings>,
    ) {
        if keys.just_pressed(KeyCode::BracketLeft) {
            settings.frame_budget_fraction = (settings.frame_budget_fraction - 0.1).clamp(0.1, 0.9);
            text.0 = format!("{:.1}", settings.frame_budget_fraction);
        }
        if keys.just_pressed(KeyCode::BracketRight) {
            settings.frame_budget_fraction = (settings.frame_budget_fraction + 0.1).clamp(0.1, 0.9);
            text.0 = format!("{:.1}", settings.frame_budget_fraction);
        }
    }

    /// Set up the 2D camera.
    fn setup_camera_2d(mut commands: Commands, viewport_height: f32) {
        let projection = Projection::Orthographic(OrthographicProjection {
            scaling_mode: bevy::render::camera::ScalingMode::FixedVertical { viewport_height },
            ..OrthographicProjection::default_2d()
        });

        commands.spawn((Camera2d, projection));
    }

    /** Keyboard controls for the 2d camera.

    * `=` resets the camera to the default.
    */
    fn camera_keyboard_control_2d(
        keys: Res<ButtonInput<KeyCode>>,
        camera: Single<(&mut Transform, &mut Projection), With<Camera2d>>,
        mut control: ResMut<CameraControl2d>,
    ) {
        let (mut transform, projection) = camera.into_inner();

        if keys.just_pressed(KeyCode::Equal) {
            if let Projection::Orthographic(ref mut orthographic) = *projection.into_inner() {
                orthographic.scale = 1.0;
            }
            control.dragging = false;
            transform.translation = Vec3::default();
        }
    }

    /** Left click and drag to pan the 2D camera.

    # Panics

    Panics when the 2D camera viewport is invalid.
    */
    fn camera_mouse_pan_control_2d(
        camera: Single<
            (&Camera, &GlobalTransform, &mut Transform, &mut Projection),
            With<Camera2d>,
        >,
        mut control: ResMut<CameraControl2d>,
        buttons: Res<ButtonInput<MouseButton>>,
        window: Single<&Window, With<PrimaryWindow>>,
    ) {
        // Firefox wasm builds do not behave well using AccumulatedMouseMotion. Use
        // absolute window coordinates and a state machine to provide consistent
        // panning behavior across all platforms.

        let (camera, global_transform, mut transform, projection) = camera.into_inner();

        let viewport_size = camera
            .logical_viewport_size()
            .unwrap_or(Vec2::new(1280.0, 720.0));

        if let Projection::Orthographic(ref mut orthographic) = *projection.into_inner() {
            if buttons.just_pressed(MouseButton::Left) {
                if let Some(world_position) = window
                    .cursor_position()
                    .and_then(|cursor| camera.viewport_to_world_2d(global_transform, cursor).ok())
                {
                    control.world_position = world_position;
                    control.dragging = true;
                    return;
                }
            }

            if !buttons.pressed(MouseButton::Left) {
                control.dragging = false;
                return;
            }

            if control.dragging
                && let Some(current_cursor_position) = window.cursor_position()
            {
                let pixel_scale = orthographic.area.size() / viewport_size;

                // Pan by placing control.world_position at the cursor position
                let desired_cursor_position = camera
                    .world_to_viewport(global_transform, Vec3::from((control.world_position, 0.0)))
                    .expect("viewport should be valid");

                let offset = (desired_cursor_position - current_cursor_position) * pixel_scale;
                transform.translation.x += offset.x;
                transform.translation.y -= offset.y;
            }
        }
    }

    /// Zoom the 2d camera using the mouse wheel or trackpad scroll gesture.
    fn camera_mouse_zoom_control_2d(
        time: Res<Time>,
        camera: Single<
            (&Camera, &GlobalTransform, &mut Transform, &mut Projection),
            With<Camera2d>,
        >,
        settings: Res<Settings>,
        mut scroll: EventReader<MouseWheel>,
        window: Single<&Window, With<PrimaryWindow>>,
    ) {
        let (camera, global_transform, mut transform, projection) = camera.into_inner();

        if let Projection::Orthographic(ref mut orthographic) = *projection.into_inner() {
            let scroll = scroll.read().map(|e| e.y).fold(0.0, |total, y| total + y);

            // The scroll events distinguish between line (mouse wheel) and pixel
            // (trackpad) events. However, In wasm builds all major browsers report
            // only pixel events. Tested on macOS, scrolling with the trackpad gave
            // consistent values across all browsers and native. However, scrolling
            // with the mouse wheel gave different scales between native and browser
            // and from browser to browser (a factor of 100 from the smallest to
            // the largest). Therefore, the best we can do is check the sign of the
            // scroll event and act scale the camera in the appropriate direction.

            let zoom_speed = settings.camera_sensitivity * CAMERA_ZOOM_SPEED * time.delta_secs();
            let delta_zoom = -zoom_speed.copysign(scroll);
            let new_scale = (orthographic.scale * (1.0 + delta_zoom))
                .clamp(settings.zoom_range.start, settings.zoom_range.end);
            let scale_ratio = new_scale / orthographic.scale;

            let world_position_result = window
                .cursor_position()
                .and_then(|cursor| camera.viewport_to_world_2d(global_transform, cursor).ok());

            let delta_translation = match world_position_result {
                None => Vec2::default(),
                Some(world_position) => {
                    (world_position - transform.translation.xy()) * (1.0 - scale_ratio)
                }
            };

            orthographic.scale = new_scale;
            transform.translation += Vec3::from((delta_translation, 0.0));
        }
    }

    /** Build the plugin.

    [`HoomdBevyPlugin`] does not implement [`Plugin`] and cannot be used with
    `add_plugins` so that the `build` method can consume `self`. This allows
    `build` to take ownership of the `simulation` field and create the appropriate
    Bevy [`Resource`].
    */
    #[expect(clippy::too_many_lines, reason = "Bevy functions are very verbose.")]
    pub fn build(self, app: &mut App) {
        representation::disk::build(app);
        representation::ellipse::build(app);

        embedded_asset!(app, "logo.png");

        let initial_camera = self.initial_settings.camera.clone();

        app.add_plugins(FrameTimeDiagnosticsPlugin::default())
            .insert_resource(ClearColor(Self::CLEAR))
            .insert_resource(FrameBudget(Duration::from_millis(9)))
            .insert_resource(self.initial_settings)
            .register_diagnostic(Diagnostic::new(Self::SPS))
            .insert_resource(self.simulation)
            .insert_state(PauseState::Running)
            .insert_state(MenuState::None)
            .add_systems(
                Startup,
                (
                    Self::setup_overlay,
                    Self::setup_debug_text,
                    Self::add_pause_text,
                    Self::add_help_text,
                    Self::add_help_reminder,
                    Self::add_logo,
                    Self::setup_options,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                (Self::remove_help_reminder, Self::remove_logo)
                    .run_if(once_after_delay(Duration::from_secs(3))),
            )
            .add_systems(Update, Self::step_simulation.in_set(AdvanceSet))
            .add_systems(
                Update,
                (
                    (Self::keyboard_overlay, Self::update_debug_text).chain(),
                    Self::keyboard_menu,
                )
                    .in_set(AlwaysInputSet),
            )
            .add_systems(
                Update,
                (
                    Self::keyboard_pause,
                    Self::keyboard_help,
                    Self::keyboard_simulation,
                )
                    .in_set(NoMenuInputSet),
            )
            .add_systems(
                Update,
                (Self::keyboard_sps, Self::keyboard_frame_budget).in_set(SettingsMenuInputSet),
            );

        match initial_camera {
            InitialCamera::Orthographic2d(initial_viewport_height) => {
                app.add_systems(
                    Update,
                    Self::camera_mouse_pan_control_2d
                        .run_if(
                            input_pressed(MouseButton::Left)
                                .or(input_just_released(MouseButton::Left)),
                        )
                        .in_set(NoMenuInputSet),
                )
                .add_systems(
                    Update,
                    Self::camera_mouse_zoom_control_2d
                        .run_if(on_event::<MouseWheel>)
                        .in_set(NoMenuInputSet),
                )
                .add_systems(
                    Update,
                    Self::camera_keyboard_control_2d.in_set(NoMenuInputSet),
                )
                .insert_resource(CameraControl2d::default())
                .add_systems(Startup, move |commands: Commands| {
                    Self::setup_camera_2d(commands, initial_viewport_height);
                });
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        app.add_systems(
            Update,
            Self::set_frame_budget.run_if(on_timer(Duration::from_millis(250))),
        );

        #[cfg(not(target_arch = "wasm32"))]
        app.add_systems(
            Update,
            (Self::keyboard_quit, Self::keyboard_screenshot).in_set(AlwaysInputSet),
        );

        app.configure_sets(
            Update,
            (
                AdvanceSet.run_if(in_state(PauseState::Running)),
                AlwaysInputSet.after(AdvanceSet),
                NoMenuInputSet
                    .run_if(in_state(MenuState::None))
                    .after(AdvanceSet),
                SettingsMenuInputSet
                    .run_if(in_state(MenuState::Settings))
                    .after(AdvanceSet),
            ),
        );
    }
}

/** Construct the default plugins.

This helper adds Bevy's `DefaultPlugins` by default. When the
`doc-example` feature is enabled, it adds a modified set of plugins
for the web.
*/
pub fn add_default_plugins(app: &mut App) {
    if cfg!(feature = "doc-example") {
        app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                canvas: Some("#hoomd-example".into()),
                fit_canvas_to_parent: true,
                ..default()
            }),
            ..default()
        }));
    } else {
        app.add_plugins(DefaultPlugins);
    }
}

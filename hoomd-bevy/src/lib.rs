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

Use [`HoomdBevyPlugin`] to create visual, interactive simulations.

# Stability

`hoomd-bevy` currently does **NOT** follow semantic versioning. First, it
is primarily intended for use to support the *hoomd-rs* examples. Second,
*Bevy* itself is under very active development and every release makes
breaking changes. You are welcome to use `hoomd-bevy` for your own interactive
applications, but keep in mind that minor releases to *hoomd-rs* may make
breaking changes in `hoomd-bevy`.

# Examples

See any one of the many *hoomd-rs* examples that use [`HoomdBevyPlugin`].
*/

use anyhow::Context;
use bevy::{
    prelude::*,
    render::view::window::screenshot::{Screenshot, save_to_disk},
    time::common_conditions::{on_timer, once_after_delay},
};
use bevy_diagnostic::{
    Diagnostic, DiagnosticPath, Diagnostics, DiagnosticsStore, FrameTimeDiagnosticsPlugin,
    RegisterDiagnostic,
};
use bevy_winit::WinitWindows;

use std::time::{Duration, Instant};

pub mod representation;

/** The model, parameters, and microstate they act on.

A [`Simulation`] type stores the microstate, all model actors, and any
macrostate parameters in fields. [`HoomdBevyPlugin`] requires that each
[`Simulation`] provide a method to advance forward one step and a method to
query the current step. Beyond that, user types are free to implement any
inherent methods necessary to manage the simulation.
*/
pub trait Simulation {
    /** Advance the simulation forward one step.

    # Errors

    When an error occurs, return an `Err` with any type that implements
    [`Error`](std::error::Error) [`HoomdBevyPlugin`] will catch the error,
    display it to the `error!` log and exit.
    */
    fn advance(&mut self) -> anyhow::Result<()>;

    /// Get the simulation step.
    fn step(&self) -> u64;
}

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
* Provide type that implements [`Simulation`] (`Sim`).
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

/// Store parameters that influence how the simulation is executed.
#[derive(Resource)]
pub struct Settings {
    /// Maximum fraction (0.0 to 1.0) of the frame time to use advancing the simulation.
    frame_budget_fraction: f32,

    /// Maximum number of steps per second to advance the simulation.
    sps_limit: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            frame_budget_fraction: 0.9,
            sps_limit: 1024.0,
        }
    }
}

/// Total time allow to advance simulation per frame.
#[derive(Resource)]
struct FrameBudget(Duration);

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

/// Mark increase SPS button.
#[derive(Component, Default)]
struct IncreaseSPS;

/// Mark decrase SPS button.
#[derive(Component, Default)]
struct DecreaseSPS;

/// Mark the SPS limit text.
#[derive(Component)]
struct SPSLimitText;

/// Mark increase frame budget button.
#[derive(Component, Default)]
struct IncreaseFrameBudget;

/// Mark decrase frame budget button.
#[derive(Component, Default)]
struct DecreaseFrameBudget;

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

/** Systems that run to process input.

Callers can optionally add input handling systems to this set. It is processed
after [`AdvanceSet`] to reduce the latency between input and result.
*/
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct InputSet;

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

    /// Display this color on UI buttons.
    pub const UI_BUTTON: Color = Color::oklch(0.6795, 0.1708, 27.77);

    /// Display this color when buttons are active.
    pub const UI_BUTTON_ACTIVE: Color = Color::oklch(0.6795, 0.1708, 144.47);

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
    fn setup_overlay(mut commands: Commands) {
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
            ChildOf(*overlay_root),
        ));
    }

    /// Add the help text UI node.
    fn add_help_text(mut commands: Commands, overlay_root: Single<Entity, With<OverlayRoot>>) {
        let text = (
            Text::new(
                "q       : Quit.
<space> : Pause the simulation.
<right> : Advance one step (while paused).
shift-F1: Show/hide the user interface.
F5      : Show/hide debugging information.
F12     : Take a screenshot (screenshot.png).
<esc>   : Open/close the settings menu.
?       : Show/hide this help text.",
            ),
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

    /// Remove paused text node.
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
    ) {
        if keys.just_pressed(KeyCode::Escape) {
            debug!("Show/hide the menu.");
            menu_root.toggle_inherited_hidden();
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
    fn keyboard_quit(mut exit: EventWriter<AppExit>, keys: Res<ButtonInput<KeyCode>>) {
        if keys.just_pressed(KeyCode::KeyQ) {
            debug!("Quitting...");
            exit.write(AppExit::Success);
        }
    }

    /// Implement keyboard commands for common operations.
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

    // TODO: how to set the camera height? Put it in settings?
    // TODO: How to set 3D cameras? Use a marker type? Or an option in the settings?

    /// Set up the 2D camera.
    fn setup_camera(mut commands: Commands) {
        let projection = Projection::Orthographic(OrthographicProjection {
            scaling_mode: bevy::render::camera::ScalingMode::FixedVertical {
                viewport_height: 10.0,
            },
            ..OrthographicProjection::default_2d()
        });

        commands.spawn((Camera2d, projection));
    }

    /// Set up the options menu.
    fn setup_options(
        mut commands: Commands,
        overlay_root: Single<Entity, With<OverlayRoot>>,
        settings: Res<Settings>,
    ) {
        let sps = (
            Node::default(),
            children![
                Self::create_button::<DecreaseSPS>("-"),
                Self::create_button::<IncreaseSPS>("+"),
                (
                    Text("Steps per second limit:\n".into()),
                    children![(TextSpan(format!("{}", settings.sps_limit)), SPSLimitText)]
                )
            ],
        );
        let frame_budget_fraction = (
            Node::default(),
            children![
                Self::create_button::<DecreaseFrameBudget>("-"),
                Self::create_button::<IncreaseFrameBudget>("+"),
                (
                    Text("Simulation time fraction:\n".into()),
                    children![(
                        TextSpan(format!("{}", settings.frame_budget_fraction)),
                        FrameBudgetText
                    )]
                )
            ],
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

    /// Handle the decrease SPS button.
    fn decrease_sps(
        interaction: Single<&Interaction, (Changed<Interaction>, With<DecreaseSPS>)>,
        mut text: Single<&mut TextSpan, With<SPSLimitText>>,
        mut settings: ResMut<Settings>,
    ) {
        if **interaction == Interaction::Pressed {
            settings.sps_limit /= 2.0;
            text.0 = format!("{}", settings.sps_limit);
        }
    }

    /// Handle the increase SPS button.
    fn increase_sps(
        interaction: Single<&Interaction, (Changed<Interaction>, With<IncreaseSPS>)>,
        mut text: Single<&mut TextSpan, With<SPSLimitText>>,
        mut settings: ResMut<Settings>,
    ) {
        if **interaction == Interaction::Pressed {
            settings.sps_limit *= 2.0;
            text.0 = format!("{}", settings.sps_limit);
        }
    }

    /// Handle the decrease frame budget button.
    fn decrease_frame_budget(
        interaction: Single<&Interaction, (Changed<Interaction>, With<DecreaseFrameBudget>)>,
        mut text: Single<&mut TextSpan, With<FrameBudgetText>>,
        mut settings: ResMut<Settings>,
    ) {
        if **interaction == Interaction::Pressed {
            settings.frame_budget_fraction = (settings.frame_budget_fraction - 0.1).clamp(0.1, 0.9);
            text.0 = format!("{:.1}", settings.frame_budget_fraction);
        }
    }

    /// Handle the increase SPS button.
    fn increase_frame_budget(
        interaction: Single<&Interaction, (Changed<Interaction>, With<IncreaseFrameBudget>)>,
        mut text: Single<&mut TextSpan, With<FrameBudgetText>>,
        mut settings: ResMut<Settings>,
    ) {
        if **interaction == Interaction::Pressed {
            settings.frame_budget_fraction = (settings.frame_budget_fraction + 0.1).clamp(0.1, 0.9);
            text.0 = format!("{:.1}", settings.frame_budget_fraction);
        }
    }

    /// Create a button with the given label
    fn create_button<Marker: Component + Default>(label: &str) -> impl Bundle {
        (
            Button,
            Node {
                width: Val::Px(35.0),
                height: Val::Px(35.0),
                border: UiRect::all(Val::Px(Self::UI_ROUNDING)),
                margin: UiRect::all(Val::Px(Self::UI_ROUNDING * 2.0 / 3.0)),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            Outline {
                width: Val::Px(Self::UI_ROUNDING / 4.0),
                offset: Val::Px(0.),
                color: Self::UI_OUTLINE,
            },
            BorderColor(Self::UI_BUTTON),
            BackgroundColor(Self::UI_BUTTON),
            Marker::default(),
            children![Text::new(label)],
        )
    }

    /** Build the plugin.

    [`HoomdBevyPlugin`] does not implement [`Plugin`] and cannot be used with
    `add_plugins` so that the `build` method can consume `self`. This allows
    `build` to take ownership of the `simulation` field and create the appropriate
    Bevy [`Resource`].
    */
    pub fn build(self, app: &mut App) {
        representation::disk::build(app);
    
        app.add_plugins(FrameTimeDiagnosticsPlugin::default())
            .insert_resource(ClearColor(Self::CLEAR))
            .insert_resource(FrameBudget(Duration::from_millis(9)))
            .insert_resource(self.initial_settings)
            .register_diagnostic(Diagnostic::new(Self::SPS))
            .insert_resource(self.simulation)
            .insert_state(PauseState::Running)
            .add_systems(Startup, Self::setup_camera)
            .add_systems(
                Startup,
                (
                    Self::setup_overlay,
                    Self::setup_debug_text,
                    Self::add_pause_text,
                    Self::add_help_text,
                    Self::add_help_reminder,
                    Self::setup_options,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                Self::remove_help_reminder.run_if(once_after_delay(Duration::from_secs(3))),
            )
            .add_systems(Update, Self::step_simulation.in_set(AdvanceSet))
            .add_systems(
                Update,
                (Self::keyboard_overlay, Self::update_debug_text)
                    .chain()
                    .in_set(InputSet),
            )
            .add_systems(
                Update,
                (
                    Self::keyboard_pause,
                    Self::keyboard_help,
                    Self::keyboard_menu,
                    Self::keyboard_simulation,
                    Self::keyboard_screenshot,
                    Self::keyboard_quit,
                    Self::decrease_sps,
                    Self::increase_sps,
                    Self::decrease_frame_budget,
                    Self::increase_frame_budget,
                )
                    .in_set(InputSet),
            )
            .add_systems(
                Update,
                Self::set_frame_budget.run_if(on_timer(Duration::from_millis(250))),
            );

        app.configure_sets(
            Update,
            (
                AdvanceSet.run_if(in_state(PauseState::Running)),
                InputSet.after(AdvanceSet),
            ),
        );
    }
}

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
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
#![allow(clippy::too_many_arguments, reason = "Bevy requires many arguments.")]
#![allow(clippy::too_many_lines, reason = "Bevy requires long functions.")]

//! Connect *hoomd-rs* simulations with the Bevy game engine.
//!
//! Use [`HoomdBevyPlugin`] to create visual, interactive simulations. Add the
//! plugin to a Bevy `App` and it will step the simulation up to a configurable
//! limit number of steps per second. To display geometry on the screen, add one
//! more more `setup` methods from [`representation`] to the `Startup` schedule.
//! Then add a `sync` method to the `Update` schedule that synchronizes the entire
//! microstate (using the helper methods from [`representation`]).
//!
//! # Examples
//!
//! Many of the examples use [`HoomdBevyPlugin`]. Find them in the [`examples`]
//! directory in the *hoomd-rs* repository.
//!
//! [`examples`]: https://github.com/glotzerlab/hoomd-rs/tree/trunk/examples
//!
//! # Embedded assets.
//!
//! `hoomd-bevy` provides the following assets:
//!
//! `embedded://hoomd_bevy/logo.png` - The HOOMD logo (512 x 512).
//!
//! # Feature flags
//!
//! `doc-example` Make examples suitable for display in a web browser.
//! `webgpu` Compile for the WebGPU platform when building for the wasm32 target.

use std::{ops::Range, time::Duration};

use anyhow::Context;
use bevy::{
    asset::embedded_asset,
    input::{
        common_conditions::{input_just_released, input_pressed},
        mouse::MouseWheel,
    },
    platform::time::Instant,
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
use bevy_egui::{
    EguiContextSettings, EguiContexts, EguiPlugin, EguiPrimaryContextPass,
    egui::{
        self,
        gui_zoom::kb_shortcuts::{ZOOM_IN, ZOOM_IN_SECONDARY, ZOOM_OUT, ZOOM_RESET},
    },
    input::{egui_wants_any_keyboard_input, egui_wants_any_pointer_input},
};
#[cfg(not(target_arch = "wasm32"))]
use bevy_winit::WINIT_WINDOWS;

use hoomd_simulation::Simulation;

pub mod representation;

/// The default color for the primary representation (in 2D).
pub const PRIMARY_COLOR: Color = Color::srgb(249.0 / 255.0, 203.0 / 255.0, 136.0 / 255.0);

/// The default color for the primary representation (darkened for 3D lighting).
pub const PRIMARY_COLOR_3D: Color = Color::srgb(0.836, 0.533, 0.211);

/// The default color for a muted representation.
pub const MUTED_COLOR: Color = Color::srgb(0.75, 0.75, 0.75);

/// The default color for the boundary representation.
pub const BOUNDARY_COLOR: Color = Color::srgb(0.0, 0.0, 0.0);

/// Camera zoom speed multiplier
const CAMERA_ZOOM_SPEED: f32 = 50.0;

/// Interface *hoomd-rs* simulations with the Bevy game engine.
///
/// [`HoomdBevyPlugin`] is used by all the *hoomd-rs* examples that create
/// interactive graphical displays of simulations. Specifically, it implements:
///
/// * Camera controls (2D and 3D separately).
/// * Simulation step and frame pacing, with a limited number of steps per second.
/// * Pause and advance by single step controls.
/// * Screenshots.
/// * A GUI that provides usage instructions, settings, and controls.
///
/// The caller must:
/// * Add the `EguiPlugin`.
/// * Provide type that implements [`Simulation`].
/// * Add a `sync` `Update` system that populates (and removes) entities for
///   rendering. See [`representation`] for helper code.
///
/// The caller may optionally:
/// * Add UI to the upper left and/or right corners of the screen.
/// * Implement custom keyboard and/or GUI controls.
///
/// To keep individual example scripts short and understandable, `hoomd-bevy` should
/// implement as much common code as possible.
///
/// # Examples
///
/// See any one of the many *hoomd-rs* examples that use [`HoomdBevyPlugin`].
///
/// [`Simulation`]: hoomd_simulation::Simulation
pub struct HoomdBevyPlugin<S> {
    /// Configuration to use at application start (may be changed later).
    pub initial_settings: Settings,
    /// The simulation to advance and display interactively.
    pub simulation: S,
}

/// State of the UI
#[derive(Default, Resource)]
pub struct UiState {
    /// Prevent the simulation from running when true.
    pause: bool,
    /// Show the debug overlay.
    show_debug: bool,
}

/// State of the options window
///
/// The options window is hidden by default.
#[derive(Default, Resource)]
struct OptionsWindowState(bool);

/// State of the parameters window
///
/// The parameters window is shown by default.
#[derive(Resource)]
pub struct ParametersWindowState(pub bool);

/// Reset the camera to the default.
#[derive(Message)]
struct ResetCamera;

/// Quit the application.
#[derive(Message)]
struct Quit;

/// Advance the simulation one step.
#[derive(Message)]
struct AdvanceSimulation;

/// Configure the initial camera view and set how the camera will be controlled.
#[derive(Clone)]
pub enum InitialCamera {
    /// Two dimensional top down camera showing the xy plane.
    ///
    /// The single field sets the height of the visible area. The width is set
    /// automatically based on the window dimensions.
    ///
    /// Controls:
    /// * Left click and drag to pan.
    /// * Scroll to zoom.
    Orthographic2d(f32),

    /// Three dimensional front down camera showing the xy plane.
    ///
    /// The single field sets the height of the visible area. The width is set
    /// automatically based on the window dimensions.
    ///
    /// Controls:
    /// * TODO
    Orthographic3d(f32),
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
            zoom_range: 0.25..10.0,
            camera_sensitivity: 0.5,
        }
    }
}

/// Total time allow to advance simulation per frame.
#[derive(Resource)]
struct FrameBudget(Duration);

/// Settings used by the 2d camera controls.
#[derive(Debug, Default, Resource)]
pub struct CameraControl2d {
    /// Coordinates clicked in the world frame.
    world_position: Vec2,

    /// Track whether the user is dragging the view.
    dragging: bool,
}

/// The overlay UI root node.
#[derive(Component)]
struct OverlayRoot;

/// Mark debug text.
#[derive(Component)]
struct DebugText;

/// Mark the logo.
#[derive(Component)]
struct Logo;

/// Systems that run to advance the simulation.
///
/// Callers should use this to execute the sync step after the simulation is advanced:
/// `app.add_systems(Update, sync_simulation.run_if(resource_changed::<MySimulation>).after(AdvanceSet));`
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdvanceSet;

/// Systems that run to process non-GUI keyboard input.
///
/// Callers must add any keyboard input handling systems to this set.
/// It is processed after [`AdvanceSet`] to reduce the latency between
/// input and result and it is skipped when the GUI is capturing
/// keyboard input.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeyboardInputSet;

/// Systems that run to process non-GUI mouse input.
///
/// Callers must add any mouse input handling systems to this set.
/// It is processed after [`AdvanceSet`] to reduce the latency between
/// input and result and it is skipped when the GUI is capturing
/// mouse input.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct MouseInputSet;

impl<Sim> HoomdBevyPlugin<Sim>
where
    Sim: Resource + Simulation,
{
    /// Bevy diagnostic that counts the number of steps executed per second.
    pub const SPS: DiagnosticPath = DiagnosticPath::const_new("sps");

    /// Clear the window to this color before rendering each frame.
    pub const CLEAR: Color = Color::oklch(0.32, 0.0, 0.0);

    /// Offset the interface from the edge of the screen.
    pub const UI_OFFSET: f32 = 12.0;

    /// Bevy system that advances the simulation forward one step.
    fn step_simulation(
        mut diagnostics: Diagnostics,
        mut exit: MessageWriter<AppExit>,
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

    /// Advance the simulation one step
    fn advance_simulation(
        simulation: ResMut<Sim>,
        mut exit: MessageWriter<AppExit>,
        mut event: MessageReader<AdvanceSimulation>,
    ) {
        let simulation = simulation.into_inner();
        for _ in event.read() {
            let result = simulation
                .advance()
                .with_context(|| format!("failed at step: {}", simulation.step()));
            if let Err(error) = result {
                error!("{error:?}");
                exit.write(AppExit::Error(1.try_into().expect("1 is non-zero")));
            }
        }
    }

    /// Test if the simulation is paused in `run_if`.
    #[must_use]
    pub fn is_paused(state: Res<UiState>) -> bool {
        state.pause
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

        ui_scale.0 = 0.6;
    }

    /// Add debug text nodes.
    fn setup_debug_text(mut commands: Commands, overlay_root: Single<Entity, With<OverlayRoot>>) {
        commands.spawn((
            Text::default(),
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(Self::UI_OFFSET),
                right: Val::Px(Self::UI_OFFSET),
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

    /// Add the logo.
    fn add_logo(mut commands: Commands, server: Res<AssetServer>) {
        commands.spawn((
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(Self::UI_OFFSET),
                right: Val::Px(Self::UI_OFFSET),
                width: Val::Px(64.0),
                height: Val::Px(64.0),
                ..default()
            },
            ImageNode {
                image: server.load("embedded://hoomd_bevy/logo.png"),
                ..default()
            },
            Logo,
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

            if let Some(fps) = diagnostic.get(&FrameTimeDiagnosticsPlugin::FPS)
                && let Some(value) = fps.smoothed()
            {
                *writer.text(debug_text, 1) = format!(" FPS: {value:.2}\n");
            }
            if let Some(sps) = diagnostic.get(&Self::SPS)
                && let Some(value) = sps.smoothed()
            {
                *writer.text(debug_text, 2) = format!(" SPS: {value:.2}\n");
            }
            *writer.text(debug_text, 3) = format!("Step: {}\n", simulation.step());
        }
    }

    /// Set the time budgeted to advancing the simulation each frame.
    ///
    /// Derive this time from the current monitor refresh rate and the
    /// `frame_budget_fraction` settings.
    #[cfg(not(target_arch = "wasm32"))]
    fn set_frame_budget(
        windows: Query<Entity, With<Window>>,
        settings: Res<Settings>,
        mut frame_budget: ResMut<FrameBudget>,
    ) {
        // adapted from: https://github.com/aevyrie/bevy_framepace/blob/main/src/lib.rs

        let new_frame_budget = match Self::detect_frame_time(windows.iter()) {
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
    fn detect_frame_time(windows: impl Iterator<Item = Entity>) -> Option<Duration> {
        WINIT_WINDOWS.with_borrow(|winit| {
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
        })
    }

    /// Set up the 2D camera.
    fn setup_camera_2d(mut commands: Commands, viewport_height: f32) {
        let projection = Projection::Orthographic(OrthographicProjection {
            scaling_mode: bevy::camera::ScalingMode::FixedVertical { viewport_height },
            ..OrthographicProjection::default_2d()
        });

        commands.spawn((Camera2d, projection));
    }

    /// Set up the 3D camera.
    fn setup_camera_3d(mut commands: Commands, viewport_height: f32) {
        let projection = Projection::Orthographic(OrthographicProjection {
            scaling_mode: bevy::camera::ScalingMode::FixedVertical { viewport_height },
            ..OrthographicProjection::default_3d()
        });

        commands.spawn((Camera3d::default(),
            projection,
            Transform::from_xyz(0.0, 0.0, -viewport_height*2.0).looking_at(Vec3::ZERO, Vec3::Y),
));
    commands.spawn((
        DirectionalLight::default(),
        Transform::from_xyz(-3.0, 3.0, -6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    }

    /// Increase the brightness of the default ambient light.
    fn setup_ambient_light(mut ambient_light: ResMut<GlobalAmbientLight>) {
       ambient_light.brightness = 100.0;
    }

    /// Keyboard controls for the 2d camera.
    ///
    /// `=` resets the camera to the default.
    fn camera_reset_2d(
        mut reset_camera: MessageReader<ResetCamera>,
        camera: Single<(&mut Transform, &mut Projection), With<Camera2d>>,
        mut control: ResMut<CameraControl2d>,
    ) {
        let (mut transform, projection) = camera.into_inner();

        if !reset_camera.is_empty() {
            if let Projection::Orthographic(ref mut orthographic) = *projection.into_inner() {
                orthographic.scale = 1.0;
            }
            control.dragging = false;
            transform.translation = Vec3::default();
        }

        reset_camera.clear();
    }

    /// Quit.
    fn quit(mut quit: MessageReader<Quit>, mut exit: MessageWriter<AppExit>) {
        if !quit.is_empty() {
            exit.write(AppExit::Success);
        }

        quit.clear();
    }

    /// Left click and drag to pan the 2D camera.
    ///
    /// # Panics
    ///
    /// Panics when the 2D camera viewport is invalid.
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
            if buttons.just_pressed(MouseButton::Left)
                && let Some(world_position) = window
                    .cursor_position()
                    .and_then(|cursor| camera.viewport_to_world_2d(global_transform, cursor).ok())
            {
                control.world_position = world_position;
                control.dragging = true;
                return;
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
        mut scroll: MessageReader<MouseWheel>,
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
            let new_scale = (orthographic.scale * (1.0 + delta_zoom)).clamp(
                1.0 / settings.zoom_range.end,
                1.0 / settings.zoom_range.start,
            );
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

    /// Build the plugin.
    ///
    /// [`HoomdBevyPlugin`] does not implement [`Plugin`] and cannot be used with
    /// `add_plugins` so that the `build` method can consume `self`. This allows
    /// `build` to take ownership of the `simulation` field and create the appropriate
    /// Bevy [`Resource`].
    ///
    /// # Panics
    ///
    /// * When `EguiPlugin` is not added before calling `build`.
    pub fn build(self, app: &mut App) {
        representation::disk::build(app);
        representation::ellipse::build(app);
        representation::hyperbolic_disk::build(app);

        embedded_asset!(app, "logo.png");

        let initial_camera = self.initial_settings.camera.clone();

        assert!(app.is_plugin_added::<EguiPlugin>());

        app.add_plugins(FrameTimeDiagnosticsPlugin::default())
            .insert_resource(ClearColor(Self::CLEAR))
            .insert_resource(FrameBudget(Duration::from_millis(9)))
            .insert_resource(self.initial_settings)
            .register_diagnostic(Diagnostic::new(Self::SPS))
            .insert_resource(self.simulation)
            .insert_resource(UiState::default())
            .insert_resource(OptionsWindowState::default())
            .insert_resource(ParametersWindowState(true))
            .add_systems(
                Startup,
                (Self::setup_overlay, Self::setup_debug_text, Self::add_logo).chain(),
            )
            .add_systems(
                Update,
                Self::remove_logo.run_if(once_after_delay(Duration::from_secs(3))),
            )
            .add_systems(Update, Self::step_simulation.in_set(AdvanceSet))
            .add_systems(
                Update,
                Self::advance_simulation.run_if(on_message::<AdvanceSimulation>),
            )
            .add_systems(Update, Self::update_debug_text.after(AdvanceSet))
            .add_systems(EguiPrimaryContextPass, Self::ui_system)
            .add_message::<ResetCamera>()
            .add_message::<AdvanceSimulation>()
            .add_message::<Quit>()
            .add_systems(Update, Self::quit.run_if(on_message::<Quit>));

        match initial_camera {
            InitialCamera::Orthographic2d(initial_viewport_height) => {
                app.add_systems(
                    Update,
                    Self::camera_mouse_pan_control_2d
                        .run_if(
                            input_pressed(MouseButton::Left)
                                .or(input_just_released(MouseButton::Left)),
                        )
                        .in_set(MouseInputSet),
                )
                .add_systems(
                    Update,
                    Self::camera_mouse_zoom_control_2d
                        .run_if(on_message::<MouseWheel>)
                        .in_set(MouseInputSet),
                )
                .add_systems(
                    Update,
                    Self::camera_reset_2d.run_if(on_message::<ResetCamera>),
                )
                .insert_resource(CameraControl2d::default())
                .add_systems(Startup, move |commands: Commands| {
                    Self::setup_camera_2d(commands, initial_viewport_height);
                });
            },
            InitialCamera::Orthographic3d(initial_viewport_height) => {
                app.add_systems(Startup, move |commands: Commands| {
                    Self::setup_camera_3d(commands, initial_viewport_height);
                }).add_systems(Startup, Self::setup_ambient_light);
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        app.add_systems(
            Update,
            Self::set_frame_budget.run_if(on_timer(Duration::from_millis(250))),
        );

        app.configure_sets(
            Update,
            (
                AdvanceSet.run_if(not(Self::is_paused)),
                KeyboardInputSet
                    .after(AdvanceSet)
                    .run_if(not(egui_wants_any_keyboard_input)),
                MouseInputSet
                    .after(AdvanceSet)
                    .run_if(not(egui_wants_any_pointer_input)),
            ),
        );
    }

    /// GUI and keyboard controls
    fn configure_ui(mut contexts: EguiContexts) -> Result {
        let context = contexts.ctx_mut()?;
        context.memory_mut(|m| {
            m.options.theme_preference = egui::ThemePreference::Dark;

            // bevy_egui overrides the egui built-in zoom. Disable it to avoid conflicts.
            m.options.zoom_with_keyboard = false;
        });

        Ok(())
    }

    /// GUI and keyboard controls
    fn ui_system(
        #[cfg(not(target_arch = "wasm32"))] mut commands: Commands,
        mut contexts: EguiContexts,
        mut context_settings: Single<&mut EguiContextSettings>,
        mut ui_state: ResMut<UiState>,
        mut options_window_state: ResMut<OptionsWindowState>,
        mut parameters_window_state: ResMut<ParametersWindowState>,
        mut settings: ResMut<Settings>,
        window: Single<&Window, With<PrimaryWindow>>,
        mut debug_text: Single<&mut Visibility, (With<DebugText>, Without<OverlayRoot>)>,
        #[cfg(not(target_arch = "wasm32"))] mut quit: MessageWriter<Quit>,
        mut reset_camera: MessageWriter<ResetCamera>,
        mut advance_simulation: MessageWriter<AdvanceSimulation>,
    ) -> Result {
        let advance_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::NONE, egui::Key::N);
        let options_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::NONE, egui::Key::M);
        let parameters_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::NONE, egui::Key::P);
        let pause_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::NONE, egui::Key::Space);
        #[cfg(not(target_arch = "wasm32"))]
        let quit_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::NONE, egui::Key::Q);
        let reset_camera_shortcut =
            egui::KeyboardShortcut::new(egui::Modifiers::NONE, egui::Key::Equals);
        let show_debug_shortcut = egui::KeyboardShortcut::new(egui::Modifiers::NONE, egui::Key::F5);
        #[cfg(not(target_arch = "wasm32"))]
        let screenshot_shortcut =
            egui::KeyboardShortcut::new(egui::Modifiers::NONE, egui::Key::F12);

        let default_width = 280.0;

        let window = egui::Window::new("⛭ Options")
            .open(&mut options_window_state.0)
            .resizable([true, false])
            .pivot(egui::Align2::LEFT_BOTTOM)
            .default_pos([
                Self::UI_OFFSET,
                window.resolution.height() - Self::UI_OFFSET,
            ])
            .collapsible(false)
            .default_width(default_width);

        window.show(contexts.ctx_mut()?, |ui| {
            ui.allocate_space(ui.available_width() * egui::vec2(1.0, 0.0));

            egui::CollapsingHeader::new("Simulation controls")
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.toggle_value(&mut ui_state.pause, "⏸ Pause (space)");
                        if ui.button("▶ Advance (n)").clicked() {
                            advance_simulation.write(AdvanceSimulation);
                        }
                    });
                    ui.add(
                        egui::Slider::new(&mut settings.sps_limit, 0.25..=32_768.0)
                            .text("Limit step rate")
                            .update_while_editing(false)
                            .logarithmic(true)
                            .suffix(" Hz"),
                    );
                });

            ui.collapsing("Camera controls", |ui| {
                match settings.camera {
                    InitialCamera::Orthographic2d(_) => {
                        ui.label("Click and drag to move the camera.");
                        ui.label("Scroll to zoom.");
                    },
                    InitialCamera::Orthographic3d(_) => {
                        ui.label("TODO.");
                        ui.label("TODO.");
                    }
                }

                ui.add(
                    egui::Slider::new(&mut settings.camera_sensitivity, 0.1..=1.0)
                        .text("Camera sensitivity")
                        .update_while_editing(false),
                );

                ui.add(
                    egui::Slider::new(&mut settings.zoom_range.end, 2.0..=100.0)
                        .text("Maximum zoom")
                        .update_while_editing(false),
                );

                ui.horizontal(|ui| {
                    if ui.button("↺ Reset (=)").clicked() {
                        reset_camera.write(ResetCamera);
                    }

                    #[cfg(not(target_arch = "wasm32"))]
                    if ui
                        .button("📷 Screenshot (F12)")
                        .on_hover_text("Write screenshot.png to the current working directory")
                        .clicked()
                    {
                        commands
                            .spawn(Screenshot::primary_window())
                            .observe(save_to_disk("screenshot.png"));
                    }
                });
            });

            ui.collapsing("More keyboard shortcuts", |ui| {
                egui::Grid::new("some_unique_id").show(ui, |ui| {
                    ui.label("m");
                    ui.label("Show/hide options");
                    ui.end_row();

                    ui.label(ui.ctx().format_shortcut(&ZOOM_IN));
                    ui.label("Zoom UI in");
                    ui.end_row();

                    ui.label(ui.ctx().format_shortcut(&ZOOM_OUT));
                    ui.label("Zoom UI out");
                    ui.end_row();

                    ui.label(ui.ctx().format_shortcut(&ZOOM_RESET));
                    ui.label("Reset UI zoom");
                    ui.end_row();
                });
            });

            ui.collapsing("Advanced settings", |ui| {
                ui.checkbox(&mut parameters_window_state.0, "Show parameters (p)");
                ui.checkbox(&mut ui_state.show_debug, "Show debug overlay (F5)");

                ui.add(
                    egui::Slider::new(&mut settings.frame_budget_fraction, 0.1..=0.9)
                        .text("Simulation fraction")
                        .update_while_editing(false),
                )
                .on_hover_text("Decrease this when FPS is limited by rendering");
            });

            #[cfg(not(target_arch = "wasm32"))]
            if ui.button("⊗ Quit (q)").clicked() {
                // Sending AppExit messages in this system causes deadlocks.
                // Send a quit message that defers AppExit until later.
                quit.write(Quit);
            }
        });

        {
            let context = contexts.ctx_mut()?;
            if !context.wants_keyboard_input() {
                if context.input_mut(|i| i.consume_shortcut(&advance_shortcut)) {
                    advance_simulation.write(AdvanceSimulation);
                }
                if context.input_mut(|i| i.consume_shortcut(&options_shortcut)) {
                    options_window_state.0 = !options_window_state.0;
                }
                if context.input_mut(|i| i.consume_shortcut(&parameters_shortcut)) {
                    parameters_window_state.0 = !parameters_window_state.0;
                }
                if context.input_mut(|i| i.consume_shortcut(&pause_shortcut)) {
                    ui_state.pause = !ui_state.pause;
                }
                if context.input_mut(|i| i.consume_shortcut(&show_debug_shortcut)) {
                    ui_state.show_debug = !ui_state.show_debug;
                }
                if context.input_mut(|i| i.consume_shortcut(&reset_camera_shortcut)) {
                    reset_camera.write(ResetCamera);
                }

                #[cfg(not(target_arch = "wasm32"))]
                if context.input_mut(|i| i.consume_shortcut(&quit_shortcut)) {
                    quit.write(Quit);
                }
                #[cfg(not(target_arch = "wasm32"))]
                if context.input_mut(|i| i.consume_shortcut(&screenshot_shortcut)) {
                    commands
                        .spawn(Screenshot::primary_window())
                        .observe(save_to_disk("screenshot.png"));
                }

                if context.input_mut(|i| i.consume_shortcut(&ZOOM_IN)) {
                    context_settings.scale_factor *= 1.125;
                }
                if context.input_mut(|i| i.consume_shortcut(&ZOOM_IN_SECONDARY)) {
                    context_settings.scale_factor *= 1.125;
                }
                if context.input_mut(|i| i.consume_shortcut(&ZOOM_OUT)) {
                    context_settings.scale_factor /= 1.125;
                }
                if context.input_mut(|i| i.consume_shortcut(&ZOOM_RESET)) {
                    context_settings.scale_factor = 1.0;
                }
            }
        }

        if **debug_text == Visibility::Hidden && ui_state.show_debug {
            debug_text.toggle_inherited_hidden();
        }
        if **debug_text != Visibility::Hidden && !ui_state.show_debug {
            debug_text.toggle_inherited_hidden();
        }

        // Ideally this would be called in a Startup schedule, but the egui context
        // doesn't exist at that point.
        Self::configure_ui(contexts)?;
        Ok(())
    }
}

/// Construct the default plugins.
///
/// This helper adds Bevy's `DefaultPlugins` by default. When the
/// `doc-example` feature is enabled, it adds a modified set of plugins
/// for the web.
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

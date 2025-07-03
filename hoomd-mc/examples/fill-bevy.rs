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
use std::time::{Duration, Instant};

const FRAME_BUDGET: Duration = Duration::from_millis(14);

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
    ))
    .insert_resource(ClearColor(Color::oklch(0.3, 0.0, 0.0)))
    .insert_resource(setup_simulation().context("failed to setup simulation")?)
    .insert_resource(Time::<Fixed>::from_hz(10_000.0))
    .add_systems(Startup, setup_scene)
    .add_systems(Update, (step_simulation_system, sync_simulation).chain());
    
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
    mut exit: EventWriter<AppExit>,
    simulation: ResMut<Simulation>) {

    let simulation = simulation.into_inner();
    let time = Instant::now();

    while time.elapsed() < FRAME_BUDGET {
        let result = step_simulation(simulation).with_context(|| format!("failed at step: {}", simulation.microstate.step()));
        if let Err(error) = result {
            error!("{error:?}");
            exit.write(AppExit::Error(1.try_into().expect("1 is non-zero")));
            break;
            }
        }
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

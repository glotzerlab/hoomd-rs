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
use hoomd_bevy::{AdvanceSet, HoomdBevyPlugin, Simulation, Settings};

use anyhow::Context;
use bevy::prelude::*;


// TODO: const background color
// TODO: const margin

fn main() -> anyhow::Result<()> {
    let simulation = Fill::new().context("failed to setup simulation")?;
    let hoomd_bevy_plugin = HoomdBevyPlugin { initial_settings: Settings::default(),
    simulation };

    
    let mut app = App::new();
    app.add_plugins(DefaultPlugins);
    hoomd_bevy_plugin.build(&mut app);
    app.add_systems(Startup, setup_disk)
    .add_systems(Update, sync_simulation.run_if(resource_changed::<Fill>).after(AdvanceSet))
    ;

    app.run();

    Ok(())
}

/// The HOOMD simulation
#[derive(Resource)]
struct Fill {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Square>,
    /// How sites interact with other sites and fields.
    hamiltonian: (CutoffPair<Isotropic<Boxcar>>, Single<Linear<Cartesian<2>>>),
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate>,
    /// Temperature set point.
    kt: f64
}

impl Fill {

/// Set up the hoomd simulation
fn new() -> anyhow::Result<Fill> {
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

    Ok(Fill { microstate, hamiltonian, translate_sweep, kt })
}
}

impl Simulation for Fill {

/// Advance the simulation forward one step.
fn advance(&mut self) -> anyhow::Result<()> {

    if self.microstate.step() % 100 == 0 {
        self.microstate.add_body(Body::point([0.0, self.microstate.boundary().l.get() / 2.0 - 0.5].into()))?;
    }

    self.translate_sweep.apply(&mut self.microstate, &self.hamiltonian, &self.kt);
    self.microstate.increment_step();
    Ok(())
    }

/// Get the current simulation step.
fn step(&self) -> u64 {
    self.microstate.step()
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

/// Copy the current positions of simulation particles to bevy entities.
fn sync_simulation(
    mut commands: Commands,
    disk: Res<Disk>,
    simulation: Res<Fill>,
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

fn setup_disk(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>, 
    mut materials: ResMut<Assets<ColorMaterial>>,
    ) {
    let mesh = meshes.add(Circle::new(0.5));
    let color = materials.add(Color::oklch(0.64, 0.14, 256.71));
    commands.insert_resource(Disk { mesh, color });
    }

use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, Settings, Simulation,
    representation::{Disk, DiskAssets},
};
use hoomd_interaction::{
    CutoffPair, Single,
    external::Linear,
    pairwise::{Boxcar, Isotropic},
};
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_microstate::{
    Body, Microstate, MicrostateBuilder, Site, boundary::Square, property::Point,
};
use hoomd_vector::Cartesian;

use anyhow::Context;
use bevy::prelude::*;

// TODO: Reset button?

fn main() -> anyhow::Result<()> {
    let simulation = Fill::new().context("failed to setup simulation")?;
    let l = simulation.microstate.boundary().l.get() as f32;
    let hoomd_bevy_plugin = HoomdBevyPlugin {
        initial_settings: Settings { viewport_height: l + 1.0, ..default()},
        simulation,
    };

    let mut app = App::new();
    app.add_plugins(DefaultPlugins);
    hoomd_bevy_plugin.build(&mut app);
    app.add_systems(Startup, Disk::setup);
    app.add_systems(
        Update,
        sync_simulation
            .run_if(resource_changed::<Fill>)
            .after(AdvanceSet),
    );

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
    kt: f64,
}

impl Fill {
    /// Set up the hoomd simulation
    fn new() -> anyhow::Result<Fill> {
        let box_height = 30.0;
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

        Ok(Fill {
            microstate,
            hamiltonian,
            translate_sweep,
            kt,
        })
    }
}

impl Simulation for Fill {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        if self.microstate.step() % 100 == 0 {
            self.microstate.add_body(Body::point(
                [0.0, self.microstate.boundary().l.get() / 2.0 - 0.5].into(),
            ))?;
        }

        self.translate_sweep
            .apply(&mut self.microstate, &self.hamiltonian, &self.kt);
        self.microstate.increment_step();
        Ok(())
    }

    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

/// Display the simulation box
fn add_box(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    )
    {
    
    }
    

/// Copy the current positions of simulation particles to bevy entities.
fn sync_simulation(
    mut commands: Commands,
    disk_assets: Res<DiskAssets>,
    query: Query<(Entity, &mut Transform), With<Disk>>,
    simulation: Res<Fill>,
) {
    let sites = simulation.microstate.sites();
    Disk::sync(
        &mut commands,
        disk_assets,
        query,
        sites,
        |site: &Site<Point<Cartesian<2>>>| -> Vec3 {
            Vec3::new(
                site.properties.position[0] as f32,
                site.properties.position[1] as f32,
                0.0,
            )
        },
        |_: &Site<Point<Cartesian<2>>>| -> f32 { 1.0f32 },
    );
}

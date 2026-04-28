//! A simulation with a single particle that is a dumbbell subjected to an external
//! constant torque

use hoomd_geometry::shape::Rectangle;
use hoomd_interaction::{
    External,
    external::ConstantTorque,
    rigid::Rigid,
};
use hoomd_md::{
    methods::{
        ConstantVolume,
        ForceAndTorqueUpdate,
        RotationalMotion,
        TranslationalMotion,
    },
    thermostat::NoThermostat,
};
use hoomd_microstate::{
    Body, Microstate, SiteKey, boundary::Periodic, property::{OrientedDynamicsPoint, Point}
};
use hoomd_simulation::Simulation;
use hoomd_spatial::AllPairs;
use hoomd_vector::{Angle, Cartesian};

use bevy_egui::EguiPlugin;
use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, InitialCamera, Settings,
    representation::RectangularBoundary,
    representation::disk::{self, Disk},
};

use anyhow::Context;
use bevy::prelude::*;

/// Mark the disk representation type
struct A;

struct Isoenergy {}

/// The state of the swimming simulation, tracked as a resource by Bevy
#[derive(Resource)]
struct Dumbbell {
    microstate: Microstate<
        OrientedDynamicsPoint<Cartesian<2>, Angle>,
        Point<Cartesian<2>>,
        AllPairs<SiteKey>,
        Periodic<Rectangle>,
    >,

    macrostate: Isoenergy,

    thermostat: NoThermostat,

    force: Rigid<External<ConstantTorque<Cartesian<2>>>>,

    integrator: ConstantVolume,
}

impl Dumbbell {
    /// Construct a new swimming simulation.
    fn new() -> anyhow::Result<Dumbbell> {
        let box_length = 30.0;

        let square = Rectangle::with_equal_edges(box_length.try_into()?);
        // let boundary = Closed(square);
        let boundary = Periodic::new(2.5, square)?;
        let mut microstate = Microstate::builder().boundary(boundary).try_build()?;

        let dumbbell_body = Body {
            properties: OrientedDynamicsPoint {
                position: Cartesian::from([0.0, 0.0]),
                momentum: Cartesian::from([0.0, 0.0]),
                net_force: Cartesian::from([0.0, 0.0]),
                mass: 1.0,
                orientation: Angle::default(),
                moment_of_inertia: 1.0,
                angular_momentum: 0.0,
                net_torque: 0.0,
            },
            sites: vec![
                Point::new(Cartesian::from([-3.0, 0.0])),
                Point::new(Cartesian::from([3.0, 0.0])),
            ],
        };
        microstate.add_body(dumbbell_body)?;

        // Model interactions (in this case, a pairwise Lennard-Jones)
        let force = Rigid(External(ConstantTorque {
            torque: 0.001,
        }));

        // Create an NVE macrostate
        let macrostate = Isoenergy {};

        // Create a constant-volume integrator
        let dt = 0.01;
        let integrator = ConstantVolume::new(dt);

        // Constant V integration requires a thermostat, even if it does nothing
        let thermostat = NoThermostat;

        Ok(Dumbbell {
            microstate,
            macrostate,
            thermostat,
            force,
            integrator,
        })
    }
}

impl Simulation for Dumbbell {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        // Evolve the system forward using the integrator
        self.integrator.integrate_translation_step_one(
            &mut self.microstate,
            &mut self.thermostat,
            &self.macrostate,
        );

        self.integrator.integrate_rotation_step_one(
            &mut self.microstate,
            &mut self.thermostat,
            &self.macrostate,
        );

        self.integrator
            .update_force_and_torque(&mut self.microstate, &self.force);

        self.integrator.integrate_translation_step_two(
            &mut self.microstate,
            &mut self.thermostat,
            &self.macrostate,
        );

        self.integrator.integrate_rotation_step_two(
            &mut self.microstate,
            &mut self.thermostat,
            &self.macrostate,
        );

        Ok(())
    }

    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

fn main() -> anyhow::Result<()> {
    let simulation = Dumbbell::new().context("failed to setup simulation")?;
    // let l = simulation.microstate.boundary().0.edge_lengths[1].get() as f32;
    let l = simulation.microstate.boundary().shape().edge_lengths[1].get() as f32;
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
        (
            //move_swimmer,
            sync_simulation
                .run_if(resource_changed::<Dumbbell>)
                .after(AdvanceSet),
        )
            .chain(),
    );

    app.run();

    Ok(())
}

/// Copy the current positions of simulation particles to bevy entities.
fn sync_simulation(
    mut commands: Commands,
    disk_representation: Res<disk::Representation<A>>,
    query: Query<(Entity, &mut Transform), With<Disk<A>>>,
    simulation: Res<Dumbbell>,
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

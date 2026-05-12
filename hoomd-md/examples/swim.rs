//! A simulation with a single particle swimming through a Lennard-Jones fluid.

use hoomd_geometry::shape::Rectangle;
use hoomd_interaction::{
    pairwise::Isotropic, univariate::LennardJones, Rigid, PairwiseCutoff
};
use hoomd_md::{
    methods::{
        ConstantVolume,
        ForceUpdate,
        TranslationalMotion,
    },
    thermostat::NoThermostat
};
use hoomd_microstate::{
    Body, Microstate, SiteKey, boundary::Periodic, property::{DynamicPoint, Momentum, Point}
};
use hoomd_simulation::{Simulation};
use hoomd_spatial::AllPairs;
use hoomd_vector::Cartesian;

use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, InitialCamera, Settings,
    representation::RectangularBoundary,
    representation::disk::{self, Disk},
};
use bevy_egui::EguiPlugin;

use anyhow::Context;
use bevy::prelude::*;

/// Mark the disk representation type
struct A;

struct Isoenergy {}

/// The state of the swimming simulation, tracked as a resource by Bevy
#[derive(Resource)]
struct Swim {
    // microstate: Microstate<DynamicPoint<Cartesian<2>>, Point<Cartesian<2>>, Closed<Rectangle>>,
    microstate: Microstate<
        DynamicPoint<Cartesian<2>>,
        Point<Cartesian<2>>,
        AllPairs<SiteKey>,
        Periodic<Rectangle>
    >,

    macrostate: Isoenergy,
    
    thermostat: NoThermostat,

    force: Rigid<PairwiseCutoff<Isotropic<LennardJones<12, 6>>>>,

    integrator: ConstantVolume,
}

impl Swim {
    /// Construct a new swimming simulation.
    fn new() -> anyhow::Result<Swim> {
        let box_length = 30.0;

        // Create a microstate with a grid of bodies and a swimmer (final body)
        let square = Rectangle::with_equal_edges(box_length.try_into()?);
        // let boundary = Closed(square);
        let boundary = Periodic::new(2.5, square)?;
        let mut builder = Microstate::builder().boundary(boundary);

        let (n_rows, n_columns) = (5, 5);
        let space = 3.0;

        for i in 0..n_rows {
            for j in 0..n_columns {
                let x = space * f64::from(i + 1) - (f64::from(1 + n_columns) * space / 2.0);
                let y = space * f64::from(j + 1) - (f64::from(1 + n_rows) * space / 2.0);
                builder = builder.bodies([
                    Body {
                        properties: DynamicPoint {
                            position: Cartesian::from([x, y]),
                            momentum: Cartesian::from([0.0, 0.0]),
                            net_force: Cartesian::from([0.0, 0.0]),
                            mass: 1.0
                        },
                        sites: vec![Point::default()]
                    }
                ]);
            }
        }

        let swimmer_x = 0.0;
        let swimmer_y = - (box_length / 2.0) * (4.0 / 5.0);
        builder = builder.bodies([
            Body {
                properties: DynamicPoint {
                    position: Cartesian::from([swimmer_x, swimmer_y]),
                    momentum: Cartesian::from([0.0, 0.0]),
                    net_force: Cartesian::from([0.0, 0.0]),
                    mass: 1.0
                },
                sites: vec![Point::default()]
            }
        ]);

        let microstate = builder.try_build()?;
     
        // Model interactions (in this case, a pairwise Lennard-Jones)
        let force = Rigid(
            PairwiseCutoff(
                Isotropic {
                    interaction: LennardJones::<12,6> {
                        epsilon: 0.01,
                        sigma: 1.0
                    },
                    r_cut: 6.0,
                }
            )
        );
    
        // Create an NVE macrostate
        let macrostate = Isoenergy{};

        // Create a constant-volume integrator
        let dt = 0.01;
        let integrator = ConstantVolume::new(dt);

        // Constant V integration requires a thermostat, even if it does nothing
        let thermostat = NoThermostat;
    
        Ok(Swim {
            microstate,
            macrostate,
            thermostat,
            force,
            integrator
        })
    }
}

impl Simulation for Swim {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {        
        // Evolve the system forward using the integrator
        self.integrator.integrate_translation_step_one(
            &mut self.microstate,
            &mut self.thermostat,
            &self.macrostate,
        );

        self.integrator
            .update_force(&mut self.microstate, &self.force);

        self.integrator.integrate_translation_step_two(
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
    let simulation = Swim::new().context("failed to setup simulation")?;
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
            move_swimmer,
            sync_simulation.run_if(resource_changed::<Swim>).after(AdvanceSet),
        ).chain(),
    );

    app.run();

    Ok(())
}

/// Copy the current positions of simulation particles to bevy entities.
fn sync_simulation(
    mut commands: Commands,
    disk_representation: Res<disk::Representation<A>>,
    query: Query<(Entity, &mut Transform), With<Disk<A>>>,
    simulation: Res<Swim>,
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

/// Move the swimmer
fn move_swimmer(
    mut simulation: ResMut<Swim>,
    kb_input: Res<ButtonInput<KeyCode>>,
) {
    // Configure the movement speed
    let dp_x = 0.005;
    let dp_y = 0.005;

    // Clone the swimmer
    let swimmer_index = simulation.microstate.bodies().len() - 1;
    let mut swimmer_body_properties = simulation
        .microstate
        .bodies()[swimmer_index]
        .item
        .properties
        .clone();

    if kb_input.pressed(KeyCode::KeyW) {
        *swimmer_body_properties.momentum_mut() += Cartesian::from([0.0, dp_y]);
    }
    if kb_input.pressed(KeyCode::KeyS) {
        *swimmer_body_properties.momentum_mut() -= Cartesian::from([0.0, dp_y]);
    }
    if kb_input.pressed(KeyCode::KeyD) {
        *swimmer_body_properties.momentum_mut() += Cartesian::from([dp_x, 0.0]);
    }
    if kb_input.pressed(KeyCode::KeyA) {
        *swimmer_body_properties.momentum_mut() -= Cartesian::from([dp_x, 0.0]);
    }

    #[allow(unused_must_use)]
    simulation.microstate.update_body_properties(swimmer_index, swimmer_body_properties);
}
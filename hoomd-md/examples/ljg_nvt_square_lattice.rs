//! A simulation with a single particle swimming through a Lennard-Jones fluid.

use bevy_egui::EguiPlugin;
use hoomd_geometry::shape::Rectangle;
use hoomd_interaction::{
    CutoffPair, NetBodyForce, TotalEnergy,
    pairwise::{Isotropic, LennardJonesGauss},
    rigid::Rigid,
};
use hoomd_md::{
    ConstantVolume, ForceUpdate, TranslationalMotion, thermalize::{
        ComAngularMomentumRemover, ComMomentumRemover, RotationalThermalizer, Thermalizer,
        TranslationalAngularMomentumModifier, TranslationalMomentumModifier,
        TranslationalThermalizer,
    }, thermostat::BussiThermostat
};
use hoomd_microstate::{
    Body, Microstate, MicrostateBuilder,
    boundary::Periodic,
    property::{DynamicsPoint, NetForce, Point, Position},
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_vector::{Cartesian, Metric};

use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, InitialCamera, Settings,
    representation::RectangularBoundary,
    representation::disk::{self, Disk},
};

use anyhow::Context;
use bevy::prelude::*;

/// Mark the disk representation type
struct A;

/// The state of the swimming simulation, tracked as a resource by Bevy
#[derive(Resource)]
struct LJG_sqaure {
    // microstate: Microstate<DynamicsPoint<Cartesian<2>>, Point<Cartesian<2>>, Closed<Rectangle>>,
    microstate: Microstate<DynamicsPoint<Cartesian<2>>, Point<Cartesian<2>>, Periodic<Rectangle>>,

    macrostate: Isothermal,

    thermostat: BussiThermostat,

    force: Rigid<CutoffPair<Isotropic<LennardJonesGauss>>>,

    integrator: ConstantVolume,
}

impl LJG_sqaure {
    /// Construct a new swimming simulation.
    fn new() -> anyhow::Result<LJG_sqaure> {
        let box_length = 16.0;
        let kT_init = 0.15;

        // LJG potential
        let force = Rigid(CutoffPair {
            r_cut: 3.0,
            evaluator: Isotropic(LennardJonesGauss {
                epsilon: 0.75,
                sigma_squared: 0.02,
                r_0: 1.41,
                scale: 1.0,
            }),
        });

        // Create a microstate with a grid of bodies and a swimmer (final body)
        let square = Rectangle::with_equal_edges(box_length.try_into()?);
        // let boundary = Closed(square);
        let boundary = Periodic::new(2.5, square)?;
        let mut builder = MicrostateBuilder::with_boundary(boundary);

        let (n_rows, n_columns) = (10, 10);
        let space = 1.0;

        for i in 0..n_rows {
            for j in 0..n_columns {
                let x = space * f64::from(i + 1) - (f64::from(1 + n_columns) * space / 2.0);
                let y = space * f64::from(j + 1) - (f64::from(1 + n_rows) * space / 2.0);
                builder = builder.bodies([Body {
                    properties: DynamicsPoint {
                        position: Cartesian::from([x, y]),
                        momentum: Cartesian::from([0.0, 0.0]),
                        net_force: Cartesian::from([0.0, 0.0]),
                        mass: 1.0,
                    },
                    sites: vec![Point::default()],
                }]);
            }
        }

        let mut microstate = builder.try_build()?;

        // Randomize the momenta of system.
        let thermalizer = Thermalizer { kT: kT_init };
        thermalizer.thermalize_translation(&mut microstate);

        // Remove com momentum and angular momentum afterwards.
        let angular_remover = ComAngularMomentumRemover {};
        let linear_remover = ComMomentumRemover {};
        angular_remover.modify(&mut microstate);
        linear_remover.modify(&mut microstate);

        // Store net body force at t=0
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            // Calculate the net force on the body
            let net_force_new = force.net_force_on_body(&microstate, body_index);

            // Calculate force at t=0
            *body_properties.net_force_mut() = net_force_new;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }

        // Create an NVT macrostate
        let macrostate = Isothermal {
            temperature: kT_init,
        };

        // Create a constant-volume integrator
        let dt = 0.005;
        let tau = 50.0 * dt;
        let integrator = ConstantVolume::new(dt);

        // Constant T integration
        let thermostat = BussiThermostat::new(tau);

        Ok(LJG_sqaure {
            microstate,
            macrostate,
            thermostat,
            force,
            integrator,
        })
    }
}

impl Simulation for LJG_sqaure {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        // Evolve the system forward using the integrator
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
        self.microstate.increment_step();

        if self.step() % 10000 == 1 {
            let ke = self.integrator.get_translational_kinetic_energy();
            let dof = self.integrator.get_translational_dof();
            let kT = 2.0 / dof * ke;

            let pe = self.force.0.total_energy(&self.microstate);
            let thermal_e = self.thermostat.get_energy();
            let hamiltonian = *ke + pe + *thermal_e;

            println!(
                "Step: {:}, kT: {:.4}, PE: {:.4}, H: {:.4}, thermostat: {:.4}",
                self.step(),
                kT,
                pe,
                hamiltonian,
                thermal_e
            );
        }

        Ok(())
    }

    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

fn main() -> anyhow::Result<()> {
    let mut simulation = LJG_sqaure::new().context("failed to setup simulation")?;
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
                .run_if(resource_changed::<LJG_sqaure>)
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
    simulation: Res<LJG_sqaure>,
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

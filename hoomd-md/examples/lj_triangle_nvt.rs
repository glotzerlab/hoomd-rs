//! A simulation with a single particle

use hoomd_geometry::shape::{Hypercuboid, Rectangle};
use hoomd_interaction::{
    CutoffPair, TotalEnergy,
    pairwise::{Isotropic, LennardJones, WeeksChandlerAnderson},
    rigid::Rigid,
};
use hoomd_md::{
    ConstantVolume, ForceAndTorqueUpdate, ForceUpdate, RotationalMotion, TranslationalMotion,
    thermalize::{
        RotationalModifier, Thermalize, TranslationalAngularMomentumModifier, TranslationalModifier,
    },
    thermostat::{BussiThermostat, MTTKThermostat, NoThermostat},
};
use hoomd_microstate::{
    Body, Microstate, MicrostateBuilder,
    boundary::{Closed, Open, Periodic},
    property::{DynamicsPoint, Momentum, OrientedDynamicsPoint, Point, Position},
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_vector::{Angle, Cartesian, Quaternion, Versor};

use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
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
struct System {
    microstate: Microstate<
        OrientedDynamicsPoint<Cartesian<2>, Angle>,
        Point<Cartesian<2>>,
        Periodic<Rectangle>,
    >,

    macrostate: Isothermal,

    thermostat: (BussiThermostat, BussiThermostat),

    force: Rigid<CutoffPair<Isotropic<LennardJones>>>,

    integrator: ConstantVolume,
}

impl System {
    /// Construct a new swimming simulation.
    fn new() -> anyhow::Result<System> {
        let kT_init = 0.2;
        let box_length = 40.0;

        let square = Rectangle::with_equal_edges(box_length.try_into()?);
        let boundary = Periodic::new(2.5, square)?;
        let mut builder = MicrostateBuilder::with_boundary(boundary);

        let (nx, ny) = (10, 10);
        let space = 2.1;

        for i in 0..nx {
            for j in 0..ny {
                let x = space * f64::from(i + 1) - (f64::from(1 + nx) * space / 2.0);
                let y = space * f64::from(j + 1) - (f64::from(1 + ny) * space / 2.0);
                builder = builder.bodies([Body {
                    properties: OrientedDynamicsPoint {
                        position: Cartesian::from([x, y]),
                        momentum: Cartesian::from([0.0, 0.0]),
                        net_force: Cartesian::from([0.0, 0.0]),
                        mass: 3.0,
                        orientation: Angle::default(),
                        moment_of_inertia: 1.0,
                        angular_momentum: 0.0,
                        net_torque: 0.0,
                    },
                    sites: vec![
                        Point::new(Cartesian::from([3.0_f64.sqrt() / 3.0, 0.0])),
                        Point::new(Cartesian::from([-(3.0_f64.sqrt()) / 6.0, 0.5])),
                        Point::new(Cartesian::from([-(3.0_f64.sqrt()) / 6.0, -0.5])),
                    ],
                }]);
            }
        }

        let mut microstate = builder.try_build()?;

        // Model interactions (in this case, a pairwise Lennard-Jones)
        let force = Rigid(CutoffPair {
            r_cut: 6.0,
            evaluator: Isotropic(LennardJones {
                epsilon: 1.0,
                sigma: 1.0,
            }),
        });

        // Randomize momenta of the whole system.
        // Remove com momentum anf angular momentum afterwards.
        let thermalizer = Thermalize { kT: kT_init };
        thermalizer.thermalize_translation(&mut microstate);
        thermalizer.thermalize_rotation(&mut microstate);
        thermalizer.remove_com_angular_momentum(&mut microstate);
        thermalizer.remove_com_momentum(&mut microstate);

        // Create an NVT macrostate
        let macrostate = Isothermal {
            temperature: kT_init,
        };

        // Create a constant-volume integrator
        let dt = 0.0025;
        let integrator = ConstantVolume::new(dt);

        // NVT simulation,
        // Notice that the thermostats for translational
        // and rotational dof are separated.
        let tau = 50.0 * dt;
        let thermostat = (BussiThermostat::new(tau), BussiThermostat::new(tau));

        Ok(System {
            microstate,
            macrostate,
            thermostat,
            force,
            integrator,
        })
    }
}

impl Simulation for System {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        // Evolve the system forward using the integrator
        self.integrator.integrate_translation_step_one(
            &mut self.microstate,
            &mut self.thermostat.0,
            &self.macrostate,
        );

        self.integrator.integrate_rotation_step_one(
            &mut self.microstate,
            &mut self.thermostat.1,
            &self.macrostate,
        );

        self.integrator
            .update_force_and_torque(&mut self.microstate, &self.force);

        self.integrator.integrate_translation_step_two(
            &mut self.microstate,
            &mut self.thermostat.0,
            &self.macrostate,
        );

        self.integrator.integrate_rotation_step_two(
            &mut self.microstate,
            &mut self.thermostat.1,
            &self.macrostate,
        );

        self.microstate.increment_step();
        if self.step() % 10000 == 1 {
            println!("==============={:}===============", self.step());
            let KE_t = self.integrator.get_translational_kinetic_energy();
            let KE_r = self.integrator.get_rotational_kinetic_energy();
            let dof_t = self.integrator.get_translational_dof();
            let dof_r = self.integrator.get_rotational_dof();
            let resorvoir_e_t = self.thermostat.0.get_energy();
            let resorvoir_e_r = self.thermostat.1.get_energy();

            let kT = 2.0 * (KE_t + KE_r) / (dof_t + dof_r);
            let pe = self.force.0.total_energy(&self.microstate);

            let total_energy = KE_t + KE_r + pe;
            let h = total_energy + resorvoir_e_t + resorvoir_e_r;

            println!("Temperature: {:}", kT);
            println!("Total energy: {:}", total_energy);
            println!("Hamiltonian: {:} \n", h);
        }

        Ok(())
    }

    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

fn main() -> anyhow::Result<()> {
    let simulation = System::new().context("failed to setup simulation")?;
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
        (sync_simulation
            .run_if(resource_changed::<System>)
            .after(AdvanceSet),)
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
    simulation: Res<System>,
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

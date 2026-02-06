//! Simple example of three-dimensional rods simualtion with MD.

#![allow(non_snake_case)]
#![allow(unused_must_use)]

use hoomd_geometry::shape::Hypercuboid;
use hoomd_interaction::{
    CutoffPair, TotalEnergy,
    pairwise::{Isotropic, LennardJones},
    rigid::Rigid,
};
use hoomd_md::{
    methods::{
        ConstantVolume,
        ForceAndTorqueUpdate,
        RotationalMotion,
        TranslationalMotion,
    },
    thermalizer::{
        ComAngularMomentumRemover, ComMomentumRemover, RotationalThermalizer, Thermalizer,
        TranslationalMomentumModifier,
        TranslationalThermalizer,
    },
    thermostat::{BussiThermostat},
};
use hoomd_microstate::{
    Body, Microstate, MicrostateBuilder,
    boundary::Periodic,
    property::{OrientedDynamicsPoint, Point},
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, Versor};

use anyhow::Context;

/// The state of the swimming simulation, tracked as a resource by Bevy
struct System {
    microstate: Microstate<
        OrientedDynamicsPoint<Cartesian<3>, Versor>,
        Point<Cartesian<3>>,
        Periodic<Hypercuboid<3>>,
    >,

    macrostate: Isothermal,

    thermostat: (BussiThermostat, BussiThermostat),

    force: Rigid<CutoffPair<Isotropic<LennardJones>>>,

    integrator: ConstantVolume,
}

impl System {
    /// Construct a new swimming simulation.
    fn new() -> anyhow::Result<System> {
        let kT_init = 0.8;
        let box_length = 40.0;

        let cube = Hypercuboid::<3>::with_equal_edges(box_length.try_into()?);
        let boundary = Periodic::new(6.0, cube)?;
        let mut builder = MicrostateBuilder::with_boundary(boundary);

        let (nx, ny, nz) = (5, 5, 5);
        let space = 2.1;

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = space * f64::from(i + 1) - (f64::from(1 + nx) * space / 2.0);
                    let y = space * f64::from(j + 1) - (f64::from(1 + ny) * space / 2.0);
                    let z = space * f64::from(k + 1) - (f64::from(1 + nz) * space / 2.0);
                    builder = builder.bodies([Body {
                        properties: OrientedDynamicsPoint {
                            position: Cartesian::from([x, y, z]),
                            momentum: Cartesian::default(),
                            net_force: Cartesian::default(),
                            mass: 2.0,
                            orientation: Versor::default(),
                            moment_of_inertia: Cartesian::from([0.5, 0.5, 0.0]),
                            angular_momentum: Cartesian::default(),
                            net_torque: Cartesian::default(),
                        },
                        sites: vec![
                            Point::new(Cartesian::from([0.0, 0.0, 0.5])),
                            Point::new(Cartesian::from([0.0, 0.0, -0.5])),
                        ],
                    }]);
                }
            }
        }

        let mut microstate = builder.try_build()?;

        // Model interactions (in this case, a pairwise Lennard-Jones)
        let force = Rigid(CutoffPair {
            r_cut: 6.0, // 2.0_f64.powf(1.0/6.0),
            evaluator: Isotropic(LennardJones {
                epsilon: 0.5,
                sigma: 1.0,
            }),
        });

        // Randomize the momenta of system.
        let thermalizer = Thermalizer { kT: kT_init };
        thermalizer.thermalize_translation(&mut microstate);
        thermalizer.thermalize_rotation(&mut microstate);

        // Remove com momentum and angular momentum afterwards.
        let angular_remover = ComAngularMomentumRemover {};
        let linear_remover = ComMomentumRemover {};
        angular_remover.modify(&mut microstate);
        linear_remover.modify(&mut microstate);

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
        let tau = PositiveReal::try_from(50.0 * dt)?;
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
    let mut simulation = System::new().context("failed to setup simulation")?;

    for _ in 0..200_000 {
        simulation.advance();
    }

    Ok(())
}

//! Example of three-dimensional LJ fluid copied from the lj_fluid.py in
//! hoomd-validation repo, using second param_list (line 62 - 69 in lj_fluid.py).

#![allow(non_snake_case)]
#![allow(unused_must_use)]

use hoomd_geometry::shape::Hypercuboid;
use hoomd_interaction::{
    PairwiseCutoff, TotalEnergy,
    pairwise::Isotropic, univariate::{LennardJones, Xplor},
    Rigid,
};
use hoomd_md::{
    methods::{
        ConstantVolume,
        ForceUpdate,
        TranslationalMotion,
    },
    thermalizer::{
        ComAngularMomentumRemover, ComMomentumRemover, Thermalizer,
        TranslationalMomentumModifier,
        TranslationalThermalizer,
    }, thermostat::NoThermostat
};
use hoomd_microstate::{
    Body, Microstate, SiteKey, boundary::Periodic, property::{DynamicPoint, Point}
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::AllPairs;
use hoomd_vector::Cartesian;

use anyhow::Context;

/// The state of the swimming simulation, tracked as a resource by Bevy
struct System {
    microstate: Microstate<
        DynamicPoint<Cartesian<3>>,
        Point<Cartesian<3>>,
        AllPairs<SiteKey>,
        Periodic<Hypercuboid<3>>
    >,

    macrostate: Isothermal,

    thermostat: NoThermostat,

    force: Rigid<PairwiseCutoff<Isotropic<Xplor<LennardJones>>>>,

    integrator: ConstantVolume,
}

impl System {
    /// Construct a new swimming simulation.
    fn new() -> anyhow::Result<System> {
        let kT_init = 1.0;
        let density = 0.9193740949934834;
        let n: f64 = 12.0;
        let box_volume = n.powi(3) / density;
        let box_length = box_volume.cbrt();

        let cube = Hypercuboid::<3>::with_equal_edges(box_length.try_into()?);
        let boundary = Periodic::new(6.0, cube)?;
        let mut builder = Microstate::builder().boundary(boundary);

        let space = box_length / n;
        assert!(
            space > 1.0,
            "Density too high to initialize on cubic lattice'!"
        );

        for i in 0..n as u32 {
            for j in 0..n as u32 {
                for k in 0..n as u32 {
                    let x = space * f64::from(i + 1) - ((1.0 + n) * space / 2.0);
                    let y = space * f64::from(j + 1) - ((1.0 + n) * space / 2.0);
                    let z = space * f64::from(k + 1) - ((1.0 + n) * space / 2.0);
                    builder = builder.bodies([Body {
                        properties: DynamicPoint {
                            position: Cartesian::from([x, y, z]),
                            momentum: Cartesian::default(),
                            net_force: Cartesian::default(),
                            mass: 1.0,
                        },
                        sites: vec![Point::default()],
                    }]);
                }
            }
        }

        let mut microstate = builder.try_build()?;

        // Model interactions (in this case, a pairwise Lennard-Jones)
        let force = Rigid(
            PairwiseCutoff(
                Isotropic {
                    interaction: Xplor {
                        f: LennardJones::<12, 6> {
                            epsilon: 1.0,
                            sigma: 1.0,
                        },
                        r_cut: 2.0_f64.powf(1.0 / 6.0),
                        r_smooth: 2.0,
                    },
                    r_cut: 2.0_f64.powf(1.0 / 6.0),
                }
            )
        );

        // Randomize the momenta of system.
        let thermalizer = Thermalizer { kT: kT_init };
        thermalizer.thermalize_translation(&mut microstate);

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
        let dt = 0.001;
        let integrator = ConstantVolume::new(dt);

        // NVE simulation
        let thermostat = NoThermostat {};

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
        self.integrator.integrate_translation_half_step_one(
            &mut self.microstate,
            &mut self.thermostat,
            &self.macrostate,
        );

        self.integrator
            .update_force(&mut self.microstate, &self.force);

        self.integrator.integrate_translation_half_step_two(
            &mut self.microstate,
            &mut self.thermostat,
            &self.macrostate,
        );

        self.microstate.increment_step();
        if self.step() % 10000 == 1 {
            println!("==============={:}===============", self.step());
            let ke = self.integrator.get_translational_kinetic_energy();
            let dof = self.integrator.get_translational_dof();
            let n = dof / 3.0 + 1.0;

            let kT = 2.0 * ke / dof;
            let pe = self.force.0.total_energy(&self.microstate);

            let total_energy = ke + pe;
            let pe_per_particle = pe / n;

            println!("Temperature: {:}", kT);
            println!("Total energy: {:}", total_energy);
            println!("Potential energy per particle: {:} \n", pe_per_particle);
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

    for _ in 0..550_000 {
        simulation.advance();
    }

    Ok(())
}

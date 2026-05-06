// ANCHOR: all
// ANCHOR: use
use anyhow::{Context, anyhow};

use hoomd_geometry::shape::Hypercuboid;
use hoomd_interaction::{
    PairwiseCutoff, TotalEnergy,
    pairwise::Isotropic, univariate::LennardJones,
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
    }, thermostat::{BussiThermostat, NoThermostat}
};
use hoomd_microstate::{
    Body, Microstate, SiteKey, boundary::Periodic, property::{DynamicsPoint, Point}
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::Cartesian;
// ANCHOR_END: use

// Remove the cfg_attr(...) line when using this code outside the hoomd-rs/examples directory.
#[cfg_attr(feature = "bevy", derive(Resource))]
// ANCHOR: simulation_struct
struct LJFluid {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<
        DynamicsPoint<Cartesian<3>>,
        Point<Cartesian<3>>,
        VecCell<SiteKey, 3>,
        Periodic<Hypercuboid<3>>
    >,
    /// How sites interact with other sites.
    force: Rigid<PairwiseCutoff<Isotropic<LennardJones>>>,
    /// Constant volume MD integrator to sample the NVT and NVE ensemble.
    integrator: ConstantVolume,
    /// Thermostat to control the temperature of the isotherm.
    thermostat: BussiThermostat,
    /// Temperature set point.
    macrostate: Isothermal,
    /// Steps to prepare the isotherm in the Equilibrate phase.
    eq_step: u64,
    /// The long range energy correction to the truncated LJ potential of each particle.
    energy_lrc: f64,
    /// The current simulation state.
    phase: Phase,
}
// ANCHOR_END: simulation_struct

// ANCHOR: phase
enum Phase {
    Equilibrate,
    SampleNVE,
}
// ANCHOR_END: phase

// ANCHOR: simulation_new
impl LJFluid {
    /// Construct a new fill simulation.
    fn new() -> anyhow::Result<LJFluid> {
        // ANCHOR_END: simulation_new
        // ANCHOR: parameters
        let epsilon: f64 = 1.0;
        let sigma: f64 = 1.0;
        let m: f64 = 1.0;

        let n: f64 = 8.0;
        let eq_step = 50_000;
        let temperature_lj = 0.85;
        let density_lj = 0.776;
        let dt_lj = 0.005;
        let tau_lj = 50.0;
        let r_cut_lj = 3.0;

        // convert to real unit
        let dt = dt_lj * sigma * (m/epsilon).sqrt();
        let kt = temperature_lj * epsilon;
        let tau = tau_lj * dt;
        let r_cut = r_cut_lj * sigma;
        let density = density_lj / sigma.powi(3);
        let box_volume = n.powi(3) / density;
        let box_length = box_volume.cbrt();      
        let macrostate = Isothermal { temperature: kt }; 
        // ANCHOR_END: parameters

        // ANCHOR: boundary
        let cube = Hypercuboid::<3>::with_equal_edges(box_length.try_into()?);
        // ANCHOR_END: boundary
        // ANCHOR: spatial_data
        let vec_cell = VecCell::builder()
            .nominal_search_radius(r_cut.try_into()?)
            .build();
        // ANCHOR_END: spatial_data
        // ANCHOR: boundary_condition
        let boundary = Periodic::new(r_cut, cube)?;
        // ANCHOR_END: boundary_condition
        // ANCHOR: microstate_builder
        let mut builder = Microstate::builder()
            .spatial_data(vec_cell)
            .boundary(boundary);
        // ANCHOR_END: microstate_builder

        // ANCHOR: particle_positions
        let space = box_length / n;

        for i in 0..n as u32 {
            for j in 0..n as u32 {
                for k in 0..n as u32 {
                    let x = space * f64::from(i + 1) - ((1.0 + n) * space / 2.0);
                    let y = space * f64::from(j + 1) - ((1.0 + n) * space / 2.0);
                    let z = space * f64::from(k + 1) - ((1.0 + n) * space / 2.0);
                    builder = builder.bodies([Body {
                        properties: DynamicsPoint {
                            position: Cartesian::from([x, y, z]),
                            momentum: Cartesian::default(),
                            net_force: Cartesian::default(),
                            mass: m,
                        },
                        sites: vec![Point::default()],
                    }]);
                }
            }
        }
        // ANCHOR_END: particle_positions

        // ANCHOR: microstate
        let mut microstate = builder.try_build()?;
        // ANCHOR_END: microstate

        // ANCHOR: pair_force
        let force = Rigid(
            PairwiseCutoff(
                Isotropic {
                    interaction: LennardJones::<12, 6> {
                            epsilon: epsilon,
                            sigma: sigma,
                        },

                    r_cut: r_cut,
                }
            )
        );
        // ANCHOR_END: pair_force

        // ANCHOR: energy_lrc
        let lj1 = 4.0 * epsilon * sigma.powi(12);
        let lj2 = 4.0 * epsilon * sigma.powi(6);
        let inv_r_cut_3 = 1.0 / r_cut.powi(3);
        let inv_r_cut_9 = 1.0 / r_cut.powi(9);
        let energy_lrc = 2.0 * std::f64::consts::PI * density * (lj1 / 9.0 * inv_r_cut_9 - lj2 / 3.0 * inv_r_cut_3);
        // ANCHOR_END: energy_lrc

        // ANCHOR: particle_momenta
        let thermalizer = Thermalizer { kT: kt };
        thermalizer.thermalize_translation(&mut microstate);

        let angular_remover = ComAngularMomentumRemover {};
        let linear_remover = ComMomentumRemover {};
        angular_remover.modify(&mut microstate);
        linear_remover.modify(&mut microstate);
        // ANCHOR_END: particle_momenta

        // ANCHOR: integrator
        let integrator = ConstantVolume::new(dt);
        // ANCHOR_END: integrator

        // ANCHOR: thermostat
        let thermostat = BussiThermostat::new(tau.try_into()?);
        // ANCHOR_END: thermostat

        // ANCHOR: struct_initialize
        Ok(LJFluid {
            microstate,
            force,
            integrator,
            thermostat,
            macrostate,
            eq_step,
            energy_lrc,
            phase: Phase::Equilibrate,
        })
    }
}
// ANCHOR_END: struct_initialize

// ANCHOR: impl_simulation
impl Simulation for LJFluid {
    // ANCHOR_END: impl_simulation
    // ANCHOR: advance
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        match self.phase {
            Phase::Equilibrate => self.nvt(),
            Phase::SampleNVE => self.nve()
        }

        self.microstate.increment_step();

        Ok(())
    }
    // ANCHOR_END: advance

    // ANCHOR: step
    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}
// ANCHOR_END: step

// ANCHOR: dummy_nve_macrostate
struct Isoenergy {}
// ANCHOR_END: dummy_nve_macrostate

// ANCHOR: simulation_protocol
impl LJFluid {
// ANCHOR_END: simulation_protocol

    // ANCHOR: properties
    fn calculate_properties(&mut self) -> (f64, f64) {
        // ANCHOR_END: properties
        // ANCHOR: potetial_energy
        let pe = self.force.0.total_energy(&self.microstate);
        let n = self.microstate.bodies().len();
        let pe_per_particle = pe / n as f64 + self.energy_lrc;
        // ANCHOR_END: potetial_energy

        // ANCHOR: current_temeprature
        let ke = self.integrator.get_translational_kinetic_energy();
        let dof = self.integrator.get_translational_dof();
        let kt = 2.0 * ke / dof;
        (pe_per_particle, kt)
    }
    // ANCHOR_END: current_temeprature

    // ANCHOR: nvt
    fn nvt(&mut self) {
        // ANCHOR_END: nvt

        // ANCHOR: state_transition
        if self.step() >= self.eq_step {
            self.phase = Phase::SampleNVE;
            println!(
                "Isotherm preparation finished at step {}.",
                self.microstate.step()
            );
            return;
        }
        // ANCHOR_END: state_transition

        // ANCHOR: first_half_integration
        self.integrator.integrate_translation_step_one(
            &mut self.microstate,
            &mut self.thermostat,
            &self.macrostate,
        );
        // ANCHOR_END: first_half_integration

        // ANCHOR: update_force
        self.integrator
            .update_force(&mut self.microstate, &self.force);
        // ANCHOR_END: update_force

        // ANCHOR: second_half_integration
        self.integrator.integrate_translation_step_two(
            &mut self.microstate,
            &mut self.thermostat,
            &self.macrostate,
        );
        // ANCHOR_END: second_half_integration
    }

    // ANCHOR: nve
    fn nve(&mut self) {
        if self.step().is_multiple_of(10_000) {
            let (pe, kt) = self.calculate_properties();

            println!(
                "NVE, Step {}, kT {}, Potential energy (w/ LRC) per particle {}" ,
                self.microstate.step() - self.eq_step,
                kt,
                pe
            );
        }

        self.integrator.integrate_translation_step_one(
            &mut self.microstate,
            &mut NoThermostat {},
            &Isoenergy {},
        );

        self.integrator
            .update_force(&mut self.microstate, &self.force);

        self.integrator.integrate_translation_step_two(
            &mut self.microstate,
            &mut NoThermostat {},
            &Isoenergy {},
        );
    }
}
// ANCHOR_END: nve

// Remove the cfg(not(...)) line when using this code outside the hoomd-rs/examples directory.
#[cfg(not(feature = "bevy"))]
// ANCHOR: main
fn main() -> anyhow::Result<()> {
    use hoomd_gsd::hoomd::HoomdGsdFile;
    use hoomd_microstate::AppendMicrostate;

    let mut simulation = LJFluid::new()?;
    // ANCHOR_END: main
    // ANCHOR: create_gsd
    let mut hoomd_gsd_file = HoomdGsdFile::create("nve-ljg-fluid.gsd")?;
    // ANCHOR_END: create_gsd

    // ANCHOR: advance
    for _ in 0..100_000 {
        simulation.advance()?;
        // ANCHOR_END: advance

        // ANCHOR: append_microstate
        if simulation.step().is_multiple_of(5_000) {
            hoomd_gsd_file.append_microstate(&simulation.microstate)?;
        }
    }

    Ok(())
}
// ANCHOR_END: append_microstate
// ANCHOR_END: all

#[cfg(feature = "bevy")]
mod nve_lj_fluid_interactive;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
#[cfg(feature = "bevy")]
use nve_lj_fluid_interactive::main;

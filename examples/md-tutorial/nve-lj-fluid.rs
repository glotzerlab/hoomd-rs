// ANCHOR: all
// ANCHOR: use
use anyhow::{Context, anyhow};

use hoomd_geometry::shape::Hypercuboid;
use hoomd_interaction::{
    PairwiseCutoff, TotalEnergy,
    pairwise::Isotropic, univariate::{LennardJones, Shifted},
    rigid::Rigid,
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
    }, thermostat::BussiThermostat
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
    force: Rigid<PairwiseCutoff<Isotropic<Shifted<LennardJones>>>>,
    /// Constant volume MD integrator to sample the NVT and NVE ensemble.
    integrator: ConstantVolume,
    /// Thermostat to control the temperature at the Equilibrate phase.
    thermostat: BussiThermostat,
    /// Temperature set point.
    macrostate: Isothermal,
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
        let kT_init = 0.851;
        let density = 0.776;
        let n: f64 = 8.0;
        let box_volume = n.powi(3) / density;
        let box_length = box_volume.cbrt();      
        let macrostate = Isothermal { temperature: kT_init }; 
        let epsilon = 1.0;
        let sigma = 1.0;
        let r_cut = 4.0 * sigma;
        let dt = 0.005;
        let tau_thermostat = 50.0 * dt;
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
                            mass: 1.0,
                        },
                        sites: vec![Point::default()],
                    }]);
                }
            }
        }
        // ANCHOR: particle_positions

        // ANCHOR: microstate
        let mut microstate = builder.try_build()?;
        // ANCHOR_END: microstate

        // ANCHOR: pair_force
        let force = Rigid(
            PairwiseCutoff(
                Isotropic {
                    interaction: Shifted {
                        f: LennardJones::<12, 6> {
                            epsilon: epsilon,
                            sigma: sigma,
                        },
                        r_shift: r_cut,
                    },
                    r_cut: r_cut,
                }
            )
        );
        // ANCHOR_END: pair_force

        // ANCHOR: particle_momenta
        let thermalizer = Thermalizer { kT: kT_init };
        thermalizer.thermalize_translation(&mut microstate);

        let angular_remover = ComAngularMomentumRemover {};
        let linear_remover = ComMomentumRemover {};
        angular_remover.modify(&mut microstate);
        linear_remover.modify(&mut microstate);
        // ANCHOR_END: particle_momenta

        // ANCHOR: integrator
        let integrator = ConstantVolume::new(dt);
        // ANCHO_END: integrator

        // ANCHOR: thermostat
        let thermostat = BussiThermostat::new(tau_thermostat.try_into()?);
        // ANCHOR: thermostat

        // ANCHOR: struct_initialize
        Ok(LJFluid {
            microstate,
            force,
            integrator,
            thermostat,
            macrostate,
        })
    }
}
// ANCHOR_END: struct_initialize

// Alex stop here ########################################################################################################

// ANCHOR: impl_simulation
impl Simulation for LJFluid {
    // ANCHOR_END: impl_simulation
    // ANCHOR: advance
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
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

// Remove the cfg(not(...)) line when using this code outside the hoomd-rs/examples directory.
#[cfg(not(feature = "bevy"))]
// ANCHOR: main
fn main() -> anyhow::Result<()> {
    let mut simulation = LJFluid::new()?;
    // TODO: Write GSD file.

    for _ in 0..100_000 {
        simulation.advance()?;
    }

    Ok(())
}
// ANCHOR_END: main
// ANCHOR_END: all

#[cfg(feature = "bevy")]
mod applying_interactions_interactive;
#[cfg(feature = "bevy")]
use applying_interactions_interactive::main;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
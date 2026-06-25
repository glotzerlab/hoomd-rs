// ANCHOR: all
use hoomd_geometry::shape::Hypercuboid;
use hoomd_interaction::{
    PairwiseCutoff,
    pairwise::Isotropic, univariate::LennardJones,
    Rigid,
};
use hoomd_md::{
    ThermalizeMomentum, TranslationalMotion, ZeroCenterAngularMomentum, ZeroCenterMomentum, method::
        ConstantVolume,
    thermostat::Bussi
};
use hoomd_microstate::{
    Body, Microstate, SiteKey, boundary::Periodic, property::{DynamicPoint, Point}
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::Cartesian;

// Remove the cfg_attr(...) line when using this code outside the hoomd-rs/examples directory.
#[cfg_attr(feature = "bevy", derive(Resource))]
struct LennardJonesFluid {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<
        DynamicPoint<Cartesian<3>>,
        Point<Cartesian<3>>,
        VecCell<SiteKey, 3>,
        Periodic<Hypercuboid<3>>
    >,
    /// How sites interact with other sites.
    interaction_model: Rigid<PairwiseCutoff<Isotropic<LennardJones>>>,
    /// Constant volume MD integrator to sample the NVT ensemble.
    constant_volume: ConstantVolume<Bussi, Bussi>,
    /// Temperature set point.
    macrostate: Isothermal,
}


impl LennardJonesFluid {
    /// Construct a new fill simulation.
    fn new() -> anyhow::Result<LennardJonesFluid> {
        let epsilon: f64 = 1.0;
        let sigma: f64 = 1.0;
        let m: f64 = 1.0;

        let n: f64 = 8.0;
        let temperature_lj = 0.85;
        let density_lj = 0.776;
        let dt_lj = 0.005;
        let tau_lj = 50.0;
        let r_cut_lj = 3.0;

        let dt = dt_lj * sigma * (m/epsilon).sqrt();
        let kt = temperature_lj * epsilon;
        let tau = tau_lj * dt;
        let r_cut = r_cut_lj * sigma;
        let density = density_lj / sigma.powi(3);
        let box_volume = n.powi(3) / density;
        let box_length = box_volume.cbrt();      
        let macrostate = Isothermal { temperature: kt }; 

        let cube = Hypercuboid::<3>::with_equal_edges(box_length.try_into()?);
        let vec_cell = VecCell::builder()
            .nominal_search_radius(r_cut.try_into()?)
            .build();
        let boundary = Periodic::new(r_cut, cube)?;
        let mut builder = Microstate::builder()
            .spatial_data(vec_cell)
            .boundary(boundary);

        let space = box_length / n;

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
                            mass: m,
                        },
                        sites: vec![Point::default()],
                    }]);
                }
            }
        }

        let mut microstate = builder.try_build()?;

        let interaction_model = Rigid(
            PairwiseCutoff(
                Isotropic {
                    interaction: LennardJones::<12, 6> {
                            epsilon,
                            sigma,
                        },

                    r_cut,
                }
            )
        );

        microstate.thermalize_momentum(kt);

        microstate.zero_center_angular_momentum();
        microstate.zero_center_momentum();

        let thermostat = Bussi::new(tau.try_into()?);
        let constant_volume = ConstantVolume::builder(dt)
            .thermostat(thermostat)
            .build();

        Ok(LennardJonesFluid {
            microstate,
            interaction_model,
            constant_volume,
            macrostate,
            })
        }
    }

impl Simulation for LennardJonesFluid {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        self.constant_volume.integrate_translation(
            &mut self.microstate,
            &self.macrostate,
            &self.interaction_model);
        self.microstate.increment_step();

        Ok(())
    }

    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

// Remove the cfg(not(...)) line when using this code outside the hoomd-rs/examples directory.
#[cfg(not(feature = "bevy"))]
fn main() -> anyhow::Result<()> {
    use hoomd_gsd::hoomd::HoomdGsdFile;
    use hoomd_microstate::AppendMicrostate;

    let mut simulation = LennardJonesFluid::new()?;
    let mut hoomd_gsd_file = HoomdGsdFile::create("nvt-lj-fluid.gsd")?;

    for _ in 0..100_000 {
        simulation.advance()?;
           hoomd_gsd_file.append_microstate(&simulation.microsconstant_volume )?;
    }
}
// ANCHOR_END: all

mod nvt_lj_fluid_interactive;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
#[cfg(feature = "bevy")]
use nvt_lj_fluid_interactive::main;

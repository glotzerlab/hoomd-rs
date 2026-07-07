// ANCHOR: all
use itertools::Itertools;

use hoomd_geometry::shape::Cuboid;
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff, Rigid, pairwise::Isotropic,
    univariate::LennardJones,
};
use hoomd_md::{
    ThermalizeMomentum, TranslationalMotion, ZeroCenterAngularMomentum,
    ZeroCenterMomentum, method::ConstantVolume, thermostat::Bussi,
};
use hoomd_microstate::{
    Body, Microstate, SiteKey,
    boundary::Periodic,
    property::{DynamicPoint, Point},
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
        Periodic<Cuboid>,
    >,
    /// How bodies interact with other bodies.
    interaction_model: Rigid<PairwiseCutoff<Isotropic<LennardJones>>>,
    /// Constant volume, constant temperature integration method..
    constant_volume: ConstantVolume<Bussi>,
    /// Temperature set point.
    macrostate: Isothermal,
}

impl LennardJonesFluid {
    /// Construct a new Lennard Jones fluid simulation.
    fn new() -> anyhow::Result<LennardJonesFluid> {
        let epsilon = 1.0;
        let sigma = 1.0;
        let r_cut = 3.0;
        let temperature = 0.85;
        let density: f64 = 0.776;
        let delta_t = 0.005;
        let n_bodies = 256;

        let box_length = (n_bodies as f64 / density).cbrt();
        let macrostate = Isothermal { temperature };

        let interaction_model = Rigid(PairwiseCutoff(Isotropic {
            interaction: LennardJones::<12, 6> { epsilon, sigma },
            r_cut,
        }));

        let cube = Cuboid::with_equal_edges(box_length.try_into()?);
        let vec_cell = VecCell::builder()
            .nominal_search_radius(
                interaction_model.maximum_interaction_range().try_into()?,
            )
            .build();
        let boundary =
            Periodic::new(interaction_model.maximum_interaction_range(), cube)?;
        let mut microstate = Microstate::builder()
            .spatial_data(vec_cell)
            .boundary(boundary)
            .try_build()?;

        let n_edge = (n_bodies as f64).cbrt().ceil();
        let spacing = box_length / n_edge;
        let n_edge = n_edge as usize;

        for index in [(0..n_edge), (0..n_edge), (0..n_edge)]
            .into_iter()
            .multi_cartesian_product()
            .take(n_bodies)
        {
            let position: Vec<_> = index
                .iter()
                .map(|x| spacing * (*x as f64) - box_length / 2.0)
                .collect();

            microstate.add_body(Body::single_site(
                DynamicPoint {
                    position: Cartesian::try_from(position)?,
                    ..Default::default()
                },
                Point::default(),
            ))?;
        }

        microstate.thermalize_momentum(temperature);
        microstate.zero_center_angular_momentum();
        microstate.zero_center_momentum();

        let thermostat = Bussi::new(0.0);
        let constant_volume = ConstantVolume::builder(delta_t)
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
            &self.interaction_model,
        );
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

    for _ in 0..40_000 {
        simulation.advance()?;
        if simulation.step().is_multiple_of(10_000) {
            hoomd_gsd_file.append_microstate(&simulation.microstate)?;
        }
    }

    Ok(())
}
// ANCHOR_END: all

#[cfg(feature = "bevy")]
mod lj_fluid_interactive;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
#[cfg(feature = "bevy")]
use lj_fluid_interactive::main;

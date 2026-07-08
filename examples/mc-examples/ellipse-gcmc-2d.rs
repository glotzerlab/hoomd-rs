// ANCHOR: all
// ANCHOR: use
use hoomd_geometry::{
    Volume,
    shape::{Ellipse, Rectangle},
};
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff, pairwise::HardShape,
};
use hoomd_mc::{GrandCanonical, Rotate, Sweep, Translate, Trial, UniformIn};
use hoomd_microstate::{
    Microstate, SiteKey, boundary::Periodic, property::OrientedPoint,
};
use hoomd_simulation::{Simulation, macrostate::IsothermalIsofugacity};
use hoomd_spatial::VecCell;
use hoomd_vector::{self, Angle, Cartesian};

type PositionVector = Cartesian<2>;
type Orientation = Angle;
type BodyProperties = OrientedPoint<PositionVector, Orientation>;
type SiteProperties = OrientedPoint<PositionVector, Orientation>;

#[cfg_attr(feature = "bevy", derive(Resource))]
struct HardEllipseGCMC {
    /// Positions and orientations of all the bodies in the simulation.
    microstate: Microstate<
        BodyProperties,
        SiteProperties,
        VecCell<SiteKey, 2>,
        Periodic<Rectangle>,
    >,
    /// How sites interact with other sites and fields.
    hamiltonian: PairwiseCutoff<HardShape<Ellipse>>,
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate<PositionVector>>,
    /// Trial moves to apply.
    rotate_sweep: Sweep<Rotate<Orientation>>,
    /// Temperature set point.
    macrostate: IsothermalIsofugacity,
    /// Temperature set point.
    gcmc: GrandCanonical<UniformIn<BodyProperties, Periodic<Rectangle>>>,
}

impl HardEllipseGCMC {
    /// Construct a new hard ellipse gcmc simulation.
    fn new() -> anyhow::Result<HardEllipseGCMC> {
        let maximum_distance = 0.07;
        let maximum_rotation = 0.3;
        let sigma = 1.0;
        let aspect = 5.0;
        let macrostate = IsothermalIsofugacity {
            temperature: 1.0,
            fugacity: 1000.0,
        };
        assert!(aspect >= 1.0);

        let ellipse = Ellipse::with_semi_axes([
            (sigma / 2.0).try_into()?,
            (sigma / aspect / 2.0).try_into()?,
        ]);
        let hamiltonian = PairwiseCutoff(HardShape(ellipse.clone()));

        let initial_box_volume = 512_f64 * ellipse.volume() / 0.4;
        let initial_box_edge_length = initial_box_volume.sqrt();
        let square =
            Rectangle::with_equal_edges(initial_box_edge_length.try_into()?);
        let periodic_square =
            Periodic::new(hamiltonian.maximum_interaction_range(), square)?;

        let vec_cell = VecCell::builder()
            .nominal_search_radius(
                hamiltonian.maximum_interaction_range().try_into()?,
            )
            .build();
        let microstate = Microstate::builder()
            .boundary(periodic_square)
            .spatial_data(vec_cell)
            .try_build()?;

        let translate =
            Translate::with_maximum_distance(maximum_distance.try_into()?);
        let translate_sweep = Sweep(translate);

        let rotate =
            Rotate::with_maximum_rotation(maximum_rotation.try_into()?);
        let rotate_sweep = Sweep(rotate);

        let distribution = UniformIn {
            boundary: microstate.boundary().clone(),
            template_sites: vec![SiteProperties::default()],
        };

        let gcmc = GrandCanonical(distribution);

        Ok(HardEllipseGCMC {
            microstate,
            hamiltonian,
            translate_sweep,
            rotate_sweep,
            macrostate,
            gcmc,
        })
    }
}

impl Simulation for HardEllipseGCMC {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        self.gcmc.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
        );

        self.translate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
        );

        self.rotate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
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

    let mut simulation = HardEllipseGCMC::new()?;
    let mut hoomd_gsd_file = HoomdGsdFile::create("hard-ellipse-gcmc.gsd")?;

    for _ in 0..100_000 {
        simulation.advance()?;

        if simulation.step().is_multiple_of(10_000) {
            hoomd_gsd_file
                .append_microstate(&simulation.microstate)?
                .end()?;
        }
    }

    Ok(())
}

// ANCHOR_END: all

#[cfg(feature = "bevy")]
mod ellipse_gcmc_2d_interactive;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
#[cfg(feature = "bevy")]
use ellipse_gcmc_2d_interactive::main;

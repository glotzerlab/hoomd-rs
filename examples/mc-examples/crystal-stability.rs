// ANCHOR: all
use std::f64::consts::PI;

use hoomd_geometry::{
    Scale, Volume, shape::{ConvexPolygon, ConvexSurfaceMesh2d, Rhomboid}
};
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff,
    pairwise::HardShape,
};
use hoomd_mc::{Rotate, Sweep, Translate, Trial};
use hoomd_microstate::{
    Body, Microstate, Replicate, SiteKey, boundary::Periodic, property::OrientedPoint
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::{Angle, Cartesian};

type PositionVector = Cartesian<2>;
type Orientation = Angle;
type BodyProperties = OrientedPoint<Cartesian<2>, Orientation>;
type SiteProperties = OrientedPoint<PositionVector, Orientation>;

#[cfg_attr(feature = "bevy", derive(Resource))]
struct HardHexagonMelt {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<
        BodyProperties,
        SiteProperties,
        VecCell<SiteKey, 2>,
        Periodic<Rhomboid>,
    >,
    /// How sites interact with other sites and fields.
    hamiltonian: PairwiseCutoff<HardShape<ConvexSurfaceMesh2d>>,
    /// Translation trial moves to apply.
    translate_sweep: Sweep<Translate<PositionVector>>,
    /// Rotation trial moves to apply.
    rotate_sweep: Sweep<Rotate<Orientation>>,
    /// Temperature set point.
    macrostate: Isothermal,
}

impl HardHexagonMelt {
    /// Construct a new hard disk self-assembly simulation.
    fn new() -> anyhow::Result<HardHexagonMelt> {
        let initial_packing_fraction = 0.9;
        let n_replicates_side = 32;
        let maximum_distance = 0.07;
        let maximum_rotation = 0.05;
        let macrostate = Isothermal { temperature: 1.0 };

        let regular_hexagon = ConvexPolygon::regular(6);
        let mesh = ConvexSurfaceMesh2d::try_from(regular_hexagon)?;
        let hamiltonian = PairwiseCutoff(HardShape(mesh.clone()));

        let unit_cell_volume = mesh.volume() / initial_packing_fraction;
        let unit_cell_edge_length = (unit_cell_volume / (PI/3.0).sin()).sqrt();
        let unit_cell_rhomboid = Rhomboid {
            extents: [
                unit_cell_edge_length.try_into()?,
                (unit_cell_edge_length *  (PI/3.0).sin()).try_into()?,
            ],
            xy: 1.0 / 3.0f64.sqrt(),
        };

        // A single unit cell is too small to allow ghost sites. Start with
        // no ghosts around the unit cell, then set the proper maximum
        // interaction range while replicating.
        let periodic_unit_cell = Periodic::new(0.0, unit_cell_rhomboid)?;

        let vec_cell = VecCell::builder()
            .nominal_search_radius(
                hamiltonian.maximum_interaction_range().try_into()?,
            )
            .build();
        let microstate = Microstate::builder()
            .boundary(periodic_unit_cell)
            .spatial_data(vec_cell)
            .bodies([Body::single_site(
                OrientedPoint { position: Cartesian::default(), orientation: Angle { theta: PI / 6.0} },
                OrientedPoint::default(),
)])
            .try_build()?
            .replicate_with_maximum_interaction_range([n_replicates_side; 2], hamiltonian.maximum_interaction_range())?;
            

        let translate =
            Translate::with_maximum_distance(maximum_distance.try_into()?);
        let translate_sweep = Sweep(translate);

        let rotate =
            Rotate::with_maximum_rotation(maximum_rotation.try_into()?);
        let rotate_sweep = Sweep(rotate);

        Ok(HardHexagonMelt {
            microstate,
            hamiltonian,
            translate_sweep,
            rotate_sweep,
            macrostate,
        })
    }
}

impl Simulation for HardHexagonMelt {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        if self.step().is_multiple_of(10_000) {
            let expanded_boundary = self.microstate.boundary().scale_volume(1.04.try_into()?);
            self.microstate = self.microstate.clone_with_boundary(expanded_boundary, |_| true)?;
        }
    
        self.equilibrate();
        self.microstate.increment_step();

        Ok(())
    }

    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl HardHexagonMelt {
    fn equilibrate(&mut self) {
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
    }
}

// Remove the cfg(not(...)) line when using this code outside the hoomd-rs/examples directory.
#[cfg(not(feature = "bevy"))]
fn main() -> anyhow::Result<()> {
    use hoomd_gsd::hoomd::HoomdGsdFile;
    use hoomd_microstate::AppendMicrostate;

    let mut simulation = HardHexagonMelt::new()?;
    let mut hoomd_gsd_file =
        HoomdGsdFile::create("crystal-stability.gsd")?;

    for _ in 0..100_000 {
        simulation.advance()?;

        if simulation.step().is_multiple_of(1_000) {
            hoomd_gsd_file
                .append_microstate(&simulation.microstate)?
                .end()?;
        }
    }

    Ok(())
}
// ANCHOR_END: all

#[cfg(feature = "bevy")]
mod crystal_stability_interactive;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
#[cfg(feature = "bevy")]
use crystal_stability_interactive::main;

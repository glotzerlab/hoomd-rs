// ANCHOR: all
use anyhow::{Context, anyhow};
use arrayvec::ArrayVec;

use hoomd_geometry::{
    Convex, IsPointInside, MapPoint, Scale, Volume, shape::{Capsule, Rectangle}
};
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff,
    pairwise::{Anisotropic, ApproximateShapeOverlap, HardShape},
    univariate::OverlapPenalty,
};
use hoomd_mc::{
    QuickCompress, QuickInsert, Rotate, Sweep, Translate, Trial, Tune, TuneOptions, UniformIn
};
use hoomd_microstate::{
    Microstate, SiteKey, Transform, boundary::{GenerateGhosts, MAX_GHOSTS, Periodic, Wrap}, property::{Orientation, OrientedPoint, Position}
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::{self, Cartesian, Versor};

type BodyProperties = OrientedPoint<Cartesian<2>, Versor>;

#[derive(Clone, Copy, Default, Position, Orientation)]
struct SiteProperties {
    position: Cartesian<3>,
    orientation: Versor,
}

#[derive(Clone, PartialEq)]
struct Boundary(Periodic<Rectangle>);

impl Transform<SiteProperties> for OrientedPoint<Cartesian<2>, Versor> {
    fn transform(&self, site_properties: &SiteProperties) -> SiteProperties {
        SiteProperties {
            position: Cartesian::from([self.position[0] + site_properties.position[0],
                self.position[1] + site_properties.position[1],
                site_properties.position[2]]),
            orientation: self.orientation,
        }
    }
}

impl Wrap<SiteProperties> for Boundary {
    fn wrap(&self, properties: SiteProperties) -> Result<SiteProperties, hoomd_microstate::boundary::Error> {
        let mut properties = properties;
        let r = properties.position_mut();

        for (coordinate, edge_length) in r.coordinates.iter_mut().zip(self.0.shape().edge_lengths).take(2) {
            let edge_length = edge_length.get();
            let lambda = *coordinate / edge_length;
            let lambda = lambda - lambda.round();
            let lambda = if lambda == 0.5 { -0.5 } else { lambda };
            *coordinate = lambda * edge_length;
        }
        Ok(properties)
    }
}

impl Wrap<BodyProperties> for Boundary {
    fn wrap(&self, properties: BodyProperties) -> Result<BodyProperties, hoomd_microstate::boundary::Error> {
        self.0.wrap(properties)
    }
}

impl GenerateGhosts<SiteProperties> for Boundary {
    fn maximum_interaction_range(&self) -> f64 {
        self.0.maximum_interaction_range()
    }

    fn generate_ghosts(&self, site_properties: &SiteProperties) -> ArrayVec<SiteProperties, MAX_GHOSTS> {
        let mut result = ArrayVec::new();

        let r = site_properties.position();
        let projected_r = Cartesian::from([r[0], r[1]]);
        let max = self.0.shape().maximal_extents();
        let min = self.0.shape().minimal_extents();

        if !self.0.shape().is_point_inside(&projected_r) {
            return result;
        }

        let new_site = |x, y| {
            let mut new_site = *site_properties;
            new_site.position_mut()[0] += x * self.0.shape().edge_lengths[0].get();
            new_site.position_mut()[1] += y * self.0.shape().edge_lengths[1].get();
            new_site
        };

        let near_left = r[0] < min[0] + self.0.maximum_interaction_range();
        let near_right = r[0] > max[0] - self.0.maximum_interaction_range();
        let near_top = r[1] > max[1] - self.0.maximum_interaction_range();
        let near_bottom = r[1] < min[1] + self.0.maximum_interaction_range();

        if near_right {
            result.push(new_site(-1.0, 0.0));
        }
        if near_left {
            result.push(new_site(1.0, 0.0));
        }
        if near_top {
            result.push(new_site(0.0, -1.0));
        }
        if near_bottom {
            result.push(new_site(0.0, 1.0));
        }
        if near_right && near_top {
            result.push(new_site(-1.0, -1.0));
        }
        if near_right && near_bottom {
            result.push(new_site(-1.0, 1.0));
        }
        if near_left && near_top {
            result.push(new_site(1.0, -1.0));
        }
        if near_left && near_bottom {
            result.push(new_site(1.0, 1.0));
        }

        result
    }
}

impl Volume for Boundary {
    fn volume(&self) -> f64 {
        self.0.volume()
    }
}

impl MapPoint<Cartesian<2>> for Boundary {
    fn map_point(&self, point: Cartesian<2>, other: &Self) -> Result<Cartesian<2>, hoomd_geometry::Error> {
        self.0.map_point(point, &other.0)
    }
}

impl Scale for Boundary {
    fn scale_length(&self, v: hoomd_utility::valid::PositiveReal) -> Self {
        Boundary(self.0.scale_length(v))
    }

    fn scale_volume(&self, v: hoomd_utility::valid::PositiveReal) -> Self {
        Boundary(self.0.scale_volume(v))
    }
}

impl Distribution<Cartesian<2>> for Boundary {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Cartesian<2> {
        self.0.sample(rng)
    }
}

impl Quasi2dCapsuleSelfAssembly {
    /// Construct a new hard tetrahedron self-assembly simulation.
    fn new() -> anyhow::Result<Quasi2dCapsuleSelfAssembly> {
        let initial_number_density = 0.1;
        let target_number_density = 0.2;
        let n_bodies = 256;
        let maximum_distance = 0.04;
        let maximum_rotation = 0.04;
        let macrostate = Isothermal { temperature: 1.0 };

        let capsule = Capsule { radius: 1.0.try_into()?, height: 5.0.try_into()?};
        let hamiltonian = PairwiseCutoff(HardShape(capsule.clone()));

        let initial_box_volume =
            n_bodies as f64 / initial_number_density;
        let initial_box_edge_length = initial_box_volume.sqrt();
        let rectangle =
            Rectangle::with_equal_edges(initial_box_edge_length.try_into()?);
        let periodic_rectangle =
            Boundary(Periodic::new(hamiltonian.maximum_interaction_range(), rectangle)?);

        let vec_cell = VecCell::builder()
            .nominal_search_radius(
                hamiltonian.maximum_interaction_range().try_into()?,
            )
            .build();
        let microstate = Microstate::builder()
            .boundary(periodic_rectangle)
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
        let quick_insert = QuickInsert::new(distribution, n_bodies);

        let target_box_volume =
            n_bodies as f64 / target_number_density;
        let quick_compress =
            QuickCompress::with_target_volume(target_box_volume.try_into()?);

        let approximate_shape_overlap = Anisotropic {
            interaction: ApproximateShapeOverlap::new(
                Convex(capsule),
                OverlapPenalty::default(),
                0.01.try_into()?,
            ),
            r_cut: hamiltonian.maximum_interaction_range(),
        };

        let overlap_penalty_hamiltonian =
            PairwiseCutoff(approximate_shape_overlap);

        Ok(Quasi2dCapsuleSelfAssembly {
            microstate,
            overlap_penalty_hamiltonian,
            hamiltonian,
            translate_sweep,
            rotate_sweep,
            quick_compress,
            quick_insert,
            macrostate,
            phase: Phase::Initialize,
        })
    }
}

#[cfg_attr(feature = "bevy", derive(Resource))]
struct Quasi2dCapsuleSelfAssembly {
    /// Positions and orientations of all the bodies in the simulation.
    microstate: Microstate<
        BodyProperties,
        SiteProperties,
        VecCell<SiteKey, 3>,
        Boundary,
    >,
    /// How sites interact with other sites and fields.
    hamiltonian: PairwiseCutoff<HardShape<Capsule<3>>>,
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate<Cartesian<2>>>,
    /// Trial moves to apply.
    rotate_sweep: Sweep<Rotate<Versor>>,
    /// Temperature set point.
    macrostate: Isothermal,
    /// Quick compress algorithm.
    quick_compress: QuickCompress<Boundary>,
    /// Quick insert algorithm.
    quick_insert: QuickInsert<UniformIn<SiteProperties, Boundary>>,
    /// How sites interact when inserted and compressed.
    overlap_penalty_hamiltonian: PairwiseCutoff<
        Anisotropic<
            ApproximateShapeOverlap<OverlapPenalty, Convex<Capsule<3>>>,
        >,
    >,
    /// The current phase of the simulation.
    phase: Phase,
}

enum Phase {
    Initialize,
    Equilibrate,
}

impl Simulation for Quasi2dCapsuleSelfAssembly {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        match self.phase {
            Phase::Initialize => {
                self.initialize().context("failed to initialize")?
            }
            Phase::Equilibrate => self.equilibrate(),
        }

        self.microstate.increment_step();

        Ok(())
    }

    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl Quasi2dCapsuleSelfAssembly {
    fn initialize(&mut self) -> anyhow::Result<()> {
        if self.quick_insert.is_complete() {
            self.quick_compress.apply(
                &mut self.microstate,
                &self.overlap_penalty_hamiltonian,
                |_| true,
            );
        } else {
            self.quick_insert
                .apply(&mut self.microstate, &self.overlap_penalty_hamiltonian);
        }

        self.translate_sweep.apply(
            &mut self.microstate,
            &self.overlap_penalty_hamiltonian,
            &Isothermal { temperature: 1.0 },
        );

        self.rotate_sweep.apply(
            &mut self.microstate,
            &self.overlap_penalty_hamiltonian,
            &Isothermal { temperature: 1.0 },
        );

        if self.quick_compress.is_complete() {
            self.translate_sweep.tune_with_options(
                &self.microstate,
                &self.hamiltonian,
                &self.macrostate,
                &TuneOptions::default(),
            );
            self.rotate_sweep.tune_with_options(
                &self.microstate,
                &self.hamiltonian,
                &self.macrostate,
                &TuneOptions::default(),
            );

            self.phase = Phase::Equilibrate;
            println!(
                "Initialization complete at step {}.",
                self.microstate.step()
            );
        }

        if self.step() >= 20_000 {
            let n = self.microstate.bodies().len();
            let target_n = self.quick_insert.target();
            let volume = self.microstate.boundary().0.volume();
            let target_volume = self.quick_compress.target_volume();
            return Err(anyhow!(
                "inserted {n}/{target_n} bodies and compressed to {volume} / {target_volume}"
            ));
        }

        Ok(())
    }

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

    let mut simulation = Quasi2dCapsuleSelfAssembly::new()?;
    let mut hoomd_gsd_file =
        HoomdGsdFile::create("hard-tetrahedron-self-assembly.gsd")?;

    for _ in 0..40_000 {
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
mod capsules_on_a_plane_interactive;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
#[cfg(feature = "bevy")]
use capsules_on_a_plane_interactive::main;
use rand::{Rng, distr::Distribution};

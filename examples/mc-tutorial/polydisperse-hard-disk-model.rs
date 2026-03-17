// ANCHOR: all
// ANCHOR: use
use anyhow::{Context, anyhow};
use hoomd_gsd::hoomd::{Dimensions, HoomdGsdFile};
use rand::{
    SeedableRng,
    distr::{Distribution, Uniform},
    rngs::StdRng,
};

use hoomd_geometry::{
    Volume,
    shape::{Circle, Rectangle},
};
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff, SitePairEnergy,
    univariate::{Expanded, OverlapPenalty, UnivariateEnergy},
};
use hoomd_mc::{
    BodyDistribution, QuickCompress, QuickInsert, Sweep, Translate, Trial, Tune,
};
use hoomd_microstate::{
    AppendMicrostate, Body, Microstate, SiteKey, Transform,
    boundary::Periodic,
    property::{Point, Position},
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, Metric};
// ANCHOR_END: use

// ANCHOR: type_aliases
type PositionVector = Cartesian<2>;
type BodyProperties = Point<PositionVector>;
// ANCHOR_END: type_aliases

// ANCHOR: site_properties
#[derive(Clone, Copy, Default, Position)]
struct SiteProperties {
    /// The site's position.
    position: PositionVector,
    /// The site's radius.
    radius: PositiveReal,
}
// ANCHOR_END: site_properties

// ANCHOR: site_transform
impl Transform<SiteProperties> for BodyProperties {
    fn transform(&self, site_properties: &SiteProperties) -> SiteProperties {
        SiteProperties {
            position: self.position + site_properties.position,
            ..*site_properties
        }
    }
}
// ANCHOR_END: site_transform

// ANCHOR: interaction_type
#[derive(MaximumInteractionRange)]
struct SitePairInteraction {
    maximum_interaction_range: f64,
}
// ANCHOR_END: interaction_type

// ANCHOR: interaction_impl
impl SitePairEnergy<SiteProperties> for SitePairInteraction {
    fn site_pair_energy(&self, a: &SiteProperties, b: &SiteProperties) -> f64 {
        let r = a.position().distance(b.position());

        if r < a.radius.get() + b.radius.get() {
            f64::INFINITY
        } else {
            0.0
        }
    }
    // ANCHOR_END: interaction_impl

    // ANCHOR: infinite_zero
    fn site_pair_energy_initial(
        &self,
        _a: &SiteProperties,
        _b: &SiteProperties,
    ) -> f64 {
        0.0
    }

    fn is_only_infinite_or_zero() -> bool {
        true
    }
}
// ANCHOR_END: infinite_zero

// ANCHOR: overlap_penalty_type
#[derive(MaximumInteractionRange)]
struct SitePairOverlapPenalty {
    maximum_interaction_range: f64,
}
// ANCHOR_END: overlap_penalty_type

// ANCHOR: overlap_penalty_impl
impl SitePairEnergy<SiteProperties> for SitePairOverlapPenalty {
    fn site_pair_energy(&self, a: &SiteProperties, b: &SiteProperties) -> f64 {
        let r = a.position().distance(b.position());
        let pair_interaction = Expanded {
            delta: a.radius.get() + b.radius.get(),
            f: OverlapPenalty::default(),
        };
        pair_interaction.energy(r)
    }
}
// ANCHOR_END: overlap_penalty_impl

// ANCHOR: body_distribution_type
struct PolydisperseBodyDistribution {
    /// Radius of each disk to insert into the microstate.
    radii: Vec<PositiveReal>,
    /// Simulation boundary.
    boundary: Periodic<Rectangle>,
}
// ANCHOR_END: body_distribution_type

// ANCHOR: body_distribution_impl
impl BodyDistribution<Body<BodyProperties, SiteProperties>>
    for PolydisperseBodyDistribution
{
    fn sample<R: rand::Rng + ?Sized>(
        &self,
        index: usize,
        rng: &mut R,
    ) -> Body<BodyProperties, SiteProperties> {
        let properties = Point {
            position: self.boundary.sample(rng),
        };
        let sites = vec![SiteProperties {
            position: Cartesian::default(),
            radius: self.radii[index],
        }];
        Body { properties, sites }
    }
}
// ANCHOR_END: body_distribution_impl

// ANCHOR: append_microstate
impl<X> AppendMicrostate<BodyProperties, SiteProperties, X, Periodic<Rectangle>>
    for HoomdGsdFile
{
    #[inline]
    fn append_microstate(
        &mut self,
        microstate: &Microstate<
            BodyProperties,
            SiteProperties,
            X,
            Periodic<Rectangle>,
        >,
    ) -> Result<hoomd_gsd::hoomd::Frame<'_>, hoomd_gsd::hoomd::AppendError>
    {
        self.append_frame(microstate.step())?
            .configuration_box(microstate.boundary().shape().to_gsd_box())?
            .configuration_dimensions(Dimensions::Two)?
            .particles_position(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.position)
                    .map(|p| [p[0], p[1], 0.0].into()),
            )?
            .particles_diameter(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.radius.get() * 2.0),
            )
    }
}
// ANCHOR_END: append_microstate

// ANCHOR: simulation_new
impl PolydisperseHardDiskModel {
    /// Construct a new hard disk self-assembly simulation.
    fn new() -> anyhow::Result<PolydisperseHardDiskModel> {
        // ANCHOR_END: simulation_new
        // ANCHOR: parameters
        let seed = 1;
        let minimum_radius = 0.1;
        let maximum_radius = 0.8;
        let initial_packing_fraction = 0.6;
        let target_packing_fraction = 0.72;
        let n_disks = 64_usize.pow(2);
        let maximum_distance = 0.07;
        let macrostate = Isothermal { temperature: 1.0 };
        // ANCHOR_END: parameters

        // ANCHOR: radii
        let radius_distribution = Uniform::new(minimum_radius, maximum_radius)?;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut radii = Vec::with_capacity(n_disks);
        for r in radius_distribution.sample_iter(&mut rng).take(n_disks) {
            radii.push(r.try_into()?);
        }
        // ANCHOR_END: radii

        // ANCHOR: particle_area
        let total_particle_area = radii.iter().fold(0.0, |total, r| {
            let circle = Circle { radius: *r };
            total + circle.volume()
        });
        // ANCHOR_END: particle_area

        // ANCHOR: hamiltonian
        let hamiltonian = PairwiseCutoff(SitePairInteraction {
            maximum_interaction_range: maximum_radius * 2.0,
        });
        let overlap_penalty_hamiltonian =
            PairwiseCutoff(SitePairOverlapPenalty {
                maximum_interaction_range: maximum_radius * 2.0,
            });
        // ANCHOR_END: hamiltonian

        // ANCHOR: microstate
        let initial_box_volume = total_particle_area / initial_packing_fraction;
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
            .seed(seed as u32)
            .boundary(periodic_square)
            .spatial_data(vec_cell)
            .try_build()?;
        // ANCHOR_END: microstate

        // ANCHOR: quick_insert
        let distribution = PolydisperseBodyDistribution {
            boundary: microstate.boundary().clone(),
            radii,
        };
        let quick_insert = QuickInsert::new(distribution, n_disks);
        // ANCHOR_END: quick_insert

        let translate =
            Translate::with_maximum_distance(maximum_distance.try_into()?);
        let translate_sweep = Sweep(translate);

        let target_box_volume = total_particle_area / target_packing_fraction;
        let quick_compress =
            QuickCompress::with_target_volume(target_box_volume.try_into()?);

        // ANCHOR: struct_initialize
        Ok(PolydisperseHardDiskModel {
            microstate,
            hamiltonian,
            translate_sweep,
            quick_insert,
            quick_compress,
            overlap_penalty_hamiltonian,
            macrostate,
            phase: Phase::Initialize,
        })
    }
}
// ANCHOR_END: struct_initialize

#[cfg_attr(feature = "bevy", derive(Resource))]
struct PolydisperseHardDiskModel {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<
        BodyProperties,
        SiteProperties,
        VecCell<SiteKey, 2>,
        Periodic<Rectangle>,
    >,
    /// How sites interact with other sites and fields.
    hamiltonian: PairwiseCutoff<SitePairInteraction>,
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate<PositionVector>>,
    /// Temperature set point.
    macrostate: Isothermal,
    /// Quick compress algorithm
    quick_compress: QuickCompress<Periodic<Rectangle>>,
    /// Quick insert algorithm.
    quick_insert: QuickInsert<PolydisperseBodyDistribution>,
    /// The current phase of the simulation.
    /// How sites interact when inserted and compressed.
    overlap_penalty_hamiltonian: PairwiseCutoff<SitePairOverlapPenalty>,
    phase: Phase,
}

enum Phase {
    Initialize,
    Equilibrate,
}

impl Simulation for PolydisperseHardDiskModel {
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

impl PolydisperseHardDiskModel {
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

        if self.quick_compress.is_complete() {
            self.translate_sweep.tune_default(
                &self.microstate,
                &self.hamiltonian,
                &self.macrostate,
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
            let volume = self.microstate.boundary().volume();
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
    }
}

// Remove the cfg(not(...)) line when using this code outside the hoomd-rs/examples directory.
#[cfg(not(feature = "bevy"))]
fn main() -> anyhow::Result<()> {
    use hoomd_gsd::hoomd::HoomdGsdFile;
    use hoomd_microstate::AppendMicrostate;

    let mut simulation = PolydisperseHardDiskModel::new()?;
    let mut hoomd_gsd_file =
        HoomdGsdFile::create("polydisperse-hard-disk-model.gsd")?;

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
mod polydisperse_hard_disk_model_interactive;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
#[cfg(feature = "bevy")]
use polydisperse_hard_disk_model_interactive::main;

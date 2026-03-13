// ANCHOR: all
// ANCHOR: use
use anyhow::{Context, anyhow};
use strum::VariantNames;
use strum_macros::VariantNames;

use hoomd_geometry::{
    Volume,
    shape::{Circle, Rectangle},
};
use hoomd_gsd::hoomd::{Dimensions, HoomdGsdFile};
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff, SitePairEnergy,
    pairwise::Isotropic,
    univariate::{
        Expanded, LennardJones, OverlapPenalty, UnivariateEnergy,
        WeeksChandlerAnderson,
    },
};
use hoomd_mc::{
    QuickCompress, QuickInsert, Sweep, Translate, Trial, Tune, UniformIn,
};
use hoomd_microstate::{AppendMicrostate, Site};
use hoomd_microstate::{
    Microstate, SiteKey, Transform,
    boundary::Periodic,
    property::{Point, Position},
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::{Cartesian, Metric};
// ANCHOR_END: use

// ANCHOR: type_aliases
type PositionVector = Cartesian<2>;
type BodyProperties = Point<PositionVector>;
// ANCHOR_END: type_aliases

// ANCHOR: type
#[derive(Clone, Copy, Default, PartialEq, VariantNames)]
enum SiteType {
    #[default]
    A,
    B,
}
// ANCHOR_END: type

// ANCHOR: site_properties
#[derive(Clone, Copy, Default, Position)]
struct SiteProperties {
    /// The site's position.
    position: PositionVector,
    /// The site's type.
    site_type: SiteType,
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
    lj_aa: LennardJones<12, 6>,
    wca_ab: WeeksChandlerAnderson,
    maximum_interaction_range: f64,
}
// ANCHOR_END: interaction_type

// ANCHOR: interaction_impl
impl SitePairEnergy<SiteProperties> for SitePairInteraction {
    fn site_pair_energy(
        &self,
        site_properties_i: &SiteProperties,
        site_properties_j: &SiteProperties,
    ) -> f64 {
        let r = site_properties_i
            .position
            .distance(&site_properties_j.position);

        match (site_properties_i.site_type, site_properties_j.site_type) {
            (SiteType::A, SiteType::A) => self.lj_aa.energy(r),
            (SiteType::A, SiteType::B) | (SiteType::B, SiteType::A) => {
                self.wca_ab.energy(r)
            }
            (SiteType::B, SiteType::B) => {
                1.0 / r.powi(12) - f64::exp(-1.0 / 2.0 * r.powi(2))
            }
        }
    }
}
// ANCHOR_END: interaction_impl

// ANCHOR: simulation_new
impl TypeDependentInteractions {
    /// Construct a new type-dependent interactions simulation.
    fn new() -> anyhow::Result<TypeDependentInteractions> {
        // ANCHOR_END: simulation_new
        // ANCHOR: parameters
        let initial_packing_fraction = 0.3;
        let target_packing_fraction = 0.5;
        let n_disks = 512;
        let maximum_distance = 0.07;
        let sigma = 1.0;
        let macrostate = Isothermal { temperature: 1.0 };
        // ANCHOR_END: parameters

        // ANCHOR: hamiltonian
        let lj_aa = LennardJones {
            epsilon: 2.0,
            sigma: 1.0,
        };
        let wca_ab = WeeksChandlerAnderson {
            epsilon: 1.0,
            sigma: 1.0,
        };
        let hamiltonian = PairwiseCutoff(SitePairInteraction {
            lj_aa,
            wca_ab,
            maximum_interaction_range: 2.5,
        });
        // ANCHOR_END: hamiltonian

        // ANCHOR: compress_hamiltonian
        let overlap_penalty = Isotropic {
            interaction: Expanded {
                delta: sigma,
                f: OverlapPenalty::default(),
            },
            r_cut: sigma,
        };

        let overlap_penalty_hamiltonian = PairwiseCutoff(overlap_penalty);
        // ANCHOR_END: compress_hamiltonian

        // ANCHOR: boundary
        let circle = Circle {
            radius: (sigma / 2.0).try_into()?,
        };
        let initial_box_volume =
            n_disks as f64 * circle.volume() / initial_packing_fraction;
        let initial_box_edge_length = initial_box_volume.sqrt();
        let square =
            Rectangle::with_equal_edges(initial_box_edge_length.try_into()?);
        let periodic_square =
            Periodic::new(hamiltonian.maximum_interaction_range(), square)?;
        // ANCHOR_END: boundary

        // ANCHOR: quick_insert
        let distribution = UniformIn {
            boundary: periodic_square.clone(),
            template_sites: vec![SiteProperties {
                site_type: SiteType::A,
                ..SiteProperties::default()
            }],
        };
        let quick_insert_a = QuickInsert::new(distribution, n_disks / 2);

        let distribution = UniformIn {
            boundary: periodic_square.clone(),
            template_sites: vec![SiteProperties {
                site_type: SiteType::B,
                ..SiteProperties::default()
            }],
        };
        let quick_insert_b =
            QuickInsert::new(distribution, n_disks - quick_insert_a.target());
        // ANCHOR_END: quick_insert

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

        let target_box_volume =
            n_disks as f64 * circle.volume() / target_packing_fraction;
        let quick_compress =
            QuickCompress::with_target_volume(target_box_volume.try_into()?);

        Ok(TypeDependentInteractions {
            microstate,
            overlap_penalty_hamiltonian,
            hamiltonian,
            translate_sweep,
            quick_insert_a,
            quick_insert_b,
            quick_compress,
            macrostate,
            phase: Phase::Initialize,
        })
    }
}

#[cfg_attr(feature = "bevy", derive(Resource))]
struct TypeDependentInteractions {
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
    /// Quick insert algorithm for A bodies.
    quick_insert_a: QuickInsert<UniformIn<SiteProperties, Periodic<Rectangle>>>,
    /// Quick insert algorithm for B bodies.
    quick_insert_b: QuickInsert<UniformIn<SiteProperties, Periodic<Rectangle>>>,
    /// Quick compress algorithm
    quick_compress: QuickCompress<Periodic<Rectangle>>,
    /// How sites interact during compression.
    overlap_penalty_hamiltonian:
        PairwiseCutoff<Isotropic<Expanded<OverlapPenalty>>>,
    /// The current phase of the simulation.
    phase: Phase,
}

enum Phase {
    Initialize,
    Equilibrate,
}

impl Simulation for TypeDependentInteractions {
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

impl TypeDependentInteractions {
    fn initialize(&mut self) -> anyhow::Result<()> {
        if self.quick_insert_a.is_complete()
            && self.quick_insert_b.is_complete()
        {
            self.quick_compress.apply(
                &mut self.microstate,
                &self.overlap_penalty_hamiltonian,
                |_| true,
            );
        } else {
            self.quick_insert_a
                .apply(&mut self.microstate, &self.overlap_penalty_hamiltonian);
            self.quick_insert_b
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
            let target_n =
                self.quick_insert_a.target() + self.quick_insert_b.target();
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
            .particles_type_id(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.site_type as u32),
            )?
            .particles_types(SiteType::VARIANTS.iter().copied())
    }
}
// ANCHOR_END: append_microstate

// Remove the cfg(not(...)) line when using this code outside the hoomd-rs/examples directory.
#[cfg(not(feature = "bevy"))]
// ANCHOR: main
fn main() -> anyhow::Result<()> {
    use hoomd_interaction::TotalEnergy;

    let mut simulation = TypeDependentInteractions::new()?;
    let mut hoomd_gsd_file =
        HoomdGsdFile::create("type-dependent-interactions.gsd")?;

    for _ in 0..100_000 {
        simulation.advance()?;
        // ANCHOR_END: main

        // ANCHOR: log_gsd
        if simulation.step().is_multiple_of(10_000) {
            hoomd_gsd_file
                .append_microstate(&simulation.microstate)?
                .log_scalar(
                    "potential_energy",
                    simulation.hamiltonian.total_energy(&simulation.microstate),
                )?
                .log_scalars(
                    "particles/site_energy",
                    simulation
                        .microstate
                        .iter_sites_tag_order()
                        .map(|s| site_energy(&simulation, s)),
                )?;
        }
    }

    Ok(())
}
// ANCHOR_END: log_gsd

#[allow(dead_code, reason = "site_energy is used in the batch mode main()")]
// ANCHOR: site_energy
fn site_energy(
    simulation: &TypeDependentInteractions,
    site: &Site<SiteProperties>,
) -> f64 {
    use hoomd_interaction::DeltaEnergyRemove;

    let body_index = simulation.microstate.body_indices()[site.body_tag]
        .expect("site's parent body should be in the microstate");

    -simulation
        .hamiltonian
        .delta_energy_remove(&simulation.microstate, body_index)
}
// ANCHOR_END: site_energy

// ANCHOR_END: all

#[cfg(feature = "bevy")]
mod type_dependent_interactions_interactive;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
#[cfg(feature = "bevy")]
use type_dependent_interactions_interactive::main;

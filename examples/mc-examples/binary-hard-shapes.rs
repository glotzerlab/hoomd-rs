// ANCHOR: all
use anyhow::{Context, anyhow};
use strum::VariantNames;
use strum_macros::VariantNames;

use hoomd_geometry::{
    BoundingSphereRadius, IntersectsAtGlobal, Scale, Volume,
    shape::{ConvexPolygon, ConvexSurfaceMesh2d, Rectangle},
};
use hoomd_gsd::hoomd::{Dimensions, HoomdGsdFile};
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff, SitePairEnergy,
};
use hoomd_mc::{
    QuickCompress, QuickInsert, Rotate, Sweep, Translate, Trial, Tune,
    TuneOptions, UniformIn,
};
use hoomd_microstate::AppendMicrostate;
use hoomd_microstate::{
    Microstate, SiteKey, Transform,
    boundary::Periodic,
    property::{Orientation, OrientedPoint, Position},
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::{Angle, Cartesian, Metric, Rotate as _, Rotation, Versor};

type PositionVector = Cartesian<2>;
type BodyProperties = OrientedPoint<PositionVector, Angle>;

#[derive(Clone, Copy, Default, PartialEq, VariantNames)]
enum SiteType {
    #[default]
    A,
    B,
}

#[derive(Clone, Copy, Default, Orientation, Position)]
struct SiteProperties {
    /// The site's position.
    position: PositionVector,
    /// The site's orientation.
    orientation: Angle,
    /// The site's type.
    site_type: SiteType,
}

impl Transform<SiteProperties> for BodyProperties {
    fn transform(&self, site_properties: &SiteProperties) -> SiteProperties {
        SiteProperties {
            position: self.position
                + self.orientation.rotate(&site_properties.position),
            orientation: self.orientation.combine(&site_properties.orientation),
            ..*site_properties
        }
    }
}

struct SitePairInteraction {
    /// The A site's shape.
    shape_a: ConvexSurfaceMesh2d,

    /// The B site's shape.
    shape_b: ConvexSurfaceMesh2d,
}

impl MaximumInteractionRange for SitePairInteraction {
    fn maximum_interaction_range(&self) -> f64 {
        let range_aa = self.shape_a.bounding_sphere_radius().get() * 2.0;
        let range_bb = self.shape_b.bounding_sphere_radius().get() * 2.0;
        let range_ab = self.shape_a.bounding_sphere_radius().get()
            + self.shape_b.bounding_sphere_radius().get();

        range_aa.max(range_bb.max(range_ab))
    }
}

impl SitePairEnergy<SiteProperties> for SitePairInteraction {
    fn site_pair_energy(
        &self,
        site_properties_i: &SiteProperties,
        site_properties_j: &SiteProperties,
    ) -> f64 {
        let (shape_i, shape_j) =
            match (site_properties_i.site_type, site_properties_j.site_type) {
                (SiteType::A, SiteType::A) => (&self.shape_a, &self.shape_a),
                (SiteType::A, SiteType::B) => (&self.shape_a, &self.shape_b),
                (SiteType::B, SiteType::A) => (&self.shape_b, &self.shape_a),
                (SiteType::B, SiteType::B) => (&self.shape_b, &self.shape_b),
            };

        if shape_i.intersects_at_global(
            shape_j,
            site_properties_i.position(),
            site_properties_i.orientation(),
            site_properties_j.position(),
            site_properties_j.orientation(),
        ) {
            f64::INFINITY
        } else {
            0.0
        }
    }

    #[inline]
    fn site_pair_energy_initial(
        &self,
        _site_properties_i: &SiteProperties,
        _site_properties_j: &SiteProperties,
    ) -> f64 {
        0.0
    }

    #[inline]
    fn is_only_infinite_or_zero() -> bool {
        true
    }
}

impl BinaryHardShapes {
    /// Construct a new type-dependent interactions simulation.
    fn new() -> anyhow::Result<BinaryHardShapes> {
        let initial_packing_fraction = 0.1;
        let target_packing_fraction = 0.7;
        let n_shapes_a = 32;
        let n_shapes_b = 32;
        let maximum_distance = 0.07;
        let maximum_rotation = 0.05;
        let macrostate = Isothermal { temperature: 1.0 };

        let square = ConvexPolygon::regular(4);
        let shape_a = ConvexSurfaceMesh2d::try_from(square)?;
        let triangle = ConvexPolygon::regular(3);
        let shape_b = ConvexSurfaceMesh2d::try_from(triangle)?;
        let scale = shape_a.vertices()[0].distance(&shape_a.vertices()[1])
            / shape_b.vertices()[0].distance(&shape_b.vertices()[1]);
        let shape_b = shape_b.scale_length(scale.try_into()?);

        let total_site_volume = n_shapes_a as f64 * shape_a.volume()
            + n_shapes_b as f64 * shape_b.volume();

        let hamiltonian =
            PairwiseCutoff(SitePairInteraction { shape_a, shape_b });

        let initial_box_volume = total_site_volume / initial_packing_fraction;
        let initial_box_edge_length = initial_box_volume.sqrt();
        let square =
            Rectangle::with_equal_edges(initial_box_edge_length.try_into()?);
        let periodic_square =
            Periodic::new(hamiltonian.maximum_interaction_range(), square)?;

        let distribution = UniformIn {
            boundary: periodic_square.clone(),
            template_sites: vec![SiteProperties {
                site_type: SiteType::A,
                ..SiteProperties::default()
            }],
        };
        let quick_insert_a = QuickInsert::new(distribution, n_shapes_a);

        let distribution = UniformIn {
            boundary: periodic_square.clone(),
            template_sites: vec![SiteProperties {
                site_type: SiteType::B,
                ..SiteProperties::default()
            }],
        };
        let quick_insert_b = QuickInsert::new(distribution, n_shapes_b);

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

        let target_box_volume = total_site_volume / target_packing_fraction;
        let quick_compress =
            QuickCompress::with_target_volume(target_box_volume.try_into()?);

        Ok(BinaryHardShapes {
            microstate,
            hamiltonian,
            translate_sweep,
            rotate_sweep,
            quick_insert_a,
            quick_insert_b,
            quick_compress,
            macrostate,
            phase: Phase::Initialize,
        })
    }
}

#[cfg_attr(feature = "bevy", derive(Resource))]
struct BinaryHardShapes {
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
    /// Rotation trial moves to apply.
    rotate_sweep: Sweep<Rotate<Angle>>,
    /// Temperature set point.
    macrostate: Isothermal,
    /// Quick insert algorithm for A bodies.
    quick_insert_a: QuickInsert<UniformIn<SiteProperties, Periodic<Rectangle>>>,
    /// Quick insert algorithm for B bodies.
    quick_insert_b: QuickInsert<UniformIn<SiteProperties, Periodic<Rectangle>>>,
    /// Quick compress algorithm
    quick_compress: QuickCompress<Periodic<Rectangle>>,
    /// The current phase of the simulation.
    phase: Phase,
}

enum Phase {
    Initialize,
    Equilibrate,
}

impl Simulation for BinaryHardShapes {
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

impl BinaryHardShapes {
    fn initialize(&mut self) -> anyhow::Result<()> {
        if self.quick_insert_a.is_complete()
            && self.quick_insert_b.is_complete()
        {
            self.quick_compress.apply(
                &mut self.microstate,
                &self.hamiltonian,
                |_| true,
            );
        } else {
            self.quick_insert_a
                .apply(&mut self.microstate, &self.hamiltonian);
            self.quick_insert_b
                .apply(&mut self.microstate, &self.hamiltonian);
        }

        self.translate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &Isothermal { temperature: 1.0 },
        );
        self.rotate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
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
        self.rotate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
        );
    }
}

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
            .particles_orientation(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.orientation.theta)
                    .map(|a| {
                        Versor::from_axis_angle(
                            [0.0, 0.0, 1.0]
                                .try_into()
                                .expect("hard-coded vector can be normalized"),
                            a,
                        )
                    }),
            )?
            .particles_type_id(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.site_type as u32),
            )?
            .particles_types(SiteType::VARIANTS.iter().copied())
    }
}

// Remove the cfg(not(...)) line when using this code outside the hoomd-rs/examples directory.
#[cfg(not(feature = "bevy"))]
fn main() -> anyhow::Result<()> {
    use hoomd_interaction::TotalEnergy;

    let mut simulation = BinaryHardShapes::new()?;
    let mut hoomd_gsd_file = HoomdGsdFile::create("binary-hard-shapes.gsd")?;

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
mod binary_hard_shapes_interactive;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
#[cfg(feature = "bevy")]
use binary_hard_shapes_interactive::main;

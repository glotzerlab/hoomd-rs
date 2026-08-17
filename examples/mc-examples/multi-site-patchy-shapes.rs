// ANCHOR: all
use std::f64::consts::PI;

use hoomd_derive::{Orientation, Position};
use hoomd_geometry::{
    Volume,
    shape::{ConvexPolygon, ConvexSurfaceMesh2d, Rectangle},
};
use hoomd_gsd::hoomd::{Dimensions, HoomdGsdFile};
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff, SitePairEnergy,
    pairwise::{HardShape, Isotropic},
    univariate::Boxcar,
};
use hoomd_mc::{Rotate, Sweep, Translate, Trial};
use hoomd_microstate::{
    AppendMicrostate, Body, Microstate, Replicate, SiteKey, Transform,
    boundary::Periodic, property::OrientedPoint,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::Rotate as _;
use hoomd_vector::{Angle, Cartesian, Rotation, Versor};

type PositionVector = Cartesian<2>;
type Orientation = Angle;
type BodyProperties = OrientedPoint<Cartesian<2>, Orientation>;

#[derive(Clone, Copy, Default, PartialEq, VariantNames)]
enum SiteType {
    #[default]
    A,
    P,
}

#[derive(Clone, Copy, Default, Position, Orientation)]
struct SiteProperties {
    /// The site's position.
    position: PositionVector,
    /// The site's orientation
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
    hard_shape_aa: HardShape<ConvexSurfaceMesh2d>,
    boxcar_bb: Isotropic<Boxcar>,
}

impl MaximumInteractionRange for SitePairInteraction {
    fn maximum_interaction_range(&self) -> f64 {
        self.boxcar_bb
            .maximum_interaction_range()
            .max(self.hard_shape_aa.maximum_interaction_range())
    }
}

impl SitePairEnergy<SiteProperties> for SitePairInteraction {
    fn site_pair_energy(
        &self,
        site_properties_i: &SiteProperties,
        site_properties_j: &SiteProperties,
    ) -> f64 {
        match (site_properties_i.site_type, site_properties_j.site_type) {
            (SiteType::A, SiteType::A) => self
                .hard_shape_aa
                .site_pair_energy(site_properties_i, site_properties_j),
            (SiteType::A, SiteType::P) | (SiteType::P, SiteType::A) => 0.0,
            (SiteType::P, SiteType::P) => self
                .boxcar_bb
                .site_pair_energy(site_properties_i, site_properties_j),
        }
    }

    fn site_pair_energy_initial(
        &self,
        site_properties_i: &SiteProperties,
        site_properties_j: &SiteProperties,
    ) -> f64 {
        match (site_properties_i.site_type, site_properties_j.site_type) {
            (SiteType::A, SiteType::A) => self
                .hard_shape_aa
                .site_pair_energy_initial(site_properties_i, site_properties_j),
            (SiteType::A, SiteType::P) | (SiteType::P, SiteType::A) => 0.0,
            (SiteType::P, SiteType::P) => self
                .boxcar_bb
                .site_pair_energy_initial(site_properties_i, site_properties_j),
        }
    }
}

#[cfg_attr(feature = "bevy", derive(Resource))]
struct MultiSitePatchyShape {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<
        BodyProperties,
        SiteProperties,
        VecCell<SiteKey, 2>,
        Periodic<Rectangle>,
    >,
    /// How sites interact with other sites and fields.
    hamiltonian: PairwiseCutoff<SitePairInteraction>,
    /// Translation trial moves to apply.
    translate_sweep: Sweep<Translate<PositionVector>>,
    /// Rotation trial moves to apply.
    rotate_sweep: Sweep<Rotate<Orientation>>,
    /// Temperature set point.
    macrostate: Isothermal,
}

impl MultiSitePatchyShape {
    /// Construct a new multi-site patchy self-assembly simulation.
    fn new() -> anyhow::Result<MultiSitePatchyShape> {
        let epsilon = -4.0;
        let sigma_patch = 0.3;
        let packing_fraction = 0.4;
        let n_replicates_side = 32;
        let maximum_distance = 0.07;
        let maximum_rotation = 0.05;
        let macrostate = Isothermal { temperature: 1.0 };
        let sites = vec![
            SiteProperties {
                position: [0.0, 0.0].into(),
                orientation: Angle::default(),
                site_type: SiteType::A,
            },
            SiteProperties {
                position: [0.0, 0.4].into(),
                orientation: Angle::default(),
                site_type: SiteType::P,
            },
            SiteProperties {
                position: [-(PI / 6.0).cos() * 0.4, -(PI / 6.0).sin() * 0.4]
                    .into(),
                orientation: Angle::default(),
                site_type: SiteType::P,
            },
            SiteProperties {
                position: [
                    -(5.0 * PI / 6.0).cos() * 0.4,
                    -(5.0 * PI / 6.0).sin() * 0.4,
                ]
                .into(),
                orientation: Angle::default(),
                site_type: SiteType::P,
            },
        ];

        let regular_hexagon = ConvexPolygon::regular(6);
        let mesh = ConvexSurfaceMesh2d::try_from(regular_hexagon)?;
        let hard_shape_aa = HardShape(mesh.clone());
        let boxcar_bb = Isotropic {
            interaction: Boxcar {
                epsilon,
                left: 0.0,
                right: sigma_patch,
            },
            r_cut: sigma_patch,
        };
        let hamiltonian = PairwiseCutoff(SitePairInteraction {
            hard_shape_aa,
            boxcar_bb,
        });

        let unit_cell_volume = mesh.volume() / packing_fraction;
        let unit_cell_edge_length = unit_cell_volume.sqrt();
        let unit_cell_rectangle =
            Rectangle::with_equal_edges(unit_cell_edge_length.try_into()?);

        let periodic_unit_cell = Periodic::new(0.0, unit_cell_rectangle)?;

        let vec_cell = VecCell::builder()
            .nominal_search_radius(
                hamiltonian.maximum_interaction_range().try_into()?,
            )
            .build();
        let microstate = Microstate::builder()
            .boundary(periodic_unit_cell)
            .spatial_data(vec_cell)
            .bodies([Body {
                properties: OrientedPoint {
                    position: Cartesian::default(),
                    orientation: Angle::default(),
                },
                sites,
            }])
            .try_build()?
            .replicate_with_maximum_interaction_range(
                [n_replicates_side; 2],
                hamiltonian.maximum_interaction_range(),
            )?;

        let translate =
            Translate::with_maximum_distance(maximum_distance.try_into()?);
        let translate_sweep = Sweep(translate);

        let rotate =
            Rotate::with_maximum_rotation(maximum_rotation.try_into()?);
        let rotate_sweep = Sweep(rotate);

        Ok(MultiSitePatchyShape {
            microstate,
            hamiltonian,
            translate_sweep,
            rotate_sweep,
            macrostate,
        })
    }
}

impl Simulation for MultiSitePatchyShape {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        self.equilibrate();
        self.microstate.increment_step();

        Ok(())
    }

    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl MultiSitePatchyShape {
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
    use hoomd_gsd::hoomd::HoomdGsdFile;
    use hoomd_microstate::AppendMicrostate;

    let mut simulation = MultiSitePatchyShape::new()?;
    let mut hoomd_gsd_file =
        HoomdGsdFile::create("multi-site-patchy-shapes.gsd")?;

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
mod multi_site_patchy_shapes_interactive;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
#[cfg(feature = "bevy")]
use multi_site_patchy_shapes_interactive::main;
use strum::VariantNames;
use strum_macros::VariantNames;

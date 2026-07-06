use hoomd_linear_algebra::matrix::Matrix;
// ANCHOR: all
use itertools::Itertools;
use strum::VariantNames;
use strum_macros::VariantNames;

use hoomd_geometry::shape::Cuboid;
use hoomd_interaction::{
    MaximumInteractionRange, PairwiseCutoff, Rigid, SitePairForceAndVirial, SitePairForceVirialAndTorque, pairwise::Isotropic, univariate::{LennardJones, WeeksChandlerAnderson}
};
use hoomd_md::{
    RotationalMotion, ThermalizeAngularMomentum, ThermalizeMomentum, ZeroCenterAngularMomentum, ZeroCenterMomentum, method::ConstantVolume, thermostat::Bussi
};
use hoomd_microstate::{
    AppendMicrostate, Body, Microstate, SiteKey, Transform, boundary::Periodic, property::{DynamicOrientedPoint, Position}
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::{Cartesian, Rotate, Versor};
use hoomd_gsd::hoomd::{Dimensions, HoomdGsdFile};

type PositionVector = Cartesian<3>;
type BodyProperties = DynamicOrientedPoint<Cartesian<3>, Versor>;

#[derive(Clone, Copy, Default, PartialEq, VariantNames)]
enum SiteType {
    #[default]
    A,
    B,
}

#[derive(Clone, Copy, Default, Position)]
struct SiteProperties {
    /// The site's position.
    position: PositionVector,
    /// The site's type.
    site_type: SiteType,
}

impl Transform<SiteProperties> for BodyProperties {
    fn transform(&self, site_properties: &SiteProperties) -> SiteProperties {
        SiteProperties {
            position: self.position
                + self.orientation.rotate(&site_properties.position),
            ..*site_properties
        }
    }
}

struct SitePairInteraction {
    wca_aa: Isotropic<WeeksChandlerAnderson>,
    lj_bb: Isotropic<LennardJones<12, 6>>,
}

impl MaximumInteractionRange for SitePairInteraction {
    fn maximum_interaction_range(&self) -> f64 {
        self.lj_bb.maximum_interaction_range().max(self.wca_aa.maximum_interaction_range())
    }
}

impl SitePairForceVirialAndTorque<SiteProperties> for SitePairInteraction {
    type Force = Cartesian<3>;

    fn site_pair_force_virial_and_torque(
        &self,
        site_properties_i: &SiteProperties,
        site_properties_j: &SiteProperties,
    ) -> (Self::Force, Matrix<3,3>, Cartesian<3>) {
        let (force, virial) = match (site_properties_i.site_type, site_properties_j.site_type) {
            (SiteType::A, SiteType::A) => self.wca_aa.site_pair_force_and_virial(site_properties_i, site_properties_j),
            (SiteType::A, SiteType::B) | (SiteType::B, SiteType::A) => (Cartesian::default(), Matrix::<3,3>::default()),
            (SiteType::B, SiteType::B) => self.lj_bb.site_pair_force_and_virial(site_properties_i, site_properties_j),
        };

        (force, virial, Cartesian::default())
    }
}

// Remove the cfg_attr(...) line when using this code outside the hoomd-rs/examples directory.
#[cfg_attr(feature = "bevy", derive(Resource))]
struct PatchyBody3D {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<
        BodyProperties,
        SiteProperties,
        VecCell<SiteKey, 3>,
        Periodic<Cuboid>,
    >,
    /// How bodies interact with other bodies.
    interaction_model: Rigid<PairwiseCutoff<SitePairInteraction>>,
    /// Constant volume, constant temperature integration method..
    constant_volume: ConstantVolume<Bussi>,
    /// Temperature set point.
    macrostate: Isothermal,
}

impl PatchyBody3D {
    /// Construct a new Lennard Jones fluid simulation.
    fn new() -> anyhow::Result<PatchyBody3D> {
        let temperature = 0.5;
        let density: f64 = 0.01;
        let delta_t = 0.005;
        let n_bodies = 16;
        let sites = vec![
            SiteProperties { position: [-0.3, -0.3, -0.3].into(), site_type: SiteType::A },
            SiteProperties { position: [-0.3, -0.3, 0.3].into(), site_type: SiteType::A },
            SiteProperties { position: [-0.3, 0.3, -0.3].into(), site_type: SiteType::A },
            SiteProperties { position: [-0.3, 0.3, 0.3].into(), site_type: SiteType::A },
            SiteProperties { position: [0.3, -0.3, -0.3].into(), site_type: SiteType::A },
            SiteProperties { position: [0.3, -0.3, 0.3].into(), site_type: SiteType::A },
            SiteProperties { position: [0.3, 0.3, -0.3].into(), site_type: SiteType::A },
            SiteProperties { position: [0.3, 0.3, 0.3].into(), site_type: SiteType::A },

            SiteProperties { position: [-0.3, 0.0, 0.0].into(), site_type: SiteType::B },
            SiteProperties { position: [0.3, 0.0, 0.0].into(), site_type: SiteType::B },
            SiteProperties { position: [0.0, 0.3, 0.0].into(), site_type: SiteType::B },
            SiteProperties { position: [0.0, -0.3, 0.0].into(), site_type: SiteType::B },
            SiteProperties { position: [0.0, 0.0, 0.3].into(), site_type: SiteType::B },
            SiteProperties { position: [0.0, 0.0, -0.3].into(), site_type: SiteType::B },
            ];

        let box_length = (n_bodies as f64 / density).cbrt();
        let macrostate = Isothermal { temperature };

        let interaction_model = Rigid(PairwiseCutoff(
            SitePairInteraction { wca_aa: Isotropic {
            interaction: WeeksChandlerAnderson { epsilon: 1.0, sigma: 1.0 },
            r_cut: 2.0_f64.powf(1.0/6.0),
        },
            lj_bb: Isotropic { interaction: LennardJones { epsilon: 1.0, sigma: 1.0 },
            r_cut: 3.0 }}));

        let cube = Cuboid::with_equal_edges(box_length.try_into()?);
        let vec_cell = VecCell::builder()
            .nominal_search_radius(interaction_model.maximum_interaction_range().try_into()?)
            .build();
        let boundary = Periodic::new(interaction_model.maximum_interaction_range(), cube)?;
        let mut microstate = Microstate::builder()
            .seed(12)
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

            microstate.add_body(Body {
                properties: DynamicOrientedPoint {
                    position: Cartesian::try_from(position)?,
                    ..Default::default()
                },
                sites: sites.clone(),
            })?;
        }

        microstate.thermalize_momentum(temperature);
        microstate.thermalize_angular_momentum(temperature);
        microstate.zero_center_angular_momentum();
        microstate.zero_center_momentum();

        let thermostat = Bussi::new(0.0);
        let constant_volume = ConstantVolume::builder(delta_t)
            .thermostat(thermostat)
            .build();

        Ok(PatchyBody3D {
            microstate,
            interaction_model,
            constant_volume,
            macrostate,
        })
    }
}

impl Simulation for PatchyBody3D {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        self.constant_volume.integrate_translation_and_rotation(
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

impl<X> AppendMicrostate<BodyProperties, SiteProperties, X, Periodic<Cuboid>>
    for HoomdGsdFile
{
    #[inline]
    fn append_microstate(
        &mut self,
        microstate: &Microstate<
            BodyProperties,
            SiteProperties,
            X,
            Periodic<Cuboid>,
        >,
    ) -> Result<hoomd_gsd::hoomd::Frame<'_>, hoomd_gsd::hoomd::AppendError>
    {
        self.append_frame(microstate.step())?
            .configuration_box(microstate.boundary().shape().to_gsd_box())?
            .configuration_dimensions(Dimensions::Three)?
            .particles_position(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.position)
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

    let mut simulation = PatchyBody3D::new()?;
    let mut hoomd_gsd_file = HoomdGsdFile::create("patchy-body-3d.gsd")?;

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
mod patchy_body_3d_interactive;
#[cfg(feature = "bevy")]
use bevy::prelude::Resource;
#[cfg(feature = "bevy")]
use patchy_body_3d_interactive::main;

#![allow(non_snake_case)]

//! Stuff for 2D simulations.

use hoomd_geometry::shape::Rectangle;
use hoomd_gsd::hoomd::{AppendError, Dimensions, Frame, HoomdGsdFile};
use hoomd_interaction::{NetBodyForceVirialAndTorque, SitePairForceVirialAndTorque, TotalEnergy};
use hoomd_md::{RotationalMotion, ThermalizeAngularMomentum, ThermalizeMomentum, ZeroCenterAngularMomentum, ZeroCenterMomentum, method::{ConstantVolume, Langevin}, thermostat::{Bussi, MartynaTuckermanTobiasKlein, NoThermostat}};
use hoomd_microstate::{AppendMicrostate, Body, SiteKey, Transform, boundary::Periodic, property::DynamicOrientedPoint};
use hoomd_simulation::macrostate::Isothermal;
use hoomd_spatial::VecCell;
use hoomd_vector::{Angle, Cartesian, Outer, Rotate, Wedge};
use itertools::Itertools;
use strum::VariantNames;

use crate::{simulation::{DUMBBELL_LJ, DumbbellInteraction, InteractionModels, Macrostate, SPHERE_LJ, SiteVariants, Thermostats}, workspace::{BodyVariants, StatePoint}};

// The dimension-specific constants.
const NDIMS: usize = 2;
const DUMBBELL_END_POSITON: Cartesian<NDIMS> = Cartesian { coordinates: [0.0, 0.25] };
const ZERO_MOI: f64 = 0.0;

// The essential types.
type Position = Cartesian<NDIMS>;
type Orientation = Angle;

pub type BodyProperties = DynamicOrientedPoint<Position, Orientation>;
type Spatial = VecCell<SiteKey, NDIMS>;
type Boundary = Periodic<Rectangle>;


/// The site properties type.
#[derive(Clone, Copy, Default, hoomd_derive::Position)]
pub struct SiteProperties {
    position: Position,
    site_type: SiteVariants,
}

impl Transform<SiteProperties> for BodyProperties {
    fn transform(&self, site_properties: &SiteProperties) -> SiteProperties {
        SiteProperties {
            position: self.position
                + self.orientation.rotate(&site_properties.position),
            site_type: site_properties.site_type
        }
    }
}

/// The microstate type.
type Microstate = hoomd_microstate::Microstate<BodyProperties, SiteProperties, Spatial, Boundary>;

impl AppendMicrostate<BodyProperties, SiteProperties, Spatial, Boundary> for HoomdGsdFile {
    #[inline]
    fn append_microstate(&mut self, microstate: &Microstate) -> Result<Frame<'_>, AppendError> {
        self.append_frame(microstate.step())?
            .configuration_box(microstate.boundary().shape().to_gsd_box())?
            .configuration_dimensions(Dimensions::Two)?
            .particles_position(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.position)
                    .map(|p| [p[0], p[1], 0.0].into()), // update this for 3D
            )?
            .particles_type_id(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.site_type as u32),
            )?
            .particles_types(SiteVariants::VARIANTS.iter().copied())
    }
}

// Implementations for interactions.
impl SitePairForceVirialAndTorque<SiteProperties> for DumbbellInteraction {
    type Force = Position;

    fn site_pair_force_virial_and_torque(
        &self,
        site_properties_i: &SiteProperties,
        site_properties_j: &SiteProperties,
    ) -> (
        Self::Force,
        <Self::Force as Outer>::Tensor,
        <Self::Force as Wedge>::Bivector,
    ) {
        let (force, virial, torque) = match (site_properties_i.site_type, site_properties_j.site_type) {
            (SiteVariants::A, SiteVariants::A) => self
                .aa
                .site_pair_force_virial_and_torque(site_properties_i, site_properties_j),
            (SiteVariants::B, SiteVariants::B) => self
                .bb
                .site_pair_force_virial_and_torque(site_properties_i, site_properties_j),
            _ => self
                .ab
                .site_pair_force_virial_and_torque(site_properties_i, site_properties_j),
        };
        (force, virial, torque)
    }
}

impl NetBodyForceVirialAndTorque<BodyProperties, SiteProperties, Spatial, Boundary>
    for InteractionModels
{
    type Force = Position;

    fn net_body_force_virial_and_torque(
        &self,
        microstate: &Microstate,
        body_index: usize,
    ) -> (
        Self::Force,
        <Self::Force as Outer>::Tensor,
        <Self::Force as hoomd_vector::Wedge>::Bivector,
    ) {
        match self {
            Self::Sphere(inner_model) => inner_model
                .net_body_force_virial_and_torque(microstate, body_index),
            Self::Dumbbell(inner_model) => inner_model
                .net_body_force_virial_and_torque(microstate, body_index),
        }
    }
}

impl TotalEnergy<Microstate> for InteractionModels {
    fn total_energy(&self, microstate: &Microstate) -> f64 {
        match self {
            Self::Sphere(inner_model) => TotalEnergy::total_energy(inner_model, microstate),
            Self::Dumbbell(inner_model) => TotalEnergy::total_energy(inner_model, microstate),
        }
    }
}

/// The integration method types.
#[derive(Debug, Clone)]
pub enum Methods {
    ConstantVolume(hoomd_md::method::ConstantVolume<Thermostats, Thermostats>),
    Langevin(hoomd_md::method::Langevin),
}

impl Methods {
    pub fn integrate(
        &mut self,
        microstate: &mut Microstate,
        macrostate: &Macrostate,
        interaction_model: &InteractionModels,
    ) {
        match self {
            Self::ConstantVolume(constant_volume) => constant_volume
                .integrate_translation_and_rotation(microstate, macrostate, interaction_model),
            Self::Langevin(langevin) => langevin
                .integrate_translation_and_rotation(microstate, macrostate, interaction_model,)
        }
    }
}

/// Parameters that determine and describe initial state and macrostate.
#[derive(Debug)]
pub struct SystemParams {
    pub particle_type: BodyVariants,
    pub particles_per_side: usize,
}

impl SystemParams {
    /// Create a new SystemParams from an entry.
    pub fn from_state_point(state_point: &StatePoint) -> Self {
        Self {
            particle_type: state_point.particle_type,
            particles_per_side: state_point.particles_per_side,
        }
    }

    /// System number density.
    pub fn density(&self) -> f64 {
        match self.particle_type {
            BodyVariants::Sphere => crate::DENSITY_SPHERE_2D,       // update this for 3D
            BodyVariants::Dumbbell => crate::DENSITY_DUMBBELL_2D,   // update this for 3D
        }
    }

    // System body template.
    pub fn body(&self, position: &Position) -> Body<BodyProperties, SiteProperties> {
        match self.particle_type {
            BodyVariants::Sphere => Body::single_site(
                BodyProperties {
                    moment_of_inertia: ZERO_MOI,
                    position: *position,
                    ..Default::default()
                },
                SiteProperties::default(),
            ),
            BodyVariants::Dumbbell => Body {
                properties: DynamicOrientedPoint {
                    position: *position,
                    ..Default::default()
                },
                sites: vec![
                    SiteProperties::default(),
                    SiteProperties {
                        position: DUMBBELL_END_POSITON,
                        site_type: SiteVariants::B,
                    }
                ],
            },
        }
    }

    /// Macrostate.
    pub fn macrostate(&self) -> Isothermal {
        Isothermal {
            temperature: crate::KT,
        }
    }

    /// Interaction model.
    pub fn interaction_model(&self) -> InteractionModels {
        match self.particle_type {
            BodyVariants::Sphere => InteractionModels::Sphere(SPHERE_LJ),
            BodyVariants::Dumbbell => InteractionModels::Dumbbell(DUMBBELL_LJ),
        }
    }
}


/// Parameters that determine and describe the simulation procedure.
pub struct ProcedureParams {
    pub method: crate::workspace::MethodVariants,
    pub thermostat: crate::workspace::ThermostatVariants,
}

impl ProcedureParams {
    /// Create a new ProcedureParams from an entry.
    pub fn from_state_point(state_point: &StatePoint) -> Self {
        Self {
            method: state_point.method,
            thermostat: state_point.thermostat,
        }
    }
}

/// Create a thermostat from procedure and system parameters.
pub fn make_thermostat(system: &SystemParams, procedure: &ProcedureParams) -> Thermostats {
    let mut rng = hoomd_rand::Counter::new(0, 0, 0).make_rng();

    let n_bodies = system.particles_per_side.pow(NDIMS.try_into().unwrap());

    match (procedure.thermostat, system.particle_type) {
        (crate::workspace::ThermostatVariants::NoThermostat, _) => {
            Thermostats::NoThermostat(NoThermostat)
        }
        (crate::workspace::ThermostatVariants::Bussi, _) => {
            Thermostats::Bussi(Bussi::new(crate::TAU))
        }
        (crate::workspace::ThermostatVariants::MTTK, BodyVariants::Sphere) => {
            Thermostats::MTTK(MartynaTuckermanTobiasKlein::thermalized(
                &mut rng,
                crate::TAU.try_into().unwrap(),
                &system.macrostate(),
                2 * n_bodies,   // TODO: check, update for 3D
            ))
        }
        (crate::workspace::ThermostatVariants::MTTK, BodyVariants::Dumbbell) => {
            Thermostats::MTTK(MartynaTuckermanTobiasKlein::thermalized(
                &mut rng,
                crate::TAU.try_into().unwrap(),
                &system.macrostate(),
                3 * n_bodies,   // TODO: check, update for 3D
            ))
        }
    }
}


/// Create an integration method from procedure and system parameters.
pub fn make_method(system: &SystemParams, procedure: &ProcedureParams) -> Methods {
    match procedure.method {
        crate::workspace::MethodVariants::ConstantVolume => Methods::ConstantVolume(
            ConstantVolume::builder(crate::DT)
                .thermostat(make_thermostat(system, procedure))
                .build(),
        ),
        crate::workspace::MethodVariants::Langevin => {
            Methods::Langevin(Langevin { delta_t: crate::DT })
        }
    }
}

// Create the initial microstate.
pub fn make_microstate(system: &SystemParams) -> anyhow::Result<Microstate> {
    // Get data for the box
    let box_volume = (system.particles_per_side as f64).powi(NDIMS.try_into()?) / system.density();
    let box_side_length = box_volume.powf(1.0 / (NDIMS as f64));

    // Create spatial data structure
    let vec_cell = VecCell::builder()
        .nominal_search_radius(crate::R_CUT.try_into()?)
        .build();

    // Create boundary condition
    let boundary = Periodic::new(
        crate::R_CUT,
        Rectangle::with_equal_edges(box_side_length.try_into()?),   // update this for 3D
    )?;

    // Create microstate from the pieces above
    let mut microstate = hoomd_microstate::Microstate::builder()
        .spatial_data(vec_cell)
        .boundary(boundary)
        .try_build()
        .unwrap();

    // Add bodies to the microstate
    let n_bodies = system.particles_per_side.pow(NDIMS.try_into()?);
    let spacing = box_side_length / (system.particles_per_side as f64);
    for index in [(0..system.particles_per_side), (0..system.particles_per_side)]   // update this for 3D
        .into_iter()
        .multi_cartesian_product()
        .take(n_bodies)
    {
        let position: Vec<_> = index
            .iter()
            .map(|x| spacing * (*x as f64) - box_side_length / 2.0)
            .collect();

            microstate.add_body(system.body(&Cartesian::try_from(position)?))?;
    }

    // Thermalize and remove collective motion
    microstate.thermalize_momentum(crate::KT);
    
    if let BodyVariants::Dumbbell = system.particle_type {
        microstate.thermalize_angular_momentum(crate::KT);
    }

    microstate.zero_center_momentum();
    microstate.zero_center_angular_momentum();

    Ok(microstate)
}

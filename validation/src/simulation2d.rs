#![allow(non_snake_case)]

use rand::Rng;

use crate::workspace::{BodyVariants, StatePoint};
use hoomd_geometry::shape::Rectangle;
use hoomd_interaction::{
    MaximumInteractionRange, NetBodyForceVirialAndTorque, PairwiseCutoff, Rigid, SitePairForceAndVirial, SitePairForceVirialAndTorque, pairwise::Isotropic, univariate::LennardJones,
};
use hoomd_md::{
    Thermostat,
    RotationalMotion,
    TranslationalKineticEnergy,
    RotationalKineticEnergy,
    method::{ConstantVolume, Langevin},
    thermostat::{Bussi, MartynaTuckermanTobiasKlein, NoThermostat},
};
use hoomd_microstate::{SiteKey, Transform, boundary::Periodic, property::{DynamicOrientedPoint, Point}};
use hoomd_simulation::macrostate::{Isothermal};
use hoomd_spatial::VecCell;
use hoomd_vector::{Angle, Cartesian, Outer, Rotate, Wedge};

/// Interaction constants
pub const SPHERE_LJ: Rigid<PairwiseCutoff<Isotropic<LennardJones<12, 6>>>> =
    Rigid(PairwiseCutoff(Isotropic {
        interaction: LennardJones {
            epsilon: crate::EPSILON,
            sigma: crate::SIGMA,
        },
        r_cut: crate::R_CUT,
    }));

pub const DUMBBELL_LJ: Rigid<PairwiseCutoff<DumbbellInteraction>> =
    Rigid(PairwiseCutoff(DumbbellInteraction {
        aa: Isotropic {
            interaction: LennardJones {
                epsilon: crate::EPSILON,
                sigma: crate::SIGMA / 5.0,
            },
            r_cut: crate::R_CUT,
        },
        bb: Isotropic {
            interaction: LennardJones {
                epsilon: crate::EPSILON / 5.0,
                sigma: crate::SIGMA,
            },
            r_cut: crate::R_CUT,
        },
        ab: Isotropic {
            interaction: LennardJones {
                epsilon: crate::EPSILON / 5.0,
                sigma: crate::SIGMA / 5.0,
            },
            r_cut: crate::R_CUT,
        },
    }));


// The essential types.
type Position = Cartesian<2>;
type Orientation = Angle;

type BodyProperties = DynamicOrientedPoint<Position, Orientation>;
type Spatial = VecCell<SiteKey, 2>;
type Boundary = Periodic<Rectangle>;

type Macrostate = Isothermal;


// The site types.
type SphereSiteProperties = Point<Position>;

#[derive(Clone, Copy, Default, PartialEq)]
enum DumbbellSiteVariants {
    #[default]
    A,
    B,
}

#[derive(Clone, Copy, Default, hoomd_derive::Position)]
struct DumbbellSiteProperties {
    position: Position,
    site_type: DumbbellSiteVariants,
}

#[derive(Clone, Copy)]
enum SiteProperties {
    Sphere(SphereSiteProperties),
    Dumbbell(DumbbellSiteProperties),
}

impl hoomd_microstate::property::Position for SiteProperties {
    type Position = Position;

    fn position(&self) -> &Self::Position {
        match self {
            Self::Sphere(inner_properties) => inner_properties.position(),
            Self::Dumbbell(inner_properties) => inner_properties.position(),
        }
    }

    fn position_mut(&mut self) -> &mut Self::Position {
        match self {
            Self::Sphere(inner_properties) => inner_properties.position_mut(),
            Self::Dumbbell(inner_properties) => inner_properties.position_mut(),
        }
    }
}

impl Transform<SiteProperties> for BodyProperties {
    fn transform(&self, site_properties: &SiteProperties) -> SiteProperties {
        match site_properties {
            SiteProperties::Sphere(inner_properties) => SiteProperties::Sphere(
                Point {
                    position: self.position + self.orientation.rotate(&inner_properties.position)
                }
            ),
            SiteProperties::Dumbbell(inner_properties) => SiteProperties::Dumbbell(
                DumbbellSiteProperties {
                    position: self.position + self.orientation.rotate(&inner_properties.position),
                    site_type: inner_properties.site_type
                }
            )
        }
    }
}


// The microstate type.
type Microstate = hoomd_microstate::Microstate<BodyProperties, SiteProperties, Spatial, Boundary>;


/// The interaction types.
struct DumbbellInteraction {
    aa: Isotropic<LennardJones<12, 6>>,
    bb: Isotropic<LennardJones<12, 6>>,
    ab: Isotropic<LennardJones<12, 6>>,
}

impl MaximumInteractionRange for DumbbellInteraction {
    fn maximum_interaction_range(&self) -> f64 {
        self.aa
            .maximum_interaction_range()
            .max(self.bb.maximum_interaction_range())
    }
}

impl SitePairForceVirialAndTorque<DumbbellSiteProperties> for DumbbellInteraction {
    type Force = Position;

    fn site_pair_force_virial_and_torque(
        &self,
        site_properties_i: &DumbbellSiteProperties,
        site_properties_j: &DumbbellSiteProperties,
    ) -> (
        Self::Force,
        <Self::Force as Outer>::Tensor,
        <Self::Force as Wedge>::Bivector,
    ) {
        let (force, virial, torque) = match (site_properties_i.site_type, site_properties_j.site_type) {
            (DumbbellSiteVariants::A, DumbbellSiteVariants::A) => self
                .aa
                .site_pair_force_virial_and_torque(site_properties_i, site_properties_j),
            (DumbbellSiteVariants::B, DumbbellSiteVariants::B) => self
                .bb
                .site_pair_force_virial_and_torque(site_properties_i, site_properties_j),
            _ => self
                .ab
                .site_pair_force_virial_and_torque(site_properties_i, site_properties_j),
        };
        (force, virial, torque)
    }
}

pub enum InteractionModels {
    Sphere(Rigid<PairwiseCutoff<Isotropic<LennardJones<12, 6>>>>),
    Dumbbell(Rigid<PairwiseCutoff<DumbbellInteraction>>),
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


// The thermostat types.
#[derive(Debug, Clone)]
pub enum Thermostats {
    NoThermostat(hoomd_md::thermostat::NoThermostat),
    Bussi(hoomd_md::thermostat::Bussi),
    MTTK(hoomd_md::thermostat::MartynaTuckermanTobiasKlein),
}

impl Thermostat<Macrostate> for Thermostats {
    fn integrate_half_step_one<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        macrostate: &Macrostate,
        delta_t: f64,
        kinetic_energy: f64,
        degrees_of_freedom: usize,
    ) -> f64 {
        match self {
            Self::NoThermostat(inner_thermostat) => inner_thermostat.integrate_half_step_one(
                rng,
                macrostate,
                delta_t,
                kinetic_energy,
                degrees_of_freedom,
            ),
            Self::Bussi(inner_thermostat) => inner_thermostat.integrate_half_step_one(
                rng,
                macrostate,
                delta_t,
                kinetic_energy,
                degrees_of_freedom,
            ),
            Self::MTTK(inner_thermostat) => inner_thermostat.integrate_half_step_one(
                rng,
                macrostate,
                delta_t,
                kinetic_energy,
                degrees_of_freedom,
            ),
        }
    }

    fn integrate_half_step_two<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        macrostate: &Macrostate,
        delta_t: f64,
        kinetic_energy: f64,
        degrees_of_freedom: usize,
    ) -> f64 {
        match self {
            Self::NoThermostat(inner_thermostat) => inner_thermostat.integrate_half_step_two(
                rng,
                macrostate,
                delta_t,
                kinetic_energy,
                degrees_of_freedom,
            ),
            Self::Bussi(inner_thermostat) => inner_thermostat.integrate_half_step_two(
                rng,
                macrostate,
                delta_t,
                kinetic_energy,
                degrees_of_freedom,
            ),
            Self::MTTK(inner_thermostat) => inner_thermostat.integrate_half_step_two(
                rng,
                macrostate,
                delta_t,
                kinetic_energy,
                degrees_of_freedom,
            ),
        }
    }
}


// The integration method types.
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
    pub ndims: usize,
    pub particle_type: BodyVariants,
    pub particles_per_side: usize,
}

impl SystemParams {
    /// Create a new SystemParams from an entry.
    pub fn from_state_point(state_point: &StatePoint) -> Self {
        Self {
            ndims: state_point.ndims,
            particle_type: state_point.particle_type,
            particles_per_side: state_point.particles_per_side,
        }
    }

    /// System number density.
    pub fn density(&self) -> f64 {
        match (self.particle_type, self.ndims) {
            (BodyVariants::Sphere, 2) => crate::DENSITY_SPHERE_2D,
            (BodyVariants::Sphere, 3) => crate::DENSITY_SPHERE_3D,
            (BodyVariants::Dumbbell, 2) => crate::DENSITY_DUMBBELL_2D,
            (BodyVariants::Dumbbell, 3) => crate::DENSITY_DUMBBELL_3D,
            _ => panic!("ndims must be 2 or 3!"),
        }
    }

    // System body template.
    // pub fn body(&self) -> Bodies {
    //     match (self.particle_type, self.ndims) {
    //         (BodyVariants::Sphere, 2) => Bodies::Sphere2D(hoomd_microstate::Body::single_site(
    //             DynamicPoint::default(),
    //             Point::default(),
    //         )),
    //         (BodyVariants::Sphere, 3) => Bodies::Sphere3D(hoomd_microstate::Body::single_site(
    //             DynamicPoint::default(),
    //             Point::default(),
    //         )),
    //         (BodyVariants::Dumbbell, 2) => Bodies::Dumbbell2D(hoomd_microstate::Body {
    //             properties: DynamicOrientedPoint::default(),
    //             sites: vec![Point::default(), Point::new(Cartesian::from([0.25, 0.0]))],
    //         }),
    //         (BodyVariants::Dumbbell, 3) => Bodies::Dumbbell3D(hoomd_microstate::Body {
    //             properties: DynamicOrientedPoint::default(),
    //             sites: vec![
    //                 Point::default(),
    //                 Point::new(Cartesian::from([0.25, 0.0, 0.0])),
    //             ],
    //         }),
    //         _ => panic!("ndims must be 2 or 3!"),
    //     }
    // }

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
    pub gsd_period: usize,
    pub sim_duration: usize,
}

impl ProcedureParams {
    /// Create a new ProcedureParams from an entry.
    pub fn from_state_point(state_point: &StatePoint) -> Self {
        Self {
            method: state_point.method,
            thermostat: state_point.thermostat,
            gsd_period: state_point.gsd_period,
            sim_duration: state_point.sim_duration,
        }
    }
}


/// Create a thermostat from procedure and system parameters.
pub fn make_thermostat(system: &SystemParams, procedure: &ProcedureParams) -> Thermostats {
    let mut rng = hoomd_rand::Counter::new(0, 0, 0).make_rng();

    match (procedure.thermostat, system.ndims, system.particle_type) {
        (crate::workspace::ThermostatVariants::NoThermostat, _, _) => {
            Thermostats::NoThermostat(NoThermostat)
        }
        (crate::workspace::ThermostatVariants::Bussi, _, _) => {
            Thermostats::Bussi(Bussi::new(crate::TAU))
        }
        (crate::workspace::ThermostatVariants::MTTK, 2, BodyVariants::Sphere) => {
            Thermostats::MTTK(MartynaTuckermanTobiasKlein::thermalized(
                &mut rng,
                crate::TAU.try_into().unwrap(),
                &system.macrostate(),
                2 * system.particles_per_side.pow(2.try_into().unwrap()),
            ))
        }
        (crate::workspace::ThermostatVariants::MTTK, 2, BodyVariants::Dumbbell) => {
            Thermostats::MTTK(MartynaTuckermanTobiasKlein::thermalized(
                &mut rng,
                crate::TAU.try_into().unwrap(),
                &system.macrostate(),
                2 * system.particles_per_side.pow(2.try_into().unwrap()),
            ))
        }
        (crate::workspace::ThermostatVariants::MTTK, 3, BodyVariants::Sphere) => {
            Thermostats::MTTK(MartynaTuckermanTobiasKlein::thermalized(
                &mut rng,
                crate::TAU.try_into().unwrap(),
                &system.macrostate(),
                3 * system.particles_per_side.pow(3.try_into().unwrap()),
            ))
        }
        (crate::workspace::ThermostatVariants::MTTK, 3, BodyVariants::Dumbbell) => {
            Thermostats::MTTK(MartynaTuckermanTobiasKlein::thermalized(
                &mut rng,
                crate::TAU.try_into().unwrap(),
                &system.macrostate(),
                3 * system.particles_per_side.pow(3.try_into().unwrap()),
            ))
        }
        _ => {
            panic!("ndims should be 2 or 3!");
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
// pub fn make_microstate<const N: usize>(
//     system: SystemParams,
// ) -> anyhow::Result<
//     Microstate<BodyProperties, Point<Positions>, VecCell<SiteKey, N>, Periodic<Boundaries>>,
// > {
//     // Get data for the box and the positions
//     let (box_side_length, positions) = match system.ndims {
//         2 => {
//             let box_volume = (system.particles_per_side as f64).powi(2) / system.density();
//             let box_side_length = box_volume.powf(1.0 / 2.0);
//             let positions = [Positions::C2(Cartesian::<2>::default()); 3];

//             (box_side_length, positions)
//         }
//         3 => {
//             let box_volume = (system.particles_per_side as f64).powi(3) / system.density();
//             let box_side_length = box_volume.powf(1.0 / 3.0);
//             let positions = [Positions::C3(Cartesian::<3>::default()); 3];

//             (box_side_length, positions)
//         }
//         _ => panic!("ndims must be 2 or 3!"),
//     };

//     // Create spatial data structure
//     let vec_cell = VecCell::builder()
//         .nominal_search_radius(crate::R_CUT.try_into()?)
//         .build();

//     // Create boundary condition
//     let boundary = match system.ndims {
//         2 => Periodic::new(
//             crate::R_CUT,
//             Boundaries::Square(Rectangle::with_equal_edges(box_side_length.try_into()?)),
//         ),
//         3 => Periodic::new(
//             crate::R_CUT,
//             Boundaries::Cube(Cuboid::with_equal_edges(box_side_length.try_into()?)),
//         ),
//         _ => panic!("ndims must be 2 or 3!"),
//     };

//     // Create microstate from the pieces above
//     let microstate = Microstate::builder()
//         .spatial_data(vec_cell)
//         .boundary(boundary)
//         .try_build()
//         .unwrap();

//     // Add bodies to the microstate
//     for position in positions {
//         let body = system.body();
//         body.position_mut() = position;
//         microstate.add_body(body);
//     }

//     Ok(microstate)
// }

// TODO: impl AppendMicrostate???

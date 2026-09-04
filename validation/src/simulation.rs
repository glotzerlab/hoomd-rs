#![allow(non_snake_case, dead_code)]

use crate::workspace::{BodyVariants, StatePoint};
use hoomd_geometry::shape::{Cuboid, Rectangle};
use hoomd_interaction::{MaximumInteractionRange, PairwiseCutoff, Rigid, SitePairForceAndVirial, pairwise::Isotropic, univariate::LennardJones};
use hoomd_linear_algebra::matrix::Matrix;
use hoomd_md::{method::{ConstantVolume, Langevin}, thermostat::{Bussi, MartynaTuckermanTobiasKlein, NoThermostat}};
use hoomd_microstate::{Body, Microstate, SiteKey, Transform, boundary::{MaximumAllowableInteractionRange, Periodic, Wrap}, property::{
    AngularMomentum, Drag, DynamicOrientedPoint, DynamicPoint, Mass, MomentOfInertia, Momentum, NetForce, NetTorque, NetVirial, Orientation, Point, Position, RotationalMotionTypes
}};
use hoomd_simulation::macrostate::Isothermal;
use hoomd_spatial::VecCell;
use hoomd_vector::{Angle, Cartesian, Outer, Versor, Wedge};

/// Interaction constants
const SPHERE_LJ: Rigid<PairwiseCutoff<Isotropic<LennardJones::<12,6>>>> = Rigid(PairwiseCutoff(Isotropic {
    interaction: LennardJones {
        epsilon: crate::EPSILON,
        sigma: crate::SIGMA
    },
    r_cut: crate::R_CUT
}));

const DUMBBELL_LJ: Rigid<PairwiseCutoff<DumbbellInteraction>> = Rigid(PairwiseCutoff(DumbbellInteraction {
    aa: Isotropic {
        interaction: LennardJones {
            epsilon: crate::EPSILON,
            sigma: crate::SIGMA / 5.0,
        },
        r_cut: crate::R_CUT
    },
    bb: Isotropic {
        interaction: LennardJones {
            epsilon: crate::EPSILON / 5.0,
            sigma: crate::SIGMA,
        },
        r_cut: crate::R_CUT
    },
    ab: Isotropic {
        interaction: LennardJones {
            epsilon: crate::EPSILON / 5.0,
            sigma: crate::SIGMA / 5.0,
        },
        r_cut: crate::R_CUT
    },
}));

/// The possible position types.
#[derive(Copy, Clone)]
enum Positions {
    C2(Cartesian<2>),
    C3(Cartesian<3>),
}

/// The possible orientation types
enum Orientations {
    C2(Angle),
    C3(Versor),
}

/// The possible body property types.
enum BodyProperties {
    C2(DynamicOrientedPoint<Cartesian<2>, Angle>),
    C3(DynamicOrientedPoint<Cartesian<3>, Versor>)
}

/// The possible site property types.
enum SiteProperties {
    C2(Point<Cartesian<2>>),
    C3(Point<Cartesian<3>>),
}

// impl Position for BodyProperties {
//     type Position = Positions;

//     fn position(&self) -> &Self::Position {
//         match self {
//             Self::C2(item) => &Positions::C2((*item.position()).clone()),
//             Self::C3(item) => &Positions::C3((*item.position()).clone()),
//         }
//     }

//     fn position_mut(&mut self) -> &mut Self::Position {
//         todo!()
//     }
// }

// impl Momentum for BodyProperties {}

// impl NetForce for BodyProperties {}

// impl NetVirial for BodyProperties {}

// impl Mass for BodyProperties {}

// impl Drag for BodyProperties {}

// impl Transform<S> for BodyProperties {}

// impl Orientation for BodyProperties {}

// impl MomentOfInertia for BodyProperties {}

// impl AngularMomentum for BodyProperties {
//     type AngularMomentum;

//     fn angular_momentum(&self) -> &Self::AngularMomentum {
//         todo!()
//     }

//     fn angular_momentum_mut(&mut self) -> &mut Self::AngularMomentum {
//         todo!()
//     }
// }

// impl NetTorque for BodyProperties {}


/// The possible boundary types.
enum Boundaries {
    Square(Rectangle),
    Cube(Cuboid),
}

impl MaximumAllowableInteractionRange for Boundaries {
    fn maximum_allowable_interaction_range(&self) -> f64 {
        match self {
            Self::Square(rectangle) => rectangle.maximum_allowable_interaction_range(),
            Self::Cube(cube) => cube.maximum_allowable_interaction_range()
        }
    }
}

impl<P> Wrap<P> for Periodic<Boundaries> {
    fn wrap(&self, properties: P) -> Result<P, hoomd_microstate::boundary::Error> {
        match self.shape() {
            Boundaries::Square(item) => {
                Periodic::new(self.maximum_interaction_range(), item)?.wrap(properties)
            },
            Boundaries::Cube(item) => {
                Periodic::new(self.maximum_interaction_range(), item)?.wrap(properties)
            },
        }
    }
}

/// The possible types of thermostat.
#[derive(Debug, Clone)]
pub enum Thermostats {
    NoThermostat(hoomd_md::thermostat::NoThermostat),
    Bussi(hoomd_md::thermostat::Bussi),
    MTTK(hoomd_md::thermostat::MartynaTuckermanTobiasKlein),
}

/// The possible types of integration methods.
#[derive(Debug, Clone)]
pub enum Methods {
    ConstantVolume(hoomd_md::method::ConstantVolume<Thermostats, Thermostats>),
    Langevin(hoomd_md::method::Langevin),
}

/// Possible types of dumbbell sites.
#[derive(Clone, Copy, Default, PartialEq)]
enum DumbbellSiteType {
    #[default]
    A,
    B,
}

/// Special properties type for dumbbell sites to track their type.
#[derive(Clone, Copy, Default, hoomd_derive::Position)]
struct DumbbellSiteProperties<const N: usize> {
    position: Cartesian<N>,
    site_type: DumbbellSiteType,
}

/// Site-specific dumbbell pairwise interaction model.
struct DumbbellInteraction {
    aa: Isotropic<LennardJones::<12, 6>>,
    bb: Isotropic<LennardJones::<12, 6>>,
    ab: Isotropic<LennardJones::<12, 6>>,
}

impl MaximumInteractionRange for DumbbellInteraction {
    fn maximum_interaction_range(&self) -> f64 {
        self.aa
            .maximum_interaction_range()
            .max(self.bb.maximum_interaction_range())
    }
}

impl<const N: usize> SitePairForceAndVirial<DumbbellSiteProperties<N>> for DumbbellInteraction {
    type Force = Cartesian<N>;
 
    fn site_pair_force_and_virial(
        &self,
        site_properties_i: &DumbbellSiteProperties<N>,
        site_properties_j: &DumbbellSiteProperties<N>,
    ) -> (Self::Force, <Self::Force as Outer>::Tensor) {
        let (force, virial) =
            match (site_properties_i.site_type, site_properties_j.site_type) {
                (DumbbellSiteType::A, DumbbellSiteType::A) => {
                    self.aa.site_pair_force_and_virial(
                        site_properties_i,
                        site_properties_j
                    )
                }
                (DumbbellSiteType::B, DumbbellSiteType::B) => {
                    self.bb.site_pair_force_and_virial(
                        site_properties_i,
                        site_properties_j
                    )
                }
                _ => {
                    self.ab.site_pair_force_and_virial(
                        site_properties_i,
                        site_properties_j
                    )
                }
            };
        (force, virial)
    }
}

/// The possible interaction model types.
enum Interactions {
    Sphere(Rigid<PairwiseCutoff<Isotropic<LennardJones::<12,6>>>>),
    Dumbbell(Rigid<PairwiseCutoff<DumbbellInteraction>>),
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
            _ => panic!("ndims must be 2 or 3!")
        }
    }

    /// System body template.
    pub fn body(&self) -> Bodies {
        match (self.particle_type, self.ndims) {
            (BodyVariants::Sphere, 2) => Bodies::Sphere2D(hoomd_microstate::Body::single_site(
                DynamicPoint::default(),
                Point::default()
            )),
            (BodyVariants::Sphere, 3) => Bodies::Sphere3D(hoomd_microstate::Body::single_site(
                DynamicPoint::default(),
                Point::default()
            )),
            (BodyVariants::Dumbbell, 2) => Bodies::Dumbbell2D(hoomd_microstate::Body {
                properties: DynamicOrientedPoint::default(),
                sites: vec![
                    Point::default(),
                    Point::new(Cartesian::from([0.25, 0.0])),
                ]
            }),
            (BodyVariants::Dumbbell, 3) => Bodies::Dumbbell3D(hoomd_microstate::Body {
                properties: DynamicOrientedPoint::default(),
                sites: vec![
                    Point::default(),
                    Point::new(Cartesian::from([0.25, 0.0, 0.0])),
                ]
            }),
            _ => panic!("ndims must be 2 or 3!")
        }
    }

    /// Macrostate.
    pub fn macrostate(&self) -> Isothermal {
        Isothermal { temperature: crate::KT }
    }

    /// Interaction model.
    pub fn interaction_model(&self) -> Interactions {
        match self.particle_type {
            BodyVariants::Sphere => Interactions::Sphere(SPHERE_LJ),
            BodyVariants::Dumbbell => Interactions::Dumbbell(DUMBBELL_LJ),
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
        (crate::workspace::ThermostatVariants::NoThermostat, _, _) => Thermostats::NoThermostat(NoThermostat),
        (crate::workspace::ThermostatVariants::Bussi, _, _) => Thermostats::Bussi(Bussi::new(crate::TAU)),
        (crate::workspace::ThermostatVariants::MTTK, 2, BodyVariants::Sphere) => Thermostats::MTTK(
            MartynaTuckermanTobiasKlein::thermalized(
                &mut rng,
                crate::TAU.try_into().unwrap(),
                &system.macrostate(),
                2 * system.particles_per_side.pow(2.try_into().unwrap()),
            )
        ),
        (crate::workspace::ThermostatVariants::MTTK, 2, BodyVariants::Dumbbell) => Thermostats::MTTK(
            MartynaTuckermanTobiasKlein::thermalized(
                &mut rng,
                crate::TAU.try_into().unwrap(),
                &system.macrostate(),
                2 * system.particles_per_side.pow(2.try_into().unwrap()),
            )
        ),
        (crate::workspace::ThermostatVariants::MTTK, 3, BodyVariants::Sphere) => Thermostats::MTTK(
            MartynaTuckermanTobiasKlein::thermalized(
                &mut rng,
                crate::TAU.try_into().unwrap(),
                &system.macrostate(),
                3 * system.particles_per_side.pow(3.try_into().unwrap()),
            )
        ),
        (crate::workspace::ThermostatVariants::MTTK, 3, BodyVariants::Dumbbell) => Thermostats::MTTK(
            MartynaTuckermanTobiasKlein::thermalized(
                &mut rng,
                crate::TAU.try_into().unwrap(),
                &system.macrostate(),
                3 * system.particles_per_side.pow(3.try_into().unwrap()),
            )
        ),
        _ => {
            panic!("ndims should be 2 or 3!");
        }
    }
}

/// Create an integration method from procedure and system parameters.
pub fn make_method(system: &SystemParams, procedure: &ProcedureParams) -> Methods {
    match procedure.method {
        crate::workspace::MethodVariants::ConstantVolume => {
            Methods::ConstantVolume(
                ConstantVolume::builder(crate::DT)
                    .thermostat(make_thermostat(system, procedure))
                    .build()
            )
        },
        crate::workspace::MethodVariants::Langevin => Methods::Langevin(
            Langevin { delta_t: crate::DT }
        ),
    }
}

/// Create the initial microstate.
fn make_microstate<const N: usize>(
    system: SystemParams
) -> anyhow::Result<Microstate<Bodies, Point<Positions>, VecCell<SiteKey, N>, Periodic<Boundaries>>>{
    // Get data for the box and the positions
    let (box_side_length, positions) = match system.ndims {
        2 => {
            let box_volume = (system.particles_per_side as f64).powi(2) / system.density();
            let box_side_length = box_volume.powf(1.0 / 2.0);
            let positions = [Positions::C2(Cartesian::<2>::default()); 3];

            (box_side_length, positions)
        },
        3 => {
            let box_volume = (system.particles_per_side as f64).powi(3) / system.density();
            let box_side_length = box_volume.powf(1.0 / 3.0);
            let positions = [Positions::C3(Cartesian::<3>::default()); 3];

            (box_side_length, positions)
        },
        _ => panic!("ndims must be 2 or 3!")
    };

    // Create spatial data structure
    let vec_cell = VecCell::builder()
        .nominal_search_radius(crate::R_CUT.try_into()?)
        .build();

    // Create boundary condition
    let boundary = match system.ndims {
        2 => Periodic::new(
            crate::R_CUT,
            Boundaries::Square(Rectangle::with_equal_edges(box_side_length.try_into()?))
        ),
        3 => Periodic::new(
            crate::R_CUT,
            Boundaries::Cube(Cuboid::with_equal_edges(box_side_length.try_into()?))
        ),
        _ => panic!("ndims must be 2 or 3!")
    };

    // Create microstate from the pieces above
    let microstate = Microstate::builder()
        .spatial_data(vec_cell)
        .boundary(boundary)
        .try_build().unwrap();

    // Add bodies to the microstate
    for position in positions {
        let body = system.body();
        body.position_mut() = position;
        microstate.add_body(body);
    }

    Ok(microstate)
}


// TODO: impl AppendMicrostate???
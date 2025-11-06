use anyhow::Context;
use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, InitialCamera, MUTED_COLOR, Settings,
    representation::{self, HyperbolicPolygonAssets, HyperbolicPolygonMaterial,HyperbolicPolygonMaterialParameters},
};
use hoomd_geometry::{hyperbolic_overlap::HyperbolicConvexPolytope, shape::EightEight};
use hoomd_interaction::{
    pairwise::HardShape, CutoffPairOverlap
};
use hoomd_manifold::{Hyperbolic, HyperbolicDisk, Minkowski};
use hoomd_mc::{Rotate, Sweep, Translate, Trial};
use hoomd_microstate::{
    Body, Microstate, MicrostateBuilder, boundary::{Open, Periodic}, property::OrientedHyperbolicPoint,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_vector::Angle;
use rand::distr::Distribution;
use rand::{SeedableRng, rngs::StdRng};

type Position = Hyperbolic<3>;
type Orientation = Angle;
type SiteProperties = OrientedHyperbolicPoint<3, Angle>;
type BodyProperties = OrientedHyperbolicPoint<3, Angle>;

/// Mark the disk representation type.
struct A;
/// Mark the ghost representation type
struct Ghost;

fn main() -> anyhow::Result<()> {
    let simulation = HyperbolicPolygonSelfAssembly::new().context("failed to setup simulation")?;
    let hoomd_bevy_plugin = HoomdBevyPlugin {
        initial_settings: Settings {
            camera: InitialCamera::Orthographic2d(2.0),
            ..default()
        },
        simulation,
    };

    let mut app = App::new();
    app.add_plugins(DefaultPlugins);
    app.add_plugins(EguiPlugin::default());
    hoomd_bevy_plugin.build(&mut app);
    app.add_systems(
        Startup,
        (|| HyperbolicPolygonMaterialParameters {
            ..default()
        })
        .pipe(representation::HyperbolicPolygon::<A>::setup),
    );
    app.add_systems(
        Startup,
        (|| HyperbolicPolygonMaterialParameters {
            background_color: MUTED_COLOR.into(),
            ..default()
        })
        .pipe(representation::HyperbolicPolygon::<Ghost>::setup),
    );
    app.add_systems(
        Update,
        (sync_simulation, sync_ghosts)
            .run_if(resource_changed::<HyperbolicPolygonSelfAssembly>)
            .after(AdvanceSet),
    );

    app.run();

    Ok(())
}

#[cfg_attr(feature = "bevy", derive(Resource))]
struct HyperbolicPolygonSelfAssembly {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<BodyProperties, SiteProperties, Open>,
    /// How sites interact with other sites and fields.
    hamiltonian: CutoffPairOverlap<HardShape<HyperbolicConvexPolytope<3>>>,
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate<Position>>,
    /// Trial moves to apply.
    rotate_sweep: Sweep<Rotate<Orientation>>,
    /// Temperature set point.
    macrostate: Isothermal,
}

const RHO: f64 = 1.0;
const PARTICLE_NUMBER: usize = 3;
const RADIUS: f64 = 0.5;

enum Phase {
    Initialize,
    Equilibrate,
}

impl HyperbolicPolygonSelfAssembly {
    /// Construct a new hard ellipsoid self-assembly simulation.
    fn new() -> anyhow::Result<HyperbolicPolygonSelfAssembly> {
        let maximum_distance = 0.005;
        let maximum_rotation = 0.000001;
        let macrostate = Isothermal { temperature: 1.0 };

        let square = HyperbolicConvexPolytope::<3>::regular(4, RADIUS, 1.0);
        let hamiltonian = CutoffPairOverlap {
            r_cut: 1.0,
            evaluator: HardShape(square.clone()),
        };

        //let boundary = Periodic::new(0.6, EightEight { skirt: 1.0_f64 })?;

        let mut microstate =
            MicrostateBuilder::with_boundary(Open).try_build()?;

            let hyp_translate = Translate::with_maximum_distance(maximum_distance.try_into()?);
            let translate_sweep = Sweep(hyp_translate);

        let rotate =
            Rotate::with_maximum_rotation(maximum_rotation.try_into()?);
        let rotate_sweep = Sweep(rotate);
        
        let initial_spacing = 1.0;
        let mut rng = StdRng::seed_from_u64(12);
        let sample_disk = HyperbolicDisk {
            disk_radius: initial_spacing.try_into()?,
            point: Hyperbolic::<3>::from_minkowski_coordinates(
                Minkowski::from([
                    0.00001,
                    0.00001,
                    f64::sqrt(2.0 * (0.00001_f64).powi(2) + RHO.powi(2)),
                ]),
                RHO,
            ),
        };
        for _n in 0..PARTICLE_NUMBER {
            let new_point: Hyperbolic<3> =
                Hyperbolic::from_minkowski_coordinates(
                    *sample_disk.sample(&mut rng).point(),
                    RHO,
                );
            let body_properties = OrientedHyperbolicPoint{position: new_point, orientation: Angle::default()};
            let site_properties = OrientedHyperbolicPoint{position: Hyperbolic::<3>::default(), orientation: Angle::default()};
            
            let body = Body {
                properties: body_properties,
                sites: vec![site_properties],
            };
            microstate.add_body(body)?;
        }

        Ok(HyperbolicPolygonSelfAssembly {
            microstate,
            hamiltonian,
            translate_sweep,
            rotate_sweep,
            macrostate,
        })
    }
}

impl Simulation for HyperbolicPolygonSelfAssembly {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
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
        self.microstate.increment_step();
        Ok(())
    }

    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}
/// Copy the current positions of simulation particles to bevy entities.
fn sync_simulation(
    mut commands: Commands,
    disk_assets: Res<HyperbolicPolygonAssets<A>>,
    query: Query<
        (Entity, &mut Transform),
        With<representation::HyperbolicPolygon<A>>,
    >,
    simulation: Res<HyperbolicPolygonSelfAssembly>,
) {
    let sites = simulation.microstate.sites();
    representation::HyperbolicPolygon::sync(
        &mut commands,
        disk_assets,
        query,
        sites
            .iter()
            .map(|site| (*site.properties.position.point(), RADIUS, site.properties.orientation.theta as f32)),
    );
}

fn sync_ghosts(
    mut commands: Commands,
    ghost_assets: Res<HyperbolicPolygonAssets<Ghost>>,
    ghost_query: Query<
        (Entity, &mut Transform),
        With<representation::HyperbolicPolygon<Ghost>>,
    >,
    simulation: Res<HyperbolicPolygonSelfAssembly>,
) {
    let ghosts = simulation.microstate.ghosts();
    representation::HyperbolicPolygon::sync(
        &mut commands,
        ghost_assets,
        ghost_query,
        ghosts
            .iter()
            .map(|site| (*site.properties.position.point(), RADIUS, site.properties.orientation.theta as f32)),
    );
}

use anyhow::{Context, anyhow};
use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, InitialCamera, MUTED_COLOR, Settings,
    representation::{
        self, HyperbolicPolygonAssets, HyperbolicPolygonMaterialParameters,
    },
};
use hoomd_geometry::{
    hyperbolic_overlap::HyperbolicConvexPolytope, shape::EightEight,
};
use hoomd_interaction::{
    PairwiseCutoff,
    pairwise::{HardShape, Isotropic},
    univariate::LennardJones,
};
use hoomd_manifold::{Hyperbolic, HyperbolicDisk, Minkowski};
use hoomd_mc::{QuickInsert, Rotate, Sweep, Translate, Trial};
use hoomd_microstate::{
    Body, Microstate, SiteKey, boundary::Periodic,
    property::OrientedHyperbolicPoint,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use hoomd_spatial::AllPairs;
use hoomd_vector::Angle;
use rand::{Rng, distr::Distribution};

type Orientation = Angle;
type SiteProperties = OrientedHyperbolicPoint<3, Angle>;
type BodyProperties = OrientedHyperbolicPoint<3, Angle>;

/// Mark the disk representation type.
struct A;
/// Mark the ghost representation type
struct Ghost;

fn main() -> anyhow::Result<()> {
    let simulation = HyperbolicPolygonSelfAssembly::new()
        .context("failed to setup simulation")?;
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
        (|| HyperbolicPolygonMaterialParameters { ..default() })
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
    microstate: Microstate<
        BodyProperties,
        SiteProperties,
        AllPairs<SiteKey>,
        Periodic<EightEight>,
    >,
    /// How sites interact with other sites and fields.
    hamiltonian: PairwiseCutoff<HardShape<HyperbolicConvexPolytope<3>>>,
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate<OrientedHyperbolicPoint<3, Angle>>>,
    /// Trial moves to apply.
    rotate_sweep: Sweep<Rotate<Orientation>>,
    /// Temperature set point.
    macrostate: Isothermal,
    /// Quick insert
    quick_insert: QuickInsert<UniformHyperbolic<SiteProperties>>,
    /// how sites interact when inserted
    insert_hamiltonian: PairwiseCutoff<Isotropic<LennardJones>>,
    /// the current phase of the simulation
    phase: Phase,
}

const RHO: f64 = 1.0;
const PARTICLE_NUMBER: usize = 20;
const START_RADIUS: f64 = 0.05; // units of rapidity
const END_RADIUS: f64 = 0.5;
const NUM_STEPS: u64 = 50_000;

enum Phase {
    Initialize,
    Crunch,
    Equilibrate,
}

#[allow(dead_code)]
struct UniformHyperbolic<S> {
    template_sites: Vec<S>,
}

impl Distribution<Body<OrientedHyperbolicPoint<3, Angle>>>
    for UniformHyperbolic<OrientedHyperbolicPoint<3, Angle>>
{
    #[inline]
    fn sample<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> Body<
        OrientedHyperbolicPoint<3, Angle>,
        OrientedHyperbolicPoint<3, Angle>,
    > {
        let initial_spacing = 1.4;
        let sample_disk = HyperbolicDisk {
            disk_radius: initial_spacing.try_into().expect("positive number"),
            point: Hyperbolic::<3>::from_minkowski_coordinates(
                Minkowski::from([
                    0.00001,
                    0.00001,
                    f64::sqrt(2.0 * (0.00001_f64).powi(2) + RHO.powi(2)),
                ]),
                RHO,
            ),
        };
        let new_point: Hyperbolic<3> = Hyperbolic::from_minkowski_coordinates(
            *sample_disk.sample(rng).point(),
            RHO,
        );
        //let new_angle: Angle = rng.random();
        let body_properties = OrientedHyperbolicPoint {
            position: new_point,
            orientation: Angle::default(), //new_angle,
        };
        let site_properties = OrientedHyperbolicPoint {
            position: Hyperbolic::<3>::default(),
            orientation: Angle::default(), //new_angle,
        };
        Body {
            properties: body_properties,
            sites: vec![site_properties],
        }
    }
}

impl HyperbolicPolygonSelfAssembly {
    /// Construct a new hard ellipsoid self-assembly simulation.
    #[allow(unused_mut)]
    fn new() -> anyhow::Result<HyperbolicPolygonSelfAssembly> {
        let maximum_distance = 0.01;
        let maximum_rotation = 0.01;
        let macrostate = Isothermal { temperature: 1.0 };

        let end_square = HyperbolicConvexPolytope::<3>::regular(4, END_RADIUS, 1.0);
        let hamiltonian = PairwiseCutoff(HardShape(end_square.clone()));

        let boundary = Periodic::new(0.6, EightEight { skirt: 1.0_f64 })?;
        //let allpairs = AllPairs
        let mut microstate = Microstate::builder()
            //.spatial_data(allpairs)
            .boundary(boundary)
            .try_build()?;

        let hyp_translate =
            Translate::with_maximum_distance(maximum_distance.try_into()?);
        let translate_sweep = Sweep(hyp_translate);

        let rotate =
            Rotate::with_maximum_rotation(maximum_rotation.try_into()?);
        let rotate_sweep = Sweep(rotate);

        let distribution = UniformHyperbolic {
            template_sites: vec![OrientedHyperbolicPoint::<3, Angle>::default()],
        };
        let quick_insert = QuickInsert::new(distribution, PARTICLE_NUMBER);

        let lj: LennardJones = LennardJones {
            epsilon: 10.0,
            sigma: START_RADIUS*2.0,
        };

        let insert_hamiltonian = PairwiseCutoff(Isotropic {
            interaction: lj,
            r_cut: 1.0,
        });

        Ok(HyperbolicPolygonSelfAssembly {
            microstate,
            hamiltonian,
            translate_sweep,
            rotate_sweep,
            macrostate,
            insert_hamiltonian,
            quick_insert,
            phase: Phase::Initialize,
        })
    }

    fn initialize(&mut self) -> anyhow::Result<()> {
        self.quick_insert
            .apply(&mut self.microstate, &self.insert_hamiltonian);

        self.translate_sweep.apply(
            &mut self.microstate,
            &self.insert_hamiltonian,
            &Isothermal { temperature: 1.0 },
        );

        self.rotate_sweep.apply(
            &mut self.microstate,
            &self.insert_hamiltonian,
            &Isothermal { temperature: 1.0 },
        );

        if self.quick_insert.is_complete() {
            self.phase = Phase::Crunch;
            println!(
                "Initialization complete at step {}.",
                self.microstate.step()
            );
        }

        if self.step() >= 10_000 {
            let n = self.microstate.bodies().len();
            let target = self.quick_insert.target();
            let step = self.microstate.step();
            return Err(anyhow!(
                "{n} of {target} bodies inserted after {step} steps"
            )); 
        }

        Ok(())
    }
}

impl Simulation for HyperbolicPolygonSelfAssembly {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        match self.phase {
            Phase::Initialize => {
                self.initialize().context("failed to initialize")?
            }
            Phase::Crunch => {
                let step = self.microstate.step();
                let radius = (END_RADIUS - START_RADIUS)*((step as f64)/(NUM_STEPS as f64)) + START_RADIUS;

                let crunch_square = HyperbolicConvexPolytope::<3>::regular(4, radius, 1.0);
                let crunch_hamiltonian = PairwiseCutoff(HardShape(crunch_square.clone()));

                self.translate_sweep.apply(
                &mut self.microstate,
                &crunch_hamiltonian,
                &Isothermal { temperature: 1.0 },
                );

                self.rotate_sweep.apply(
                &mut self.microstate,
                &crunch_hamiltonian,
                &Isothermal { temperature: 1.0 },
                );

                if step > NUM_STEPS {
                    self.phase = Phase::Equilibrate;
                }
            }
            Phase::Equilibrate => {
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
        //println!("orientation: {} ",self.microstate.bodies()[0].item.properties.orientation.theta);
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
    let step = simulation.microstate.step();
    let radius = if step <= NUM_STEPS {(END_RADIUS - START_RADIUS)*(step as f64/NUM_STEPS as f64) + START_RADIUS} else {END_RADIUS};
    representation::HyperbolicPolygon::sync(
        &mut commands,
        disk_assets,
        query,
        sites.iter().map(|site| {
            (
                *site.properties.position.point(),
                radius,
                site.properties.orientation.theta as f32,
            )
        }),
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
    let step = simulation.microstate.step();
    let radius = if step <= NUM_STEPS {(END_RADIUS - START_RADIUS)*(step as f64/NUM_STEPS as f64) + START_RADIUS} else {END_RADIUS};
    representation::HyperbolicPolygon::sync(
        &mut commands,
        ghost_assets,
        ghost_query,
        ghosts.iter().map(|site| {
            (
                *site.properties.position.point(),
                radius,
                site.properties.orientation.theta as f32,
            )
        }),
    );
}

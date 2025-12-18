use anyhow::Context;
use bevy::prelude::*;
use bevy_egui::EguiPlugin;
use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, InitialCamera, Settings,
    representation::{self, HyperbolicDiskAssets, HyperbolicDiskMaterial},
};
use hoomd_geometry::shape::EightEight;
use hoomd_interaction::{
    CutoffPair,
    pairwise::{Isotropic, LennardJones},
};
use hoomd_manifold::{Hyperbolic, HyperbolicDisk, Minkowski};
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_microstate::{
    Body, Microstate, MicrostateBuilder, boundary::Periodic, property::Point,
};
use hoomd_simulation::{Simulation, macrostate::Isothermal};
use rand::distr::Distribution;
use rand::{SeedableRng, rngs::StdRng};

/// Mark the disk representation type.
struct A;
/// Mark the ghost representation type
struct Ghost;

const RHO: f64 = 1.0;
const PARTICLE_NUMBER: usize = 200;
const DIAMETER: f64 = 0.15; //in Hyperbolic metric

fn main() -> anyhow::Result<()> {
    let simulation = Fill::new().context("failed to setup simulation")?;
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
        (|| HyperbolicDiskMaterial::default())
            .pipe(representation::HyperbolicDisk::<A>::setup),
    );
    app.add_systems(
        Startup,
        (|| HyperbolicDiskMaterial::ghost())
            .pipe(representation::HyperbolicDisk::<Ghost>::setup),
    );
    app.add_systems(
        Update,
        (sync_simulation, sync_ghosts)
            .run_if(resource_changed::<Fill>)
            .after(AdvanceSet),
    );

    app.run();

    Ok(())
}

#[derive(Resource)]
struct Fill {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<
        Point<Hyperbolic<3>>,
        Point<Hyperbolic<3>>,
        Periodic<EightEight>,
    >,
    /// How sites interact with other sites and fields.
    hamiltonian: CutoffPair<Isotropic<LennardJones>>,
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate<Point<Hyperbolic<3>>>>,
    /// Temperature set point.
    macrostate: Isothermal,
}

impl Fill {
    /// Set up the hoomd simulation
    fn new() -> anyhow::Result<Fill> {
        let boundary = Periodic::new(0.6, EightEight { skirt: 1.0_f64 })?;
        let mut microstate =
            MicrostateBuilder::with_boundary(boundary).try_build()?;

        let initial_spacing = 1.0;
        let mut rng = StdRng::seed_from_u64(23);
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
            microstate.add_body(Body::point(new_point))?;
        }

        let lj: LennardJones = LennardJones {
            epsilon: 10.0,
            sigma: 0.15,
        };

        let evaluator = Isotropic(lj);
        let cutoff_pair = CutoffPair {
            r_cut: 0.5,
            evaluator,
        };

        let macrostate = Isothermal { temperature: 1.0 };

        let hamiltonian = cutoff_pair;
        let d = 0.01;

        let hyp_translate = Translate::with_maximum_distance(d.try_into()?);
        let translate_sweep = Sweep(hyp_translate);

        Ok(Fill {
            microstate,
            hamiltonian,
            translate_sweep,
            macrostate,
        })
    }
}

impl Simulation for Fill {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        self.translate_sweep.apply(
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
    disk_assets: Res<HyperbolicDiskAssets<A>>,
    query: Query<
        (Entity, &mut Transform),
        With<representation::HyperbolicDisk<A>>,
    >,
    simulation: Res<Fill>,
) {
    let sites = simulation.microstate.sites();
    representation::HyperbolicDisk::sync(
        &mut commands,
        disk_assets,
        query,
        sites
            .iter()
            .map(|site| (*site.properties.position.point(), DIAMETER)),
    );
}

fn sync_ghosts(
    mut commands: Commands,
    ghost_assets: Res<HyperbolicDiskAssets<Ghost>>,
    ghost_query: Query<
        (Entity, &mut Transform),
        With<representation::HyperbolicDisk<Ghost>>,
    >,
    simulation: Res<Fill>,
) {
    let ghosts = simulation.microstate.ghosts();
    representation::HyperbolicDisk::sync(
        &mut commands,
        ghost_assets,
        ghost_query,
        ghosts
            .iter()
            .map(|site| (*site.properties.position.point(), DIAMETER)),
    );
}

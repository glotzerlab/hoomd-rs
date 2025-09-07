use anyhow::Context;
use bevy::prelude::*;
use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, Settings, Simulation,
    representation::{self, HyperbolicDiskAssets, HyperbolicDiskMaterial},
};
use hoomd_geometry::shape::EightEight;
use hoomd_interaction::{
    CutoffPair,
    pairwise::{Isotropic, LennardJonesGauss},
};
use hoomd_manifold::{HyperbolicDisk, Hyperboloid, Minkowski};
use hoomd_mc::{HyperbolicTranslate, Sweep, Trial};
use hoomd_microstate::{
    Body, Microstate, MicrostateBuilder, boundary::Periodic, property::Point,
};
use rand::distr::Distribution;
use rand::{SeedableRng, rngs::StdRng};

/// Mark the disk representation type.
struct A;
/// Mark the ghost representation type
struct Ghost;

const RHO: f64 = 1.0;
const PARTICLE_NUMBER: usize = 1118;
const DIAMETER: f64 = 0.15; //in hyperboloid metric

fn main() -> anyhow::Result<()> {
    let simulation = Fill::new().context("failed to setup simulation")?;
    let hoomd_bevy_plugin = HoomdBevyPlugin {
        initial_settings: Settings {
            viewport_height: 2.0_f32,
            ..default()
        },
        simulation,
    };

    let mut app = App::new();
    app.add_plugins(DefaultPlugins);
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

/// The HOOMD simulation
#[derive(Resource)]
struct Fill {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<
        Point<Hyperboloid<3>>,
        Point<Hyperboloid<3>>,
        Periodic<EightEight>,
    >,
    /// How sites interact with other sites and fields.
    hamiltonian: CutoffPair<Isotropic<LennardJonesGauss>>,
    /// Trial moves to apply.
    translate_sweep: Sweep<HyperbolicTranslate>,
    /// Temperature set point.
    kt: f64,
}

impl Fill {
    /// Set up the hoomd simulation
    fn new() -> anyhow::Result<Fill> {
        let boundary = Periodic::new(0.6, EightEight { skirt: 1.0_f64 })?;
        let mut microstate =
            MicrostateBuilder::with_boundary(boundary).try_build()?;

        let initial_spacing = 2.0;
        let mut rng = StdRng::seed_from_u64(23);
        let sample_disk = HyperbolicDisk {
            r: initial_spacing.try_into()?,
            point: Minkowski::from([
                0.00001,
                0.00001,
                f64::sqrt(2.0 * (0.00001_f64).powi(2) + RHO.powi(2)),
            ]),
            skirt: RHO,
        };
        for _n in 0..PARTICLE_NUMBER {
            let new_point: Hyperboloid<3> =
                Hyperboloid::from(&sample_disk.sample(&mut rng).point);
            microstate.add_body(Body::point(new_point))?;
        }

        let ljg: LennardJonesGauss = LennardJonesGauss {
            epsilon: 1.8,
            sigma_squared: 0.02,
            r_0: 1.52,
            scale: 0.1,
        };

        let evaluator = Isotropic(ljg);
        let cutoff_pair = CutoffPair {
            r_cut: 0.5,
            evaluator,
        };

        let mut kt = 0.6;

        if microstate.step() < 20_000_u64 {
            kt = 0.6;
        } else if microstate.step() < 2_020_000_u64 {
            kt = 0.6 - (0.5 / 2_000_000.0) * (microstate.step() as f64);
        } else {
            kt = 0.1;
        }
        let hamiltonian = cutoff_pair;
        let d = 0.01;

        let hyp_translate = HyperbolicTranslate {
            maximum_distance: d.try_into()?,
            skirt: RHO,
        };
        let translate_sweep = Sweep(hyp_translate);

        Ok(Fill {
            microstate,
            hamiltonian,
            translate_sweep,
            kt,
        })
    }
}

impl Simulation for Fill {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        self.translate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.kt,
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
            .map(|site| (site.properties.position.point, DIAMETER)),
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
            .map(|site| (site.properties.position.point, DIAMETER)),
    );
}

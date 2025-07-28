use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, Settings, Simulation,
    representation::{self, HyperbolicDiskAssets, HyperbolicDiskMaterial},
};
use rand::distr::Distribution;
use hoomd_mc::{Sweep, Translate, Trial};
use rand::{rngs::StdRng, SeedableRng};
use hoomd_microstate::{Body, Microstate, MicrostateBuilder, boundary::Open, property::Point};
use hoomd_manifold::{Minkowski, HyperbolicTranslate, Hyperboloid, CurvedIsotropic, HyperbolicDisk};
use hoomd_vector::Cartesian;
use hoomd_interaction::{
    CutoffPair, pairwise::LennardJones};
use libm::{cosh, sinh, acosh};
use anyhow::Context;
use bevy::prelude::*;

/// Mark the disk representation type.
struct A;
const RHO: f64 = 1.0; 
const PARTICLE_NUMBER : usize = 100;
const RAD_SQ : f64 = 0.1;

fn main() -> anyhow::Result<()> {
    let simulation = Fill::new().context("failed to setup simulation")?;
    let hoomd_bevy_plugin = HoomdBevyPlugin {
        initial_settings: Settings {
            viewport_height: 2.0 as f32,
            ..default()
        },
        simulation,
    };

    let mut app = App::new();
    app.add_plugins(DefaultPlugins);
    hoomd_bevy_plugin.build(&mut app);
    app.add_systems(Startup, (|| HyperbolicDiskMaterial::default()).pipe(representation::HyperbolicDisk::<A>::setup));
    app.add_systems(
        Update,
        sync_simulation
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
    microstate: Microstate<Point<Minkowski<3>>, Point<Minkowski<3>>>,
    /// How sites interact with other sites and fields.
    hamiltonian: CutoffPair<CurvedIsotropic<LennardJones>>,
    /// Trial moves to apply.
    translate_sweep: Sweep<HyperbolicTranslate>,
    /// Temperature set point.
    kt: f64,
}

impl Fill {
    /// Set up the hoomd simulation
    fn new() -> anyhow::Result<Fill> {
        let mut microstate = MicrostateBuilder::with_boundary(Open)
    //.bodies([Body::point(Minkowski::from([1.0, -2.0, sqrt(5.0)])),
    //    Body::point(Minkowski::from([1.0, -1.0, sqrt(3.0)])),
    //    Body::point(Minkowski::from([-1.0, -2.0, sqrt(5.0)])),
    //    Body::point(Minkowski::from([-1.0, -1.0, sqrt(3.0)]))])
    .try_build()?;

    let initial_spacing = 1.0;
    let mut rng = StdRng::seed_from_u64(23);
    let sample_disk = HyperbolicDisk{
        r: initial_spacing.try_into()?, 
        point: Minkowski::from([0.00001,0.00001,f64::sqrt(2.0*(0.00001_f64).powi(2) + RHO.powi(2))]),
        skirt: RHO,}; 
    for _n in 0..PARTICLE_NUMBER {
        let new_point: Minkowski<3> = sample_disk.sample(&mut rng);
        microstate.add_body(Body::point(new_point))?;
    }
    
    let lj : LennardJones = LennardJones {
        epsilon: 10.0,
        sigma: 0.5,
    };

    let evaluator = CurvedIsotropic(lj, RHO);
    let cutoff_pair = CutoffPair {
        r_cut: 10.0,
        evaluator,
    };

    let kt = 1.0;
    let hamiltonian = cutoff_pair;
    let d = 0.05;

    let translate = HyperbolicTranslate {
        maximum_distance: d.try_into()?,
        skirt: RHO,
    };
    let translate_sweep = Sweep { local: translate };

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
        self.translate_sweep
            .apply(&mut self.microstate, &self.hamiltonian, &self.kt);
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
    query: Query<(Entity, &mut Transform), With<representation::HyperbolicDisk<A>>>,
    simulation: Res<Fill>,
) {
    let sites = simulation.microstate.sites();
    representation::HyperbolicDisk::sync(
        &mut commands,
        disk_assets,
        query,
        sites.iter().map(|site| {
            (
                site.properties.position,
                RAD_SQ,
            )
        }),
    );
}

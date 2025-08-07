// ANCHOR: use
use rand::{Rng, seq::IndexedRandom};
use std::iter;

use hoomd_geometry::IsPointInside;
use hoomd_interaction::Zero;
use hoomd_mc::{LocalTrial, Sweep, Trial};
use hoomd_microstate::{
    Body, Microstate, MicrostateBuilder, boundary::Closed, property::Point,
};
use hoomd_vector::{Cartesian, Vector};
// ANCHOR_END: use

use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, Settings, Simulation,
    representation::disk::{self, Disk},
};

use anyhow::Context;
use bevy::prelude::*;

// ANCHOR: boundary_struct
/// Closed circular boundary condition.
struct Circle {
    radius: f64,
}
// ANCHOR_END: boundary_struct

// ANCHOR: boundary_all
impl IsPointInside<Cartesian<2>> for Circle {
    fn is_point_inside(&self, point: &Cartesian<2>) -> bool {
        point.distance(&[0.0, 0.0].into()) < self.radius
    }
}
// ANCHOR_END: boundary_all

// ANCHOR: local_trial_all
// ANCHOR: local_trial_struct
/// Take fixed steps left, right, down, or up.
struct Discrete;
// ANCHOR_END: local_trial_struct

impl LocalTrial<Point<Cartesian<2>>> for Discrete {
    // ANCHOR: local_trial_fn
    fn propose<R: Rng>(
        &self,
        rng: &mut R,
        body_properties: Point<Cartesian<2>>,
    ) -> Point<Cartesian<2>> {
        // ANCHOR_END: local_trial_fn
        // ANCHOR: local_trial_steps
        let steps = [
            [0.0, -1.0].into(),
            [0.0, 1.0].into(),
            [-1.0, 0.0].into(),
            [1.0, 0.0].into(),
        ];
        // ANCHOR_END: local_trial_steps

        // ANCHOR: local_trial_mut
        let mut trial = body_properties;
        trial.position += *steps
            .choose(rng)
            .expect("steps should have at least 1 element");
        trial
        // ANCHOR_END: local_trial_mut
    }
}
// ANCHOR_END: local_trial_all

#[derive(Resource)]
// ANCHOR: simulation_struct
struct CustomRandomWalk {
    microstate: Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Closed<Circle>>,
    hamiltonian: Zero,
    translate_sweep: Sweep<Discrete>,
    kt: f64,
}
// ANCHOR_END: simulation_struct

// ANCHOR: simulation_new
impl CustomRandomWalk {
    /// Construct a new random walk simulation.
    fn new() -> anyhow::Result<CustomRandomWalk> {
        let kt = 1.0;
        let n = 1000;

        // ANCHOR: microstate
        let circle = Circle { radius: 50.0 };

        let microstate = MicrostateBuilder::with_boundary(Closed(circle))
            .bodies(iter::repeat_n(Body::point(Cartesian::default()), n))
            .try_build()?;
        // ANCHOR_END: microstate

        // ANCHOR: sweep
        let translate_sweep = Sweep(Discrete);
        // ANCHOR_END: sweep

        let hamiltonian = Zero;

        Ok(CustomRandomWalk {
            microstate,
            hamiltonian,
            translate_sweep,
            kt,
        })
    }
}
// ANCHOR_END: simulation_new

// ANCHOR: impl_simulation
impl Simulation for CustomRandomWalk {
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
// ANCHOR_END: impl_simulation

/// Mark the disk representation type.
struct A;

fn main() -> anyhow::Result<()> {
    let simulation =
        CustomRandomWalk::new().context("failed to setup simulation")?;
    let hoomd_bevy_plugin = HoomdBevyPlugin {
        initial_settings: Settings {
            viewport_height: 110.0,
            ..default()
        },
        simulation,
    };

    let mut app = App::new();
    hoomd_bevy::add_default_plugins(&mut app);
    hoomd_bevy_plugin.build(&mut app);
    app.add_systems(
        Startup,
        (|| disk::MaterialParameters::default()).pipe(Disk::<A>::setup),
    );
    app.add_systems(
        Update,
        sync_simulation
            .run_if(resource_changed::<CustomRandomWalk>)
            .after(AdvanceSet),
    );

    app.run();

    Ok(())
}

/// Copy the current positions of simulation particles to bevy entities.
fn sync_simulation(
    mut commands: Commands,
    disk_representation: Res<disk::Representation<A>>,
    query: Query<(Entity, &mut Transform), With<Disk<A>>>,
    simulation: Res<CustomRandomWalk>,
) {
    let sites = simulation.microstate.sites();
    Disk::sync(
        &mut commands,
        disk_representation,
        query,
        sites.iter().map(|site| {
            (
                Vec3::new(
                    site.properties.position[0] as f32,
                    site.properties.position[1] as f32,
                    0.0,
                ),
                1.0f32,
            )
        }),
    );
}

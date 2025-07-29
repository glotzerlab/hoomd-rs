// ANCHOR: use
use hoomd_interaction::{
    CutoffPair, Single, TotalEnergy,
    external::Linear,
    pairwise::{Boxcar, Isotropic},
};
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_microstate::{
    Body, Microstate, MicrostateBuilder, boundary::Square, property::Point,
};
use hoomd_vector::Cartesian;
// ANCHOR_END: use

use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, Settings, Simulation,
    representation::disk::{self, Disk},
    representation::RectangularBoundary,
};

use anyhow::Context;
use bevy::prelude::*;

// ANCHOR: simulation_new
impl Fill {
    /// Construct a new fill simulation.
    fn new() -> anyhow::Result<Fill> {
        let box_height = 30.0;
        let kt = 1.0;
        let d = 0.15;
        let alpha = 10.0;
        let epsilon = 1000.0;
        let sigma = 1.0;

        let microstate = MicrostateBuilder::with_boundary(Square {
            l: box_height.try_into()?,
        })
        .try_build()?;

        // ANCHOR: external
        let linear = Single(Linear {
            alpha,
            plane_origin: Cartesian::default(),
            plane_normal: [0.0, 1.0].try_into()?,
        });
        // ANCHOR_END: external

        // ANCHOR: pair
        let boxcar = Boxcar {
            epsilon,
            left: 0.0,
            right: sigma,
        };
        let isotropic = Isotropic(boxcar);
        let cutoff_pair = CutoffPair {
            r_cut: sigma,
            evaluator: isotropic,
        };
        // ANCHOR_END: pair

        // ANCHOR: hamiltonian
        let hamiltonian = (linear, cutoff_pair);
        // ANCHOR_END: hamiltonian

        let translate = Translate {
            maximum_distance: d.try_into()?,
        };
        let translate_sweep = Sweep(translate);

        Ok(Fill {
            microstate,
            hamiltonian,
            translate_sweep,
            kt,
        })
    }
}
// ANCHOR_END: simulation_new

// ANCHOR: impl_simulation
impl Simulation for Fill {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        // ANCHOR: add
        if self.microstate.step() % 100 == 0 {
            self.microstate.add_body(Body::point(
                [0.0, self.microstate.boundary().l.get() / 2.0 - 0.5].into(),
            ))?;
        }
        // ANCHOR_END: add

        // ANCHOR: apply
        self.translate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.kt,
        );
        self.microstate.increment_step();
        // ANCHOR_END: apply

        // ANCHOR: reset
        if self.hamiltonian.1.total_energy(&self.microstate) > 20_000.0 {
            self.microstate.clear();
        }
        // ANCHOR_END: reset

        Ok(())
    }

    /// Get the current simulation step.
    fn step(&self) -> u64 {
        self.microstate.step()
    }
}
// ANCHOR_END: impl_simulation

#[derive(Resource)]
// ANCHOR: simulation_struct
struct Fill {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Square>,
    /// How sites interact with other sites and fields.
    hamiltonian: (Single<Linear<Cartesian<2>>>, CutoffPair<Isotropic<Boxcar>>),
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate>,
    /// Temperature set point.
    kt: f64,
}
// ANCHOR_END: simulation_struct

/// Mark the disk representation type.
struct A;

fn main() -> anyhow::Result<()> {
    let simulation = Fill::new().context("failed to setup simulation")?;
    let l = simulation.microstate.boundary().l.get() as f32;
    let hoomd_bevy_plugin = HoomdBevyPlugin {
        initial_settings: Settings {
            viewport_height: l + 1.0,
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
        Startup,
        (move || RectangularBoundary {
            width: l,
            height: l,
            ..default()
        })
        .pipe(RectangularBoundary::setup),
    );
    app.add_systems(
        Update,
        sync_simulation
            .run_if(resource_changed::<Fill>)
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
    simulation: Res<Fill>,
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

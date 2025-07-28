// ANCHOR: use
use rand::{Rng, seq::IndexedRandom};
use std::f64::consts::PI;

use hoomd_interaction::{
    CutoffPair, Single, TotalEnergy,
    external::Linear,
    pairwise::{Boxcar, Isotropic},
};
use hoomd_mc::{LocalTrial, Sweep, Trial};
use hoomd_microstate::{
    Body, Microstate, MicrostateBuilder, boundary::Square, property::{OrientedPoint, Point},
};
use hoomd_vector::{Angle, Cartesian};
// ANCHOR_END: use

use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, Settings, Simulation,
    representation::disk::{self, Disk},
    representation::RectangularBoundary,
};

use anyhow::Context;
use bevy::prelude::*;
use bevy::render::storage::ShaderStorageBuffer;
use std::iter;

// TODO: fix spelling error!
// TODO: Type alias for body and site properties

// ANCHOR: simulation_new
impl Tetronimos {
    /// Construct a new tetronimo simulation.
    fn new() -> anyhow::Result<Tetronimos> {
        let box_height = 30.0;
        let kt = 1.0;
        let alpha = 1.0;

        let microstate = MicrostateBuilder::<OrientedPoint<Cartesian<2>, Angle>, Point<Cartesian<2>>, Square>::with_boundary(Square {
            l: box_height.try_into()?,
        })
        .try_build()?;

        let linear = Single(Linear {
            alpha,
            plane_origin: Cartesian::default(),
            plane_normal: [0.0, 1.0].try_into()?,
        });

        let boxcar = Boxcar {
            epsilon: 1000.0,
            left: 0.0,
            right: 1.0,
        };
        let isotropic = Isotropic(boxcar);
        let cutoff_pair = CutoffPair {
            r_cut: 1.0,
            evaluator: isotropic,
        };

        let hamiltonian = (linear, cutoff_pair);

        // ANCHOR: trial_moves
        let translate_sweep = Sweep(Discrete);
        // ANCHOR_END: trial_moves
        
        // ANCHOR: template_sites
        let template_sites = vec![
            // square
            vec![Point::new([-0.5, -0.5].into()),
                 Point::new([0.5, -0.5].into()),
                 Point::new([0.5, 0.5].into()),
                 Point::new([-0.5, 0.5].into())],
            // line
            vec![Point::new([-1.5, 0.5].into()),
                 Point::new([-0.5, 0.5].into()),
                 Point::new([0.5, 0.5].into()),
                 Point::new([1.5, 0.5].into())],
            // T
            vec![Point::new([-1.5, -0.5].into()),
                 Point::new([-0.5, -0.5].into()),
                 Point::new([0.5, -0.5].into()),
                 Point::new([-0.5, 0.5].into())],
            // L1
            vec![Point::new([-1.5, -0.5].into()),
                 Point::new([-0.5, -0.5].into()),
                 Point::new([0.5, -0.5].into()),
                 Point::new([0.5, 0.5].into())],
            // L2
            vec![Point::new([-1.5, 0.5].into()),
                 Point::new([-0.5, 0.5].into()),
                 Point::new([0.5, 0.5].into()),
                 Point::new([0.5, -0.5].into())],
            ];

        Ok(Tetronimos {
            microstate,
            hamiltonian,
            translate_sweep,
            kt,
            template_sites,
        })
    }
}
// ANCHOR_END: simulation_new

// ANCHOR: impl_simulation
impl Simulation for Tetronimos {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        // ANCHOR: add
        if self.microstate.step() % 100 == 0 {
            let properties = OrientedPoint {
                position: [0.0, self.microstate.boundary().l.get() / 2.0 - 2.0].into(),
                orientation: Angle::from(0.0),
            };
            let mut rng = self.microstate.counter().make_rng();
            let sites = self.template_sites.choose(&mut rng)
                .expect("template_sites should have at least 1 element")
                .clone();
            self.microstate.add_body(Body { sites, properties })?;
            self.microstate.increment_substep();
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

// ANCHOR: local_trial_all
// ANCHOR: local_trial_struct
/// Take fixed steps left, right, down, or up.
struct Discrete;
// ANCHOR_END: local_trial_struct

impl LocalTrial<OrientedPoint<Cartesian<2>, Angle>> for Discrete {
    // ANCHOR: local_trial_fn
    fn propose<R: Rng>(
        &self,
        rng: &mut R,
        body_properties: OrientedPoint<Cartesian<2>, Angle>,
    ) -> OrientedPoint<Cartesian<2>, Angle> {
        // ANCHOR_END: local_trial_fn
        // ANCHOR: local_trial_steps
        let translate_steps = [
            [0.0, -1.0].into(),
            [0.0, 1.0].into(),
            [-1.0, 0.0].into(),
            [1.0, 0.0].into(),
        ];
        let rotate_steps = [-PI/2.0, PI/2.0];
        // ANCHOR_END: local_trial_steps

        // ANCHOR: local_trial_mut
        let mut trial = body_properties;
        if rng.random_bool(0.9) {
            trial.position += *translate_steps
                .choose(rng)
                .expect("translate_steps should have at least 1 element");
        } else {
            trial.orientation.theta += *rotate_steps
                .choose(rng)
                .expect("rotate_steps should have at least 1 element");
        }
        trial
        // ANCHOR_END: local_trial_mut
    }
}
// ANCHOR_END: local_trial_all

#[derive(Resource)]
// ANCHOR: simulation_struct
struct Tetronimos {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<OrientedPoint<Cartesian<2>, Angle>, Point<Cartesian<2>>, Square>,
    /// How sites interact with other sites and fields.
    hamiltonian: (Single<Linear<Cartesian<2>>>, CutoffPair<Isotropic<Boxcar>>),
    /// Trial moves to apply.
    translate_sweep: Sweep<Discrete>,
    /// Temperature set point.
    kt: f64,
    /// Tetronimo shapes.
    template_sites: Vec<Vec<Point<Cartesian<2>>>>,
    
}
// ANCHOR_END: simulation_struct

/// Mark the disk representation type.
struct A;

fn main() -> anyhow::Result<()> {
    let simulation = Tetronimos::new().context("failed to setup simulation")?;
    let l = simulation.microstate.boundary().l.get() as f32;
    let hoomd_bevy_plugin = HoomdBevyPlugin {
        initial_settings: Settings {
            sps_limit: 64.0,
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
        ((|| disk::MaterialParameters::default()).pipe(Disk::<A>::setup), setup_colors).chain(),
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
            .run_if(resource_changed::<Tetronimos>)
            .after(AdvanceSet),
    );

    app.run();

    Ok(())
}

/// Set the tetronimo colors.
fn setup_colors(
    disk_representation: ResMut<disk::Representation<A>>,
    mut materials: ResMut<Assets<disk::Material>>,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    let material = materials.get_mut(disk_representation.material()).expect("Disk::setup should have added the material");

    let color_buffer = buffers.get_mut(&material.background_colors).expect("Disk::setup should have added the storage buffer");

    let color_wheel = (0..360*4).step_by(39).map(|i| Color::oklch(0.75, 0.1246, (i % 360) as f32));
    let linear_color_wheel = color_wheel.map(LinearRgba::from);
    let duplicate = linear_color_wheel.flat_map(|v| iter::repeat_n(v, 4));
    color_buffer.set_data(duplicate.collect::<Vec<_>>());
}
/// Copy the current positions of simulation particles to bevy entities.
fn sync_simulation(
    mut commands: Commands,
    disk_representation: Res<disk::Representation<A>>,
    query: Query<(Entity, &mut Transform), With<Disk<A>>>,
    simulation: Res<Tetronimos>,
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

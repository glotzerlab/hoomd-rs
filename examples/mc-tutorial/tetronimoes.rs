// ANCHOR: use
use rand::{Rng, seq::IndexedRandom};
use std::f64::consts::PI;

use hoomd_geometry::shape::Rectangle;
use hoomd_interaction::{
    CutoffPair, Single, TotalEnergy,
    external::Linear,
    pairwise::{Boxcar, Isotropic},
};
use hoomd_mc::{LocalTrial, Sweep, Trial};
use hoomd_microstate::{
    Body, Microstate, MicrostateBuilder,
    boundary::Closed,
    property::{OrientedPoint, Point},
};
use hoomd_vector::{Angle, Cartesian};
// ANCHOR_END: use

use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, Settings, Simulation,
    representation::RectangularBoundary,
    representation::disk::{self, Disk},
};

use anyhow::Context;
use bevy::prelude::*;
use bevy::render::storage::ShaderStorageBuffer;
use std::iter;

// ANCHOR: type_aliases
type PositionVector = Cartesian<2>;
type BodyProperties = OrientedPoint<PositionVector, Angle>;
type SiteProperties = Point<PositionVector>;
// ANCHOR_END: type_aliases

// ANCHOR: local_trial_all
/// Take fixed steps left, right, down, up, rotate left, or rotate right.
struct DiscreteRotateOrTranslate;

impl LocalTrial<BodyProperties> for DiscreteRotateOrTranslate {
    fn propose<R: Rng>(
        &self,
        rng: &mut R,
        body_properties: BodyProperties,
    ) -> BodyProperties {
        // ANCHOR: local_trial_steps
        let translate_steps = [
            [0.0, -1.0].into(),
            [0.0, 1.0].into(),
            [-1.0, 0.0].into(),
            [1.0, 0.0].into(),
        ];
        let rotate_steps = [-PI / 2.0, PI / 2.0];
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

// ANCHOR: simulation_new
impl Tetronimoes {
    /// Construct a new tetronimo simulation.
    fn new() -> anyhow::Result<Tetronimoes> {
        let box_height = 30.0;
        let kt = 1.0;
        let alpha = 1.0;
        let epsilon = 1000.0;
        let sigma = 1.0;

        let square = Rectangle::with_equal_edges(box_height.try_into()?);
        let microstate = MicrostateBuilder::<
            BodyProperties,
            SiteProperties,
            Closed<Rectangle>,
        >::with_boundary(Closed(square))
        .try_build()?;

        let linear = Single(Linear {
            alpha,
            plane_origin: Cartesian::default(),
            plane_normal: [0.0, 1.0].try_into()?,
        });

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

        let hamiltonian = (linear, cutoff_pair);

        // ANCHOR: trial_moves
        let sweep = Sweep(DiscreteRotateOrTranslate);
        // ANCHOR_END: trial_moves

        // ANCHOR: template_sites
        let template_sites = vec![
            // square
            vec![
                Point::new([-0.5, -0.5].into()),
                Point::new([0.5, -0.5].into()),
                Point::new([0.5, 0.5].into()),
                Point::new([-0.5, 0.5].into()),
            ],
            // line
            vec![
                Point::new([-1.5, 0.5].into()),
                Point::new([-0.5, 0.5].into()),
                Point::new([0.5, 0.5].into()),
                Point::new([1.5, 0.5].into()),
            ],
            // T
            vec![
                Point::new([-1.5, -0.5].into()),
                Point::new([-0.5, -0.5].into()),
                Point::new([0.5, -0.5].into()),
                Point::new([-0.5, 0.5].into()),
            ],
            // L1
            vec![
                Point::new([-1.5, -0.5].into()),
                Point::new([-0.5, -0.5].into()),
                Point::new([0.5, -0.5].into()),
                Point::new([0.5, 0.5].into()),
            ],
            // L2
            vec![
                Point::new([-1.5, 0.5].into()),
                Point::new([-0.5, 0.5].into()),
                Point::new([0.5, 0.5].into()),
                Point::new([0.5, -0.5].into()),
            ],
        ];
        // ANCHOR_END: template_sites

        Ok(Tetronimoes {
            microstate,
            hamiltonian,
            sweep,
            kt,
            template_sites,
        })
    }
}
// ANCHOR_END: simulation_new

#[derive(Resource)]
// ANCHOR: simulation_struct
struct Tetronimoes {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<BodyProperties, SiteProperties, Closed<Rectangle>>,
    /// How sites interact with other sites and fields.
    hamiltonian: (Single<Linear<PositionVector>>, CutoffPair<Isotropic<Boxcar>>),
    /// Trial moves to apply.
    sweep: Sweep<DiscreteRotateOrTranslate>,
    /// Temperature set point.
    kt: f64,
    /// Tetronimo shapes.
    template_sites: Vec<Vec<Point<PositionVector>>>,
}
// ANCHOR_END: simulation_struct

// ANCHOR: impl_simulation
impl Simulation for Tetronimoes {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        // ANCHOR: add
        if self.microstate.step() % 100 == 0 {
            let mut rng = self.microstate.counter().make_rng();
            let sites = self
                .template_sites
                .choose(&mut rng)
                .expect("template_sites should have at least 1 element")
                .clone();

            let properties = OrientedPoint {
                position: [
                    0.0,
                    self.microstate.boundary().0.edge_lengths[1].get() / 2.0
                        - 2.0,
                ]
                .into(),
                orientation: Angle::from(0.0),
            };

            self.microstate.add_body(Body { sites, properties })?;
            self.microstate.increment_substep();
        }
        // ANCHOR_END: add

        self.sweep
            .apply(&mut self.microstate, &self.hamiltonian, &self.kt);
        self.microstate.increment_step();

        if self.hamiltonian.1.total_energy(&self.microstate) > 20_000.0 {
            self.microstate.clear();
        }

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
        Tetronimoes::new().context("failed to setup simulation")?;
    let l = simulation.microstate.boundary().0.edge_lengths[1].get() as f32;
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
        (
            (|| disk::MaterialParameters::default()).pipe(Disk::<A>::setup),
            setup_colors,
        )
            .chain(),
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
            .run_if(resource_changed::<Tetronimoes>)
            .after(AdvanceSet),
    );

    app.run();

    Ok(())
}

/// Set the tetronimo colors.
fn setup_colors(
    disk_representation: ResMut<disk::Representation<A>>,
    mut materials: ResMut<Assets<disk::Material>>,
    buffers: ResMut<Assets<ShaderStorageBuffer>>,
) {
    let material = materials
        .get_mut(disk_representation.material())
        .expect("Disk::setup should have added the material");

    let color_wheel = (0..360 * 4)
        .step_by(39)
        .map(|i| Color::oklch(0.75, 0.1246, (i % 360) as f32));
    let linear_color_wheel = color_wheel.map(LinearRgba::from);
    let duplicate = linear_color_wheel.flat_map(|v| iter::repeat_n(v, 4));
    material.set_background_colors(buffers, &duplicate.collect());
}

/// Copy the current positions of simulation particles to bevy entities.
fn sync_simulation(
    mut commands: Commands,
    disk_representation: Res<disk::Representation<A>>,
    query: Query<(Entity, &mut Transform), With<Disk<A>>>,
    simulation: Res<Tetronimoes>,
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

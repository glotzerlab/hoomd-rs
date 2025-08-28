// ANCHOR: use
use hoomd_geometry::{
    shape::{Cuboid, Ellipse},
};
use hoomd_interaction::{
    CutoffPair, CutoffPairOverlap,
    pairwise::{Anisotropic, ApproximateShapeOverlap, HardShape, OverlapPenalty},
};
use hoomd_mc::{QuickInsert, Rotate, Sweep, Translate, Trial, UniformIn};
use hoomd_microstate::{
    Microstate, MicrostateBuilder, boundary::Periodic, property::OrientedPoint,
};
use hoomd_vector::{self, Angle, Cartesian};
// ANCHOR_END: use

use hoomd_bevy::{
    AdvanceSet, HoomdBevyPlugin, MUTED_COLOR, Settings, Simulation,
    representation::RectangularBoundary, representation::ellipse,
};

use anyhow::Context;
use bevy::prelude::*;

enum Phase {
    Initialization,
    Equilibration,
}

// ANCHOR: simulation_new
impl HardEllipseSelfAssembly {
    /// Construct a new fill simulation.
    fn new() -> anyhow::Result<HardEllipseSelfAssembly> {
        let box_height = 14.0;
        let kt = 1.0;
        let d = 0.05;
        let a = 0.1;
        let sigma = 1.0;

        let square = Cuboid::with_equal_edges(box_height.try_into()?);
        let microstate =
            MicrostateBuilder::with_boundary(Periodic::new(sigma, square)?)
                .try_build()?;

        // ANCHOR: pair
        let ellipse = Ellipse {
            axes: [0.5.try_into()?, (0.5 / 5.0).try_into()?],
        };
        let cutoff_pair = CutoffPairOverlap {
            r_cut: sigma,
            evaluator: HardShape(ellipse.clone()),
        };
        // ANCHOR_END: pair

        // ANCHOR: hamiltonian
        let hamiltonian = cutoff_pair;
        // ANCHOR_END: hamiltonian

        let approximate_shape_overlap = Anisotropic(ApproximateShapeOverlap::new(
            ellipse,
            OverlapPenalty::default(),
            0.01.try_into()?));
            
        let insert_hamiltonian = CutoffPair {
            r_cut: sigma,
            evaluator: approximate_shape_overlap,
        };

        let translate = Translate {
            maximum_distance: d.try_into()?,
        };
        let translate_sweep = Sweep(translate);

        let rotate = Rotate {
            maximum_rotation: a.try_into()?,
        };
        let rotate_sweep = Sweep(rotate);

        let distribution = UniformIn {
            boundary: *microstate.boundary(),
            template_sites: vec![OrientedPoint::default()],
        };
        let quick_insert = QuickInsert::new(distribution, 820);

        Ok(HardEllipseSelfAssembly {
            microstate,
            insert_hamiltonian,
            hamiltonian,
            translate_sweep,
            rotate_sweep,
            quick_insert,
            kt,
            phase: Phase::Initialization,
        })
    }
}
// ANCHOR_END: simulation_new

// ANCHOR: impl_simulation
impl Simulation for HardEllipseSelfAssembly {
    /// Advance the simulation forward one step.
    fn advance(&mut self) -> anyhow::Result<()> {
        let n = self.microstate.sites().len();

        match self.phase {
            Phase::Initialization => {
                self.quick_insert.apply(
                    &mut self.microstate,
                    &self.insert_hamiltonian,
                    &self.translate_sweep,
                    &1.0,
                );

                if self.quick_insert.is_complete() {
                    self.phase = Phase::Equilibration;
                    println!("{}: Complete", self.microstate.step());
                }
            }
            Phase::Equilibration => {
                self.translate_sweep.apply(
                    &mut self.microstate,
                    &self.hamiltonian,
                    &self.kt,
                );
            }
        }

        let n_new = self.microstate.sites().len();
        if n_new != n {
            println!("{}: {n_new}", self.microstate.step());
        }

        self.rotate_sweep.apply(
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

#[derive(Resource)]
// ANCHOR: simulation_struct
struct HardEllipseSelfAssembly {
    /// Positions of all the bodies in the simulation.
    microstate: Microstate<
        OrientedPoint<Cartesian<2>, Angle>,
        OrientedPoint<Cartesian<2>, Angle>,
        Periodic<Cuboid<2>>,
    >,
    /// How sites interact when inserted.
    insert_hamiltonian: CutoffPair<Anisotropic<ApproximateShapeOverlap<OverlapPenalty, Ellipse>>>,
    /// How sites interact with other sites and fields.
    hamiltonian: CutoffPairOverlap<HardShape<Ellipse>>,
    /// Trial moves to apply.
    translate_sweep: Sweep<Translate>,
    /// Trial moves to apply.
    rotate_sweep: Sweep<Rotate>,
    /// Quick insert
    quick_insert: QuickInsert<
        UniformIn<OrientedPoint<Cartesian<2>, Angle>, Periodic<Cuboid<2>>>,
    >,
    /// Temperature set point.
    kt: f64,
    phase: Phase,
}
// ANCHOR_END: simulation_struct

/// Mark the ellipse representation type.
struct A;

/// Mark the ghost representation type.
struct Ghost;

fn main() -> anyhow::Result<()> {
    let simulation =
        HardEllipseSelfAssembly::new().context("failed to setup simulation")?;
    let l =
        simulation.microstate.boundary().shape().edge_lengths[1].get() as f32;
    let hoomd_bevy_plugin = HoomdBevyPlugin {
        initial_settings: Settings {
            viewport_height: l + 2.0,
            ..default()
        },
        simulation,
    };

    let mut app = App::new();
    hoomd_bevy::add_default_plugins(&mut app);
    hoomd_bevy_plugin.build(&mut app);
    app.add_systems(
        Startup,
        (|| ellipse::MaterialParameters {
            outline_width: 0.025,
            ..default()
        })
        .pipe(ellipse::Ellipse::<A>::setup),
    );
    app.add_systems(
        Startup,
        (|| ellipse::MaterialParameters {
            outline_width: 0.025,
            background_color: MUTED_COLOR.into(),
            ..default()
        })
        .pipe(ellipse::Ellipse::<Ghost>::setup),
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
        (sync_sites, sync_ghosts)
            .run_if(resource_changed::<HardEllipseSelfAssembly>)
            .after(AdvanceSet),
    );

    app.run();

    Ok(())
}

/// Copy the current positions of simulation sites to bevy entities.
fn sync_sites(
    mut commands: Commands,
    site_representation: Res<ellipse::Representation<A>>,
    site_query: Query<(Entity, &mut Transform), With<ellipse::Ellipse<A>>>,
    simulation: Res<HardEllipseSelfAssembly>,
) {
    let sites = simulation.microstate.sites();
    ellipse::Ellipse::sync(
        &mut commands,
        site_representation,
        site_query,
        sites.iter().map(|site| {
            (
                Vec3::new(
                    site.properties.position[0] as f32,
                    site.properties.position[1] as f32,
                    0.0,
                ),
                site.properties.orientation.theta as f32,
                1.0,
                1.0 / 5.0,
            )
        }),
    );
}

/// Copy the current positions of simulation ghosts to bevy entities.
fn sync_ghosts(
    mut commands: Commands,
    ghost_representation: Res<ellipse::Representation<Ghost>>,
    ghost_query: Query<(Entity, &mut Transform), With<ellipse::Ellipse<Ghost>>>,
    simulation: Res<HardEllipseSelfAssembly>,
) {
    let ghosts = simulation.microstate.ghosts();
    ellipse::Ellipse::sync(
        &mut commands,
        ghost_representation,
        ghost_query,
        ghosts.iter().map(|site| {
            (
                Vec3::new(
                    site.properties.position[0] as f32,
                    site.properties.position[1] as f32,
                    0.0,
                ),
                site.properties.orientation.theta as f32,
                1.0,
                1.0 / 5.0,
            )
        }),
    );
}

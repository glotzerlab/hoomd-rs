// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Disk.
*/

use bevy::prelude::*;
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;

// TODO: introduce phantom marker types to differentiate different disks.

/** Represent an entity with a 2D disk in the plane z=0.

To use:
* Add [`setup`](Self::setup) to the `Startup` schedule.
* Call [`sync`](Self::sync) in an `Update` schedule that runs after `AdvanceSet`.
*/
#[derive(Component)]
pub struct Disk;

/// Assets that represent a Disk in the scene.
#[derive(Resource)]
pub struct DiskAssets {
    /// The disk's mesh.
    mesh: Handle<Mesh>,
    /// The disk's color.
    color: Handle<ColorMaterial>,
}

impl Disk {
/** Create assets to render disks.

Disks are currently fixed to a diameter of 1.0. Provide a non-unit scale in
[`sync`] to render disks of different sizes.

[This technique](https://www.reddit.com/r/bevy/comments/1bwq9a0/plugin_system_initialization_pattern/)
would allow for configurable setup at the cost of more boilerplate code.
*/
pub fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>, 
    mut materials: ResMut<Assets<ColorMaterial>>,
    ) {
    let mesh = meshes.add(Circle::new(0.5));
    let color = materials.add(Color::oklch(0.64, 0.14, 256.71));
    commands.insert_resource(DiskAssets { mesh, color });
    }

/// Copy the current positions of simulation particles to bevy entities.
pub fn sync<T, I, F1, F2>(
    commands: &mut Commands,
    disk_assets: Res<DiskAssets>,
    query: Query<(Entity, &mut Transform), With<Self>>,
    sites: I,
    position: F1,
    diameter: F2
    ) where
I: IntoIterator<Item = T>,
F1: Fn(&T) -> Vec3,
F2: Fn(&T) -> f32,
    {
    for item in &mut query.into_iter().zip_longest(sites) {
        match item {
            Both((_, mut transform), item) => {
                transform.translation = position(&item);
                transform.scale = Vec3::splat(diameter(&item));
            }
            Left((entity, _)) => commands.entity(entity).despawn(),
            Right(item) => {
            commands.spawn((
                Mesh2d(disk_assets.mesh.clone()),
                MeshMaterial2d(disk_assets.color.clone()),
                Transform::from_translation(
                    position(&item)
                ).with_scale(Vec3::splat(diameter(&item))),
                Self,
            ));    
            },
    }
}
}
}

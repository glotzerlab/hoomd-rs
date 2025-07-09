// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Disk.
*/

use bevy::{
    prelude::*,
    asset::embedded_asset,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderRef},
    render::texture::TRANSPARENT_IMAGE_HANDLE,
    sprite::{AlphaMode2d, Material2d, Material2dPlugin},
};
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;

// TODO: introduce phantom marker types to differentiate different disks.
// TODO: Use closure to initialize disks at a given radius.

/// Location of the shader imlplementation
const SHADER_ASSET_PATH: &str = "embedded://hoomd_bevy/representation/disk.wgsl";

/** Represent an entity with a 2D disk in the plane z=0.

Disks are fixed to a diameter of 1.0. Provide a non-unit scale in [`sync`]
to render disks of different sizes.

To use:
* Add [`setup`](Self::setup) to the `Startup` schedule.
* Call [`sync`](Self::sync) in an `Update` schedule that runs after `AdvanceSet`.
*/
#[derive(Component)]
pub struct Disk;

/// Assets that represent a Disk in the scene.
#[derive(Resource)]
pub struct DiskAssets {
    /// The disk mesh.
    mesh: Handle<Mesh>,
    /// The disk material.
    material: Handle<DiskMaterial>,
}

/// Initialize needed plugins and add assets for this representation.
pub(crate) fn build(app: &mut App) {
    app.add_plugins(Material2dPlugin::<DiskMaterial>::default());
    embedded_asset!(app, "disk.wgsl");
}

impl Disk {
    /** Create assets to render disks.

    [This technique](https://www.reddit.com/r/bevy/comments/1bwq9a0/plugin_system_initialization_pattern/)
    would allow for configurable setup at the cost of more boilerplate code.
    */
    pub fn setup(
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<DiskMaterial>>,
        server: Res<AssetServer>,
    ) {
        let mesh = meshes.add(Rectangle::new(1.0, 1.0));
        let material = materials.add(DiskMaterial { texture: server.load("embedded://hoomd_bevy/logo.png"), ..default()});
        commands.insert_resource(DiskAssets { mesh, material });
    }

    /// Copy the current positions of simulation particles to bevy entities.
    pub fn sync<'a, T, I, F1, F2>(
        commands: &mut Commands,
        disk_assets: Res<DiskAssets>,
        query: Query<(Entity, &mut Transform), With<Self>>,
        sites: I,
        position: F1,
        diameter: F2,
    ) where
        T: 'a,
        I: IntoIterator<Item = &'a T>,
        F1: Fn(&T) -> Vec3,
        F2: Fn(&T) -> f32,
    {
        for item in &mut query.into_iter().zip_longest(sites) {
            match item {
                Both((_, mut transform), item) => {
                    transform.translation = position(item);
                    transform.scale = Vec3::splat(diameter(item));
                }
                Left((entity, _)) => commands.entity(entity).despawn(),
                Right(item) => {
                    commands.spawn((
                        Mesh2d(disk_assets.mesh.clone()),
                        MeshMaterial2d(disk_assets.material.clone()),
                        Transform::from_translation(position(item))
                            .with_scale(Vec3::splat(diameter(item))),
                        Self,
                    ));
                }
            }
        }
    }
}

/// Control how disks are rendered.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct DiskMaterial {
    /// Color applied to the interior of the disk.
    #[uniform(0)]
    background_color: LinearRgba,

    /// Color applied to the outline.
    #[uniform(1)]
    outline_color: LinearRgba,

    /// Width of the outline.
    #[uniform(2)]
    outline_width: f32,

    /// Texture to apply. Blended with `color`.
    #[texture(3)]
    #[sampler(4)]
    texture: Handle<Image>,
}

impl Default for DiskMaterial {
    fn default() -> Self {
        Self {
        background_color: Color::oklch(0.6795, 0.1708, 255.71).into(),
        outline_color: Color::linear_rgb(0.0, 0.0, 0.0).into(),
        outline_width: 0.05,
        texture: TRANSPARENT_IMAGE_HANDLE,}
    }
}

impl Material2d for DiskMaterial {
    fn fragment_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Blend
    }
}

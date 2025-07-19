// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Disk.
*/

use bevy::{
    asset::embedded_asset,
    prelude::*,
    reflect::TypePath,
    render::render_resource::{AsBindGroup, ShaderRef},
    render::texture::TRANSPARENT_IMAGE_HANDLE,
    sprite::{AlphaMode2d, Material2d, Material2dPlugin},
};
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use std::marker::PhantomData;

use crate::PRIMARY_COLOR;

/// Location of the shader implementation
const SHADER_ASSET_PATH: &str = "embedded://hoomd_bevy/representation/disk.wgsl";

/** Represent an entity with a 2D disk in the xy plane.

The base representation has a diameter of 1.0. Provide a non-unit diameter in
[`sync`] to render disks of different sizes. Nominally, the z coordinate of the
disks should be set to 0. Choose a different value to control the back to front
draw order.

All disks of the same type must have the same material. To display disks with
different colors, outline widths, or textures, `setup` and `sync` multiple types
of disks with different marker types.

To use:
* Add [`setup`](Self::setup) to the `Startup` schedule.
* Call [`sync`](Self::sync) in an `Update` schedule that runs after `AdvanceSet`.
*/
#[derive(Component)]
pub struct Disk<T> {
    /// Mark the type of the disk.
    marker: PhantomData<T>,
}

/// Assets that represent a Disk in the scene.
#[derive(Resource)]
pub struct DiskAssets<T> {
    /// The disk mesh.
    mesh: Handle<Mesh>,
    /// The disk material.
    material: Handle<DiskMaterial>,
    /// Mark the type of the disk assets.
    marker: PhantomData<T>,
}

/// Initialize needed plugins and add assets for this representation.
pub(crate) fn build(app: &mut App) {
    app.add_plugins(Material2dPlugin::<DiskMaterial>::default());
    embedded_asset!(app, "disk.wgsl");
}

impl<T: Send + Sync + 'static> Disk<T> {
    /** Create assets to render disks.
     */
    pub fn setup(
        material: In<DiskMaterial>,
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<DiskMaterial>>,
    ) {
        let mesh = meshes.add(Rectangle::new(1.0, 1.0));
        let material = materials.add(material.0);
        commands.insert_resource(DiskAssets::<T> {
            mesh,
            material,
            marker: PhantomData,
        });
    }

    /// Copy the current positions of simulation particles to bevy entities.
    pub fn sync<I>(
        commands: &mut Commands,
        disk_assets: Res<DiskAssets<T>>,
        query: Query<(Entity, &mut Transform), With<Self>>,
        disks: I,
    ) where
        I: IntoIterator<Item = (Vec3, f32)>,
    {
        for item in &mut query.into_iter().zip_longest(disks) {
            match item {
                Both((_, mut transform), (position, diameter)) => {
                    transform.translation = position;
                    transform.scale = Vec3::splat(diameter);
                }
                Left((entity, _)) => commands.entity(entity).despawn(),
                Right((position, diameter)) => {
                    commands.spawn((
                        Mesh2d(disk_assets.mesh.clone()),
                        MeshMaterial2d(disk_assets.material.clone()),
                        Transform::from_translation(position).with_scale(Vec3::splat(diameter)),
                        Self {
                            marker: PhantomData,
                        },
                    ));
                }
            }
        }
    }
}

/** Control how disks are rendered.

[`DiskMaterial`] mixes the texture (which defaults to fully transparent) with
the background color using the texture alpha. It ignores the background alpha.

Control the draw order using the z coordinate. The draw order is non-deterministic
for all disks at the same z value.
*/
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct DiskMaterial {
    /// Color applied to the interior of the disk.
    #[uniform(0)]
    pub background_color: LinearRgba,

    /// Color applied to the outline.
    #[uniform(1)]
    pub outline_color: LinearRgba,

    /// Width of the outline.
    #[uniform(2)]
    pub outline_width: f32,

    /// Factor to scale the texture by.
    #[uniform(3)]
    pub texture_scale: f32,

    /// Texture to apply. Blended with `color`.
    #[texture(4)]
    #[sampler(5)]
    pub texture: Handle<Image>,
}

impl Default for DiskMaterial {
    fn default() -> Self {
        Self {
            background_color: PRIMARY_COLOR.into(),
            outline_color: Color::linear_rgb(0.0, 0.0, 0.0).into(),
            outline_width: 0.05,
            texture: TRANSPARENT_IMAGE_HANDLE,
            texture_scale: 1.2,
        }
    }
}

impl Material2d for DiskMaterial {
    fn fragment_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Mask(0.5)
    }
}

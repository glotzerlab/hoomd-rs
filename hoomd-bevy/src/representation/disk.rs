// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! An outlined circle.

The [`Disk`] representation is a circle of pixels with a configurable
outline color and an optional texture map.
*/

use bevy::{
    asset::embedded_asset,
    prelude::*,
    reflect::TypePath,
    render::{
        mesh::MeshTag,
        render_resource::{AsBindGroup, ShaderRef},
        storage::ShaderStorageBuffer,
    },
    sprite::{AlphaMode2d, Material2d, Material2dPlugin},
};
#[cfg(all(target_arch = "wasm32", not(feature = "webgpu")))]
use bevy::{
    render::{
        mesh::MeshVertexBufferLayoutRef,
        render_resource::{RenderPipelineDescriptor, SpecializedMeshPipelineError},
    },
    sprite::Material2dKey,
};
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use std::marker::PhantomData;

use crate::PRIMARY_COLOR;

/// Location of the shader implementation
const SHADER_ASSET_PATH: &str = "embedded://hoomd_bevy/representation/disk.wgsl";

/** Represent an entity with a 2D disk in the xy plane.

The base representation has a diameter of 1.0. Provide a non-unit diameter
in [`sync`](Self::sync) to render disks of different sizes. Nominally, the z
coordinate of the disks should be set to 0. Choose a different value to control
the back to front draw order.

All disks of the same type must have the same material. To display disks with
different colors, outline widths, or textures, call `setup` and `sync` multiple
types of disks with different marker types.

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
pub struct Representation<T> {
    /// The disk mesh.
    mesh: Handle<Mesh>,
    /// The disk material.
    material: Handle<Material>,
    /// Mark the type of the disk assets.
    marker: PhantomData<T>,
}

impl<T> Representation<T> {
    /// Get the material
    #[must_use]
    pub fn material(&self) -> &Handle<Material> {
        &self.material
    }
}

/// Initialize needed plugins and add assets for this representation.
pub(crate) fn build(app: &mut App) {
    app.add_plugins(Material2dPlugin::<Material>::default());
    embedded_asset!(app, "disk.wgsl");
}

impl<T: Send + Sync + 'static> Disk<T> {
    /** Create assets to render disks.
     */
    pub fn setup(
        material: In<MaterialParameters>,
        mut commands: Commands,
        #[cfg(not(all(target_arch = "wasm32", not(feature = "webgpu"))))] mut buffers: ResMut<
            Assets<ShaderStorageBuffer>,
        >,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<Material>>,
        asset_server: Res<AssetServer>,
    ) {
        #[cfg(all(target_arch = "wasm32", not(feature = "webgpu")))]
        let background_colors = [material.0.background_color; 1024];

        #[cfg(not(all(target_arch = "wasm32", not(feature = "webgpu"))))]
        let background_colors =
            buffers.add(ShaderStorageBuffer::from([material.0.background_color]));

        let mesh = meshes.add(Rectangle::new(1.0, 1.0));
        let material = Material {
            background_colors,
            #[cfg(all(target_arch = "wasm32", not(feature = "webgpu")))]
            n_background_colors: 1,
            outline_color: material.0.outline_color,
            outline_width: material.0.outline_width,
            texture_scale: material.0.texture_scale,
            texture: material.0.texture_asset.map(|t| asset_server.load(t)),
        };
        let material = materials.add(material);

        commands.insert_resource(Representation::<T> {
            mesh,
            material,
            marker: PhantomData,
        });
    }

    /// Copy the current positions of simulation particles to bevy entities.
    pub fn sync<I>(
        commands: &mut Commands,
        disk_representation: Res<Representation<T>>,
        query: Query<(Entity, &mut Transform), With<Self>>,
        disks: I,
    ) where
        I: IntoIterator<Item = (Vec3, f32)>,
    {
        for (tag, item) in &mut query.into_iter().zip_longest(disks).enumerate() {
            match item {
                Both((_, mut transform), (position, diameter)) => {
                    transform.translation = position;
                    transform.scale = Vec3::splat(diameter);
                }
                Left((entity, _)) => commands.entity(entity).despawn(),
                Right((position, diameter)) => {
                    commands.spawn((
                        MeshTag(tag as u32),
                        Mesh2d(disk_representation.mesh.clone()),
                        MeshMaterial2d(disk_representation.material.clone()),
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

/// Initialize [`Material`] with these settings.
pub struct MaterialParameters {
    /// Color applied to the interior of the disk.
    pub background_color: LinearRgba,

    /// Color applied to the outline.
    pub outline_color: LinearRgba,

    /// Width of the outline.
    pub outline_width: f32,

    /// Factor to scale the texture by.
    pub texture_scale: f32,

    /// Name of the texture asset.
    pub texture_asset: Option<String>,
}

impl Default for MaterialParameters {
    fn default() -> Self {
        Self {
            background_color: PRIMARY_COLOR.into(),
            outline_color: Color::linear_rgb(0.0, 0.0, 0.0).into(),
            outline_width: 0.05,
            texture_asset: None,
            texture_scale: 1.2,
        }
    }
}

/** Control how disks are rendered.

Disks are always opaque and alpha in any texture or background color is ignored.

By default [`Material`] is initialized with only one background
color. Color the instances differently by setting more than one color
with [`set_background_colors`]. The color of each disk is given by
`background_colors[tag % len(background_colors)]` so you may set fewer colors
than there are disks. [`sync`] assigns `tag` values in increasing order to each
primitive.

The `background_color` tints the texture by multiplication. With a `None`
texture (the default), `background_color` sets the exact color of the disk.

Set the initial material by piping `MaterialParameters` into [`Disk::setup`].
After it is initialized, change the material during execution via the `material`
field in`ResMut<disk::Representation<A>>`.

[`sync`]: Disk::sync
[`set_background_colors`]: Material::set_background_colors
*/
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct Material {
    /// Color applied to the outline.
    #[uniform(0)]
    outline_color: LinearRgba,

    /// Width of the outline.
    #[uniform(0)]
    outline_width: f32,

    /// Factor to scale the texture by.
    #[uniform(0)]
    texture_scale: f32,

    /// Number of background colors in fixed size array.
    #[uniform(0)]
    #[cfg(all(target_arch = "wasm32", not(feature = "webgpu")))]
    n_background_colors: u32,

    /// Texture to apply. Tinted by `background_color`.
    #[texture(1)]
    #[sampler(2)]
    texture: Option<Handle<Image>>,

    /// Color applied to the interior of the disk (indexed by disk % array size).
    #[uniform(3)]
    #[cfg(all(target_arch = "wasm32", not(feature = "webgpu")))]
    background_colors: [LinearRgba; 1024],

    /// Color applied to the interior of the disk (indexed by disk % array size).
    #[cfg(not(all(target_arch = "wasm32", not(feature = "webgpu"))))]
    #[storage(3, read_only)]
    background_colors: Handle<ShaderStorageBuffer>,
}

impl Material {
    /** Set new background colors.

    # Panics

    WebGL2 builds (identified by the `wasm32` target without the `webgpu`
    feature) support only 1024 background colors.

    Desktop target builds or `wasm32` target builds with `webgpu` support
    a much larger number of colors and will not panic.
    */
    pub fn set_background_colors(
        &mut self,
        mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
        colors: &Vec<LinearRgba>,
    ) {
        #[cfg(all(target_arch = "wasm32", not(feature = "webgpu")))]
        {
            if colors.len() > 1024 {
                panic!(
                    "webgl2 builds support up to 1024 colors, got {}",
                    colors.len()
                );
            }
            self.background_colors[..colors.len()].copy_from_slice(&colors);
            self.n_background_colors = colors.len() as u32;
        }

        #[cfg(not(all(target_arch = "wasm32", not(feature = "webgpu"))))]
        {
            let color_buffer = buffers
                .get_mut(&self.background_colors)
                .expect("Disk::setup should have added the storage buffer");

            color_buffer.set_data(colors);
        }
    }
}

impl Material2d for Material {
    fn fragment_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }

    fn vertex_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Mask(0.5)
    }

    #[cfg(all(target_arch = "wasm32", not(feature = "webgpu")))]
    fn specialize(
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: Material2dKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.vertex.shader_defs.push("WEBGL2".into());

        Ok(())
    }
}

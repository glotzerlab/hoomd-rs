// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! An outlined ellipse.
//!
//! The [`Ellipse`] representation is a ellipse of pixels with a configurable
//! outline color. Each ellipse can have a different aspect ratio.

use bevy::{
    asset::embedded_asset,
    mesh::MeshTag,
    prelude::*,
    reflect::TypePath,
    render::{render_resource::AsBindGroup, storage::ShaderStorageBuffer},
    shader::ShaderRef,
    sprite_render::{AlphaMode2d, Material2d, Material2dPlugin},
};
#[cfg(all(target_arch = "wasm32", not(feature = "webgpu")))]
use bevy::{
    mesh::MeshVertexBufferLayoutRef,
    render::render_resource::{RenderPipelineDescriptor, SpecializedMeshPipelineError},
    sprite_render::Material2dKey,
};
use itertools::{
    EitherOrBoth::{Both, Left, Right},
    Itertools,
};
use std::marker::PhantomData;

use crate::PRIMARY_COLOR;

/// Location of the shader implementation
const SHADER_ASSET_PATH: &str = "embedded://hoomd_bevy/representation/ellipse.wgsl";

/// Represent an entity with a 2D ellipse in the xy plane.
///
/// The base representation has semi-axes (0.5, 0.5). Provide per-item axes
/// in [`sync`](Self::sync) to render ellipses of different sizes and aspect ratios.
/// Nominally, the z coordinate of the ellipses should be set to 0. Choose a different
/// value to control the back to front draw order.
///
/// To use:
/// * Add [`setup`](Self::setup) to the `Startup` schedule.
/// * Call [`sync`](Self::sync) in an `Update` schedule that runs after `AdvanceSet`.
#[derive(Component)]
pub struct Ellipse<T> {
    /// Mark the type of the ellipse.
    marker: PhantomData<T>,
}

/// Assets that represent a ellipse in the scene.
#[derive(Resource)]
pub struct Representation<T> {
    /// The ellipse mesh.
    mesh: Handle<Mesh>,
    /// The ellipse material.
    material: Handle<Material>,
    /// Mark the type of the ellipse assets.
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
    embedded_asset!(app, "ellipse.wgsl");
}

impl<T: Send + Sync + 'static> Ellipse<T> {
    /// Create assets to render ellipses.
    pub fn setup(
        material: In<MaterialParameters>,
        mut commands: Commands,
        #[cfg(not(all(target_arch = "wasm32", not(feature = "webgpu"))))] mut buffers: ResMut<
            Assets<ShaderStorageBuffer>,
        >,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<Material>>,
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
        ellipse_representation: Res<Representation<T>>,
        query: Query<(Entity, &mut Transform), With<Self>>,
        ellipses: I,
    ) where
        I: IntoIterator<Item = (Vec3, f32, f32, f32)>,
    {
        for (tag, item) in &mut query.into_iter().zip_longest(ellipses).enumerate() {
            match item {
                Both((_, mut transform), (position, theta, a, b)) => {
                    transform.translation = position;
                    transform.rotation = Quat::from_rotation_z(theta);
                    transform.scale = Vec3::new(a, b, 1.0);
                }
                Left((entity, _)) => commands.entity(entity).despawn(),
                Right((position, theta, a, b)) => {
                    commands.spawn((
                        MeshTag(tag as u32),
                        Mesh2d(ellipse_representation.mesh.clone()),
                        MeshMaterial2d(ellipse_representation.material.clone()),
                        Transform::from_translation(position)
                            .with_scale(Vec3::new(a, b, 1.0))
                            .with_rotation(Quat::from_rotation_z(theta)),
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
    /// Color applied to the interior of the ellipses.
    pub background_color: LinearRgba,

    /// Color applied to the outline.
    pub outline_color: LinearRgba,

    /// Width of the outline.
    pub outline_width: f32,
}

impl Default for MaterialParameters {
    fn default() -> Self {
        Self {
            background_color: PRIMARY_COLOR.into(),
            outline_color: Color::linear_rgb(0.0, 0.0, 0.0).into(),
            outline_width: 0.05,
        }
    }
}

/// Control how ellipses are rendered.
///
/// Ellipses are always opaque and alpha in any background color is ignored.
///
/// By default [`Material`] is initialized with only one background
/// color. Color the instances differently by setting more than one color
/// with [`set_background_colors`]. The color of each ellipse is given by
/// `background_colors[tag % len(background_colors)]` so you may set fewer colors
/// than there are ellipses. [`sync`] assigns `tag` values in increasing order to each
/// primitive.
///
/// The `background_color` tints the texture by multiplication. With a `None`
/// texture (the default), `background_color` sets the exact color of the ellipses.
///
/// Set the initial material by piping `MaterialParameters` into [`Ellipse::setup`].
/// After it is initialized, change the material during execution via the `material`
/// field in`ResMut<ellipse::Representation<A>>`.
///
/// [`sync`]: Ellipse::sync
/// [`set_background_colors`]: Material::set_background_colors
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct Material {
    /// Color applied to the outline.
    #[uniform(0)]
    outline_color: LinearRgba,

    /// Width of the outline.
    #[uniform(0)]
    outline_width: f32,

    /// Number of background colors in fixed size array.
    #[uniform(0)]
    #[cfg(all(target_arch = "wasm32", not(feature = "webgpu")))]
    n_background_colors: u32,

    /// Color applied to the interior of the ellipse (indexed by ellipse % array size).
    #[uniform(1)]
    #[cfg(all(target_arch = "wasm32", not(feature = "webgpu")))]
    background_colors: [LinearRgba; 1024],

    /// Color applied to the interior of the ellipse (indexed by ellipse % array size).
    #[cfg(not(all(target_arch = "wasm32", not(feature = "webgpu"))))]
    #[storage(1, read_only)]
    background_colors: Handle<ShaderStorageBuffer>,
}

impl Material {
    /// Set new background colors.
    ///
    /// # Panics
    ///
    /// WebGL2 builds (identified by the `wasm32` target without the `webgpu`
    /// feature) support only 1024 background colors.
    ///
    /// Desktop target builds or `wasm32` target builds with `webgpu` support
    /// a much larger number of colors and will not panic.
    pub fn set_background_colors(
        &mut self,
        #[allow(
            unused_variables,
            unused_mut,
            reason = "Not used in all build configurations."
        )]
        mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
        colors: &[LinearRgba],
    ) {
        #[cfg(all(target_arch = "wasm32", not(feature = "webgpu")))]
        {
            assert!(
                colors.len() <= 1024,
                "webgl2 builds support up to 1024 colors, got {}",
                colors.len()
            );
            self.background_colors[..colors.len()].copy_from_slice(colors);
            self.n_background_colors = colors.len() as u32;
        }

        #[cfg(not(all(target_arch = "wasm32", not(feature = "webgpu"))))]
        {
            let color_buffer = buffers
                .get_mut(&self.background_colors)
                .expect("Ellipse::setup should have added the storage buffer");

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

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Hyperbolic disks in Bevy.

use crate::PRIMARY_COLOR;
use bevy::{
    asset::embedded_asset,
    prelude::*,
    reflect::TypePath,
    render::render_resource::AsBindGroup,
    shader::ShaderRef,
    sprite_render::{AlphaMode2d, Material2d, Material2dPlugin},
};
use hoomd_manifold::{Hyperbolic, Minkowski};
use itertools::{
    EitherOrBoth::{Both, Left, Right},
    Itertools,
};
use std::marker::PhantomData;

/// Location of the shader implementation
const SHADER_ASSET_PATH: &str = "embedded://hoomd_bevy/representation/hyperbolic_polygon.wgsl";

/// Represent an entity with a 2D regular polygon in hyperbolic space.
#[derive(Component)]
pub struct HyperbolicPolygon<T> {
    /// Mark the type of the disk.
    marker: PhantomData<T>,
}

/// Assets that represent a Disk in the scene.
#[derive(Resource)]
pub struct HyperbolicPolygonAssets<T> {
    /// The polygon mesh.
    mesh: Handle<Mesh>,
    /// The polygon material.
    material: Handle<HyperbolicPolygonMaterial>,
    /// Mark the type of the polygon assets.
    marker: PhantomData<T>,
}

/// Initialize needed plugins and add assets for this representation.
pub(crate) fn build(app: &mut App) {
    app.add_plugins(Material2dPlugin::<HyperbolicPolygonMaterial>::default());
    embedded_asset!(app, "hyperbolic_polygon.wgsl");
}

impl<T: Send + Sync + 'static> HyperbolicPolygon<T> {
    /// Create assets to render polygons.
    pub fn setup(
        material: In<HyperbolicPolygonMaterialParameters>,
        mut commands: Commands,
        //#[cfg(not(all(target_arch = "wasm32", not(feature = "webgpu"))))] mut buffers: ResMut<
        //    Assets<ShaderStorageBuffer>,
        //>,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<HyperbolicPolygonMaterial>>,
        asset_server: Res<AssetServer>,
    ) {
        //#[cfg(not(all(target_arch = "wasm32", not(feature = "webgpu"))))]
        let n_sides = material.0.n_sides;

        let mesh = meshes.add(Rectangle::new(1.0, 1.0));
        let material = HyperbolicPolygonMaterial {
            n_sides,
            background_color: material.0.background_color,
            outline_color: material.0.outline_color,
            outline_width: material.0.outline_width,
            texture_scale: material.0.texture_scale,
            texture: material.0.texture_asset.map(|t| asset_server.load(t)),
        };
        let material = materials.add(material);
        commands.insert_resource(HyperbolicPolygonAssets::<T> {
            mesh,
            material,
            marker: PhantomData,
        });
    }

    /// Copy the current positions of simulation particles to bevy entities.
    pub fn sync<I>(
        commands: &mut Commands,
        disk_assets: Res<HyperbolicPolygonAssets<T>>,
        query: Query<(Entity, &mut Transform), With<Self>>,
        disks: I,
    ) where
        I: IntoIterator<Item = (Minkowski<3>, f64, f32)>,
    {
        for item in &mut query.into_iter().zip_longest(disks) {
            match item {
                Both((_, mut transform), (position, radius, theta)) => {
                    let (poincare_position, max_projected_radius) =
                        poincare(&position, radius, theta);
                    // let rad_arg = RHO * (radius / RHO).sinh() / (1.0 + (radius / RHO).cosh());
                    // let poincare_radius = (0.5)
                    //    * (1.0 + 2.0 * rad_arg.powi(2) / (1.0 - (rad_arg.powi(2)))).acosh() as f32;
                    transform.translation = Vec3::from_array(poincare_position);
                    transform.scale = Vec3::from_array([
                        max_projected_radius * 2.0,
                        max_projected_radius * 2.0,
                        radius as f32, // radius in units of rapidity,
                    ]);
                    // transform.rotation = Quat::from_rotation_z(theta);
                }
                Left((entity, _)) => commands.entity(entity).despawn(),
                Right((position, radius, theta)) => {
                    let (poincare_position, max_projected_radius) =
                        poincare(&position, radius, theta);
                    // let rad_arg = RHO * (radius / RHO).sinh() / (1.0 + (radius / RHO).cosh());
                    // let poincare_radius = (0.5)
                    //    * (1.0 + 2.0 * rad_arg.powi(2) / (1.0 - (rad_arg.powi(2)))).acosh() as f32;
                    commands.spawn((
                        Mesh2d(disk_assets.mesh.clone()),
                        MeshMaterial2d(disk_assets.material.clone()),
                        Transform::from_translation(Vec3::from_array(poincare_position))
                            .with_scale(Vec3::from_array([
                                max_projected_radius * 2.0,
                                max_projected_radius * 2.0,
                                radius as f32, // radius in units of rapidity,
                            ])),
                        Self {
                            marker: PhantomData,
                        },
                    ));
                }
            }
        }
    }
}

/// Project coordinates to Poincaré disk
fn poincare(point: &Minkowski<3>, radius: f64, angle: f32) -> ([f32; 3], f32) {
    let pt = Hyperbolic::from_minkowski_coordinates(*point);
    let proj = pt.to_poincare();
    let v = radius;
    let eta = (point.coordinates[2]).acosh();
    let edge_proj = ((eta - v).sinh()) / (1.0 + (eta - v).cosh());
    let rad_proj = ((eta).sinh()) / (1.0 + (eta).cosh()) - edge_proj;
    ([proj[0] as f32, proj[1] as f32, angle], rad_proj as f32)
}

/// Control how hyperbolic polygons are rendered.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct HyperbolicPolygonMaterial {
    /// Color applied to the interior of the polygon.
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
    pub texture: Option<Handle<Image>>,
    /// Color applied to the interior of the disk (indexed by disk % array size).
    //#[cfg(not(all(target_arch = "wasm32", not(feature = "webgpu"))))]
    #[uniform(6)]
    pub n_sides: f32,
}

/// Material Parameters for hyperbolic polygon.
pub struct HyperbolicPolygonMaterialParameters {
    /// Number of sides of the polygon.
    pub n_sides: f32,
    /// Color applied to the interior of the polygon.
    pub background_color: LinearRgba,
    /// Color applied to the outline.
    pub outline_color: LinearRgba,
    /// Width of the outline.
    pub outline_width: f32,
    /// Factor to scale the texture by,.
    pub texture_scale: f32,
    /// Name of the texture asset.
    pub texture_asset: Option<String>,
}

impl Default for HyperbolicPolygonMaterialParameters {
    fn default() -> Self {
        Self {
            n_sides: 0.0_f32,
            background_color: PRIMARY_COLOR.into(),
            outline_color: Color::linear_rgb(0.0, 0.0, 0.0).into(),
            outline_width: 0.005,
            texture_asset: None,
            texture_scale: 1000.0,
        }
    }
}

impl HyperbolicPolygonMaterialParameters {
    /// color for ghost particles
    #[must_use]
    pub fn ghost() -> Self {
        Self {
            background_color: Color::linear_rgb(0.5, 0.5, 0.5).into(),
            outline_color: Color::linear_rgb(0.0, 0.0, 0.0).into(),
            outline_width: 0.005,
            texture_asset: None,
            texture_scale: 1.2,
            n_sides: 4.0_f32,
        }
    }
}

impl Material2d for HyperbolicPolygonMaterial {
    fn fragment_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }

    fn vertex_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        AlphaMode2d::Mask(0.5)
    }
}

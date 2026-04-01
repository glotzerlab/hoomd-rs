// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! A 2D mesh
//!
//! The [`PlaneMesh`] representation places a bevy `Mesh2d` at each site.

use bevy::prelude::*;
use itertools::{
    EitherOrBoth::{Both, Left, Right},
    Itertools,
};
use std::marker::PhantomData;

/// Represent each entity with a triangle mesh in 2D.
///
/// Each entity is an instanced copy of the given mesh. Provide the position and
/// orientation of each mesh to [`sync`](Self::sync).
///
/// All triangle meshes of the same type must have the same material. To display
/// meshes with different materials, call `setup` and `sync` for multiple types of
/// triangle meshes with different marker types.
///
/// To use:
/// * Add [`setup`](Self::setup) to the `Startup` schedule.
/// * Call [`sync`](Self::sync) in an `Update` schedule that runs after `AdvanceSet`.
#[derive(Component)]
pub struct PlaneMesh<T> {
    /// Mark the type of the disk.
    marker: PhantomData<T>,
}

/// Assets that represent a 3D mesh in the scene.
#[derive(Resource)]
pub struct Representation<T> {
    /// The mesh.
    mesh: Handle<Mesh>,
    /// The material.
    material: Handle<ColorMaterial>,
    /// Mark the type of the triangle mesh assets.
    marker: PhantomData<T>,
}

impl<T> Representation<T> {
    /// Get the material
    #[must_use]
    pub fn material(&self) -> &Handle<ColorMaterial> {
        &self.material
    }
}

impl<T: Send + Sync + 'static> PlaneMesh<T> {
    /// Create assets to render instanced triangle meshes.
    pub fn setup(
        mesh_material: In<(Mesh, ColorMaterial)>,
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<ColorMaterial>>,
    ) {
        let (mesh, material) = mesh_material.0;
        let mesh = meshes.add(mesh);
        let material = materials.add(material.clone());

        commands.insert_resource(Representation::<T> {
            mesh,
            material,
            marker: PhantomData,
        });
    }

    /// Copy the current positions of simulation particles to bevy entities.
    pub fn sync<I>(
        commands: &mut Commands,
        plane_mesh_representation: Res<Representation<T>>,
        query: Query<(Entity, &mut Transform), With<Self>>,
        triangle_meshes: I,
    ) where
        I: IntoIterator<Item = (Vec3, f32)>,
    {
        for item in &mut query.into_iter().zip_longest(triangle_meshes) {
            match item {
                Both((_, mut transform), (position, theta)) => {
                    transform.translation = position;
                    transform.rotation = Quat::from_rotation_z(theta);
                }
                Left((entity, _)) => commands.entity(entity).despawn(),
                Right((position, theta)) => {
                    commands.spawn((
                        Mesh2d(plane_mesh_representation.mesh.clone()),
                        MeshMaterial2d(plane_mesh_representation.material.clone()),
                        Transform::from_translation(position)
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

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `RectangularBoundary`.

use bevy::{
    asset::RenderAssetUsages,
    render::mesh::{Indices, VertexAttributeValues},
    prelude::*,
    render::render_resource::PrimitiveTopology,
};

use crate::BOUNDARY_COLOR;
use hoomd_geometry::shape::EightEight;

/// Represent the simulation boundary in hyperbolic space.
#[derive(Component)]
pub struct EightEightBoundary {
    /// Width of the octagon edges.
    pub thickness: f32,
    /// Color of the octagon.
    pub color: Color,
    /// The scale of the octagon.
    pub scale: f32,
}

impl Default for EightEightBoundary {
    fn default() -> Self {
        Self {
            thickness: 0.1,
            color: BOUNDARY_COLOR,
            scale: 1.0_f32,
        }
    }
}

impl EightEightBoundary {
    /// Create entities that render the boundaries. 
    pub fn setup(
        eighteight_boundary: In<Self>,
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<ColorMaterial>>,
    ) {
        let mesh = meshes.add(Self::create_octagon());
        let material = materials.add(ColorMaterial::from_color(eighteight_boundary.color));

        let thickness = eighteight_boundary.thickness;
        let scale = eighteight_boundary.scale;

        let shape = (
            Transform::from_xyz(0.0, 0.0, 0.0).with_scale(Vec3::new(
                scale,
                scale,
                1.0,
            )),
            Mesh2d(mesh.clone()),
            MeshMaterial2d(material.clone()),
        );

        commands.spawn((
            eighteight_boundary.0,
            Transform::default(),
            Visibility::Visible,
            children![shape],
        ));
    }
    fn create_octagon() -> Mesh {
        // Create a new mesh using a triangle list topology, where each set of 3 vertices composes a triangle.
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
            // Add 8 vertices, each with its own position attribute (coordinate in
            // 3D space), for each of the corners of the parallelogram.
            .with_inserted_attribute(
                Mesh::ATTRIBUTE_POSITION,
                VertexAttributeValues::Float32x3(EightEight::boundary_points(1000, 1.0)
                    .iter()
                    .map(|(x,y)| [*x as f32, *y as f32, 0.0_f32])
                    .collect::<Vec<_>>()
                    //.push([0.0_f32, 0.0_f32, 0.0_f32])
            ))
            // Assign a UV coordinate to each vertex.
            .with_inserted_attribute(
                Mesh::ATTRIBUTE_UV_0,
                VertexAttributeValues::Float32x3(EightEight::boundary_points(1000, 1.0)
                    .iter()
                    .map(|(x,y)| [*x as f32, *y as f32, 0.0_f32])
                    .collect()
                    //.push([0.0_f32, 0.0_f32, 0.0_f32])
            )
            )
            // Assign normals (everything points outwards)
            .with_inserted_attribute(
                Mesh::ATTRIBUTE_NORMAL,
                vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
            )
            // After defining all the vertices and their attributes, build each triangle using the
            // indices of the vertices that make it up in a counter-clockwise order.
            .with_inserted_indices(Indices::U32(
                Self::triangles()
            ))
    }
    fn triangles() -> Vec<u32> {
        let mut vect: Vec<u32> = vec![];
        for n in 0..1000 {
            vect.push(1000);
            vect.push(n);
            vect.push(n+1);
        }
        vect
    }
}

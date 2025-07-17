// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Graphical elements that depict the boundary conditions.
*/

use bevy::prelude::*;

use crate::BOUNDARY_COLOR;

/** Represent an simulation boundary with a thin lined rectangle.

The lines are a fixed thickness in world coordinates. A default
[`RectangularBoundary`] has:
* `width`: 1.0
* `height`: 1.0
* `thickness`: 0.1
* `color`: [`BOUNDARY_COLOR`](crate::BOUNDARY_COLOR)

To use:
* Add [`setup`](Self::setup) to the `Startup` schedule.
* (if needed) Call [`sync`](Self::sync) in an `Update` schedule that runs after `AdvanceSet`.
*/
#[derive(Component)]
pub struct RectangularBoundary {
    /// Extent of the rectangle's open space in the x direction.
    pub width: f32,

    /// Extent of the rectangle's open space in the y direction.
    pub height: f32,

    /// Width of the rectangle edges.
    pub thickness: f32,

    /// Color of the rectangle.
    pub color: Color,
    }

impl Default for RectangularBoundary {
    fn default() -> Self {
        Self { width: 1.0, height: 1.0, thickness: 0.1, color: BOUNDARY_COLOR, }
    }
}

impl RectangularBoundary {
    /** Create assets to render rectangular boundaries.
    */
    pub fn setup(
        rectangular_boundary: In<Self>,
        mut commands: Commands,
        mut meshes: ResMut<Assets<Mesh>>,
        mut materials: ResMut<Assets<ColorMaterial>>,
    ) {
        let mesh = meshes.add(Rectangle::new(1.0, 1.0));
        let material = materials.add(ColorMaterial::from_color(rectangular_boundary.color));

        let height = rectangular_boundary.height;
        let width = rectangular_boundary.width;
        let thickness = rectangular_boundary.thickness;
        let half_thickness = thickness/2.0;
        let double_thickness = thickness*2.0;

        let left = (Transform::from_xyz(-width/2.0-half_thickness, 0.0, 0.0)
                        .with_scale(Vec3::new(thickness, height+double_thickness, 1.0)),
                    Mesh2d(mesh.clone()),
                    MeshMaterial2d(material.clone()));
        let right = (Transform::from_xyz(width/2.0+half_thickness, 0.0, 0.0)
                        .with_scale(Vec3::new(thickness, height+double_thickness, 1.0)),
                    Mesh2d(mesh.clone()),
                    MeshMaterial2d(material.clone()));
        let bottom = (Transform::from_xyz(0.0, -height/2.0-half_thickness, 0.0)
                        .with_scale(Vec3::new(width+double_thickness, thickness, 1.0)),
                    Mesh2d(mesh.clone()),
                    MeshMaterial2d(material.clone()));
        let top = (Transform::from_xyz(0.0, height/2.0+half_thickness, 0.0)
                        .with_scale(Vec3::new(width+double_thickness, thickness, 1.0)),
                    Mesh2d(mesh.clone()),
                    MeshMaterial2d(material.clone()));

        commands.spawn((rectangular_boundary.0,
            Transform::default(),
            Visibility::Visible,
            children![left, right, bottom, top]));
    }

    /// Copy the current positions of simulation particles to bevy entities.
    pub fn sync(
        entity_rectangle: Single<(Entity, &Self)>,
        width: f32,
        height: f32,
    )
    {
    let (entity, rectangle) = *entity_rectangle;
    }
}

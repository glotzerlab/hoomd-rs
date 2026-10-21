// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `OrientedSphericalPoint`

use super::{Orientation, Point, Position};
use crate::Transform;
use hoomd_manifold::Spherical;
use hoomd_vector::Angle;
use rand::{
    Rng,
    distr::{Distribution, Uniform},
};
use robust::{Coord, orient2d};
use std::f64::consts::PI;

/// The position and orientation of an extended body in spherical space. The 
/// orientation is defined as the angle of the body in the stereographic 
/// projection through the "north pole" of the sphere.
///
/// Use [`OrientedSphericalPoint`] as a [`Body`](crate::Body) or
/// [`Site`](crate::Site) property type.
///
/// # Example
///
/// ```
/// use hoomd_manifold::Spherical;
/// use hoomd_microstate::property::OrientedSphericalPoint;
/// use hoomd_vector::Angle;
///
/// let point = OrientedSphericalPoint {
///     position: Spherical::<3>::from_polar_coordinates(0.1, 0.2),
///     orientation: Angle::from(0.3),
/// };
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OrientedSphericalPoint<const N: usize, R> {
    /// The location of the extended body in the system frame.
    pub position: Spherical<N>,
    /// Orientation of the body in the global frame.
    pub orientation: R,
}

impl OrientedSphericalPoint<3, Angle> {
    /// Compute the change in orientation associated with a transformation from a given
    /// position. Isometries of the 2-sphere can be expressed as a zenith rotation 
    /// conjugated by an azimuthal rotation. These $`SO(3)`$ operations can be mapped 
    /// to $`SU(2)`$ operations, where they act on the stereographic projected points 
    /// as the Möbius transformation
    /// ```math
    /// g(z) = \begin{bmatrix} \cos(\theta/2) & e^{i\phi} \sin(\theta/2) \\ -e^{-i\theta}\sin(\theta/2) & \cos(\theta/2) \end{bmatrix}
    /// ```
    /// This transformation induces a rotation of angle $`\Delta\psi`$ in the tangent
    /// bundle, where
    /// ```math
    /// \Delta \phi = \operatorname{Arg} [g'(z)] = -2 \operatorname{Arg}[\cos(\theta/2) - e^{-i\phi}\sin(\theta/2)z]
    /// ```
    #[inline]
    #[must_use]
    pub fn deck_transform(zenith: f64, azimuth: f64, position: &Spherical<3>) -> f64 {
        let theta_over_two = zenith/2.0;
        let stereo = position.stereographic_projection_from_north_pole();
        -2.0 * (((theta_over_two.sin())
            * ((azimuth.sin()) * stereo[0] - (azimuth.cos()) * stereo[1]))
            .atan2(
                (theta_over_two.cos())
                    - (theta_over_two.sin())
                        * ((azimuth.cos()) * stereo[0] + (azimuth.sin()) * stereo[1]),
            ))
    }
}

impl<const N: usize, R> Position for OrientedSphericalPoint<N, R> {
    type Position = Spherical<N>;

    #[inline]
    fn position(&self) -> &Spherical<N> {
        &self.position
    }

    #[inline]
    fn position_mut(&mut self) -> &mut Self::Position {
        &mut self.position
    }
}

impl<const N: usize, R> Orientation for OrientedSphericalPoint<N, R> {
    type Rotation = R;

    #[inline]
    fn orientation(&self) -> &R {
        &self.orientation
    }

    #[inline]
    fn orientation_mut(&mut self) -> &mut R {
        &mut self.orientation
    }
}

impl Transform<OrientedSphericalPoint<3, Angle>> for OrientedSphericalPoint<3, Angle> {
    #[inline]
    fn transform(&self, site_properties: &OrientedSphericalPoint<3, Angle>) -> OrientedSphericalPoint<3, Angle> {
        // TODO
        *self
    }
}
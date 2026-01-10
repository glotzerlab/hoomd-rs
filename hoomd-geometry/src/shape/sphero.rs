// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Sphero`]

use serde::{Deserialize, Serialize};

use crate::{BoundingSphereRadius, SupportMapping};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::InnerProduct;

/// Round a shape with a given radius.
///
/// [`Sphero`] modifies a given shape by sweeping it with a hypersphere of the
/// given radius. The resulting [`Sphero<S>`] type is a shape itself. If `S`
/// implements [`crate::SupportMapping`], then [`Sphero<S>`] can be used in
/// [`IntersectsAt`](crate::IntersectsAt) tests with other convex shapes. See
/// the full list of implementations below to see what other traits [`Sphero<S>`]
/// implements for a given `S`.
///
/// # Example
///
/// Test if a circle overlaps with a rounded rectangle:
/// ```
/// use hoomd_geometry::{
///     Convex, IntersectsAt,
///     shape::{Circle, Rectangle, Sphero},
/// };
/// use hoomd_vector::{Angle, Cartesian};
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let circle = Convex(Circle {
///     radius: 0.5.try_into()?,
/// });
/// let rectangle = Rectangle {
///     edge_lengths: [3.0.try_into()?, 2.0.try_into()?],
/// };
/// let rounded_rectangle = Convex(Sphero {
///     shape: rectangle,
///     rounding_radius: 0.5.try_into()?,
/// });
///
/// assert!(rounded_rectangle.intersects_at(
///     &circle,
///     &[2.4, 0.0].into(),
///     &Angle::default()
/// ));
/// assert!(!rounded_rectangle.intersects_at(
///     &circle,
///     &[0.0, 2.4].into(),
///     &Angle::default()
/// ));
/// assert!(circle.intersects_at(
///     &rounded_rectangle,
///     &[0.0, 2.4].into(),
///     &Angle::from(PI / 2.0)
/// ));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sphero<S> {
    /// The shape to round.
    pub shape: S,
    /// The radius of the rounding hypersphere.
    pub rounding_radius: PositiveReal,
}

impl<S, V> SupportMapping<V> for Sphero<S>
where
    S: SupportMapping<V>,
    V: InnerProduct,
{
    #[inline]
    fn support_mapping(&self, n: &V) -> V {
        self.shape.support_mapping(n) + *n / n.norm() * self.rounding_radius.get()
    }
}

impl<S> BoundingSphereRadius for Sphero<S>
where
    S: BoundingSphereRadius,
{
    #[inline]
    fn bounding_sphere_radius(&self) -> PositiveReal {
        (self.shape.bounding_sphere_radius().get() + self.rounding_radius.get())
            .try_into()
            .expect("expression should evaluate to a positive real")
    }
}

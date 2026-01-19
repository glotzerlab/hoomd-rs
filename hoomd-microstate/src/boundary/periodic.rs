// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement periodic boundary conditions.

use serde::{Deserialize, Serialize};
use std::fmt;

use hoomd_geometry::{MapPoint, Scale, Volume};
use hoomd_utility::valid::PositiveReal;
use rand::{Rng, distr::Distribution};

use super::{Error, MaximumAllowableInteractionRange};

mod cuboid;
mod eighteight;

/// Describe a simulation space that repeats in one or more directions.
///
/// [`Periodic`] is a newtype that wraps a shape. Use it to set the `boundary`
/// for a [`Microstate`].
///
/// When bodies or sites exit the shape through one of the periodic sides of the
/// shape, they are wrapped to the other side. Similarly, sites that are within
/// the interaction range of one of the periodic sides appear as ghost sites just
/// outside the opposite side. Depending on the shape type `T`, `Periodic<T>` might
/// implement fully periodic boundaries or ones that are periodic in some directions
/// and closed in others.
///
/// [`Periodic`] is implemented for the following shapes:
/// * [`EightEight`]
/// * [`Hypercuboid<2>`] (also known as [`Rectangle`])
/// * [`Hypercuboid<3>`] (also known as [`Cuboid`])
///
/// [`EightEight`]: hoomd_geometry::shape::EightEight
/// [`Hypercuboid<2>`]: hoomd_geometry::shape::Hypercuboid
/// [`Hypercuboid<3>`]: hoomd_geometry::shape::Hypercuboid
/// [`Cuboid`]: hoomd_geometry::shape::Cuboid
/// [`Rectangle`]: hoomd_geometry::shape::Rectangle
/// [`Microstate`]: crate::Microstate
///
/// # Example
///
/// ```
/// use hoomd_geometry::shape::Rectangle;
/// use hoomd_microstate::boundary::Periodic;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let periodic =
///     Periodic::new(2.5, Rectangle::with_equal_edges(10.0.try_into()?))?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Periodic<T> {
    /// The largest interaction distance between two sites.
    maximum_interaction_range: f64,

    /// Bound the points that belong to the primary image.
    shape: T,
}

impl<T> Periodic<T>
where
    T: MaximumAllowableInteractionRange,
{
    /// Construct a new periodic boundary condition.
    ///
    /// # Errors
    ///
    /// [`Error::InteractionRangeTooLarge`] when `maximum_interaction_range` is
    /// larger than the maximum allowable interaction range by the shape.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_microstate::boundary::Periodic;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let periodic =
    ///     Periodic::new(2.5, Rectangle::with_equal_edges(10.0.try_into()?))?;
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn new(maximum_interaction_range: f64, shape: T) -> Result<Self, Error> {
        if maximum_interaction_range > shape.maximum_allowable_interaction_range() {
            return Err(Error::InteractionRangeTooLarge(
                maximum_interaction_range,
                shape.maximum_allowable_interaction_range(),
            ));
        }

        Ok(Self {
            maximum_interaction_range,
            shape,
        })
    }
}

impl<T> Periodic<T> {
    /// Access the boundary's shape.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_microstate::boundary::Periodic;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let periodic =
    ///     Periodic::new(2.5, Rectangle::with_equal_edges(10.0.try_into()?))?;
    ///
    /// assert_eq!(periodic.shape().edge_lengths[0].get(), 10.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn shape(&self) -> &T {
        &self.shape
    }

    /// Access the boundary's maximum interaction range.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_microstate::boundary::Periodic;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let periodic =
    ///     Periodic::new(2.5, Rectangle::with_equal_edges(10.0.try_into()?))?;
    ///
    /// assert_eq!(periodic.maximum_interaction_range(), 2.5);
    /// # Ok(())
    /// # }
    /// ```
    #[expect(
        clippy::same_name_method,
        reason = "MaximumInteractionRange is a trait in hoomd-interaction"
    )]
    #[inline]
    pub fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }
}

impl<T, V> Distribution<V> for Periodic<T>
where
    T: Distribution<V>,
{
    /// Generate points uniformly distributed in the wrapped shape.
    ///
    /// # Example
    ///
    /// ```
    /// use rand::{SeedableRng, distr::Distribution, rngs::StdRng};
    ///
    /// use hoomd_geometry::{IsPointInside, shape::Hypercuboid};
    /// use hoomd_microstate::boundary::Periodic;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let cuboid = Hypercuboid {
    ///     edge_lengths: [6.0.try_into()?, 8.0.try_into()?],
    /// };
    /// let periodic = Periodic::new(2.5, cuboid)?;
    /// let mut rng = StdRng::seed_from_u64(1);
    ///
    /// let point = periodic.sample(&mut rng);
    /// assert!(periodic.shape().is_point_inside(&point));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> V {
        self.shape.sample(rng)
    }
}

impl<T> Scale for Periodic<T>
where
    T: fmt::Debug + Scale + MaximumAllowableInteractionRange,
{
    /// Scale the wrapped shape.
    ///
    /// # Panics
    ///
    /// When scaling the wrapped shape, `scale_length` will panic if
    /// the scaled maximum allowable interaction range is smaller than
    /// `maximum_interaction_range`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::Rectangle};
    /// use hoomd_microstate::boundary::Periodic;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let periodic =
    ///     Periodic::new(2.5, Rectangle::with_equal_edges(10.0.try_into()?))?;
    ///
    /// let scaled_periodic = periodic.scale_length(0.5.try_into()?);
    ///
    /// assert_eq!(scaled_periodic.maximum_interaction_range(), 2.5);
    /// assert_eq!(scaled_periodic.shape().edge_lengths[0].get(), 5.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ```should_panic
    /// use hoomd_geometry::{Scale, shape::Rectangle};
    /// use hoomd_microstate::boundary::Periodic;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let periodic =
    ///     Periodic::new(2.5, Rectangle::with_equal_edges(10.0.try_into()?))?;
    ///
    /// let scaled_periodic = periodic.scale_length(0.2.try_into()?);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_length(&self, v: PositiveReal) -> Self {
        let new_shape = self.shape.scale_length(v);
        assert!(
            new_shape.maximum_allowable_interaction_range() >= self.maximum_interaction_range,
            "The scaled periodic boundary {new_shape:?} is too small for the maximum interaction range {}",
            self.maximum_interaction_range
        );

        Self {
            shape: new_shape,
            ..*self
        }
    }

    /// Scale the wrapped shape.
    ///
    /// # Panics
    ///
    /// When scaling the wrapped shape, `scale_length` will panic if
    /// the scaled maximum allowable interaction range is smaller than
    /// `maximum_interaction_range`.
    ///
    /// # Examples
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::Rectangle};
    /// use hoomd_microstate::boundary::Periodic;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let periodic =
    ///     Periodic::new(2.5, Rectangle::with_equal_edges(10.0.try_into()?))?;
    ///
    /// let scaled_periodic = periodic.scale_volume(4.0.try_into()?);
    ///
    /// assert_eq!(scaled_periodic.maximum_interaction_range(), 2.5);
    /// assert_eq!(scaled_periodic.shape().edge_lengths[0].get(), 20.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ```should_panic
    /// use hoomd_geometry::{Scale, shape::Rectangle};
    /// use hoomd_microstate::boundary::Periodic;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let periodic =
    ///     Periodic::new(2.5, Rectangle::with_equal_edges(10.0.try_into()?))?;
    ///
    /// let scaled_periodic = periodic.scale_volume(0.2.try_into()?);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_volume(&self, v: PositiveReal) -> Self {
        let new_shape = self.shape.scale_volume(v);
        assert!(
            new_shape.maximum_allowable_interaction_range() >= self.maximum_interaction_range,
            "The scaled periodic boundary {new_shape:?} is too small for the maximum interaction range {}",
            self.maximum_interaction_range
        );

        Self {
            shape: new_shape,
            ..*self
        }
    }
}

impl<P, T> MapPoint<P> for Periodic<T>
where
    T: MapPoint<P>,
{
    /// Map points in from the wrapped shape into another periodic boundary.
    ///
    /// # Errors
    ///
    /// [`hoomd_geometry::Error::PointOutsideShape`] when `point` is outside
    /// `self.shape()`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{MapPoint, shape::Rectangle};
    /// use hoomd_microstate::boundary::Periodic;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let periodic_a =
    ///     Periodic::new(2.5, Rectangle::with_equal_edges(10.0.try_into()?))?;
    /// let periodic_b =
    ///     Periodic::new(2.5, Rectangle::with_equal_edges(20.0.try_into()?))?;
    ///
    /// let mapped_point =
    ///     periodic_a.map_point(Cartesian::from([-1.0, 1.0]), &periodic_b);
    ///
    /// assert_eq!(mapped_point, Ok(Cartesian::from([-2.0, 2.0])));
    /// assert_eq!(
    ///     periodic_a.map_point(Cartesian::from([-100.0, 1.0]), &periodic_b),
    ///     Err(hoomd_geometry::Error::PointOutsideShape)
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn map_point(&self, point: P, other: &Self) -> Result<P, hoomd_geometry::Error> {
        self.shape.map_point(point, &other.shape)
    }
}

impl<T> Volume for Periodic<T>
where
    T: Volume,
{
    /// Volume of the wrapped shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use hoomd_geometry::{Volume, shape::Rectangle};
    /// use hoomd_microstate::boundary::Periodic;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let periodic =
    ///     Periodic::new(2.5, Rectangle::with_equal_edges(10.0.try_into()?))?;
    ///
    /// let volume = periodic.volume();
    ///
    /// assert_eq!(volume, 100.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn volume(&self) -> f64 {
        self.shape.volume()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use hoomd_geometry::shape::Rectangle;

    #[test]
    fn interaction_range_validation() {
        let rectangle = Rectangle {
            edge_lengths: [
                10.0.try_into()
                    .expect("hard-coded constant should be positive"),
                6.0.try_into()
                    .expect("hard-coded constant should be positive"),
            ],
        };

        let result = Periodic::new(1.0, rectangle.clone());
        assert!(result.is_ok());

        let result = Periodic::new(3.0, rectangle.clone());
        assert!(result.is_ok());

        let result = Periodic::new(3.0f64.next_up(), rectangle);
        assert!(matches!(result, Err(Error::InteractionRangeTooLarge(_, _))));
    }
}

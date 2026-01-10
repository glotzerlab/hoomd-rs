// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Closed

use arrayvec::ArrayVec;
use rand::{Rng, distr::Distribution};
use serde::{Deserialize, Serialize};

use super::{Error, GenerateGhosts, MAX_GHOSTS, Wrap};
use crate::property::Position;
use hoomd_geometry::{IsPointInside, MapPoint, Scale, Volume};
use hoomd_utility::valid::PositiveReal;

/// Restrict points to the inside of a shape.
///
/// [`Closed`] is a newtype that wraps a shape. It prevents bodies and sites from
/// existing outside the shape. Bodies and sites are never wrapped, and there are no
/// ghost sites.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Closed<T>(pub T);

impl<BS, T, P> Wrap<BS> for Closed<T>
where
    BS: Position<Position = P>,
    T: IsPointInside<P>,
{
    #[inline]
    fn wrap(&self, properties: BS) -> Result<BS, Error> {
        if self.0.is_point_inside(properties.position()) {
            Ok(properties)
        } else {
            Err(Error::CannotWrapProperties)
        }
    }
}

impl<S, T> GenerateGhosts<S> for Closed<T>
where
    S: Default,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        f64::INFINITY
    }

    #[inline]
    fn generate_ghosts(&self, _site_properties: &S) -> ArrayVec<S, MAX_GHOSTS> {
        ArrayVec::new()
    }
}

impl<T, P> Distribution<P> for Closed<T>
where
    T: Distribution<P>,
{
    /// Generate points uniformly distributed in the wrapped shape.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{IsPointInside, shape::Sphere};
    /// use hoomd_microstate::boundary::Closed;
    /// use rand::{SeedableRng, distr::Distribution, rngs::StdRng};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let sphere = Closed(Sphere {
    ///     radius: 5.0.try_into()?,
    /// });
    /// let mut rng = StdRng::seed_from_u64(1);
    ///
    /// let point = sphere.sample(&mut rng);
    /// assert!(sphere.0.is_point_inside(&point));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> P {
        self.0.sample(rng)
    }
}

impl<T> Scale for Closed<T>
where
    T: Scale,
{
    /// Scale the wrapped shape.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::Sphere};
    /// use hoomd_microstate::boundary::Closed;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let sphere = Closed(Sphere {
    ///     radius: 5.0.try_into()?,
    /// });
    ///
    /// let scaled_sphere = sphere.scale_length(0.5.try_into()?);
    ///
    /// assert_eq!(scaled_sphere.0.radius.get(), 2.5);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_length(&self, v: PositiveReal) -> Self {
        Self(self.0.scale_length(v))
    }

    /// Scale the wrapped shape.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::Rectangle};
    /// use hoomd_microstate::boundary::Closed;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let closed = Closed(Rectangle::with_equal_edges(10.0.try_into()?));
    ///
    /// let scaled_closed = closed.scale_volume(4.0.try_into()?);
    ///
    /// assert_eq!(scaled_closed.0.edge_lengths[0].get(), 20.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_volume(&self, v: PositiveReal) -> Self {
        Self(self.0.scale_volume(v))
    }
}

impl<P, T> MapPoint<P> for Closed<T>
where
    T: MapPoint<P>,
{
    /// Map a point in the wrapped shape to another.
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
    /// use hoomd_microstate::boundary::Closed;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let closed_a = Closed(Rectangle::with_equal_edges(10.0.try_into()?));
    /// let closed_b = Closed(Rectangle::with_equal_edges(20.0.try_into()?));
    ///
    /// let mapped_point =
    ///     closed_a.map_point(Cartesian::from([-1.0, 1.0]), &closed_b);
    ///
    /// assert_eq!(mapped_point, Ok(Cartesian::from([-2.0, 2.0])));
    /// assert_eq!(
    ///     closed_a.map_point(Cartesian::from([-100.0, 1.0]), &closed_b),
    ///     Err(hoomd_geometry::Error::PointOutsideShape)
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn map_point(&self, point: P, other: &Self) -> Result<P, hoomd_geometry::Error> {
        self.0.map_point(point, &other.0)
    }
}

impl<T> Volume for Closed<T>
where
    T: Volume,
{
    /// Volume of the wrapped shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use hoomd_geometry::{Volume, shape::Rectangle};
    /// use hoomd_microstate::boundary::Closed;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let closed = Closed(Rectangle::with_equal_edges(10.0.try_into()?));
    ///
    /// let volume = closed.volume();
    ///
    /// assert_eq!(volume, 100.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn volume(&self) -> f64 {
        self.0.volume()
    }
}

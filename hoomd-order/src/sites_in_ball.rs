// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `SitesInBall`

use hoomd_microstate::{Microstate, Site, SiteKey, property::Position};
use hoomd_spatial::PointsNearBall;
use hoomd_vector::Metric;

/// Find all sites in a [`Microstate`] with positions inside a ball of radius *r*.
///
/// In other words, find all neighboring sites within a specific distance of a given
/// point.
///
/// When the given [`Microstate`] has periodic boundary conditions, *r* must
/// be less than or equal to the boundary's maximum interaction range. Exercise caution,
/// as this condition is not validated. Set *r* too large and [`SitesInBall`] will
/// find the correct sites near the center of the boundary, but will miss some neighbors
/// near the edge. TODO: revisit *why* this is not an error - it seems that Microstate could
/// check this and panic....
///
/// Construct [`SitesInBall`] with [`near_site`] to find the neighbors of a site
/// (excluding the site itself). Construct [`SitesInBall`] with [`near_point`] to find
/// all sites (no exclusions) within a distance of the given point.
///
/// After construction, call one or more `iter_` methods to iterate over the
/// matching sites. When the [`Microstate`] has periodic boundary conditions,
/// matches may be ghost sites with positions outside the boundary: no wrapping
/// is necessary as [`Microstate`] has already accounted for it.
///
/// # Examples
///
/// For example, use [`iter_site_positions`] to compute an order parameter
/// of all sites near another site:
/// ```
/// use hoomd_microstate::{Body, Microstate};
/// use hoomd_order::SitesInBall;
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::new();
/// microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])))?;
/// microstate.add_body(Body::point(Cartesian::from([1.0, 0.0])))?;
/// microstate.add_body(Body::point(Cartesian::from([0.0, 1.0])))?;
/// microstate.add_body(Body::point(Cartesian::from([-1.0, 0.0])))?;
/// microstate.add_body(Body::point(Cartesian::from([0.0, -1.0])))?;
///
/// let near_site_0 = SitesInBall::near_site(&microstate, 0, 1.5);
///
/// let psi_0 = hoomd_order::k_atic_psi(4.0, near_site_0.point(), near_site_0.iter_site_positions().copied());
/// # Ok(())
/// # }
/// ```
/// 
/// Use [`iter_sites`] to perform additional filtering. For example, compute an order parameter
/// only on neighboring sites of type *A*:
/// ```
/// use hoomd_microstate::{Body, Microstate, Transform, property::{Point, Position}};
/// use hoomd_order::SitesInBall;
/// use hoomd_vector::Cartesian;
///
/// type PositionVector = Cartesian<2>;
/// type BodyProperties = Point<PositionVector>;
///
/// #[derive(Clone, Copy, Default, PartialEq)]
/// enum SiteType {
///     #[default]
///     A,
///     B,
/// }
///
/// #[derive(Clone, Copy, Default, Position)]
/// struct SiteProperties {
///     /// The site's position.
///     position: PositionVector,
///     /// The site's type.
///     site_type: SiteType,
/// }
///
/// impl Transform<SiteProperties> for BodyProperties {
///     fn transform(&self, site_properties: &SiteProperties) -> SiteProperties {
///         SiteProperties {
///             position: self.position + site_properties.position,
///             ..*site_properties
///         }
///     }
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::new();
/// microstate.add_body(Body::single_site(Point::new(Cartesian::from([0.0, 0.0])), SiteProperties::default()))?;
/// microstate.add_body(Body::single_site(Point::new(Cartesian::from([1.0, 0.0])), SiteProperties::default()))?;
/// microstate.add_body(Body::single_site(Point::new(Cartesian::from([0.0, 1.0])), SiteProperties::default()))?;
/// microstate.add_body(Body::single_site(Point::new(Cartesian::from([-1.0, 0.0])), SiteProperties::default()))?;
/// microstate.add_body(Body::single_site(Point::new(Cartesian::from([0.0, -1.0])), SiteProperties::default()))?;
/// microstate.add_body(Body::single_site(Point::new(Cartesian::from([1.0, 1.0])), SiteProperties { site_type: SiteType::A, ..Default::default() }))?;
/// microstate.add_body(Body::single_site(Point::new(Cartesian::from([-1.0, -1.0])), SiteProperties { site_type: SiteType::A, ..Default::default() }))?;
///
/// let near_site_0 = SitesInBall::near_site(&microstate, 0, 1.5);
///
/// let psi_0_a = hoomd_order::k_atic_psi(4.0, near_site_0.point(), near_site_0.iter_sites()
///     .filter(|s| s.properties.site_type == SiteType::A)
///     .map(|s| *s.properties.position()));
/// # Ok(())
/// # }
/// ```
///
/// [`near_site`]: Self::near_site
/// [`near_point`]: Self::near_point
/// [`iter_site_positions`]: Self::iter_site_positions
/// [`iter_sites`]: Self::iter_sites
pub struct SitesInBall<'a, P, B, S, X, C> {
    /// The center of the ball.
    point: P,
    /// Radius of the ball.
    r: f64,
    /// Ignore sites with this tag.
    ignore_site_tag: Option<usize>,
    /// Search for sites in this microstate.
    microstate: &'a Microstate<B, S, X, C>,
}

impl<P, B, S, X, C> SitesInBall<'_, P, B, S, X, C> {
    /// The point at the center of the ball.
    ///
    /// # Example
    /// ```
    /// use hoomd_microstate::{Body, Microstate};
    /// use hoomd_order::SitesInBall;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.add_body(Body::point(Cartesian::from([1.0, 2.0])))?;
    ///
    /// let near_site_0 = SitesInBall::near_site(&microstate, 0, 1.5);
    /// assert_eq!(near_site_0.point(), &[1.0, 2.0].into());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn point(&self) -> &P {
        &self.point
    }

    /// The radius of the ball.
    ///
    /// # Example
    /// ```
    /// use hoomd_microstate::{Body, Microstate};
    /// use hoomd_order::SitesInBall;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.add_body(Body::point(Cartesian::from([1.0, 2.0])))?;
    ///
    /// let near_site_0 = SitesInBall::near_site(&microstate, 0, 1.5);
    /// assert_eq!(near_site_0.r(), 1.5);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn r(&self) -> f64 {
        self.r
    }
}

impl<'a, P, B, S, X, C> SitesInBall<'a, P, B, S, X, C> where
    P: Copy + Metric,
    S: Position<Position = P>,
    X: PointsNearBall<P, SiteKey>,
{
    /// Match sites in the given microstate with positions that are within a distance
    /// *r* of the given site. The given site is *excluded* from the iterator despite the
    /// fact that it a distance of zero from itself.
    ///
    /// # Example
    /// ```
    /// use hoomd_microstate::{Body, Microstate};
    /// use hoomd_order::SitesInBall;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])))?;
    /// microstate.add_body(Body::point(Cartesian::from([1.0, 0.0])))?;
    /// microstate.add_body(Body::point(Cartesian::from([2.0, 0.0])))?;
    ///
    /// let near_site_1 = SitesInBall::near_site(&microstate, 1, 1.5);
    /// assert_eq!(near_site_1.iter_sites().count(), 2);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn near_site(microstate: &'a Microstate<B, S, X, C>, site_index: usize, r: f64) -> Self {
        Self {
            point: *microstate.sites()[site_index].properties.position(),
            r,
            ignore_site_tag: Some(microstate.sites()[site_index].site_tag),
            microstate
        }
    }

    /// Match sites in the given microstate with positions that are within a distance
    /// *r* of the given point in space, including sites at a distance of zero.
    ///
    /// # Example
    /// ```
    /// use hoomd_microstate::{Body, Microstate};
    /// use hoomd_order::SitesInBall;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])))?;
    /// microstate.add_body(Body::point(Cartesian::from([1.0, 0.0])))?;
    /// microstate.add_body(Body::point(Cartesian::from([2.0, 0.0])))?;
    ///
    /// let near_position_1_0 = SitesInBall::near_point(&microstate, [1.0, 0.0].into(), 1.5);
    /// assert_eq!(near_position_1_0.iter_sites().count(), 3);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// TODO: What should happen when `point` is outside boundary? Should we construct
    /// a temporary point site and wrap it? That would make `near_point` fallible, but perhaps
    /// that is a good thing.
    #[inline]
    pub fn near_point(microstate: &'a Microstate<B, S, X, C>, point: P, r: f64) -> Self {
        Self {
            point,
            r,
            ignore_site_tag: None,
            microstate
        }
    }

    /// Iterate over all matching sites.
    ///
    /// # Examples
    /// ```
    /// use hoomd_microstate::{Body, Microstate, Transform, property::{Point, Position}};
    /// use hoomd_order::SitesInBall;
    /// use hoomd_vector::Cartesian;
    ///
    /// type PositionVector = Cartesian<2>;
    /// type BodyProperties = Point<PositionVector>;
    ///
    /// #[derive(Clone, Copy, Default, PartialEq)]
    /// enum SiteType {
    ///     #[default]
    ///     A,
    ///     B,
    /// }
    ///
    /// #[derive(Clone, Copy, Default, Position)]
    /// struct SiteProperties {
    ///     /// The site's position.
    ///     position: PositionVector,
    ///     /// The site's type.
    ///     site_type: SiteType,
    /// }
    ///
    /// impl Transform<SiteProperties> for BodyProperties {
    ///     fn transform(&self, site_properties: &SiteProperties) -> SiteProperties {
    ///         SiteProperties {
    ///             position: self.position + site_properties.position,
    ///             ..*site_properties
    ///         }
    ///     }
    /// }
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.add_body(Body::single_site(Point::new(Cartesian::from([0.0, 0.0])), SiteProperties::default()))?;
    /// microstate.add_body(Body::single_site(Point::new(Cartesian::from([1.0, 0.0])), SiteProperties::default()))?;
    /// microstate.add_body(Body::single_site(Point::new(Cartesian::from([0.0, 1.0])), SiteProperties::default()))?;
    /// microstate.add_body(Body::single_site(Point::new(Cartesian::from([-1.0, 0.0])), SiteProperties::default()))?;
    /// microstate.add_body(Body::single_site(Point::new(Cartesian::from([0.0, -1.0])), SiteProperties::default()))?;
    /// microstate.add_body(Body::single_site(Point::new(Cartesian::from([1.0, 1.0])), SiteProperties { site_type: SiteType::A, ..Default::default() }))?;
    /// microstate.add_body(Body::single_site(Point::new(Cartesian::from([-1.0, -1.0])), SiteProperties { site_type: SiteType::A, ..Default::default() }))?;
    ///
    /// let near_site_0 = SitesInBall::near_site(&microstate, 0, 1.5);
    ///
    /// let psi_0_a = hoomd_order::k_atic_psi(4.0, near_site_0.point(), near_site_0.iter_sites()
    ///     .filter(|s| s.properties.site_type == SiteType::A)
    ///     .map(|s| *s.properties.position()));
    /// # Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn iter_sites(&self) -> impl Iterator<Item = &Site<S>> {
        self.microstate.iter_sites_near(&self.point, self.r)
            .filter(|s| {
                match self.ignore_site_tag {
                    None => true,
                    Some(site_tag) => site_tag != s.site_tag,
            }})
            .filter(|s| self.point.distance_squared(s.properties.position()) < self.r.powi(2) )
    }

    /// Iterate over all matching site positions.
    ///
    /// # Example
    /// ```
    /// use hoomd_microstate::{Body, Microstate};
    /// use hoomd_order::SitesInBall;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])))?;
    /// microstate.add_body(Body::point(Cartesian::from([1.0, 0.0])))?;
    /// microstate.add_body(Body::point(Cartesian::from([0.0, 1.0])))?;
    /// microstate.add_body(Body::point(Cartesian::from([-1.0, 0.0])))?;
    /// microstate.add_body(Body::point(Cartesian::from([0.0, -1.0])))?;
    ///
    /// let near_site_0 = SitesInBall::near_site(&microstate, 0, 1.5);
    ///
    /// let psi_0 = hoomd_order::k_atic_psi(4.0, near_site_0.point(), near_site_0.iter_site_positions().copied());
    /// # Ok(())
    /// # }
    /// ```
    #[inline(always)]
    pub fn iter_site_positions(&self) -> impl Iterator<Item = &P> {
        self.iter_sites().map(|s| s.properties.position())
    }

    /// Iterate over all matching site tags.
    #[inline(always)]
    pub fn iter_site_tags(&self) -> impl Iterator<Item = usize> {
        self.iter_sites().map(|s| s.site_tag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_microstate::Body;
    use hoomd_vector::Cartesian;

    #[test]
    fn near_site() -> anyhow::Result<()> {
        let mut microstate = Microstate::new();
        microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])))?;
        microstate.add_body(Body::point(Cartesian::from([1.0, 0.0])))?;
        microstate.add_body(Body::point(Cartesian::from([2.0, 0.0])))?;
        microstate.add_body(Body::point(Cartesian::from([1.0, 2.0])))?;
        microstate.add_body(Body::point(Cartesian::from([1.0, -2.0])))?;
        microstate.add_body(Body::point(Cartesian::from([1.0, 0.0])))?;

        let near_site_1 = SitesInBall::near_site(&microstate, 1, 1.5);
        let sites = near_site_1.iter_sites().collect::<Vec<_>>();
        let mut sorted_sites = sites.clone();
        sorted_sites.sort_by_key(|a| a.site_tag);

        itertools::assert_equal(sorted_sites.iter().map(|s| s.site_tag), [0, 2, 5]);
        itertools::assert_equal(near_site_1.iter_site_tags(), sites.iter().map(|s| s.site_tag));
        itertools::assert_equal(near_site_1.iter_site_positions().copied(), sites.iter().map(|s| s.properties.position));
        
        Ok(())        
    }
    #[test]
    fn near_point() -> anyhow::Result<()> {
        let mut microstate = Microstate::new();
        microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])))?;
        microstate.add_body(Body::point(Cartesian::from([1.0, 0.0])))?;
        microstate.add_body(Body::point(Cartesian::from([2.0, 0.0])))?;
        microstate.add_body(Body::point(Cartesian::from([1.0, 2.0])))?;
        microstate.add_body(Body::point(Cartesian::from([1.0, -2.0])))?;
        microstate.add_body(Body::point(Cartesian::from([1.0, 0.0])))?;

        let near_point = SitesInBall::near_point(&microstate, [1.0, 0.0].into(), 1.5);
        let sites = near_point.iter_sites().collect::<Vec<_>>();
        let mut sorted_sites = sites.clone();
        sorted_sites.sort_by_key(|a| a.site_tag);

        itertools::assert_equal(sorted_sites.iter().map(|s| s.site_tag), [0, 1, 2, 5]);
        itertools::assert_equal(near_point.iter_site_tags(), sites.iter().map(|s| s.site_tag));
        itertools::assert_equal(near_point.iter_site_positions().copied(), sites.iter().map(|s| s.properties.position));
        
        Ok(())        
    }
}

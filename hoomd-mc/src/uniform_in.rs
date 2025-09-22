// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `UniformIn`

use hoomd_microstate::{
    Body,
    property::{OrientedPoint, Point},
};

use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};

/// Generate bodies uniformly in the given boundary condition.
///
/// Give [`UniformIn`] a template vector of sites and it will randomly generate
/// bodies uniformly distributed in the given boundary. Each generated body will
/// have the same sites (cloned from `template_sites`) and random body properties
/// sampled in the given `boundary`.
///
/// # Example
///
/// Place points at random locations in the boundary:
/// ```
/// use hoomd_geometry::{IsPointInside, shape::Rectangle};
/// use hoomd_mc::UniformIn;
/// use hoomd_microstate::{Body, boundary::Closed, property::Point};
/// use hoomd_vector::Cartesian;
///
/// use rand::{SeedableRng, distr::Distribution, rngs::StdRng};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let rectangle = Closed(Rectangle::with_equal_edges(5.0.try_into()?));
/// let mut rng = StdRng::seed_from_u64(1);
///
/// let uniform_in = UniformIn {
///     boundary: rectangle,
///     template_sites: vec![Point::new(Cartesian::from([0.0, 0.0]))],
/// };
///
/// let body: Body<Point<Cartesian<2>>, Point<Cartesian<2>>> =
///     uniform_in.sample(&mut rng);
/// assert!(
///     uniform_in
///         .boundary
///         .0
///         .is_point_inside(&body.properties.position)
/// );
/// # Ok(())
/// # }
/// ```
///
/// Place oriented bodies at random locations in the boundary and give them random
/// orientations:
/// ```
/// use rand::{SeedableRng, distr::Distribution, rngs::StdRng};
/// use std::f64::consts::PI;
///
/// use hoomd_geometry::{IsPointInside, shape::Rectangle};
/// use hoomd_mc::UniformIn;
/// use hoomd_microstate::{
///     Body,
///     boundary::Closed,
///     property::{OrientedPoint, Point},
/// };
/// use hoomd_vector::{Angle, Cartesian};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let rectangle = Closed(Rectangle::with_equal_edges(5.0.try_into()?));
/// let mut rng = StdRng::seed_from_u64(1);
///
/// let uniform_in = UniformIn {
///     boundary: rectangle,
///     template_sites: vec![
///         Point::new(Cartesian::from([-1.0, 0.0])),
///         Point::new(Cartesian::from([1.0, 0.0])),
///     ],
/// };
///
/// let body: Body<OrientedPoint<Cartesian<2>, Angle>, Point<Cartesian<2>>> =
///     uniform_in.sample(&mut rng);
/// assert!(
///     uniform_in
///         .boundary
///         .0
///         .is_point_inside(&body.properties.position)
/// );
/// assert!(body.properties.orientation.theta < 2.0 * PI);
/// # Ok(())
/// # }
/// ```
pub struct UniformIn<S, C> {
    /// Generate bodies inside this boundary.
    pub boundary: C,

    /// Give each generated body these sites.
    pub template_sites: Vec<S>,
}

/// Randomly place point bodies in a given boundary.
///
/// `sample` chooses the *body's* position randomly in the given boundary. Sites,
/// therefore, may be placed outside the boundary. Callers should reject insertions
/// appropriately when `add_body` fails.
impl<V, S, C> Distribution<Body<Point<V>, S>> for UniformIn<S, C>
where
    S: Clone,
    C: Distribution<V>,
{
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Body<Point<V>, S> {
        let properties = Point {
            position: self.boundary.sample(rng),
        };
        let sites = self.template_sites.clone();
        Body { properties, sites }
    }
}

/// Randomly place oriented bodies in a given boundary.
///
/// `sample` chooses the *body's* position randomly in the given boundary and also
/// assigns a *uniform random orientation*. Sites, therefore, may be placed outside
/// the boundary. Callers should reject insertions appropriately when `add_body`
/// fails.
impl<V, O, S, C> Distribution<Body<OrientedPoint<V, O>, S>> for UniformIn<S, C>
where
    S: Clone,
    C: Distribution<V>,
    StandardUniform: Distribution<O>,
{
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Body<OrientedPoint<V, O>, S> {
        let properties = OrientedPoint {
            position: self.boundary.sample(rng),
            orientation: rng.random(),
        };
        let sites = self.template_sites.clone();
        Body { properties, sites }
    }
}

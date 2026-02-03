// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `HardShape`

use serde::{Deserialize, Serialize};

use crate::{MaximumInteractionRange, SitePairEnergy};
use hoomd_geometry::{BoundingSphereRadius, IntersectsAt, hyperbolic_overlap::SeparatingPlanes};
use hoomd_manifold::Hyperbolic;
use hoomd_microstate::property::{Orientation, Position};
use hoomd_vector::{self, Angle, Cartesian, Metric, Rotate, Rotation};

/// Infinite energy when sites overlap, 0 when they don't (*not differentiable*).
///
/// [`HardShape`] represents each site with a hard orientable shape.
///
/// The generic type names are:
/// * `G`: The [`shape`](hoomd_geometry::shape) type.
///
/// # Example
///
/// ```
/// use hoomd_geometry::{Convex, shape::Rectangle};
/// use hoomd_interaction::pairwise::HardShape;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let square = Rectangle::with_equal_edges(1.0.try_into()?);
/// let hard_shape = HardShape(Convex(square));
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HardShape<G>(pub G);

impl<S, G, const N: usize, R> SitePairEnergy<S, Cartesian<N>> for HardShape<G>
where
    S: Position<Position = Cartesian<N>> + Orientation<Rotation = R>,
    R: Rotation + Rotate<Cartesian<N>>,
    G: IntersectsAt<G, Cartesian<N>, R> + BoundingSphereRadius,
{
    /// Compute the energy contribution from a pair of sites.
    ///
    /// A pair of hard shapes contributes an infinite energy when they overlap,
    /// and zero when they do not.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{Convex, shape::Rectangle};
    /// use hoomd_interaction::{SitePairEnergy, pairwise::HardShape};
    /// use hoomd_microstate::property::OrientedPoint;
    /// use hoomd_vector::{Angle, Cartesian};
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let square = Rectangle::with_equal_edges(1.0.try_into()?);
    /// let hard_shape = HardShape(Convex(square));
    ///
    /// let a = OrientedPoint {
    ///     position: Cartesian::from([1.0, -1.0]),
    ///     orientation: Angle::from(PI / 2.0),
    /// };
    /// let b = OrientedPoint {
    ///     position: Cartesian::from([2.0, 0.0]),
    ///     orientation: Angle::from(PI / 4.0),
    /// };
    ///
    /// assert_eq!(hard_shape.site_pair_energy(&a, &b), 0.0);
    ///
    /// let c = OrientedPoint {
    ///     position: Cartesian::from([1.5, -0.5]),
    ///     orientation: Angle::from(PI / 4.0),
    /// };
    ///
    /// assert_eq!(hard_shape.site_pair_energy(&a, &c), f64::INFINITY);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn site_pair_energy(&self, site_properties_i: &S, site_properties_j: &S) -> f64 {
        // Use the global form of `intersects_at` so that the circumsphere early
        // rejection test can be performed before the expensive transformation
        // into the local coordinate system.
        if self.0.intersects_at_global(
            &self.0,
            site_properties_i.position(),
            site_properties_i.orientation(),
            site_properties_j.position(),
            site_properties_j.orientation(),
        ) {
            f64::INFINITY
        } else {
            0.0
        }
    }

    /// Evaluate the energy contribution from a pair of sites *in the initial state*.
    ///
    /// Hard shapes are assumed to be non-overlapping in the initial state.
    /// This method always returns zero.
    #[inline]
    fn site_pair_energy_initial(&self, _site_properties_i: &S, _site_properties_j: &S) -> f64 {
        0.0
    }

    #[inline]
    fn is_only_infinite_or_zero() -> bool {
        true
    }
}

impl<G> MaximumInteractionRange for HardShape<G>
where
    G: BoundingSphereRadius,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.0.bounding_sphere_radius().get() * 2.0
    }
}

/// Model infinitely hard spheres when used with [`PairwiseCutoff`] (*not differentiable*).
///
/// [`HardSphere`] represents each site as a hard sphere with a diameter given
/// by the `diameter` field.
///
/// [`PairwiseCutoff`]: crate::PairwiseCutoff
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HardSphere {
    /// The sphere's diameter.
    pub diameter: f64,
}

impl<S, P> SitePairEnergy<S, P> for HardSphere
where
    S: Position<Position = P>,
    P: Metric,
{
    /// Compute the energy contribution from a pair of sites.
    ///
    /// The interaction energy is infinite when two spheres overlap and zero
    /// when they do not.
    #[inline]
    fn site_pair_energy(&self, site_properties_i: &S, site_properties_j: &S) -> f64 {
        if site_properties_i
            .position()
            .distance_squared(site_properties_j.position())
            < self.diameter.powi(2)
        {
            f64::INFINITY
        } else {
            0.0
        }
    }

    /// Evaluate the energy contribution from a pair of sites *in the initial state*.
    ///
    /// Hard shapes are assumed to be non-overlapping in the initial state.
    /// This implementation always returns zero.
    #[inline]
    fn site_pair_energy_initial(&self, _site_properties_i: &S, _site_properties_j: &S) -> f64 {
        0.0
    }

    #[inline]
    fn is_only_infinite_or_zero() -> bool {
        true
    }
}

impl MaximumInteractionRange for HardSphere {
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.diameter
    }
}

impl<G, S> SitePairEnergy<S, Hyperbolic<3>> for HardShape<G>
where
    S: Position<Position = Hyperbolic<3>> + Orientation<Rotation = Angle>,
    G: SeparatingPlanes<G, Hyperbolic<3>, Angle>,
{
    /// Test whether two sites in two-dimensional hyperbolic space overlap.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::hyperbolic_overlap::HyperbolicConvexPolytope;
    /// use hoomd_interaction::{SitePairEnergy, pairwise::HardShape};
    /// use hoomd_manifold::Hyperbolic;
    /// use hoomd_microstate::property::OrientedHyperbolicPoint;
    /// use hoomd_vector::Angle;
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let square = HyperbolicConvexPolytope::<3>::regular(4, 1.0, 1.0);
    /// let hard_shape = HardShape(square);
    ///
    /// let a = OrientedHyperbolicPoint {
    ///     position: Hyperbolic::<3>::default(),
    ///     orientation: Angle::default(),
    /// };
    /// let b = OrientedHyperbolicPoint {
    ///     position: Hyperbolic::<3>::from_polar_coordinates(3.0, 2.0, 1.0),
    ///     orientation: Angle::from(0.4),
    /// };
    ///
    /// assert_eq!(hard_shape.site_pair_energy(&a, &b), 0.0);
    ///
    /// let c = OrientedHyperbolicPoint {
    ///     position: Hyperbolic::<3>::from_polar_coordinates(0.49, 2.3, 1.0),
    ///     orientation: Angle::from(0.4),
    /// };
    ///
    /// assert_eq!(hard_shape.site_pair_energy(&a, &c), f64::INFINITY);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn site_pair_energy(&self, site_properties_i: &S, site_properties_j: &S) -> f64 {
        let x_i = site_properties_i.position();
        let r_i = site_properties_i.orientation();
        let x_j = site_properties_j.position();
        let r_j = site_properties_j.orientation();
        if self.0.intersects_at(x_i, r_i, x_j, r_j) {
            f64::INFINITY
        } else {
            0.0
        }
    }
}

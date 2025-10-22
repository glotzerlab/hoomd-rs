// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `HardShape`

use crate::SitePairOverlap;
use hoomd_geometry::{IntersectsAt, hyperbolic_overlap::SeparatingPlanes};
use hoomd_manifold::Hyperbolic;
use hoomd_microstate::property::{Orientation, Position};
use hoomd_vector::{Angle, Cartesian, Metric, self, Rotate, Rotation, Vector};

/// Infinite energy when sites overlap, 0 when they don't (*not differentiable*).
///
/// [`HardShape`] represents each site with a hard shape.
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
pub struct HardShape<G>(pub G);

impl<S, G, R, const N: usize> SitePairOverlap<S, Cartesian<N>> for HardShape<G>
where
    S: Position<Position = Cartesian<N>> + Orientation<Rotation = R>,
    R: Rotation + Rotate<Cartesian<N>>,
    G: IntersectsAt<G, Cartesian<N>, R>,
{
    /// Test whether two sites overlap.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{Convex, shape::Rectangle};
    /// use hoomd_interaction::{SitePairOverlap, pairwise::HardShape};
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
    /// assert!(!hard_shape.site_pair_overlap(&a, &b));
    ///
    /// let c = OrientedPoint {
    ///     position: Cartesian::from([1.5, -0.5]),
    ///     orientation: Angle::from(PI / 4.0),
    /// };
    ///
    /// assert!(hard_shape.site_pair_overlap(&a, &c));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn site_pair_overlap(&self, site_properties_i: &S, site_properties_j: &S) -> bool {
        let (v_ij, o_ij) = hoomd_vector::pair_system_to_local(
            site_properties_i.position(),
            site_properties_i.orientation(),
            site_properties_j.position(),
            site_properties_j.orientation(),
        );
        self.0.intersects_at(&self.0, &v_ij, &o_ij)
    }
}

impl<S, G> SitePairOverlap<S, Hyperbolic<3>> for HardShape<G>
where
    S: Position<Position = Hyperbolic<3>> + Orientation<Rotation = Angle>,
    G: SeparatingPlanes<G, Hyperbolic<3>, Angle>,
{
    /// Test whether two sites overlap. 
    /// TODO
    #[inline]
    fn site_pair_overlap(&self, site_properties_i: &S, site_properties_j: &S) -> bool {
        false
    }
}

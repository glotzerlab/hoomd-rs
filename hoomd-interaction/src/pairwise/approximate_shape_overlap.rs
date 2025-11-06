// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`ApproximateShapeOverlap`].

use super::AnisotropicEnergy;
use crate::univariate::UnivariateEnergy;
use hoomd_geometry::IntersectsAt;
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{InnerProduct, Rotate, Rotation};

/// Apply an isotropic potential evaluated at the approximate signed overlap distance.
///
/// Use [`ApproximateShapeOverlap`] combined with [`OverlapPenalty`] and
/// `QuickInsert` to quickly insert new bodies into the microstate or
/// `QuickCompress` to quickly compress the system to a target density.
/// In both cases, [`ApproximateShapeOverlap`] will allow partial overlaps
/// that can be resolved by later trial moves.
///
/// [`ApproximateShapeOverlap`] approximates the distance `r` that shape j
/// must be moved to remove any overlaps. If the shapes are already separate,
/// `r` is 0. Then it returns `isotropic.energy(-r)`. `resolution` sets the
/// steps between `r` values.
///
/// The overlap distance is *approximate* and expensive to evaluate. Do not
/// use this potential for equilibration or sampling. It is suitable *only*
/// for use during a brief initialization phase when `QuickInsert` is adding
/// bodies or `QuickCompress` is compressing the system.
///
/// [`OverlapPenalty`]: crate::univariate::OverlapPenalty
///
/// # Example
///
/// ```
/// use hoomd_geometry::{Convex, shape::ConvexPolygon};
/// use hoomd_interaction::{
///     pairwise::ApproximateShapeOverlap, univariate::OverlapPenalty,
/// };
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let approximate_shape_overlap = ApproximateShapeOverlap::new(
///     Convex(ConvexPolygon::regular(6)),
///     OverlapPenalty::default(),
///     0.01.try_into()?,
/// );
/// # Ok(())
/// # }
/// ```
pub struct ApproximateShapeOverlap<E, A, B = A> {
    /// The site i's shape.
    pub shape_i: A,
    /// The site j's shape.
    pub shape_j: B,
    /// Potential to apply as a function of the signed distance between shapes.
    pub isotropic: E,
    /// Resolve the separation distance in multiples of this resolution.
    pub resolution: PositiveReal,
}

impl<E, A, B, V, R> AnisotropicEnergy<V, R> for ApproximateShapeOverlap<E, A, B>
where
    E: UnivariateEnergy,
    V: InnerProduct,
    R: Rotation + Rotate<V>,
    A: IntersectsAt<B, V, R>,
{
    /// Evaluate the energy contribution from a pair of sites.
    ///
    /// ```
    /// use hoomd_geometry::{Convex, shape::ConvexPolygon};
    /// use hoomd_interaction::{
    ///     SitePairEnergy,
    ///     pairwise::{Anisotropic, ApproximateShapeOverlap},
    ///     univariate::OverlapPenalty,
    /// };
    /// use hoomd_microstate::property::OrientedPoint;
    /// use hoomd_vector::{Angle, Cartesian};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let approximate_shape_overlap = Anisotropic(ApproximateShapeOverlap::new(
    ///     Convex(ConvexPolygon::regular(6)),
    ///     OverlapPenalty::default(),
    ///     0.01.try_into()?,
    /// ));
    ///
    /// let a = OrientedPoint {
    ///     position: Cartesian::from([0.0, 0.0]),
    ///     orientation: Angle::default(),
    /// };
    /// let b = OrientedPoint {
    ///     position: Cartesian::from([1.2, 0.0]),
    ///     orientation: Angle::default(),
    /// };
    ///
    /// let energy = approximate_shape_overlap.site_pair_energy(&a, &b);
    /// assert!(energy >= 100.0);
    ///
    /// let c = OrientedPoint {
    ///     position: Cartesian::from([1.6, 0.0]),
    ///     orientation: Angle::default(),
    /// };
    ///
    /// let energy = approximate_shape_overlap.site_pair_energy(&a, &c);
    /// assert!(energy >= 0.0);
    /// assert!(energy < f64::INFINITY);
    ///
    /// let d = OrientedPoint {
    ///     position: Cartesian::from([2.0, 0.0]),
    ///     orientation: Angle::default(),
    /// };
    ///
    /// let energy = approximate_shape_overlap.site_pair_energy(&a, &d);
    /// assert_eq!(energy, 0.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn energy(&self, r_ij: &V, o_ij: &R) -> f64 {
        let r = self.shape_i.approximate_separation_distance(
            &self.shape_j,
            r_ij,
            o_ij,
            self.resolution,
        );

        self.isotropic.energy(-r)
    }
}

impl<E, G> ApproximateShapeOverlap<E, G> {
    /// Construct a new approximate shape overlap potential.
    ///
    /// `new()` sets both `shape_i` and `shape_j` to `shape`. Use struct
    /// initialization syntax when you need to set these separately.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{Convex, shape::ConvexPolygon};
    /// use hoomd_interaction::{
    ///     pairwise::ApproximateShapeOverlap, univariate::OverlapPenalty,
    /// };
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let approximate_shape_overlap = ApproximateShapeOverlap::new(
    ///     Convex(ConvexPolygon::regular(6)),
    ///     OverlapPenalty::default(),
    ///     0.01.try_into()?,
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn new(shape: G, isotropic: E, resolution: PositiveReal) -> Self
    where
        G: Clone,
    {
        Self {
            shape_i: shape.clone(),
            shape_j: shape,
            isotropic,
            resolution,
        }
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `Anisotropic`

use super::AnisotropicEnergy;
use crate::{MaximumInteractionRange, SitePairEnergy};
use hoomd_microstate::property::{Orientation, Position};
use hoomd_vector::{Rotate, Rotation, Vector};

/// Compute anisotropic properties from a pair of sites.
///
/// [`Anisotropic`] provides a single implementation that computes pairwise
/// interactions that are a function of the sites' positions and orientations.
/// It fills the gap between traits like [`SitePairEnergy`] which operates on
/// site properties and [`AnisotropicEnergy`] which is a function only of the
/// relative position and orientation.
///
/// Use [`Anisotropic`] with [`PairwiseCutoff`] in MD and MC simulations.
///
/// [`PairwiseCutoff`]: crate::PairwiseCutoff
///
/// # Example
///
/// ```
/// use hoomd_interaction::{
///     pairwise::{AngularMask, Anisotropic, angular_mask::Patch},
///     univariate::Boxcar,
/// };
/// use hoomd_vector::Angle;
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let boxcar = Boxcar {
///     epsilon: -1.0,
///     left: 1.0,
///     right: 1.5,
/// };
/// let masks = [Patch {
///     director: [1.0, 0.0].try_into()?,
///     cos_delta: (PI / 8.0).cos(),
/// }];
///
/// let angular_mask = Anisotropic {
///     interaction: AngularMask::new(boxcar, masks),
///     r_cut: 1.5,
/// };
/// # Ok(())
/// # }
/// ```
pub struct Anisotropic<E> {
    /// The site-site interaction.
    pub interaction: E,
    /// Maximum distance between two interacting sites.
    pub r_cut: f64,
}

impl<V, R, S, E> SitePairEnergy<S> for Anisotropic<E>
where
    S: Position<Position = V> + Orientation<Rotation = R>,
    V: Vector,
    R: Rotation + Rotate<V>,
    E: AnisotropicEnergy<V, R>,
{
    /// Compute the pair energy between two sites.
    ///
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_interaction::{
    ///     SitePairEnergy,
    ///     pairwise::{AngularMask, Anisotropic, angular_mask::Patch},
    ///     univariate::Boxcar,
    /// };
    /// use hoomd_microstate::property::OrientedPoint;
    /// use hoomd_vector::{Angle, Cartesian};
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let boxcar = Boxcar {
    ///     epsilon: -1.0,
    ///     left: 1.0,
    ///     right: 1.5,
    /// };
    /// let masks = [Patch {
    ///     director: [1.0, 0.0].try_into()?,
    ///     cos_delta: (PI / 8.0).cos(),
    /// }];
    ///
    /// let angular_mask = Anisotropic {
    ///     interaction: AngularMask::new(boxcar, masks),
    ///     r_cut: 1.5,
    /// };
    ///
    /// let a = OrientedPoint {
    ///     position: Cartesian::from([0.0, 0.0]),
    ///     orientation: Angle::from(0.0),
    /// };
    /// let b = OrientedPoint {
    ///     position: Cartesian::from([1.0, 0.0]),
    ///     orientation: Angle::from(0.0),
    /// };
    /// let energy = angular_mask.site_pair_energy(&a, &b);
    /// assert_eq!(energy, 0.0);
    ///
    /// let c = OrientedPoint {
    ///     position: Cartesian::from([1.0, 0.0]),
    ///     orientation: Angle::from(PI),
    /// };
    /// let energy = angular_mask.site_pair_energy(&a, &c);
    /// assert_eq!(energy, -1.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn site_pair_energy(&self, site_properties_i: &S, site_properties_j: &S) -> f64 {
        let r = site_properties_i
            .position()
            .distance(site_properties_j.position());
        if r >= self.r_cut {
            return 0.0;
        }

        let (r_ab, o_ab) = hoomd_vector::pair_system_to_local(
            site_properties_i.position(),
            site_properties_i.orientation(),
            site_properties_j.position(),
            site_properties_j.orientation(),
        );
        self.interaction.energy(&r_ab, &o_ab)
    }
}

impl<E> MaximumInteractionRange for Anisotropic<E> {
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.r_cut
    }
}

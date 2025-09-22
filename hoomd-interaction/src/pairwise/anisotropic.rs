// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `Anisotropic`

use super::AnisotropicEnergy;
use crate::SitePairEnergy;
use hoomd_microstate::property::{Orientation, Position};
use hoomd_vector::{Rotate, Rotation, Vector};

/// Compute anisotropic properties from a pair of sites.
///
/// [`Anisotropic`] is a newtype that provides a single implementation to compute
/// pairwise properties. It fills the gap between traits like [`SitePairEnergy`]
/// which operates on site properties and [`AnisotropicEnergy`] which is a function
/// only of the the relative position and orientation.
///
/// Use [`Anisotropic`] with [`CutoffPair`](crate::CutoffPair) in MD and MC
/// simulations.
///
/// # Example
///
/// ```
/// use hoomd_interaction::pairwise::{
///     AngularMask, Anisotropic, Boxcar, angular_mask::Patch,
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
/// let angular_mask = Anisotropic(AngularMask::new(boxcar, masks));
/// # Ok(())
/// # }
/// ```
pub struct Anisotropic<E>(pub E);

impl<V, R, S, E> SitePairEnergy<S> for Anisotropic<E>
where
    S: Position<Vector = V> + Orientation<Rotation = R>,
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
    ///     pairwise::{AngularMask, Anisotropic, Boxcar, angular_mask::Patch},
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
    /// let angular_mask = Anisotropic(AngularMask::new(boxcar, masks));
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
        let (r_ab, o_ab) = hoomd_vector::pair_system_to_local(
            site_properties_i.position(),
            site_properties_i.orientation(),
            site_properties_j.position(),
            site_properties_j.orientation(),
        );
        self.0.energy(&r_ab, &o_ab)
    }
}

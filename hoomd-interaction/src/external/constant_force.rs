// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`ConstantForce`]

use serde::{Deserialize, Serialize};

use hoomd_microstate::property::Position;
use hoomd_vector::{InnerProduct, Unit, Wedge};

use crate::{SiteForce, SiteForceAndTorque};

use super::super::SiteEnergy;

/// Apply the same force to every site, independent of the site's properties.
///
/// The force is:
/// ```math
/// \vec{F} = -\alpha \hat{n}
/// ```
/// which is consistent with the potential energy:
/// ```math
/// U = \alpha \cdot \hat{n} \cdot ( \vec{r} - \vec{p} )
/// ```
///
/// The plane origin `p` sets the 0 energy reference, the plane normal `n`,
/// sets the direction of the force, and `alpha` is the the interaction strength.
///
/// # Generics
///
/// * `V`: The type used to represent the position and normal vectors.
///
/// # Example
///
/// Basic usage:
///
/// ```
/// use hoomd_interaction::external::ConstantForce;
/// use hoomd_vector::{Cartesian, Unit};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let constant_force = ConstantForce {
///     alpha: 2.0,
///     plane_origin: [0.0, -10.0].into(),
///     plane_normal: [0.0, 1.0].try_into()?,
/// };
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstantForce<V> {
    /// Interaction strength $`[\mathrm{energy}] [\mathrm{length}]^{-1}`$.
    pub alpha: f64,
    /// Point on the plane where U=0 $`[\mathrm{length}]`$.
    pub plane_origin: V,
    /// Vector normal to the plane *(unitless)*.
    pub plane_normal: Unit<V>,
}

impl<V> ConstantForce<V>
where
    V: InnerProduct,
{
    /// Compute the energy of a point in the linear field.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_interaction::external::ConstantForce;
    /// use hoomd_vector::{Cartesian, Unit};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let constant_force = ConstantForce {
    ///     alpha: 2.0,
    ///     plane_origin: [0.0, -10.0].into(),
    ///     plane_normal: [0.0, 1.0].try_into()?,
    /// };
    ///
    /// let energy = constant_force.energy(&[0.0, 0.0].into());
    /// assert_eq!(energy, 20.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn energy(&self, r: &V) -> f64 {
        self.alpha * self.plane_normal.get().dot(&(*r - self.plane_origin))
    }

    /// The force vector that acts on all sites.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_interaction::external::ConstantForce;
    /// use hoomd_vector::{Cartesian, Unit};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let constant_force = ConstantForce {
    ///     alpha: 2.0,
    ///     plane_origin: [0.0, -10.0].into(),
    ///     plane_normal: [0.0, 1.0].try_into()?,
    /// };
    ///
    /// let energy = constant_force.force();
    /// assert_eq!(energy, [0.0, -2.0].into());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn force(&self) -> V {
        *self.plane_normal.get() * self.alpha * -1.0
    }
}

impl<S, P> SiteEnergy<S> for ConstantForce<P>
where
    S: Position<Position = P>,
    P: InnerProduct,
{
    #[inline]
    fn site_energy(&self, site_properties: &S) -> f64 where {
        self.energy(site_properties.position())
    }
}

impl<S, V> SiteForce<S> for ConstantForce<V> where
V: InnerProduct,
{
    type Force = V;

    #[inline]
    fn site_force(&self, _site_properties: &S) -> Self::Force {
        self.force()
    }
}

impl<S, V> SiteForceAndTorque<S> for ConstantForce<V> where
V: InnerProduct + Wedge,
V::Bivector: Default,
{
    type Force = V;
    type Torque = V::Bivector;

    #[inline]
    fn site_force_and_torque(&self, _site_properties: &S) -> (Self::Force, Self::Torque) {
        (self.force(), V::Bivector::default())
    }
}

#[cfg(test)]
mod tests {
    use hoomd_vector::Cartesian;

    use super::*;
    use approxim::assert_relative_eq;
    use rstest::*;

    #[rstest]
    fn energy_2d(
        #[values(1.0, 0.0, -2.0)] alpha: f64,
        #[values([0.0, 0.0], [-10.0, 15.0], [16.0, 3.0])] plane_origin: [f64; 2],
        #[values([1.0, 1.0], [-1.0, 0.2], [-5.0, -1.0])] plane_normal: [f64; 2],
    ) {
        let n = Unit::<Cartesian<2>>::try_from(plane_normal)
            .expect("hard-coded vector should have non-zero length");

        let linear = ConstantForce {
            plane_origin: plane_origin.into(),
            plane_normal: n,
            alpha,
        };

        let p = linear.plane_origin + *n.get() * 5.0;
        assert_relative_eq!(linear.energy(&p), 5.0 * alpha, epsilon = 1e-6);
    }
}

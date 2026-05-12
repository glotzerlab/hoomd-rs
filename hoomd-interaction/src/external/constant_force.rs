// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`ConstantForce`]

use serde::{Deserialize, Serialize};

use hoomd_microstate::property::Position;
use hoomd_vector::{InnerProduct, Wedge};

use crate::{SiteForce, SiteForceAndTorque};

use super::super::SiteEnergy;

/// Apply the same force to every site, independent of the site's properties.
///
/// The field `force` sets the force vector $` \vec{F} `$. The corresponding
/// potential energy $` U `$ is:
/// ```math
/// U = - \vec{F} \cdot ( \vec{r} - \vec{r}_0 )
/// ```
/// The vector $` \vec{r}_0 `$ sets the reference plane where $` U = 0 `$.
///
/// # Generics
///
/// * `V`: The type used to represent the position and force vectors.
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
///     force: Cartesian::from([0.0, -2.0]),
///     r_0: Cartesian::from([0.0, -10.0]),
/// };
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstantForce<V> {
    /// Force vector $`[\mathrm{energy}] [\mathrm{length}]^{-1}`$.
    pub force: V,

    ///  $` \vec{r}_0 `$ $`[\mathrm{length}]`$: A point on the plane where $` U = 0 `$.
    pub r_0: V,
}

impl<V> ConstantForce<V>
where
    V: InnerProduct,
{
    /// Compute the energy of a point in a constant force field.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_interaction::external::ConstantForce;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let constant_force = ConstantForce {
    ///     force: Cartesian::from([0.0, -2.0]),
    ///     r_0: Cartesian::from([0.0, -10.0]),
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
        let magnitude = self.force.norm();

        if magnitude == 0.0 {
            return 0.0;
        }
        
        let direction = self.force / magnitude;
        -magnitude * direction.dot(&(*r - self.r_0))
    }

    /// The force vector that acts on all sites.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_interaction::external::ConstantForce;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let constant_force = ConstantForce {
    ///     force: Cartesian::from([0.0, -2.0]),
    ///     r_0: Cartesian::from([0.0, -10.0]),
    /// };
    ///
    /// let force = constant_force.force();
    /// assert_eq!(force, [0.0, -2.0].into());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn force(&self) -> V {
        self.force
    }
}

impl<S, P> SiteEnergy<S> for ConstantForce<P>
where
    S: Position<Position = P>,
    P: InnerProduct,
{
    /// Evaluate the energy contribution of a single site.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_interaction::{external::ConstantForce, SiteEnergy};
    /// use hoomd_vector::Cartesian;
    /// use hoomd_microstate::property::Point;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let constant_force = ConstantForce {
    ///     force: Cartesian::from([0.0, -2.0]),
    ///     r_0: Cartesian::from([0.0, -10.0]),
    /// };
    ///
    /// let a = Point { position: Cartesian::from([0.0, 0.0]) };
    /// let b = Point { position: Cartesian::from([0.0, 3.0]) };
    ///
    /// let energy_0 = constant_force.site_energy(&a);
    /// assert_eq!(energy_0, 20.0);
    //
    /// let energy_1 = constant_force.site_energy(&b);
    /// assert_eq!(energy_1, 26.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn site_energy(&self, site_properties: &S) -> f64 where {
        self.energy(site_properties.position())
    }
}

impl<S, V> SiteForce<S> for ConstantForce<V> where
V: InnerProduct,
{
    type Force = V;

    /// Evaluate the force as a function of a single site's properties.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_interaction::{external::ConstantForce, SiteForce};
    /// use hoomd_vector::Cartesian;
    /// use hoomd_microstate::property::Point;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let constant_force = ConstantForce {
    ///     force: Cartesian::from([0.0, -2.0]),
    ///     r_0: Cartesian::from([0.0, -10.0]),
    /// };
    ///
    /// let a = Point { position: Cartesian::from([0.0, 0.0]) };
    /// let b = Point { position: Cartesian::from([0.0, 3.0]) };
    ///
    /// let force_0 = constant_force.site_force(&a);
    /// assert_eq!(force_0, [0.0, -2.0].into());
    ///
    /// let force_1 = constant_force.site_force(&b);
    /// assert_eq!(force_1, [0.0, -2.0].into());
    /// # Ok(())
    /// # }
    /// ```
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

    /// Evaluate the force and/or torque as a function of a single site's properties.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_interaction::{external::ConstantForce, SiteForceAndTorque};
    /// use hoomd_vector::Cartesian;
    /// use hoomd_microstate::property::Point;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let constant_force = ConstantForce {
    ///     force: Cartesian::from([0.0, -2.0]),
    ///     r_0: Cartesian::from([0.0, -10.0]),
    /// };
    ///
    /// let a = Point { position: Cartesian::from([0.0, 0.0]) };
    /// let b = Point { position: Cartesian::from([0.0, 3.0]) };
    ///
    /// let (force_0, torque_0) = constant_force.site_force_and_torque(&a);
    /// assert_eq!(force_0, [0.0, -2.0].into());
    /// assert_eq!(torque_0, 0.0);
    ///
    /// let (force_1, torque_1) = constant_force.site_force_and_torque(&b);
    /// assert_eq!(force_1, [0.0, -2.0].into());
    /// assert_eq!(torque_1, 0.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn site_force_and_torque(&self, _site_properties: &S) -> (V, V::Bivector) {
        (self.force(), V::Bivector::default())
    }
}

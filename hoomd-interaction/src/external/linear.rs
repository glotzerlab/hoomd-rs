// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Linear`]
 */

use super::super::SiteEnergy;
use hoomd_microstate::property::Position;
use hoomd_vector::{InnerProduct, Unit};

/** Linear potential based on position.

```math
U = \alpha \cdot \vec{n} \cdot ( \vec{r} - \vec{p} )
```

Computes a linear external potential at a point in space relative to the plane
origin `p`, plane normal `n`, and the interaction strength `alpha`.

# Example

Basic usage:

```
use hoomd_interaction::external::Linear;
use hoomd_vector::{Cartesian, Unit};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let linear = Linear { alpha: 2.0,
    plane_origin: [0.0, -10.0].into(),
    plane_normal: [0.0, 1.0].try_into()?,
};
# Ok(())
# }
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Linear<V> {
    /// Interaction strength *(\[energy\] \[length\]^(-1))*.
    pub alpha: f64,
    /// Point on the plane where U=0 *(\[length\])*.
    pub plane_origin: V,
    /// Vector normal to the plane *(unitless)*.
    pub plane_normal: Unit<V>,
}

impl<V> Linear<V>
where
    V: InnerProduct,
{
    /** Compute the energy of a point in the linear field.

    # Example

    ```
    use hoomd_interaction::external::Linear;
    use hoomd_vector::{Cartesian, Unit};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let linear = Linear { alpha: 2.0,
        plane_origin: [0.0, -10.0].into(),
        plane_normal: [0.0, 1.0].try_into()?,
    };

    let energy = linear.energy(&[0.0, 0.0].into());
    assert_eq!(energy, 20.0);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn energy(&self, r: &V) -> f64 {
        self.alpha * self.plane_normal.get().dot(&(*r - self.plane_origin))
    }
}

impl<S, V> SiteEnergy<S> for Linear<V>
where
    S: Position<Vector = V>,
    V: InnerProduct,
{
    #[inline]
    fn site_energy(&self, site_properties: &S) -> f64 where {
        self.energy(site_properties.position())
    }
}

#[cfg(test)]
mod tests {
    use hoomd_vector::Cartesian;

    use super::*;
    use ::approx::assert_relative_eq;
    use rstest::*;

    #[rstest]
    fn energy_2d(
        #[values(1.0, 0.0, -2.0)] alpha: f64,
        #[values([0.0, 0.0], [-10.0, 15.0], [16.0, 3.0])] plane_origin: [f64; 2],
        #[values([1.0, 1.0], [-1.0, 0.2], [-5.0, -1.0])] plane_normal: [f64; 2],
    ) {
        let n = Unit::<Cartesian<2>>::try_from(plane_normal)
            .expect("hard-coded vector should have non-zero length");

        let linear = Linear {
            plane_origin: plane_origin.into(),
            plane_normal: n,
            alpha,
        };

        let p = linear.plane_origin + *n.get() * 5.0;
        assert_relative_eq!(linear.energy(&p), 5.0 * alpha, epsilon = 1e-6);
    }
}

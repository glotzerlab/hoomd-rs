// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Isotropic
 */

use super::IsotropicEnergy;
use crate::SitePairEnergy;
use hoomd_microstate::property::Position;
use hoomd_vector::Vector;

/** Compute isotropic properties from a pair of sites

[`Isotropic`] is a newtype that provides a single implementation to compute
pairwise properties. It fills the gap between traits like [`SitePairEnergy`]
which operates on site properties and [`IsotropicEnergy`] which is a function
only of the separation distance.

Use [`Isotropic`] with [`CutoffPair`](crate::CutoffPair) in MD and MC
simulations.

# Example

```
use hoomd_interaction::{SitePairEnergy, pairwise::{Isotropic, LennardJones}};
use hoomd_microstate::property::Point;
use hoomd_vector::Cartesian;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let a = Point { position: Cartesian::from([0.0, 0.0]) };
let b = Point { position: Cartesian::from([0.0, 2.0 * 2.0_f64.powf(1.0/6.0)]) };

let lennard_jones: LennardJones = LennardJones { epsilon: 1.5, sigma: 2.0 };
let lennard_jones = Isotropic(lennard_jones);

let energy = lennard_jones.site_pair_energy(&a, &b);
assert_eq!(energy, -1.5);
# Ok(())
# }
```
*/
pub struct Isotropic<E>(pub E);

impl<V, S, E> SitePairEnergy<S> for Isotropic<E>
where
    S: Position<Vector = V>,
    V: Vector,
    E: IsotropicEnergy,
{
    #[inline]
    fn site_pair_energy(&self, a: &S, b: &S) -> f64 {
        self.0.energy((a.position()).distance(b.position()))
    }
}

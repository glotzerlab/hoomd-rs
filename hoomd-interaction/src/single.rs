// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Single
*/

use crate::{TotalEnergy, SiteEnergy};
use hoomd_microstate::Microstate;

/** Compute system properties given a [`SiteEnergy`]

[`Single`] is a newtype that provides a single implementation for system
properties, like [`TotalEnergy`], for all types that implement [`SiteEnergy`].

Use types that implement [`SiteEnergy`], such as one from [`external`] or your
own custom type, directly when you only need to call `site_energy`. Wrap the
type in `Single` to use it with MC simulations or to compute the total energy.

TODO: Reword this when Single also implements `SiteForce`.

# Example

```
use hoomd_interaction::{Single, TotalEnergy, external::Linear};
use hoomd_microstate::{Microstate, Body};
use hoomd_microstate::property::{Point, Position};
use hoomd_vector::Cartesian;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = Microstate::new();
microstate.extend_bodies([Body::point(Cartesian::from([1.0, 0.0])),
                              Body::point(Cartesian::from([-1.0, 2.0]))])?;

let linear = Single(Linear{ alpha: 1.0,
    plane_origin: Cartesian::default(),
    plane_normal: [0.0, 1.0].try_into()? });

let total_energy = linear.total_energy(&microstate);
assert_eq!(total_energy, 2.0);
# Ok(())
# }
```
*/
pub struct Single<E>(pub E);

impl<B, S, C, E> TotalEnergy<Microstate<B, S, C>> for Single<E>
where
    E: SiteEnergy<S>,
{
    /** Compute the total energy of the microstate contributed by functions of a single site.

    The sum over sites differs from HOOMD-blue where external energies are
    evaluated only at the body centers. In general, hoomd-rs interactions apply
    to sites. Use a custom implementation to compute energies over body centers.
    */
    #[inline]
    fn total_energy(&self, microstate: &Microstate<B, S, C>) -> f64 {
        microstate
            .sites()
            .iter()
            .fold(0.0, |total, s| total + self.0.site_energy(&s.properties))
    }
}

impl<E, S> SiteEnergy<S> for Single<E>
where
    E: SiteEnergy<S>,
{
    #[inline]
    fn site_energy(&self, site_properties: &S) -> f64 {
        self.0.site_energy(site_properties)
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Particle interactions and physical models that apply them to microstates.

TODO: Expand documentation.
 */

pub mod external;
pub mod pairwise;

use hoomd_microstate::Microstate;

// TODO: should M and S be trait parameters? Or parameters on the methods in the traits?

/** Compute the total external energy of the microstate.

*/
pub trait TotalEnergy<M> {
    /** Compute the total external energy of the microstate.

    */
    #[must_use]
    fn total_energy(&self, microstate: &M) -> f64;
}

/** Compute the energy contribution of a single site.

TODO: expand documentation.
*/
pub trait SiteEnergy<S> {
    #[must_use]
    fn site_energy(&self, site_properties: &S) -> f64;
}

/**

# Example

```
use hoomd_interaction::{Single, TotalEnergy, external::Linear};
use hoomd_microstate::{Microstate, Body};
use hoomd_microstate::property::{Point, Position};
use hoomd_vector::Cartesian;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = Microstate::new();
microstate.extend_bodies([Body::point(Cartesian::from([1.0, 0.0])),
                          Body::point(Cartesian::from([-1.0, 2.0]))]);

let linear = Single::new(Linear{ alpha: 1.0,
    plane_origin: Cartesian::default(),
    plane_normal: [0.0, 1.0].try_into()? });

let total_energy = linear.total_energy(&microstate);
assert_eq!(total_energy, 2.0);
# Ok(())
# }
```

TODO: Naming is hard. Is there a better name? Single goes along with CutoffPairs
that hold each hold an inner type that implements SitePairEnergy (and other future types).
*/
pub struct Single<E> {
    /// Evaluate the energy/force/etc... on a single site.
    pub inner: E,
}

impl<E> Single<E> {
    pub fn new(inner: E) -> Self {
        Self { inner }
    }
}

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
        microstate.sites().iter().fold(0.0, |total, s| {
            total + self.inner.site_energy(&s.properties)
        })
    }
}

impl<E, S> SiteEnergy<S> for Single<E>
where
    E: SiteEnergy<S>,
{
    #[inline]
    fn site_energy(&self, site_properties: &S) -> f64 {
        self.inner.site_energy(site_properties)
    }
}

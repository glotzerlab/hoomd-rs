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

/** Compute the total energy of a potential applied to the microstate.

The `TotalEnergy` trait describes a type that can compute the energy of a
given microstate. Depending on the type, `total_energy` might compute the total
potential energy of the system or a single term, such as the Lennard-Jones
potential energy.

TODO: Provide a `LennardJones` example.
*/
pub trait TotalEnergy<M> {
    /// Compute the energy.
    #[must_use]
    fn total_energy(&self, microstate: &M) -> f64;
}

/** Compute the energy contribution of a single site.

The `SiteEnergy` trait describes a type that can compute the energy contribution
of a site to the system's total energy, *as a function only of that site's
properties*.

The [`external`] module provides a number of commonly used implementations.
Combine them with [`Single`] newtype for use with MC and MD simulations or to
compute system-wide properties.

## Examples

Implement a custom site energy function:

```
use hoomd_interaction::{Single, TotalEnergy, SiteEnergy};
use hoomd_microstate::{Microstate, Body};
use hoomd_microstate::property::{Point, Position};
use hoomd_vector::Cartesian;

struct Custom {
    a: f64,
    b: f64,
}

impl<S> SiteEnergy<S> for Custom
where
    S: Position<Cartesian<2>>
{
    fn site_energy(&self, site_properties: &S) -> f64 {
        self.a * (site_properties.position()[0] / self.b).cos()
    }
}

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = Microstate::new();
microstate.extend_bodies([Body::point(Cartesian::from([1.0, 0.0])),
                              Body::point(Cartesian::from([-1.0, 2.0]))])?;

let custom_evaluator = Custom { a: 1.0, b: 10.0 };
let site_energy = custom_evaluator.site_energy(&microstate.sites()[0].properties);

let custom = Single(custom_evaluator);
let total_energy = custom.total_energy(&microstate);
# Ok(())
# }
```
*/
pub trait SiteEnergy<S> {
    /// Evaluate the energy contribution of a single site.
    #[must_use]
    fn site_energy(&self, site_properties: &S) -> f64;
}

/** Compute system-wide properties given a [`SiteEnergy`]

`Single` is a newtype that provides a single implementation for system-wide
properties, like [`TotalEnergy`], for all types that implement [`SiteEnergy`].
It also reimplements [`SiteEnergy`] by forwarding the call to the inner type.

Use types that implement [`SiteEnergy`], such as one from [`external`] or your
own custom type, directly when you only need to call `site_energy`. Wrap the
type in `Single` to use it with MC simulations or to call `total_energy`.

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

impl<V, B, S, C, E> TotalEnergy<Microstate<V, B, S, C>> for Single<E>
where
    E: SiteEnergy<S>,
{
    /** Compute the total energy of the microstate contributed by functions of a single site.

    The sum over sites differs from HOOMD-blue where external energies are
    evaluated only at the body centers. In general, hoomd-rs interactions apply
    to sites. Use a custom implementation to compute energies over body centers.
    */
    #[inline]
    fn total_energy(&self, microstate: &Microstate<V, B, S, C>) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_microstate::boundary::Open;
    use hoomd_microstate::property::{Point, Position};
    use hoomd_microstate::{Body, Microstate};
    use hoomd_vector::Cartesian;
    use rstest::*;

    struct TestSE;

    impl<S> SiteEnergy<S> for TestSE
    where
        S: Position<Cartesian<2>>,
    {
        fn site_energy(&self, site_properties: &S) -> f64 {
            site_properties.position()[0] + site_properties.position()[1]
        }
    }

    #[fixture]
    fn microstate() -> Microstate<Cartesian<2>, Point<Cartesian<2>>, Point<Cartesian<2>>, Open> {
        let mut microstate = Microstate::new();
        microstate
            .extend_bodies([
                Body::point(Cartesian::from([1.0, 0.0])),
                Body::point(Cartesian::from([-1.0, 3.0])),
            ])
            .expect("valid bodies");
        microstate
    }

    #[rstest]
    fn single_total(
        microstate: Microstate<Cartesian<2>, Point<Cartesian<2>>, Point<Cartesian<2>>, Open>,
    ) {
        let test_se = TestSE;
        let single = Single(test_se);

        assert_eq!(single.total_energy(&microstate), 3.0);
    }

    #[rstest]
    fn single_site(
        microstate: Microstate<Cartesian<2>, Point<Cartesian<2>>, Point<Cartesian<2>>, Open>,
    ) {
        let test_se = TestSE;
        let single = Single(test_se);

        assert_eq!(single.site_energy(&microstate.sites()[0].properties), 1.0);
        assert_eq!(single.site_energy(&microstate.sites()[1].properties), 2.0);
    }
}

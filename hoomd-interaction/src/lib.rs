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

mod single;
pub use single::Single;

mod cutoff_pair;
pub use cutoff_pair::CutoffPair;

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

The generic type names are:
* `S`: The `Site::properties` type.

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
    S: Position<Vector = Cartesian<2>>
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

/** Compute the energy contribution from a pair of sites.

The `SitePairEnergy` trait describes a type that can compute the energy
contribution from a pair of sites to the system's total energy, *as a function
only of those site's properties*.

The [`pairwise`] module provides a number of commonly used implementations.
Combine them with the [`CutoffPair`] and [`Isotropic`]/[`Anisotropic`], newtypes
for use with MC and MD simulations or to compute system-wide properties.

## Examples

TODO Implement a custom site pair energy function:

*/
pub trait SitePairEnergy<S> {
    /// Evaluate the energy contribution of a single site.
    fn site_pair_energy(&self, a: &S, b: &S) -> f64;
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
        S: Position<Vector = Cartesian<2>>,
    {
        fn site_energy(&self, site_properties: &S) -> f64 {
            site_properties.position()[0] + site_properties.position()[1]
        }
    }

    #[fixture]
    fn microstate() -> Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open> {
        let mut microstate = Microstate::new();
        microstate
            .extend_bodies([
                Body::point(Cartesian::from([1.0, 0.0])),
                Body::point(Cartesian::from([-1.0, 3.0])),
            ])
            .expect("hard-coded bodies should be in the boundary");
        microstate
    }

    #[rstest]
    fn single_total(
        microstate: Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open>,
    ) {
        let test_se = TestSE;
        let single = Single(test_se);

        assert_eq!(single.total_energy(&microstate), 3.0);
    }

    #[rstest]
    fn single_site(
        microstate: Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open>,
    ) {
        let test_se = TestSE;
        let single = Single(test_se);

        assert_eq!(single.site_energy(&microstate.sites()[0].properties), 1.0);
        assert_eq!(single.site_energy(&microstate.sites()[1].properties), 2.0);
    }

    // TODO: Test CutoffPair
}

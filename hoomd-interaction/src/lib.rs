// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Particle interactions and physical models that apply to microstates.

TODO: Expand documentation.
 */

pub mod external;
pub mod pairwise;

mod single;
pub use single::Single;

mod cutoff_pair;
pub use cutoff_pair::CutoffPair;

mod hamiltonian;

/** Compute the total energy of a potential applied to the microstate.

The `TotalEnergy` trait describes a type that can compute the energy of a
given microstate. Depending on the type, `total_energy` might compute the total
potential energy of the system or a single term, such as the Lennard-Jones
potential energy.

# Example

```
use hoomd_interaction::{CutoffPair, SitePairEnergy, TotalEnergy,
    pairwise::{Isotropic, LennardJones}};
use hoomd_microstate::{Microstate, Body};
use hoomd_microstate::property::{Point, Position};
use hoomd_vector::{Cartesian, Vector};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = Microstate::new();
// Place two pairs of particles separated by a large distance.
microstate.extend_bodies([Body::point(Cartesian::from([0.0, 0.0])),
                          Body::point(Cartesian::from([1.0, 0.0])),
                          Body::point(Cartesian::from([0.0, 5.0])),
                          Body::point(Cartesian::from([-1.0, 5.0])),
                        ])?;

let lennard_jones: LennardJones = LennardJones { epsilon: 1.5, sigma: 1.0 / 2.0_f64.powf(1.0/6.0) };
let lennard_jones = Isotropic(lennard_jones);
let cutoff_pair = CutoffPair { r_cut: 2.5, evaluator: lennard_jones, };

let total_energy = cutoff_pair.total_energy(&microstate);
assert_eq!(total_energy, -3.0);
# Ok(())
# }
```
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
Combine them with the [`CutoffPair`] and the [`Isotropic`](pairwise::Isotropic)
or [`Anisotropic`](pairwise::Anisotropic) newtypes for use with MC and MD
simulations or to compute system-wide properties.

## Examples

Implement a custom site energy function:

```
use hoomd_interaction::{CutoffPair, TotalEnergy, SitePairEnergy};
use hoomd_microstate::{Microstate, Body};
use hoomd_microstate::property::{Point, Position};
use hoomd_vector::{Cartesian, InnerProduct};

struct Custom {
    epsilon: f64,
}

impl<S> SitePairEnergy<S> for Custom
where
    S: Position<Vector = Cartesian<2>>
{
    fn site_pair_energy(&self, a: &S, b: &S) -> f64 {
        self.epsilon * a.position().dot(&b.position())
    }
}

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = Microstate::new();
microstate.extend_bodies([Body::point(Cartesian::from([1.0, 0.0])),
                          Body::point(Cartesian::from([0.0, 1.0]))])?;

let evaluator = Custom { epsilon: 1.0 };
let site_pair_energy = evaluator.site_pair_energy(
    &microstate.sites()[0].properties,
    &microstate.sites()[1].properties);

let custom = CutoffPair { r_cut: 2.5, evaluator };
let total_energy = custom.total_energy(&microstate);
# Ok(())
# }
```
*/
pub trait SitePairEnergy<S> {
    /// Evaluate the energy contribution from a pair of sites.
    fn site_pair_energy(&self, a: &S, b: &S) -> f64;
}

/** Compute the force on a single site.
TODO: Add SiteForce documentation
*/
pub trait SiteForce<V, S> {
    /// Evaluate the force on a single site.
    fn site_force(&self, site_properties: &S) -> V;
}

/** Compute the Force on one site from another site.
TODO: Add SitePairForce documentation
*/
pub trait SitePairForce<V, S> {
    /// Evaluate the force on site a from site b.
    fn site_pair_force(&self, a: &S, b: &S) -> V;
}

/** Compute the torque on a single site.
TODO: Add SiteTorque documentation
*/
pub trait SiteTorque<V, S> {
    /// Evaluate the torque on a single site.
    fn site_torque(&self, site_properties: &S) -> V;
}


/** Compute the torque on one site from another site.
TODO: Add SitePairTorque documentation
*/
pub trait SitePairTorque<V, S> {
    /// Evaluate the torque contribution from a pair of sites.
    fn site_pair_torque(&self, a: &S, b: &S) -> V;
}

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_microstate::boundary::Open;
    use hoomd_microstate::property::{Point, Position};
    use hoomd_microstate::{Body, Microstate};
    use hoomd_vector::Cartesian;
    use rstest::*;

    mod single {
        use super::*;

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
        fn single_total(microstate: Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open>) {
            let test_se = TestSE;
            let single = Single(test_se);

            assert_eq!(single.total_energy(&microstate), 3.0);
        }

        #[rstest]
        fn single_site(microstate: Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open>) {
            let test_se = TestSE;
            let single = Single(test_se);

            assert_eq!(single.site_energy(&microstate.sites()[0].properties), 1.0);
            assert_eq!(single.site_energy(&microstate.sites()[1].properties), 2.0);
        }
    }

    mod cutoff_pair {
        use super::*;
        use crate::pairwise::Isotropic;

        #[fixture]
        fn microstate() -> Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open> {
            let mut microstate = Microstate::new();
            microstate
                .extend_bodies([
                    Body::point(Cartesian::from([0.0, 0.0])),
                    Body::point(Cartesian::from([1.0, 0.0])),
                    Body::point(Cartesian::from([0.0, 5.0])),
                    Body::point(Cartesian::from([1.0, 5.0])),
                ])
                .expect("hard-coded bodies should be in the boundary");
            microstate
        }

        #[rstest]
        fn blanket_fn(microstate: Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open>) {
            // Ensure that closures can be used as IsotropicEnergy
            let cutoff_pair = CutoffPair {
                r_cut: 2.0,
                evaluator: Isotropic(|r| 1.0 / (r * 2.0)),
            };

            // Two pairs at a distance of 1.0 each with energy 1/2.
            assert_eq!(cutoff_pair.total_energy(&microstate), 1.0);
        }

        #[rstest]
        fn large_r_cut(microstate: Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open>) {
            // Ensure that CutoffPair respects the r_cut value set.
            let cutoff_pair = CutoffPair {
                r_cut: 5.0_f64.next_up(),
                evaluator: Isotropic(|r| 1.0 / (r * 2.0)),
            };

            // Two pairs at a distance of 1.0 each with energy 1/2.
            // Plus two pairs at a distance of 5.0 with energy 1/10
            assert_eq!(cutoff_pair.total_energy(&microstate), 1.2);
        }

        #[test]
        fn body_exclusion() {
            // Ensure that CutoffPair excludes pairs in the same body.
            let body_a = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [
                    Point::new(Cartesian::from([1.0, 1.0])),
                    Point::new(Cartesian::from([1.0, -1.0])),
                    Point::new(Cartesian::from([-1.0, 1.0])),
                    Point::new(Cartesian::from([-1.0, -1.0])),
                ]
                .into(),
            };
            let body_b = Body {
                properties: Point::new(Cartesian::from([3.0, 0.0])),
                sites: body_a.sites.clone(),
            };

            let mut microstate = Microstate::new();
            microstate
                .extend_bodies([body_a, body_b])
                .expect("hard-coded bodies should be in the boundary");

            let cutoff_pair = CutoffPair {
                r_cut: 1.0_f64.next_up(),
                evaluator: Isotropic(|_r| 1.0),
            };

            // Of all the pairs a distance 1.0 apart, only 2 are interbody pairs.
            assert_eq!(cutoff_pair.total_energy(&microstate), 2.0);
        }
    }
}

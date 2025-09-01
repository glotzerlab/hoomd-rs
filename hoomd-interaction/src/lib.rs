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

use hoomd_microstate::{Body, Microstate};

pub mod external;
pub mod pairwise;

mod cutoff_pair;
mod hamiltonian;
mod single;
mod zero;

pub use cutoff_pair::CutoffPair;
pub use single::Single;
pub use zero::Zero;

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

let lennard_jones: LennardJones = LennardJones { epsilon: 1.5, sigma: 1.0 / 2.0_f64.powf(1.0 / 6.0) };
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
* `S`: The [`Site::properties`](hoomd_microstate::Site) type.

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
    S: Position<Metric = Cartesian<2>>
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
Combine them with the [`CutoffPair`] and the [`Isotropic`] or [`Anisotropic`]
newtypes for use with MC and MD simulations or to compute system-wide
properties.

The generic type names are:
* `S`: The [`Site::properties`](hoomd_microstate::Site) type.

TODO: Fix anisotropic link when implemented.

[`Isotropic`]: pairwise::Isotropic
[`Anisotropic`]: pairwise::Isotropic

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
    S: Position<Metric = Cartesian<2>>
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

/** Compute the change energy as a function of a single modified body.

Some trial moves apply to a single body at a time and use a Hamiltonian that
implements `DeltaEnergyOne` to efficiently compute the change in energy.

The generic type names are:
* `B`: The [`Body::properties`](hoomd_microstate::Body) type.
* `S`: The [`Site::properties`](hoomd_microstate::Site) type.
* `C`: The [`boundary`](hoomd_microstate::boundary) condition type.

See the [Implementors](#implementors) section below for examples.
*/
pub trait DeltaEnergyOne<B, S, C> {
    /** Compute the change in energy.

    `initial_microstate` describes the initial configuration and `final_body`
    describes the new body configuration. In the final configuration, the
    body may have changed properties and/or sites. The index `body_index`
    identifies which body in `initial_microstate` is changing.

    Returns:
    ```math
    \Delta E = E_\mathrm{final} - E_\mathrm{initial}
    ```
    */
    #[must_use]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64;
}

/** Compute the change energy when a single body is inserted.

Some trial moves insert a single body at a time and use a Hamiltonian that
implements `DeltaEnergyInsert` to efficiently compute the change in energy.

The generic type names are:
* `B`: The [`Body::properties`](hoomd_microstate::Body) type.
* `S`: The [`Site::properties`](hoomd_microstate::Site) type.
* `C`: The [`boundary`](hoomd_microstate::boundary) condition type.

See the [Implementors](#implementors) section below for examples.
*/
pub trait DeltaEnergyInsert<B, S, C> {
    /** Compute the change in energy.

    `initial_microstate` describes the initial configuration and `new_body`
    describes the new body configuration. The final configuration includes
    all bodies in the initial microstate and `new_body`.

    Returns:
    ```math
    \Delta E = E_\mathrm{final} - E_\mathrm{initial}
    ```
    */
    #[must_use]
    fn delta_energy_insert(
        &self,
        initial_microstate: &Microstate<B, S, C>,
        new_body: &Body<B, S>,
    ) -> f64;
}

/** Compute the change energy when a single body is removed.

Some trial moves remove a single body at a time and use a Hamiltonian that
implements `DeltaEnergyRemove` to efficiently compute the change in energy.

The generic type names are:
* `B`: The [`Body::properties`](hoomd_microstate::Body) type.
* `S`: The [`Site::properties`](hoomd_microstate::Site) type.
* `C`: The [`boundary`](hoomd_microstate::boundary) condition type.

See the [Implementors](#implementors) section below for examples.
*/
pub trait DeltaEnergyRemove<B, S, C> {
    /** Compute the change in energy.

    `initial_microstate` describes the initial configuration and `body_index` is
    the index of the body to remove. The final configuration includes all bodies
    in the initial microstate except the body previously at `body_index`.

    Returns:
    ```math
    \Delta E = E_\mathrm{final} - E_\mathrm{initial}
    ```
    */
    #[must_use]
    fn delta_energy_remove(
        &self,
        initial_microstate: &Microstate<B, S, C>,
        body_index: usize,
    ) -> f64;
}

// TODO: More doc examples for all implementors.

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Apply the Metropolis Monte Carlo simulation method to systems of particles.

TODO: Expand documentation.
 */

use hoomd_microstate::{Body, Microstate, Tagged};
use rand::Rng;
use std::ops::AddAssign;

mod external;
mod sweep;
mod translate;

pub use sweep::Sweep;
pub use translate::Translate;

/** Propose trial moves in the microstate, evaluate the changes in energy and accept or reject accordingly.

`Trial` describes a type that applies trial moves to microstates. One call to
`apply` will attempt one or more individual trial moves to the microstate. It
evaluates the change in energy with the given `hamiltonian`, then accepts or
rejects the trial based on the `state` parameters.

Each type of trial move in *hoomd-rs* implements the `Trial` trait so that they
may be used as generic arguments in higher level functions.

See [`Sweep`] or any of the other implementations of `Trial` for examples.
*/
pub trait Trial<M, H> {
    /** Represent the number of accepted and rejected individual trial moves.

    Most implementations of `Trial` will use [`crate::Count`] directly. Some
    may provide more granular detail broken down by move type.
    */
    type Count;

    /// Represent the macrostate parameter(s) used when evaluating move acceptance.
    type Macrostate;

    /** Apply the trial move(s).

    A given type that implements `Trial` may perform one or many trial moves
    in a single call to `apply`. The returned value informs the caller how many
    trial moves were accepted and rejected (possibly broken down by type).
    */
    fn apply(&self, microstate: &mut M, hamiltonian: &H, state: &Self::Macrostate) -> Self::Count;
}

/** Propose a new configuration for given body properties.
*/
pub trait LocalTrial<B> {
    #[must_use]
    fn propose<R: Rng>(&self, rng: &mut R, body_properties: B) -> B;
}

/** Compute the change in energy after making changes to a single body in the microstate.
*/
pub trait DeltaEnergyOne<B, S, C> {
    #[must_use]
    fn delta_energy_one(
        &self,
        microstate: &Microstate<B, S, C>,
        new_body: &Tagged<Body<B, S>>,
    ) -> f64;
}

/** Set the energy of any system to 0.

[`Zero`] is useful for trivial code examples that demonstrate MC simulations.
It returns 0 for all delta energies.
*/
pub struct Zero;

impl<B, S, C> DeltaEnergyOne<B, S, C> for Zero {
    #[inline]
    fn delta_energy_one(
        &self,
        _microstate: &Microstate<B, S, C>,
        _new_body: &Tagged<Body<B, S>>,
    ) -> f64 {
        0.0
    }
}

/** Accepted and rejected trial moves.

A [`Trial`] reports the number moves it accepts and rejects via `Count`
(or some variation on `Count`). `Count` implements [`AddAssign`] and convenience
methods that compute often used properties, like the acceptance rate.

# Example

Count the total number of trial moves performed over a number of sweeps:

```
use hoomd_mc::{Count, Sweep, Translate, Trial, Zero};
use hoomd_microstate::property::Position;
use hoomd_microstate::{Body, Microstate};
use hoomd_vector::Cartesian;

let mut microstate = Microstate::new();
microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])));
let translate_sweep = Sweep::new(Translate::with_maximum_distance(0.1));

let mut count = Count::default();

for _ in 0..1_000 {
    count += translate_sweep.apply(&mut microstate, &Zero, &1.0);
    microstate.increment_step();
}

assert_eq!(count.total(), 1_000);
```
*/
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Count {
    /// The number of accepted moves.
    pub accepted: u64,
    /// The number of rejected moves.
    pub rejected: u64,
}

impl Count {
    /** The total number of trial moves.

    # Example

    ```
    use hoomd_mc::Count;
    
    let count = Count { accepted: 2_000, rejected: 8_000 };
    let total = count.total();

    assert_eq!(total, 10_000);
    ```
    */
    #[inline]
    #[must_use]
    pub fn total(&self) -> u64 {
        self.accepted + self.rejected
    }

    /** The fraction of moves that were accepted.

    The acceptance ratio is the ratio of accepted moves to total moves.
    `acceptance_ratio` returns `Some(ratio)` when the number of total moves is
    nonzero and `None` when there are 0 moves.

    # Example

    ```
    use hoomd_mc::Count;
    
    let count = Count { accepted: 2_000, rejected: 8_000 };
    let acceptance_ratio = count.acceptance_ratio();

    assert_eq!(acceptance_ratio, Some(0.2));
    ```

    ```
    use hoomd_mc::Count;
    
    let count = Count::default();
    let acceptance_ratio = count.acceptance_ratio();

    assert_eq!(acceptance_ratio, None);
    ```
    */
    #[inline]
    #[must_use]
    pub fn acceptance_ratio(&self) -> Option<f64> {
        let total = self.total();
        
        if total > 0 {
            Some(self.accepted as f64 / total as f64)
        } else {
            None
        }
    }
}

impl AddAssign for Count {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.accepted += rhs.accepted;
        self.rejected += rhs.rejected;
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement `CutoffPair`
*/

use crate::{SitePairEnergy, TotalEnergy};
use hoomd_microstate::{Microstate, property::Position};

use hoomd_vector::Vector;

/** Compute system properties given a [`SitePairEnergy`].

[`CutoffPair`] provides a single implementation for system properties, like
[`TotalEnergy`], for all types that implement [`SitePairEnergy`].

Use types that implement [`SitePairEnergy`], such as
[`Isotropic`](crate::pairwise::Isotropic) or your own custom type, directly
when you only need to call `site_pair_energy`. Combine these types with
[`CutoffPair`] to enable MC simulations or to compute the total energy of
a microstate.

TODO: Reword this when [`CutoffPair`] also implements `SitePairForce`.

[`CutoffPair`] sums properties over pairs that meet all of these conditions:
* separated by a distance less than `r_cut`.
* pairs that belong to different bodies.

# Example

Basic usage:
```
use hoomd_interaction::{CutoffPair,
    pairwise::{Isotropic, LennardJones}};

let lennard_jones: LennardJones = LennardJones { epsilon: 1.5, sigma: 2.0 };
let evaluator = Isotropic(lennard_jones);
let cutoff_pair = CutoffPair { r_cut: 2.5, evaluator };
```

Set a custom potential using a closure:
```
use hoomd_interaction::{CutoffPair, pairwise::Isotropic};

let cutoff_pair = CutoffPair {
    r_cut: 3.0,
    evaluator: Isotropic(|r: f64| 1.0 / (r.powi(12))),
};
```

Implement a custom potential via a type:
```
use hoomd_interaction::{CutoffPair, pairwise::{Isotropic, IsotropicEnergy}};

struct Custom {
    a: f64,
}

impl IsotropicEnergy for Custom {
    fn energy(&self, r: f64) -> f64 {
        self.a / r.powi(12)
    }
}

let custom = Custom { a: 2.0 };
let cutoff_pair = CutoffPair {
    r_cut: 2.0,
    evaluator: Isotropic(custom),
};
```
*/
pub struct CutoffPair<E> {
    /// The distance beyond which all pairwise interactions evaluate to 0.
    pub r_cut: f64,

    /// Computes the pairwise energies and forces.
    pub evaluator: E,
}

impl<V, B, S, C, E> TotalEnergy<Microstate<B, S, C>> for CutoffPair<E>
where
    E: SitePairEnergy<S>,
    S: Position<Vector = V>,
    V: Vector,
{
    /** Compute the total energy of the microstate contributed by functions on pairs of sites.

    ```math
    U_\mathrm{total} = \sum_{i=0}^{N-1}\sum_{j=i}^{N-1} U\left(\left|\vec{r_j} - \vec{r_i}\right|\right) \left[ \left|\vec{r_j} - \vec{r_i}\right| \lt r_\mathrm{cut} \right]\left[b_i \ne b_j\right]
    ```
    where `U(r)` is the potential computed by [`CutoffPair::evaluator`], `b_i`
    is the body tag that holds site *i*, and `[]` denotes the Iverson bracket.

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

    let lennard_jones: LennardJones = LennardJones { epsilon: 1.5,
        sigma: 1.0 / 2.0_f64.powf(1.0/6.0) };
    let evaluator = Isotropic(lennard_jones);
    let cutoff_pair = CutoffPair { r_cut: 2.5, evaluator };

    // The potential energy is set to 0 beyond r_cut when computed by `CutoffPair`.
    let total_energy = cutoff_pair.total_energy(&microstate);
    assert_eq!(total_energy, -3.0);

    // However, individual pairwise `site_pair_energy` evaluations are always computed.
    let a = &microstate.sites()[0].properties;
    let b = &microstate.sites()[2].properties;
    assert_eq!((*a.position() - *b.position()).norm(), 5.0);
    assert!(cutoff_pair.evaluator.site_pair_energy(a, b) < 0.0);
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn total_energy(&self, microstate: &Microstate<B, S, C>) -> f64 {
        let mut total = 0.0;
        for site_i in microstate.sites() {
            for site_j in microstate
                .iter_sites_near(site_i.properties.position(), self.r_cut)
                .filter(|s| site_i.site_tag < s.site_tag && site_i.body_tag != s.body_tag)
            {
                total += self
                    .evaluator
                    .site_pair_energy(&site_i.properties, &site_j.properties);
            }
        }

        total
    }
}

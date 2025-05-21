// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement `CutoffPair`
*/

use crate::{SitePairEnergy, TotalEnergy};
use hoomd_microstate::{Microstate, property::Position};

use hoomd_vector::Vector;

/** Compute system properties given a [`SitePairEnergy`]

[`CutoffPair`] provides a single implementation for system properties, like
[`TotalEnergy`], for all types that implement [`SitePairEnergy`].

Use types that implement [`SitePairEnergy`], such as one from
[`pairwise`](crate::pairwise) or your own custom type, directly when you only
need to call `site_pair_energy`. Wrap the type in `CutoffPair` to use it with MC
simulations or to compute the total energy.

TODO: Reword this when `CutoffPair` also implements `SitePairForce`.

[`CutoffPair`] sums properties over all:
* pairs that are separated by a distance less than `r_cut`.
* pairs that belong to different bodies.

# Example

Basic usage:
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

// The potential energy is set to 0 beyond r_cut by `CutoffPair`.
// Individual pairwise `site_pair_energy` evaluations are always computed.
let a = &microstate.sites()[0].properties;
let b = &microstate.sites()[2].properties;
assert_eq!((*a.position() - *b.position()).norm(), 5.0);
assert!(cutoff_pair.evaluator.site_pair_energy(a, b) < 0.0);
# Ok(())
# }
```

Set a one-off custom potential:
```
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

    <!-- U_\mathrm{total} = \sum_{i=0}^{N-1}\sum_{j=i}^{N-1} U\left(\left|\vec{r_j} - \vec{r_i}\right|\right) \left[ \left|\vec{r_j} - \vec{r_i}\right| \lt r_\mathrm{cut} \right]\left[b_i \ne b_j\right] -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><msub><mi>U</mi><mpadded lspace="0"><mi>total</mi></mpadded></msub><mo>=</mo><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>0</mn></mrow><mrow><mi>N</mi><mo>−</mo><mn>1</mn></mrow></munderover></mrow><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>j</mi><mo>=</mo><mi>i</mi></mrow><mrow><mi>N</mi><mo>−</mo><mn>1</mn></mrow></munderover></mrow><mi>U</mi><mrow><mo fence="true" form="prefix">(</mo><mrow><mo fence="true" form="prefix">|</mo><mover><msub><mi>r</mi><mi>j</mi></msub><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo>−</mo><mover><msub><mi>r</mi><mi>i</mi></msub><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo fence="true" form="postfix">|</mo></mrow><mo fence="true" form="postfix">)</mo></mrow><mrow><mo fence="true" form="prefix">[</mo><mrow><mo fence="true" form="prefix">|</mo><mover><msub><mi>r</mi><mi>j</mi></msub><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo>−</mo><mover><msub><mi>r</mi><mi>i</mi></msub><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo fence="true" form="postfix">|</mo></mrow><mo>&lt;</mo><msub><mi>r</mi><mpadded lspace="0"><mi>cut</mi></mpadded></msub><mo fence="true" form="postfix">]</mo></mrow><mrow><mo fence="true" form="prefix">[</mo><msub><mi>b</mi><mi>i</mi></msub><mo>≠</mo><msub><mi>b</mi><mi>j</mi></msub><mo fence="true" form="postfix">]</mo></mrow></mrow></math>
    where `U(r)` is the potential computed by [`CutoffPair::evaluator`], `b_i`
    is the body tag that holds site *i*, and `[]` denotes the Iverson bracket.
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

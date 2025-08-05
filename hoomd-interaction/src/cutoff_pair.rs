// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement `CutoffPair`
*/

use crate::{DeltaEnergyOne, SitePairEnergy, TotalEnergy};
use hoomd_microstate::{Body, Microstate, Transform, boundary::Boundary, property::Position};
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
    U_\mathrm{total} = \sum_{i=0}^{N-1}\sum_{j=i+1}^{N-1} U\left(s_i, s_j \right) \left[ \left|\vec{r}_j - \vec{r}_i\right| \lt r_\mathrm{cut} \right]\left[b_i \ne b_j\right]
    ```
    where $`U(s_i, s_j)`$ is the potential computed by [`CutoffPair::evaluator`],
    $`s_i`$ is the full set of site properties for site i, $`\vec{r}_i`$ is
    the position of site i, $`b_i`$ is the body tag that holds site *i*, and
    $`\left| \right|`$ denotes the Iverson bracket.

    # Example
    ```
    use hoomd_interaction::{CutoffPair, SitePairEnergy, TotalEnergy,
        pairwise::{Isotropic, LennardJones}};
    use hoomd_microstate::{Microstate, Body};
    use hoomd_microstate::property::{Point, Position};
    use hoomd_vector::{Cartesian, InnerProduct};

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

/** Evaluate the change in energy due to functions that act on single sites.

# Example

```
use hoomd_interaction::{CutoffPair, DeltaEnergyOne, pairwise::{Boxcar, Isotropic}};
use hoomd_microstate::{Microstate, Body, property::Point};
use hoomd_vector::Cartesian;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = Microstate::new();
microstate.extend_bodies([Body::point(Cartesian::from([0.0, 0.0])),
    Body::point(Cartesian::from([1.0, 0.0])),
])?;


let epsilon = 2.0;
let (left,right) = (0.0, 1.5);
let boxcar = Boxcar { epsilon, left, right };
let evaluator = Isotropic(boxcar);
let cutoff_pair = CutoffPair { r_cut: 1.5, evaluator };

let delta_energy = cutoff_pair.delta_energy_one(&microstate, 0,
    &Body::point([-1.0, 0.0].into()));
assert_eq!(delta_energy, -2.0);
# Ok(())
# }
```
*/
impl<V, B, S, C, E> DeltaEnergyOne<B, S, C> for CutoffPair<E>
where
    E: SitePairEnergy<S>,
    B: Transform<S>,
    S: Position<Vector = V>,
    C: Boundary<V, B, S>,
    V: Vector,
{
    #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
        let body_tag = initial_microstate.bodies()[body_index].tag;

        // CutoffPair cannot implement site_energy to centrally calculate the
        // energy of one site with the rest of the system because the resulting
        // TotalEnergy calculation would double count interactions. Therefore,
        // this (and other codes) that need to sum over specific pairs implement
        // the necessary loops and call `site_pair_energy` directly.
        let site_energy = |site_properties: &S| {
            initial_microstate
                .iter_sites_near(site_properties.position(), self.r_cut)
                .filter(|s| body_tag != s.body_tag)
                .fold(0.0, |total, site_j| {
                    total
                        + self
                            .evaluator
                            .site_pair_energy(site_properties, &site_j.properties)
                })
        };

        let mut energy_final = 0.0;
        for s in &final_body.sites {
            match initial_microstate
                .boundary()
                .wrap_site(final_body.properties.transform(s))
            {
                Err(_) => return f64::INFINITY,
                Ok(wrapped_site) => energy_final += site_energy(&wrapped_site),
            }
        }

        let energy_initial = initial_microstate
            .iter_body_sites(body_index)
            .fold(0.0, |total, site_i| total + site_energy(&site_i.properties));

        energy_final - energy_initial
    }
}

// TODO: implement site_pair_energy for CutoffPair. It needs to apply
// the r_cut and body exclusions first, then forward the call to the inner type.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        TotalEnergy,
        pairwise::{Isotropic, LennardJones},
    };
    use hoomd_microstate::boundary::Open;
    use hoomd_microstate::{MicrostateBuilder, boundary::Square, property::Point};
    use hoomd_vector::Cartesian;

    use ::approx::assert_relative_eq;
    use rand::{Rng, SeedableRng, distr::Uniform, rngs::StdRng};
    use rstest::*;
    use std::f64::consts::PI;

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

    mod delta_energy {
        use super::*;

        #[test]
        fn site_outside() {
            let square = Square {
                l: 4.0
                    .try_into()
                    .expect("hard-coded constant should be positive"),
            };

            let body = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
            };
            let mut final_body = body.clone();
            final_body.properties.position[0] = 1.0;

            let microstate = MicrostateBuilder::with_boundary(square)
                .bodies([body])
                .try_build()
                .expect("the hard-coded bodies should be in the boundary");

            let energy = CutoffPair {
                r_cut: 0.0,
                evaluator: Isotropic(|_r| 0.0),
            };

            assert_eq!(
                energy.delta_energy_one(&microstate, 0, &final_body),
                f64::INFINITY
            );
        }

        #[test]
        fn body_exclusion() {
            // Ensure that CutoffPair.delta_energy_one excludes pairs in the same body.
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
            let body_a_final = Body {
                properties: Point::new(Cartesian::from([-1.0, 0.0])),
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
            // Moving body 0 to the left results in a -2.0 energy difference.
            assert_eq!(
                cutoff_pair.delta_energy_one(&microstate, 0, &body_a_final),
                -2.0
            );
        }

        #[test]
        fn random_moves() {
            // Ensure that CutoffPair.delta_energy_one is consistent with TotalEnergy
            let body_template = Body {
                properties: Point::new(Cartesian::from([0.0, 0.0])),
                sites: [
                    Point::new(Cartesian::from([0.0, 1.0])),
                    Point::new(Cartesian::from([-1.0, 1.0])),
                    Point::new(Cartesian::from([-1.0, -1.0])),
                ]
                .into(),
            };
            let body_a = Body {
                properties: Point::new(Cartesian::from([3.0, 0.0])),
                sites: body_template.sites.clone(),
            };
            let body_b = body_template.clone();

            let microstate_initial = MicrostateBuilder::new()
                .bodies([body_a, body_b])
                .try_build()
                .expect("hard-coded bodies should be in the boundary");

            let mut microstate_final = microstate_initial.clone();
            let lennard_jones: LennardJones = LennardJones {
                epsilon: 1.5,
                sigma: 1.25,
            };
            let cutoff_pair = CutoffPair {
                r_cut: 5.0,
                evaluator: Isotropic(lennard_jones),
            };

            assert!(cutoff_pair.total_energy(&microstate_initial) != 0.0);

            // Use `LennardJones` for validation because it is a varies with r and
            // will therefore show some changes for any moves (unlike `BoxCar`).
            // However, we need to avoid numerical errors when two sites get
            // too close. Randomly move the 2nd particle around within a
            // well-defined space where there will be no such overlaps.
            let mut rng = StdRng::seed_from_u64(0);
            let r_distribution =
                Uniform::new(3.0, 6.0).expect("hard-coded constants should be valid");
            let theta_distribution =
                Uniform::new(0.0, 2.0 * PI).expect("hard-coded constants should be valid");

            let mut new_body = body_template.clone();
            for _ in 0..1024 {
                let r = rng.sample(r_distribution);
                let theta = rng.sample(theta_distribution);
                new_body.properties.position = [r * theta.cos(), r * theta.sin()].into();

                let delta_energy_one =
                    cutoff_pair.delta_energy_one(&microstate_initial, 0, &new_body);
                microstate_final
                    .update_body_properties(0, new_body.properties)
                    .expect("generated bodies should be inside open boundaries");
                let delta_energy_total = cutoff_pair.total_energy(&microstate_final)
                    - cutoff_pair.total_energy(&microstate_initial);

                assert_relative_eq!(delta_energy_one, delta_energy_total, epsilon = 1e-10);
            }
        }
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement `DeltaEnergy*` for `cutoff_pair` potentials.
*/

use super::DeltaEnergyOne;
use hoomd_interaction::{CutoffPair, SitePairEnergy};
use hoomd_microstate::{Body, Microstate, Transform, boundary::Boundary, property::Position};
use hoomd_vector::Vector;

/** Evaluate the change in energy due to functions that act on single sites.

# Example

```
use hoomd_interaction::{CutoffPair, pairwise::{Boxcar, Isotropic}};
use hoomd_mc::DeltaEnergyOne;
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
    C: Boundary<V>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_interaction::{
        TotalEnergy,
        pairwise::{Isotropic, LennardJones},
    };
    use hoomd_microstate::{MicrostateBuilder, boundary::Square, property::Point};
    use hoomd_vector::Cartesian;

    use ::approx::assert_relative_eq;
    use rand::{Rng, SeedableRng, distr::Uniform, rngs::StdRng};
    use std::f64::consts::PI;

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
        let r_distribution = Uniform::new(3.0, 6.0).expect("hard-coded constants should be valid");
        let theta_distribution =
            Uniform::new(0.0, 2.0 * PI).expect("hard-coded constants should be valid");

        let mut new_body = body_template.clone();
        for _ in 0..1024 {
            let r = rng.sample(r_distribution);
            let theta = rng.sample(theta_distribution);
            new_body.properties.position = [r * theta.cos(), r * theta.sin()].into();

            let delta_energy_one = cutoff_pair.delta_energy_one(&microstate_initial, 0, &new_body);
            microstate_final
                .update_body_properties(0, new_body.properties)
                .expect("generated bodies should be inside open boundaries");
            let delta_energy_total = cutoff_pair.total_energy(&microstate_final)
                - cutoff_pair.total_energy(&microstate_initial);

            assert_relative_eq!(delta_energy_one, delta_energy_total, epsilon = 1e-10);
        }
    }
}

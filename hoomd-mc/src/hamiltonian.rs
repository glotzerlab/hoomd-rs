// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement `DeltaEnergy*` for varying lengths of tuples.
*/

use super::DeltaEnergyOne;
use hoomd_microstate::{Body, Microstate};

/** Sum two delta energy terms.

# Example

```
use hoomd_interaction::{CutoffPair, Single, external::Linear, pairwise::{Boxcar, Isotropic}};
use hoomd_mc::DeltaEnergyOne;
use hoomd_microstate::{Microstate, Body, property::Point};
use hoomd_vector::Cartesian;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = Microstate::new();
microstate.extend_bodies([Body::point(Cartesian::from([0.0, 0.0])),
    Body::point(Cartesian::from([1.0, 0.0])),
])?;


let epsilon = 2.0;
let (a,b) = (0.0, 1.5);
let boxcar = Boxcar { epsilon, a, b };
let evaluator = Isotropic(boxcar);
let cutoff_pair = CutoffPair { r_cut: 1.5, evaluator };

let linear = Single(Linear{ alpha: 10.0,
plane_origin: Cartesian::default(),
plane_normal: [0.0, 1.0].try_into()? });

let hamiltonian = (cutoff_pair, linear);

let delta_energy = hamiltonian.delta_energy_one(&microstate, 0,
    &Body::point([-1.0, 0.0].into()));
assert_eq!(delta_energy, -2.0);
# Ok(())
# }
```
*/
impl<B, S, C, E1, E2> DeltaEnergyOne<B, S, C> for (E1, E2)
where
    E1: DeltaEnergyOne<B, S, C>,
    E2: DeltaEnergyOne<B, S, C>,
{
    #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
        let mut total = self
            .0
            .delta_energy_one(initial_microstate, body_index, final_body);
        if total != f64::INFINITY {
            total += self
                .1
                .delta_energy_one(initial_microstate, body_index, final_body);
        }
        total
    }
}

// TODO: Test

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement `*Energy` for varying lengths of tuples.
 */

use super::{DeltaEnergyInsert, DeltaEnergyOne, TotalEnergy};
use hoomd_microstate::{Body, Microstate};

/** Sum two delta energy terms.

# Example

```
use hoomd_interaction::{CutoffPair, DeltaEnergyOne, Single, external::Linear, pairwise::{Boxcar, Isotropic}};
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
let cutoff_pair = CutoffPair { r_cut: right, evaluator };

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

/** Sum two total energy terms.

# Example

```
use hoomd_interaction::{CutoffPair, Single, TotalEnergy,
    external::Linear,
    pairwise::{Boxcar, Isotropic}};
use hoomd_microstate::{Microstate, Body, property::Point};
use hoomd_vector::Cartesian;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = Microstate::new();
microstate.extend_bodies([
    Body::point(Cartesian::from([0.0, 4.0])),
    Body::point(Cartesian::from([1.0, 4.0])),
])?;


let epsilon = 2.0;
let (left,right) = (0.0, 1.5);
let boxcar = Boxcar { epsilon, left, right };
let evaluator = Isotropic(boxcar);
let cutoff_pair = CutoffPair { r_cut: right, evaluator };

let linear = Single(Linear{ alpha: 1.0,
    plane_origin: Cartesian::default(),
    plane_normal: [0.0, 1.0].try_into()? });

let hamiltonian = (cutoff_pair, linear);

let total_energy = hamiltonian.total_energy(&microstate);
assert_eq!(total_energy, 10.0);
# Ok(())
# }
```
*/
impl<M, E1, E2> TotalEnergy<M> for (E1, E2)
where
    E1: TotalEnergy<M>,
    E2: TotalEnergy<M>,
{
    #[inline]
    fn total_energy(&self, microstate: &M) -> f64 {
        let mut total = self.0.total_energy(microstate);
        if total != f64::INFINITY {
            total += self.1.total_energy(microstate);
        }
        total
    }
}

/** Sum two delta energy insert.

# Example

```
use hoomd_interaction::{CutoffPair, Single, DeltaEnergyInsert,
    external::Linear,
    pairwise::{Boxcar, Isotropic}};
use hoomd_microstate::{Microstate, Body, property::Point};
use hoomd_vector::Cartesian;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = Microstate::new();
microstate.extend_bodies([
    Body::point(Cartesian::from([0.0, 4.0])),
])?;

let epsilon = 2.0;
let (left,right) = (0.0, 1.5);
let boxcar = Boxcar { epsilon, left, right };
let evaluator = Isotropic(boxcar);
let cutoff_pair = CutoffPair { r_cut: right, evaluator };

let linear = Single(Linear{ alpha: 1.0,
    plane_origin: Cartesian::default(),
    plane_normal: [0.0, 1.0].try_into()? });

let hamiltonian = (cutoff_pair, linear);

let new_body = Body::point(Cartesian::from([1.0, 4.0]));
let delta_energy = hamiltonian.delta_energy_insert(&microstate, &new_body);
assert_eq!(delta_energy, 6.0);
# Ok(())
# }
```
*/
impl<B, S, C, E1, E2> DeltaEnergyInsert<B, S, C> for (E1, E2)
where
    E1: DeltaEnergyInsert<B, S, C>,
    E2: DeltaEnergyInsert<B, S, C>,
{
    #[inline]
    fn delta_energy_insert(
        &self,
        initial_microstate: &Microstate<B, S, C>,
        new_body: &Body<B, S>,
    ) -> f64 {
        let mut total = self.0.delta_energy_insert(initial_microstate, new_body);
        if total != f64::INFINITY {
            total += self.1.delta_energy_insert(initial_microstate, new_body);
        }
        total
    }
}

// FUTURE: Expand macros for 1,2,3,4,... types using macros. Add more unit tests at that time.

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement `TotalEnergy` for varying lengths of tuples.
*/

use super::TotalEnergy;

impl<M, E1, E2> TotalEnergy<M> for (E1, E2)
where
    E1: TotalEnergy<M>,
    E2: TotalEnergy<M>,
    {
    #[inline]
    fn total_energy(
        &self,
        microstate: &M) -> f64 {
        let mut total = self.0.total_energy(microstate);
        if total != f64::INFINITY {
            total += self.1.total_energy(microstate);
        }
        total
    }
}

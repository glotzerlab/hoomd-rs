// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement `OverlapPenalty`.
 */

use super::IsotropicEnergy;

/** Monotonically non-decreasing potential to move particles apart.

[`OverlapPenalty`] is specifically designed to work with the `QuickInsert`
and `QuickCompress` protocols to quickly prepare states with non-overlapping
particles. Combine it with (TODO: Implement) to compute an energy that penalizes
hard particle overlaps.

The potential has three regions:
```math
U(r) = \begin{cases}
\infty & r < -d_\mathrm{max} \\
\frac{1}{2} kr^2 + \varepsilon_\mathrm{shoulder} & r < 0 \\
0 & r \ge 0
\end{cases}
```
The first region describes a completely hard interaction when sites overlap
too far. This prevents `QuickInsert` from creating too much strain with an
insertion. The second part applies a harmonic potential that allows trial moves
to gradually resolve overlaps. In the third region, sites are allowed to move
freely when not overlapping. The shoulder potential prevents trial moves from
creating new overlaps.

# Example

```
use hoomd_interaction::pairwise::OverlapPenalty;

let overlap_penalty = OverlapPenalty::default();
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverlapPenalty {
    /// Spring stiffness $`[\mathrm{energy}] [\mathrm{length}]^{-2}`$.
    pub k: f64,

    /// The largest overlap distance to allow  $`[\mathrm{length}]`$.
    pub maximum_allowed_overlap: f64,

    /// Height of the potential as $`r`$ approaches 0 from the left $`[\mathrm{energy}]`$
    pub epsilon_shoulder: f64,
}

impl Default for OverlapPenalty {
    /** Default overlap penalty parameters.

    The default values are tuned for use with `QuickInsert` and `QuickCompress`
    applied to systems of spherical particles with diameter approximately 1.

    * $`k = 1000`$
    * $`d_\mathrm{max} = 0.2`$
    * $`\varepsilon_\mathrm{shoulder} = 100`$

    # Example

    ```
    use hoomd_interaction::pairwise::OverlapPenalty;

    let overlap_penalty = OverlapPenalty::default();
    ```
    */
    #[inline]
    fn default() -> Self {
        Self {
            k: 1000.0,
            maximum_allowed_overlap: 0.2,
            epsilon_shoulder: 100.0,
        }
    }
}

impl IsotropicEnergy for OverlapPenalty {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        match r {
            x if x < -self.maximum_allowed_overlap => f64::INFINITY,
            x if x < 0.0 => 0.5 * self.k * r.powi(2) + self.epsilon_shoulder,
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shoulder() {
        let k = 0.0;
        let epsilon_shoulder = 15.0;
        let maximum_allowed_overlap =  1.0;
    
        let overlap_penalty = OverlapPenalty { k, maximum_allowed_overlap, epsilon_shoulder };

        assert_eq!(overlap_penalty.energy(1.0), 0.0);
        assert_eq!(overlap_penalty.energy(0.0), 0.0);
        assert_eq!(overlap_penalty.energy(0.0f64.next_down()), epsilon_shoulder);
        assert_eq!(overlap_penalty.energy(-0.5), epsilon_shoulder);
    }

    #[test]
    fn core() {
        let k = 0.0;
        let epsilon_shoulder = 0.0;
        let maximum_allowed_overlap =  0.5;
    
        let overlap_penalty = OverlapPenalty { k, maximum_allowed_overlap, epsilon_shoulder };

        assert_eq!(overlap_penalty.energy(1.0), 0.0);
        assert_eq!(overlap_penalty.energy(0.0), 0.0);
        assert_eq!(overlap_penalty.energy(0.0f64.next_down()), 0.0);
        assert_eq!(overlap_penalty.energy(-0.5), 0.0);
        assert_eq!(overlap_penalty.energy((-0.5f64).next_down()), f64::INFINITY);
        assert_eq!(overlap_penalty.energy(-1.0), f64::INFINITY);
    }

    #[test]
    fn harmonic() {
        let k = 6.0;
        let epsilon_shoulder = 0.0;
        let maximum_allowed_overlap =  10.0;
    
        let overlap_penalty = OverlapPenalty { k, maximum_allowed_overlap, epsilon_shoulder };

        assert_eq!(overlap_penalty.energy(1.0), 0.0);
        assert_eq!(overlap_penalty.energy(0.0), 0.0);
        assert_eq!(overlap_penalty.energy(-0.125), 0.5 * k * 0.125f64.powi(2));
    }
}

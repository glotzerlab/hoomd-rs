// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement `AlwaysTrue`
 */

use crate::SitePairOverlap;

/** All site pairs overlap.

Use [`AlwaysTrue`] with [`CutoffPairOverlap`] to implement hard sphere overlap
checks. See [`CutoffPairOverlap`] for examples.

[`CutoffPairOverlap`]: crate::CutoffPairOverlap
*/
pub struct AlwaysTrue;

impl<S> SitePairOverlap<S> for AlwaysTrue {
    /// Return true.
    #[inline]
    fn site_pair_overlap(&self, _a: &S, _b: &S) -> bool {
        true
    }
}

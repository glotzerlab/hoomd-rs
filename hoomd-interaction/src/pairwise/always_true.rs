// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `AlwaysTrue`

use crate::SitePairOverlap;

/// All site pairs overlap (*not differentiable*).
///
/// Use [`AlwaysTrue`] with [`CutoffPairOverlap`] to implement hard sphere overlap
/// checks. See [`CutoffPairOverlap`] for examples.
///
/// [`CutoffPairOverlap`]: crate::CutoffPairOverlap
pub struct AlwaysTrue;

impl<S, V> SitePairOverlap<S, V> for AlwaysTrue {
    /// Return true.
    #[inline]
    fn site_pair_overlap(&self, _site_properties_i: &S, _site_properties_j: &S) -> bool {
        true
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `AllPairs`

use std::{fmt, hash::Hash};

use hoomd_utility::valid::PositiveReal;
use rustc_hash::FxHashSet;

use super::{PointUpdate, PointsNearBall, WithSearchRadius};

/// Check all pairs.
///
/// [`AllPairs`] is extremely slow when used with `CutoffPair`.
/// Prefer [`VecCell`] or [`HashCell`] when possible. When not possible,
/// TODO: Mention `PairwiseCutoffall`.
///
/// [`VecCell`]: crate::VecCell
/// [`HashCell`]: crate::HashCell
#[derive(Clone)]
pub struct AllPairs<K> {
    /// Store all keys currently in the spatial data.
    keys: FxHashSet<K>,
}

impl<K> Default for AllPairs<K>
where
    K: Copy + Eq + Hash,
{
    #[inline]
    fn default() -> Self {
        Self {
            keys: FxHashSet::default(),
        }
    }
}

impl<K> WithSearchRadius for AllPairs<K>
where
    K: Copy + Eq + Hash,
{
    #[inline]
    fn with_search_radius(_radius: PositiveReal) -> Self {
        Self::default()
    }
}

impl<P, K> PointUpdate<P, K> for AllPairs<K>
where
    K: Copy + Eq + Hash,
{
    #[inline]
    fn insert(&mut self, key: K, _position: P) {
        self.keys.insert(key);
    }

    #[inline]
    fn remove(&mut self, key: &K) {
        self.keys.remove(key);
    }

    #[inline]
    fn len(&self) -> usize {
        self.keys.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    #[inline]
    fn contains_key(&self, key: &K) -> bool {
        self.keys.contains(key)
    }

    #[inline]
    fn clear(&mut self) {
        self.keys.clear();
    }
}

impl<P, K> PointsNearBall<P, K> for AllPairs<K>
where
    K: Copy + Eq + Hash,
{
    #[inline]
    fn points_near_ball(&self, _position: &P, _radius: f64) -> impl Iterator<Item = K> {
        self.keys.iter().copied()
    }
}

impl<K> fmt::Display for AllPairs<K> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AllPairs")
    }
}

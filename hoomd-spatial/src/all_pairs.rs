// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::hash::Hash;

use rustc_hash::FxHashSet;

use super::{PointUpdate, PointsInBall};


/// Check all pairs.
#[derive(Clone)]
pub struct AllPairs<K> {
    keys: FxHashSet<K>
}

impl<K> Default for AllPairs<K> where
K: Copy + Eq + Hash
{
    fn default() -> Self {
        Self {
            keys: FxHashSet::default(),
        }
    }
}

impl<P, K> PointUpdate<P, K> for AllPairs<K> where
K: Copy + Eq + Hash
{
    #[inline]
    fn insert(&mut self, key: K, _position: P) {
        self.keys.insert(key);
    }

    #[inline]
    fn remove(&mut self, key: &K) {
        self.keys.remove(&key);
    }

    #[inline]
    fn clear(&mut self) {
        self.keys.clear()
    }
}

impl<P, K> PointsInBall<P, K> for AllPairs<K> where
K: Copy + Eq + Hash
{
    #[inline]
    fn points_potentially_in_ball(&self, _position: &P, _radius: f64) -> impl Iterator<Item=K> {
        self.keys.iter().copied()
    }
}

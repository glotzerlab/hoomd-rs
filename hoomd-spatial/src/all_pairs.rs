// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use super::{PointUpdate, PointsInBall};

/// Check all pairs.
///
/// [`AllPairs`] is a marker type that indicates there is no spatial data structure.
/// `Microstate` iterates over *all* sites when its generic type `X` is `AllPairs`.
#[derive(Clone)]
pub struct AllPairs;

impl<P, K> PointUpdate<P, K> for AllPairs
{
    #[inline]
    fn insert(&mut self, _key: K, _position: P) {
    }

    #[inline]
    fn remove(&mut self, _key: &K) {
    }

    #[inline]
    fn clear(&mut self) {
    }
}

impl<P, K> PointsInBall<P, K> for AllPairs
{
    #[inline]
    fn points_potentially_in_ball<I: Iterator<Item=K>>(&self, position: &P, radius: f64, all_keys: I) -> impl Iterator<Item=K> {
        all_keys
    }
}

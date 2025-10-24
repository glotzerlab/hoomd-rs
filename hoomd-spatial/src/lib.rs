// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

//! Spatial data structures.
//!
//! TODO: Document

// TODO: PostiveReal for cell width?

mod all_pairs;
mod hash_cell;
mod vec_cell;

pub use all_pairs::AllPairs;
pub use hash_cell::HashCell;
pub use vec_cell::VecCell;

pub trait PointUpdate<P, K> {
    /// Insert a point identified by a key.
    fn insert(&mut self, key: K, position: P);

    /// Remove the point with the given key.
    fn remove(&mut self, key: &K);

    /// Remove all points.
    fn clear(&mut self);
}

pub trait PointsInBall<P, K> {
    /// Find all the points that *may* be in the given ball.
    fn points_potentially_in_ball(&self, position: &P, radius: f64) -> impl Iterator<Item=K>;
    }

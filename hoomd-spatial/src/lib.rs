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
//! Use a spatial data structure to efficiently find points near a position in space.
//! `Microstate` maintains a spatial data structure internally for use with
//! pair potential computations. Specifically, `Microstate` interoperates with:
//! * [`VecCell`] is the highest performance spatial data structure. Use it in typical
//!   simulations where points remain inside a fixed boundary and have a roughly
//!   uniform density throughout.
//! * [`HashCell`] is only slightly slower than [`VecCell`]. [`HashCell`] will use less
//!   memory in simulations with open boundaries and/or when there are large regions
//!   of space with no points.
//! * [`AllPairs`] is very slow. Use it only when necessary.
//!
//! You can also utilize any of these data structures directly. They operate by
//! mapping keys (of a generic type) to points in space. Add points to a spatial
//! data structure with [`PointUpdate::insert`], which also updates the position of
//! already added points. Remove points with [`PointUpdate::remove`].
//!
//! [`PointsInBall::points_in_ball`] takes a position and a radius
//! and returns an iterator that will yield all of the inserted points within
//! the given ball. It *may* yield additional points that you need to filter out.

use hoomd_utility::valid::PositiveReal;

mod all_pairs;
mod hash_cell;
mod vec_cell;

pub use all_pairs::AllPairs;
pub use hash_cell::{HashCell, HashCellBuilder};
pub use vec_cell::{VecCell, VecCellBuilder};

/// Allow incremental updates to points in the spatial data.
pub trait PointUpdate<P, K> {
    /// Insert or update a point identified by a key.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::{PointUpdate, VecCell};
    ///
    /// let mut vec_cell = VecCell::default();
    ///
    /// vec_cell.insert(0, [1.25, 2.5].into());
    /// ```
    fn insert(&mut self, key: K, position: P);

    /// Remove the point with the given key.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::{PointUpdate, VecCell};
    ///
    /// let mut vec_cell = VecCell::default();
    /// vec_cell.insert(0, [1.25, 2.5].into());
    ///
    /// vec_cell.remove(&0)
    /// ```
    fn remove(&mut self, key: &K);

    /// Get the number of points in the spatial data structure.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::{PointUpdate, VecCell};
    ///
    /// let mut vec_cell = VecCell::default();
    /// vec_cell.insert(0, [1.25, 2.5].into());
    ///
    /// assert_eq!(vec_cell.len(), 1)
    /// ```
    fn len(&self) -> usize;

    /// Test if the spatial data structure is empty.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::{PointUpdate, VecCell};
    ///
    /// let mut vec_cell = VecCell::<usize, 2>::default();
    ///
    /// assert!(vec_cell.is_empty());
    /// ```
    fn is_empty(&self) -> bool;

    /// Test if the spatial data structure contains a key.
    /// ```
    /// use hoomd_spatial::{PointUpdate, VecCell};
    ///
    /// let mut vec_cell = VecCell::default();
    /// vec_cell.insert(0, [1.25, 2.5].into());
    ///
    /// assert!(vec_cell.contains_key(&0));
    /// ```
    fn contains_key(&self, key: &K) -> bool;

    /// Remove all points.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::{PointUpdate, VecCell};
    ///
    /// let mut vec_cell = VecCell::default();
    /// vec_cell.insert(0, [1.25, 2.5].into());
    ///
    /// vec_cell.clear();
    /// ```
    fn clear(&mut self);
}

/// Find all points in the given ball.
pub trait PointsInBall<P, K> {
    /// Find all the points that *may* be in the given ball.
    ///
    /// `points_in_ball` will iterate over all points in the given ball *and
    /// possibly others as well*. The spatial data structure may iterate over
    /// the points in any order.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::{PointUpdate, PointsInBall, VecCell};
    ///
    /// let mut vec_cell = VecCell::default();
    /// vec_cell.insert(0, [1.25, 0.0].into());
    /// vec_cell.insert(1, [3.25, 0.75].into());
    /// vec_cell.insert(2, [-10.0, 12.0].into());
    ///
    /// for key in vec_cell.points_in_ball(&[2.0, 0.0].into(), 1.0) {
    ///     println!("{key}");
    /// }
    /// ```
    /// Prints (in any order):
    /// ```text
    /// 0
    /// 1
    /// ```
    fn points_in_ball(&self, position: &P, radius: f64) -> impl Iterator<Item = K>;
}

/// Construct a spatial data structure capable of searching up to the given radius.
pub trait WithSearchRadius {
    /// Construct a spatial data structure that can search *at least* as far as the given radius.
    fn with_search_radius(radius: PositiveReal) -> Self;
}

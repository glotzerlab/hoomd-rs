// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the necessary conversions are necessary and have been checked"
)]
#![expect(
    clippy::cast_sign_loss,
    reason = "the necessary conversions are necessary and have been checked"
)]

//! Implement `VecCell`

use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::{array, cmp::Eq, fmt, f64::consts::PI, hash::Hash, iter, marker::PhantomData, mem};

use log::trace;
use rustc_hash::FxHashMap;

use hoomd_manifold::Spherical;
use hoomd_utility::valid::PositiveReal;

use super::{PointUpdate, PointsNearBall, WithSearchRadius};

use crate::{IndexFromPosition, hash_cell::CellIndex};

/// TODO: docs
#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SphericalVecCell<K, const D: usize>
where
    K: Eq + Hash,
{
    /// The Euclidean width of each cell.
    cell_width: PositiveReal,

    /// A map from cell indices to cell contents.
    keys_map: Vec<Vec<K>>,

    /// A map from particle indices to cell indices.
    cell_index: FxHashMap<K, CellIndex<D>>,

    /// The shape of `keys_map` is `(half_extent * 2 + 1).powi(D)`.
    half_extent: u32,

    /// Pre-computed stencils.
    #[serde_as(as = "Vec<Vec<[_; D]>>")]
    stencils: Vec<Vec<[i64; D]>>,
}

/// Iterate over cell indices in row major order.
struct CellIndexIterator<const D: usize> {
    /// The current cell index.
    cell_index: [i64; D],
    /// The half extent of the cube.
    half_extent: u32,
}

impl<const D: usize> CellIndexIterator<D> {
    /// Iterate over a cube.
    ///
    /// The cube extends from `[-half_extent, -half_extent, ..., -half_extent]` to
    /// `[half_extent, half_extent, ..., half_extent]`.
    fn cube(half_extent: u32) -> Self {
        let mut cell_index = [-i64::from(half_extent); D];
        cell_index[D - 1] -= 1;
        Self {
            cell_index,
            half_extent,
        }
    }

    /// Increment the cell index.
    #[inline]
    fn increment_cell_index(&mut self) -> Option<[i64; D]> {
        self.cell_index[D - 1] += 1;

        for i in (0..D).rev() {
            if self.cell_index[i] > self.half_extent.into() {
                if i == 0 {
                    return None;
                }

                self.cell_index[i] = -(i64::from(self.half_extent));
                self.cell_index[i - 1] += 1;
            }
        }

        Some(self.cell_index)
    }
}

impl<const D: usize> Iterator for CellIndexIterator<D> {
    type Item = [i64; D];

    fn next(&mut self) -> Option<Self::Item> {
        self.increment_cell_index()
    }
}

/// Generate the stencil for a given radius.
fn generate_stencil<const D: usize>(radius: u32) -> Vec<[i64; D]> {
    assert!(radius >= 1, "cell list must have a minimum radius of 1");

    let mut result = CellIndexIterator::cube(radius).collect::<Vec<_>>();
    result.sort_by(|a, b| {
        let r_a = a.iter().map(|x| x.pow(2));
        let r_b = b.iter().map(|x| x.pow(2));
        r_a.cmp(r_b)
    });
    result
}

/// Generate the stencils up to a given radius.
pub(crate) fn generate_all_stencils<const D: usize>(max_radius: u32) -> Vec<Vec<[i64; D]>> {
    assert!(max_radius >= 1, "cell list must have a minimum radius of 1");
    let mut result = Vec::new();

    // A cell list will never use radius=0, so stencil[i] stores the stencil needed for
    // radius i+1.
    for radius in 0..max_radius {
        result.push(generate_stencil(radius + 1));
    }
    result
}

/// TODO: docs
pub struct SphericalVecCellBuilder<K, const D: usize> {
    /// Most commonly used search radius, in Euclidean metric.
    nominal_search_radius: PositiveReal,

    /// Largest possible search radius, in Euclidean metric.
    maximum_search_radius: f64,

    /// Track the key type.
    phantom_key: PhantomData<K>,
}

impl<K, const D: usize> SphericalVecCellBuilder<K, D>
where
    K: Copy + Eq + Hash,
{
    /// TODO
    #[inline]
    #[must_use]
    pub fn nominal_search_radius(mut self, nominal_search_radius: PositiveReal) -> Self {
        self.nominal_search_radius = nominal_search_radius;
        self
    }

    /// TODO
    #[inline]
    #[must_use]
    pub fn maximum_search_radius(mut self, maximum_search_radius: f64) -> Self {
        self.maximum_search_radius = maximum_search_radius;
        self
    }

    /// TODO
    #[inline]
    #[must_use]
    pub fn build(self) -> SphericalVecCell<K, D> {
        let maximum_stencil_radius =
            (self.maximum_search_radius / self.nominal_search_radius.get()).ceil() as u32;
        let half_extent: u32 = 1;

        SphericalVecCell {
            cell_width: self.nominal_search_radius,
            keys_map: iter::repeat_n(Vec::new(), (half_extent * 2 + 1).pow(D as u32) as usize)
                .collect(),
            cell_index: FxHashMap::default(),
            half_extent,
            stencils: generate_all_stencils(maximum_stencil_radius.max(1)),
        }
    }
}

impl<K, const D: usize> Default for SphericalVecCell<K, D>
where
    K: Copy + Eq + Hash,
{
    /// TODO
    #[inline]
    fn default() -> Self {
        Self::builder().build()
    }
}

impl<K, const D: usize> WithSearchRadius for SphericalVecCell<K, D>
where
    K: Copy + Eq + Hash,
{
    /// TODO
    #[inline]
    fn with_search_radius(spherical_radius: PositiveReal) -> Self {
        let euclidean_radius = (spherical_radius.get().clamp(0.0, PI/2.0)).sin();
        Self::builder()
            .nominal_search_radius(
                PositiveReal::try_from(euclidean_radius)
                .expect("positive number given previous check")
            )
            .maximum_search_radius(euclidean_radius)
            .build()
    }
}

impl<K, const D: usize> SphericalVecCell<K, D>
where
    K: Eq + Hash,
{
    /// Compute the cell index given a `Spherical` position in space.
    #[inline]
    fn cell_index_from_position(&self, position: &Spherical<D>) -> [i64; D] {
        std::array::from_fn(|j| (position.coordinates()[j] / self.cell_width.get()).floor() as i64)
    }

    /// Compute the vector index from a cell index
    ///
    /// Returns `None` when the index is out of bounds.
    #[inline]
    fn map_index_from_cell(half_extent: u32, cell_index: &[i64; D]) -> Option<usize> {
        assert!(D > 1);

        let mut vec_index: usize = 0;
        let mut width = 1;

        for i in (0..D).rev() {
            let needed_extent = cell_index[i].unsigned_abs();
            if needed_extent > u64::from(half_extent) {
                return None;
            }
            let v: usize = (cell_index[i] + i64::from(half_extent))
                .try_into()
                .expect("cell index should be in bounds");

            vec_index += v * width;
            width *= (half_extent * 2 + 1) as usize;
        }
        Some(vec_index)
    }

    /// Get the keys in a given cell index
    #[inline]
    fn get_keys(&self, cell_index: &[i64; D]) -> &[K] {
        let index = Self::map_index_from_cell(self.half_extent, cell_index)
            .expect("cell_index should be in bounds");
        &self.keys_map[index]
    }
}

impl<K, const D: usize> SphericalVecCell<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Construct a `VecCell` builder.
    ///
    /// Use the builder to set any or all parameters and construct a [`VecCell`].
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_spatial::VecCell;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let vec_cell = VecCell::<usize, 3>::builder()
    ///     .nominal_search_radius(2.5.try_into()?)
    ///     .maximum_search_radius(7.5)
    ///     .build();
    /// # Ok(())
    /// # }
    /// ```
    #[expect(
        clippy::missing_panics_doc,
        reason = "hard-coded constant will never panic"
    )]
    #[inline]
    #[must_use]
    pub fn builder() -> SphericalVecCellBuilder<K, D> {
        SphericalVecCellBuilder {
            nominal_search_radius: 1.0
                .try_into()
                .expect("hard-coded constant is a positive real"),
            maximum_search_radius: 1.0,
            phantom_key: PhantomData,
        }
    }

    /// Remove excess capacity from dynamically allocated arrays.
    ///
    /// At this time, `shrink_to_fit` only reduces the memory utilized by the
    /// cell contents. It does not shrink the `D`-dimensional storage to match
    /// the range spanned by points currently in the data structure.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        for keys in &mut self.keys_map {
            keys.shrink_to_fit();
        }
        self.keys_map.shrink_to_fit();
        self.cell_index.shrink_to_fit();
    }

    /// Double the number of cells stored along each axis until it includes the target.
    fn expand_to(&mut self, target: u32) {
        if self.half_extent >= target {
            return;
        }

        let mut new_half_extent = self.half_extent.min(1) * 2;

        while new_half_extent < target {
            new_half_extent *= 2;
        }

        trace!("Expanding to {}^{} cells", new_half_extent * 2 + 1, D);

        let mut new_keys_map: Vec<Vec<K>> =
            iter::repeat_n(Vec::new(), (new_half_extent * 2 + 1).pow(D as u32) as usize).collect();
        let old_half_extent = self.half_extent;
        let old_keys_map = &mut self.keys_map;

        for old_cell_index in CellIndexIterator::cube(old_half_extent) {
            let old_vec_index = Self::map_index_from_cell(old_half_extent, &old_cell_index)
                .expect("cell_index should be consistent with keys_map");
            let new_vec_index = Self::map_index_from_cell(new_half_extent, &old_cell_index)
                .expect("old_cell_index should be inside the new keys_map");
            new_keys_map[new_vec_index] = mem::take(&mut old_keys_map[old_vec_index]);
        }

        self.half_extent = new_half_extent;
        self.keys_map = new_keys_map;
    }
}

impl<K, const D: usize> PointUpdate<Spherical<D>, K> for SphericalVecCell<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Insert or update a point identified by a key.
    ///
    /// TODO
    #[inline]
    fn insert(&mut self, key: K, position: Spherical<D>) {
        let cell_index = self.cell_index_from_position(&position);
        let old_cell_index = self.cell_index.insert(key, CellIndex(cell_index));
        let map_index =
            Self::map_index_from_cell(self.half_extent, &cell_index).unwrap_or_else(|| {
                let max_half_extent = cell_index
                    .iter()
                    .map(|x| x.unsigned_abs())
                    .reduce(u64::max)
                    .expect("D should be greater than 1");
                self.expand_to(
                    max_half_extent
                        .try_into()
                        .expect("max extent cannot exceed u32::MAX"),
                );
                Self::map_index_from_cell(self.half_extent, &cell_index)
                    .expect("cell_index should be in the expanded VecCell")
            });

        // This checks if old_cell_index is None or if it is different from the new cell index.
        if old_cell_index != Some(CellIndex(cell_index)) {
            // Add the particle index to the new cell index vector.
            self.keys_map[map_index].push(key);

            if let Some(old_cell_index) = old_cell_index {
                // If the particle was in a different cell, we need to remove it from the old cell.
                let old_map_index = Self::map_index_from_cell(self.half_extent, &old_cell_index.0)
                    .expect("cell_index and keys_map should agree");
                let old_keys = &mut self.keys_map[old_map_index];
                if let Some(pos) = old_keys.iter().position(|x| *x == key) {
                    old_keys.swap_remove(pos);
                }
            }
        }
    }

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
    #[inline]
    fn remove(&mut self, key: &K) {
        let cell_index = self.cell_index.remove(key);
        if let Some(cell_index) = cell_index {
            let map_index = Self::map_index_from_cell(self.half_extent, &cell_index.0);
            if let Some(map_index) = map_index {
                let keys = &mut self.keys_map[map_index];
                if let Some(idx) = keys.iter().position(|x| x == key) {
                    keys.swap_remove(idx);
                }
            }
        }
    }

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
    #[inline]
    fn len(&self) -> usize {
        self.cell_index.len()
    }

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
    #[inline]
    fn is_empty(&self) -> bool {
        self.cell_index.is_empty()
    }

    /// Test if the spatial data structure contains a key.
    /// ```
    /// use hoomd_spatial::{PointUpdate, VecCell};
    ///
    /// let mut vec_cell = VecCell::default();
    /// vec_cell.insert(0, [1.25, 2.5].into());
    ///
    /// assert!(vec_cell.contains_key(&0));
    /// ```
    #[inline]
    fn contains_key(&self, key: &K) -> bool {
        self.cell_index.contains_key(key)
    }

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
    #[inline]
    fn clear(&mut self) {
        self.cell_index.clear();
        for keys in &mut self.keys_map {
            keys.clear();
        }
    }
}

/// Iterate over keys in the cell list around a given center cell.
struct PointsIterator<'a, K, const D: usize>
where
    K: Eq + Hash,
{
    /// Keys of the current cell iteration (None if the cell is empty)
    keys: Option<&'a Vec<K>>,

    /// The cell list we are iterating in.
    cell_list: &'a SphericalVecCell<K, D>,

    /// Current location of the iteration in the cell.
    index_in_current_cell: usize,

    /// Current location of the iteration in the stencil.
    current_stencil: usize,

    /// Cell offsets to iterate over.
    stencil: &'a [[i64; D]],

    /// The cell at the center of the iteration.
    center: [i64; D],
}

impl<K, const D: usize> Iterator for PointsIterator<'_, K, D>
where
    K: Copy + Eq + Hash,
{
    type Item = K;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(keys) = self.keys
                && self.index_in_current_cell < keys.len()
            {
                let last_index = self.index_in_current_cell;
                self.index_in_current_cell += 1;
                return Some(keys[last_index]);
            }

            self.index_in_current_cell = 0;
            self.current_stencil += 1;

            if self.current_stencil >= self.stencil.len() {
                return None;
            }

            let cell_index =
                array::from_fn(|i| self.center[i] + self.stencil[self.current_stencil][i]);
            let map_index =
                SphericalVecCell::<K, D>::map_index_from_cell(self.cell_list.half_extent, &cell_index);
            self.keys = map_index.map(|index| &self.cell_list.keys_map[index]);
        }
    }
}

impl<const D: usize, K> PointsNearBall<Spherical<D>, K> for SphericalVecCell<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Find all the points that *might* be in the given ball.
    ///
    /// `points_near_ball` will iterate over all points in the given ball *and
    /// possibly others as well*. [`VecCell`] may iterate over the points in
    /// any order.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::{PointUpdate, PointsNearBall, VecCell};
    ///
    /// let mut vec_cell = VecCell::default();
    /// vec_cell.insert(0, [1.25, 0.0].into());
    /// vec_cell.insert(1, [3.25, 0.75].into());
    /// vec_cell.insert(2, [-10.0, 12.0].into());
    ///
    /// for key in vec_cell.points_near_ball(&[2.0, 0.0].into(), 1.0) {
    ///     println!("{key}");
    /// }
    /// ```
    /// Prints (in any order):
    /// ```text
    /// 0
    /// 1
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when `radius` is larger than the *maximum search radius*
    /// provided at construction, rounded up to the nearest integer multiple
    /// of the *nominal search radius*.
    #[inline]
    fn points_near_ball(&self, position: &Spherical<D>, radius: f64) -> impl Iterator<Item = K> {
        let stencil_index = (radius / self.cell_width.get()).ceil() as usize - 1;
        assert!(
            stencil_index < self.stencils.len(),
            "search radius must be less than or equal to the maximum search radius"
        );

        let center = self.cell_index_from_position(position);
        let stencil = &self.stencils[stencil_index];
        let map_index = Self::map_index_from_cell(
            self.half_extent,
            &array::from_fn(|i| center[i] + stencil[0][i]),
        );

        PointsIterator {
            keys: map_index.map(|index| &self.keys_map[index]),
            cell_list: self,
            index_in_current_cell: 0,
            current_stencil: 0,
            stencil,
            center,
        }
    }
}

impl<K, const D: usize> fmt::Display for SphericalVecCell<K, D>
where
    K: Eq + Hash,
{
    /// Summarize the contents of the cell list.
    ///
    /// This is a slow operation. It is meant to be printed to logs only
    /// occasionally, such as at the end of a benchmark or simulation.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_spatial::VecCell;
    /// use log::info;
    ///
    /// let vec_cell = VecCell::<usize, 3>::default();
    ///
    /// info!("{vec_cell}");
    /// ```
    #[allow(
        clippy::missing_inline_in_public_items,
        reason = "no need to inline display"
    )]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let largest_cell_size = self.keys_map.iter().map(Vec::len).fold(0, usize::max);

        writeln!(f, "VecCell<K, {D}>:")?;
        writeln!(
            f,
            "- {} total cells with {} cells on each side.",
            self.keys_map.len(),
            self.half_extent * 2 + 1
        )?;
        writeln!(f, "- {} points.", self.cell_index.len())?;
        writeln!(
            f,
            "- Nominal, maximum search radii: {}, {}",
            self.cell_width,
            self.cell_width.get() * self.stencils.len() as f64
        )?;
        write!(f, "- Largest cell length: {largest_cell_size}")
    }
}

impl<K, const D: usize> IndexFromPosition<Spherical<D>> for SphericalVecCell<K, D>
where
    K: Eq + Hash,
{
    type Location = [i64; D];

    #[inline]
    fn location_from_position(&self, position: &Spherical<D>) -> Self::Location {
        self.cell_index_from_position(position)
    }
}
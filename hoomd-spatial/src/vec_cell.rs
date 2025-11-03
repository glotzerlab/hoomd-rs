// Copyright (c) 2024-2025 The Regents of the University of Michigan.
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

use std::{array, cmp::Eq, fmt, hash::Hash, iter, marker::PhantomData, mem};

use log::trace;
use rustc_hash::FxHashMap;

use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Cartesian;

use super::{PointUpdate, PointsInBall, WithSearchRadius};

/// Bucket sort points into cubes with [`Vec`]-backed storage
///
/// [`VecCell`] is a spatial data structure that can efficiently update one
/// point at a time (see [`PointUpdate`]) and find all stored points that are
/// *potentially* near a position in space (see [`PointsInBall`]).
///
/// Points are stored in [`VecCell`] by key. The caller can choose a generic key
/// type `K` appropriately. For example, `K` could be `usize` and map to an
/// array index. Or it could be an enum that distinguishes between primary and
/// ghost site positions.
///
/// Internally, [`VecCell`] stores a `D`-dimensional vector of cells that
/// partition space. Each bucket is a hypercube with an edge length equal to
/// the *nominal search radius*. When searching for points near a position in
/// space, `VecCell` first determines which bucket that point is in. Then it
/// iterates over all the keys in all the cells within the given search radius.
/// To perform this iteration efficiently, [`VecCell`] pre-computes a set of
/// stencils up to the given *maximum search radius* given at const
///
/// In systems with low density fluctuations, the time complexity of [`VecCell`]
/// is O(1). More generally, and important when density fluctuations are
/// present, the time complexity scales with the maximum number of points in any
/// of the cells.
///
/// [`VecCell`] performs best when the search radius is equal to the nominal
/// search radius. In that case, 2D searches loop over 9 cells and 3D searches
/// loop over 27. The caller must filter the output of this iteration because
/// the resulting hypercube covers more volume than the given hypersphere.
/// When searching with a radius larger than the nominal, many more cells must
/// be iterated leading to slower performance. At the same time, the stencil
/// covers less excess volume and therefore fewer points must be filtered.
///
/// [`VecCell`] stores each cell at a unique index backed by storage in [`Vec`].
/// This performs very well in dense, compact systems. However, the memory
/// utilization scales with the *volume* of the cube that spans from the most
/// negative point to the most positive point. In dilute systems, [`VecCell`]
/// can therefore waste a lot of memory storing empty cells. Use [`HashCell`]
/// and occasionally call its `shrink_to_fit` method to limit the memory
/// required to model dilute systems.
///
/// [`HashCell`]: crate::HashCell
///
/// # Examples
///
/// The default [`VecCell`] set both the *nominal* and *maximum* search radii to 1.0.
/// ```
/// use hoomd_spatial::VecCell;
///
/// let vec_cell = VecCell::<usize, 3>::default();
/// ```
///
/// Use the builder API to set any or all parameters:
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
pub struct VecCell<K, const D: usize> {
    /// The width of each cell.
    cell_width: PositiveReal,

    /// A map from cell indices to cell contents.
    keys_map: Vec<Vec<K>>,

    /// A map from particle indices to cell indices.
    cell_index: FxHashMap<K, [i64; D]>,

    /// The shape of `keys_map` is `(half_extent * 2 + 1).powi(D)`.
    half_extent: u32,

    /// Pre-computed stencils.
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

/// Construct a [`VecCell`] with given parameters.
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
pub struct VecCellBuilder<K, const D: usize> {
    /// Most commonly used search radius.
    nominal_search_radius: PositiveReal,

    /// Largest possible search radius.
    maximum_search_radius: f64,

    /// Track the key type.
    phantom_key: PhantomData<K>,
}

impl<K, const D: usize> VecCellBuilder<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Choose the most commonly used search radius.
    ///
    /// [`VecCell`] performs the best when searching for points within the
    /// *nominal search radius* of a given position.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::VecCell;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let vec_cell = VecCell::<usize, 3>::builder()
    ///     .nominal_search_radius(2.5.try_into()?)
    ///     .build();
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn nominal_search_radius(mut self, nominal_search_radius: PositiveReal) -> Self {
        self.nominal_search_radius = nominal_search_radius;
        self
    }

    /// Choose the largest search radius.
    ///
    /// The maximum radius is rounded up to the nearest integer multiple of the
    /// *nominal search radius*. [`VecCell`] will panic when asked to search for
    /// points within a radius larger than the maximum.
    ///
    /// # Example
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
    #[inline]
    #[must_use]
    pub fn maximum_search_radius(mut self, maximum_search_radius: f64) -> Self {
        self.maximum_search_radius = maximum_search_radius;
        self
    }

    /// Construct the [`VecCell`] with the chosen parameters.
    ///
    /// # Example
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
    #[inline]
    #[must_use]
    pub fn build(self) -> VecCell<K, D> {
        let maximum_stencil_radius =
            (self.maximum_search_radius / self.nominal_search_radius.get()).ceil() as u32;
        let half_extent: u32 = 1;

        VecCell {
            cell_width: self.nominal_search_radius,
            keys_map: iter::repeat_n(Vec::new(), (half_extent * 2 + 1).pow(D as u32) as usize)
                .collect(),
            cell_index: FxHashMap::default(),
            half_extent,
            stencils: generate_all_stencils(maximum_stencil_radius.max(1)),
        }
    }
}

impl<K, const D: usize> Default for VecCell<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Construct a default [`VecCell`].
    ///
    /// The default sets both the *nominal* and *maximum* search radii to 1.0.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_spatial::VecCell;
    ///
    /// let vec_cell = VecCell::<usize, 3>::default();
    /// ```
    #[inline]
    fn default() -> Self {
        Self::builder().build()
    }
}

impl<K, const D: usize> WithSearchRadius for VecCell<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Construct a [`VecCell`] with the given search radius.
    ///
    /// Set both the *nominal* and *maximum* search radii to `radius`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_spatial::{VecCell, WithSearchRadius};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let vec_cell = VecCell::<usize, 3>::with_search_radius(2.5.try_into()?);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn with_search_radius(radius: PositiveReal) -> Self {
        Self::builder()
            .nominal_search_radius(radius)
            .maximum_search_radius(radius.get())
            .build()
    }
}

impl<K, const D: usize> VecCell<K, D> {
    /// Compute the cell index given a position in space.
    #[inline]
    fn cell_index_from_position(&self, position: &Cartesian<D>) -> [i64; D] {
        std::array::from_fn(|j| (position.coordinates[j] / self.cell_width.get()).floor() as i64)
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
    #[cfg(test)]
    #[inline]
    fn get_keys(&self, cell_index: &[i64; D]) -> &[K] {
        let index = Self::map_index_from_cell(self.half_extent, cell_index)
            .expect("cell_index should be in bounds");
        &self.keys_map[index]
    }
}

impl<K, const D: usize> VecCell<K, D>
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
    pub fn builder() -> VecCellBuilder<K, D> {
        VecCellBuilder {
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

impl<K, const D: usize> PointUpdate<Cartesian<D>, K> for VecCell<K, D>
where
    K: Copy + Eq + Hash,
{
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
    #[inline]
    fn insert(&mut self, key: K, position: Cartesian<D>) {
        let cell_index = self.cell_index_from_position(&position);
        let old_cell_index = self.cell_index.insert(key, cell_index);
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
        if old_cell_index != Some(cell_index) {
            // Add the particle index to the new cell index vector.
            self.keys_map[map_index].push(key);

            if let Some(old_cell_index) = old_cell_index {
                // If the particle was in a different cell, we need to remove it from the old cell.
                let old_map_index = Self::map_index_from_cell(self.half_extent, &old_cell_index)
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
            let map_index = Self::map_index_from_cell(self.half_extent, &cell_index);
            if let Some(map_index) = map_index {
                let keys = &mut self.keys_map[map_index];
                if let Some(idx) = keys.iter().position(|x| x == key) {
                    keys.swap_remove(idx);
                }
            }
        }
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
struct PointsIterator<'a, K, const D: usize> {
    /// Keys of the current cell iteration (None if the cell is empty)
    keys: Option<&'a Vec<K>>,

    /// The cell list we are iterating in.
    cell_list: &'a VecCell<K, D>,

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
    K: Copy,
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
                VecCell::<K, D>::map_index_from_cell(self.cell_list.half_extent, &cell_index);
            self.keys = map_index.map(|index| &self.cell_list.keys_map[index]);
        }
    }
}

impl<const D: usize, K> PointsInBall<Cartesian<D>, K> for VecCell<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Find all the points that *may* be in the given ball.
    ///
    /// `points_potentially_in_ball` will iterate over all points in the given ball.
    /// It may include any number of points inserted into the spatial data structure
    /// that are *not* in the ball. Filter the output as needed.
    ///
    /// [`VecCell`] may iterate over the points in any order.
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
    /// for key in vec_cell.points_potentially_in_ball(&[2.0, 0.0].into(), 1.0) {
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
    fn points_potentially_in_ball(
        &self,
        position: &Cartesian<D>,
        radius: f64,
    ) -> impl Iterator<Item = K> {
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

impl<K, const D: usize> fmt::Display for VecCell<K, D> {
    /// Summarize the contents of the cell list.
    ///
    /// This is a slow operation. It is meant to be printed to logs only
    /// occasionally, such as at the end of a benchmark or simulation.
    ///
    /// # Example
    ///
    /// ```
    /// use log::info;
    /// use hoomd_spatial::VecCell;
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
        writeln!(f, "- {} total cells with {} cells on each side.", self.keys_map.len(),self.half_extent*2 + 1)?;
        writeln!(f, "- {} points.", self.cell_index.len())?;
        writeln!(f, "- Nominal, maximum search radii: {}, {}", self.cell_width, self.cell_width.get() * self.stencils.len() as f64)?;
        write!(f, "- Largest cell length: {largest_cell_size}")
    }
}

#[expect(
    clippy::used_underscore_binding,
    reason = "Used for const parameterization."
)]
#[cfg(test)]
mod tests {
    use assert2::{assert, check};
    use rand::{
        Rng, SeedableRng,
        distr::{Distribution, Uniform},
        rngs::StdRng,
    };
    use rstest::*;

    use super::*;
    use hoomd_vector::{Metric, distribution::Ball};

    #[test]
    fn test_increment_cell_index() {
        let mut cells = CellIndexIterator::cube(1);
        check!(cells.next() == Some([-1, -1]));
        check!(cells.next() == Some([-1, 0]));
        check!(cells.next() == Some([-1, 1]));
        check!(cells.next() == Some([0, -1]));
        check!(cells.next() == Some([0, 0]));
        check!(cells.next() == Some([0, 1]));
        check!(cells.next() == Some([1, -1]));
        check!(cells.next() == Some([1, 0]));
        check!(cells.next() == Some([1, 1]));
        check!(cells.next() == None);
        check!(cells.next() == None);
        check!(cells.next() == None);
        check!(cells.next() == None);

        let mut c = CellIndexIterator {
            cell_index: [1, 2, 2],
            half_extent: 2,
        };
        check!(c.increment_cell_index() == Some([2, -2, -2]));
        let mut c = CellIndexIterator {
            cell_index: [0, 1, 2],
            half_extent: 2,
        };
        check!(c.increment_cell_index() == Some([0, 2, -2]));
        let mut c = CellIndexIterator {
            cell_index: [0, 0, -2],
            half_extent: 2,
        };
        check!(c.increment_cell_index() == Some([0, 0, -1]));
        let mut c = CellIndexIterator {
            cell_index: [2, 2, 2],
            half_extent: 2,
        };
        check!(c.increment_cell_index() == None);
    }

    #[test]
    fn test_cell_index() {
        let cell_list = VecCell::<usize, 3>::builder()
            .nominal_search_radius(
                2.0.try_into()
                    .expect("hard-coded constant should be positive"),
            )
            .build();
        check!(cell_list.cell_index_from_position(&[0.0, 0.0, 0.0].into()) == [0, 0, 0]);
        check!(cell_list.cell_index_from_position(&[2.0, 0.0, 0.0].into()) == [1, 0, 0]);
        check!(cell_list.cell_index_from_position(&[0.0, 2.0, 0.0].into()) == [0, 1, 0]);
        check!(cell_list.cell_index_from_position(&[0.0, 0.0, 2.0].into()) == [0, 0, 1]);
        check!(cell_list.cell_index_from_position(&[-41.5, 18.5, -0.125].into()) == [-21, 9, -1]);
    }

    #[test]
    fn test_insert_one() {
        let mut cell_list = VecCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));

        assert!(cell_list.cell_index.get(&0) == Some(&[0, 0]));

        let keys = cell_list.get_keys(&[0, 0]);
        assert!(keys.len() == 1);
        assert!(keys.contains(&0));
    }

    #[test]
    fn test_insert_many() {
        let mut cell_list = VecCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.995, 0.897]));
        cell_list.insert(2, Cartesian::from([-0.125, 3.25]));

        check!(cell_list.cell_index.get(&0) == Some(&[0, 0]));
        check!(cell_list.cell_index.get(&1) == Some(&[0, 0]));
        check!(cell_list.cell_index.get(&2) == Some(&[-1, 3]));

        let keys = cell_list.get_keys(&[0, 0]);
        assert!(keys.len() == 2);
        check!(keys.contains(&0));
        check!(keys.contains(&1));

        let keys = cell_list.get_keys(&[-1, 3]);
        assert!(keys.len() == 1);
        check!(keys.contains(&2));
    }

    #[test]
    fn test_insert_again_same() {
        let mut cell_list = VecCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(0, Cartesian::from([0.25, 0.5]));
        cell_list.insert(0, Cartesian::from([0.5, 0.75]));

        check!(cell_list.cell_index.get(&0) == Some(&[0, 0]));

        let keys = cell_list.get_keys(&[0, 0]);
        assert!(keys.len() == 1);
        check!(keys.contains(&0));
    }

    #[test]
    fn test_insert_again_different() {
        let mut cell_list = VecCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.25, 0.5]));
        cell_list.insert(1, Cartesian::from([-0.5, -0.75]));

        check!(cell_list.cell_index.get(&0) == Some(&[0, 0]));
        check!(cell_list.cell_index.get(&1) == Some(&[-1, -1]));

        let keys = cell_list.get_keys(&[0, 0]);
        assert!(keys.len() == 1);
        check!(keys.contains(&0));

        let keys = cell_list.get_keys(&[-1, -1]);
        assert!(keys.len() == 1);
        check!(keys.contains(&1));
    }

    #[test]
    fn test_remove() {
        let mut cell_list = VecCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.995, 0.897]));
        cell_list.insert(2, Cartesian::from([-0.125, 3.25]));

        cell_list.remove(&1);
        cell_list.remove(&2);

        check!(cell_list.cell_index.get(&0) == Some(&[0, 0]));
        check!(cell_list.cell_index.get(&1) == None);
        check!(cell_list.cell_index.get(&2) == None);

        let keys = cell_list.get_keys(&[0, 0]);
        assert!(keys.len() == 1);
        check!(keys.contains(&0));

        let keys = cell_list.get_keys(&[-1, 3]);
        check!(keys.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut cell_list = VecCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.995, 0.897]));
        cell_list.insert(2, Cartesian::from([-0.125, 3.25]));

        cell_list.clear();

        check!(cell_list.cell_index.is_empty());
        for keys in cell_list.keys_map {
            check!(keys.is_empty());
        }
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut cell_list = VecCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.995, 0.897]));
        cell_list.insert(2, Cartesian::from([-0.125, 3.25]));
        cell_list.insert(3, Cartesian::from([10.995, -12.897]));
        cell_list.insert(4, Cartesian::from([15.125, 19.25]));

        cell_list.remove(&1);
        cell_list.remove(&2);
        cell_list.remove(&3);
        cell_list.remove(&4);

        cell_list.shrink_to_fit();

        let filled_cell_index =
            VecCell::<usize, 2>::map_index_from_cell(cell_list.half_extent, &[0, 0])
                .expect("hard-coded cell is valid");
        for (i, keys) in cell_list.keys_map.iter().enumerate() {
            if i == filled_cell_index {
                check!(keys.capacity() == 1);
                check!(keys.len() == 1);
            } else {
                check!(keys.capacity() == 0);
                check!(keys.len() == 0);
            }
        }

        let keys = cell_list.get_keys(&[0, 0]);
        assert!(keys.len() == 1);
        check!(keys.contains(&0));
    }

    #[rstest]
    fn consistency() {
        const N_STEPS: usize = 65_536;
        let mut rng = StdRng::seed_from_u64(0);
        let mut reference = FxHashMap::default();

        let cell_width = 0.5;
        let mut cell_list = VecCell::builder()
            .nominal_search_radius(
                cell_width
                    .try_into()
                    .expect("hard-coded cell with should be positive"),
            )
            .build();
        let position_distribution = Ball {
            radius: 20.0.try_into().expect("hardcoded value should be positive"),
        };
        let key_distribution =
            Uniform::new(0, N_STEPS / 4).expect("hardcoded distribution should be valid");

        for _ in 0..N_STEPS {
            // Add more keys than removing
            if rng.random_bool(0.7) {
                let position: Cartesian<3> = position_distribution.sample(&mut rng);
                let key = key_distribution.sample(&mut rng);

                cell_list.insert(key, position);
                reference.insert(key, cell_list.cell_index_from_position(&position));
            } else {
                let key = key_distribution.sample(&mut rng);
                cell_list.remove(&key);
                reference.remove(&key);
            }
        }

        // Validate that cell_index contains the expected keys and that
        // keys_map is consistent.
        assert!(cell_list.cell_index.len() == reference.len());
        for (reference_key, reference_value) in reference.drain() {
            let value = cell_list.cell_index.get(&reference_key);
            check!(value == Some(&reference_value));

            let keys = cell_list.get_keys(&reference_value);
            check!(keys.contains(&reference_key));
        }

        // Ensure that there are no extra values in keys_map.
        let total = cell_list.keys_map.iter().map(Vec::len).sum();
        check!(cell_list.cell_index.len() == total);
        check!(total > 2000);
    }

    #[test]
    fn test_outside() {
        let mut cell_list = VecCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.995, 0.897]));
        cell_list.insert(2, Cartesian::from([8.125, 0.0]));

        check!(cell_list.half_extent == 8);
        let potential_neighbors: Vec<_> = cell_list
            .points_potentially_in_ball(&[9.125, 0.0].into(), 1.0)
            .collect();
        assert!(potential_neighbors.len() == 1);
        check!(potential_neighbors[0] == 2);
    }

    #[rstest]
    #[case::d_2(PhantomData::<VecCell<usize, 2>>)]
    #[case::d_3(PhantomData::<VecCell<usize, 3>>)]
    fn test_points_in_ball<const D: usize>(
        #[case] _d: PhantomData<VecCell<usize, D>>,
        #[values(1.0, 0.5, 0.25)] nominal_search_radius: f64,
    ) {
        let mut rng = StdRng::seed_from_u64(0);
        let mut reference = Vec::new();

        let mut cell_list = VecCell::builder()
            .nominal_search_radius(
                nominal_search_radius
                    .try_into()
                    .expect("hardcoded value should be positive"),
            )
            .maximum_search_radius(1.0)
            .build();
        let position_distribution = Ball {
            radius: 12.0.try_into().expect("hardcoded value should be positive"),
        };

        let n = 2048;

        for key in 0..n {
            let position: Cartesian<D> = position_distribution.sample(&mut rng);

            cell_list.insert(key, position);
            reference.push(position);
        }

        let mut n_neighbors = 0;
        for p_i in &reference {
            let potential_neighbors: Vec<_> =
                cell_list.points_potentially_in_ball(p_i, 1.0).collect();

            for (j, p_j) in reference.iter().enumerate() {
                if p_i.distance(p_j) <= 1.0 {
                    check!(potential_neighbors.contains(&j));
                    n_neighbors += 1;
                }
            }
        }
        check!(n_neighbors >= n * 2);
    }
}

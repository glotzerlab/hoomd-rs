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

//! Implement `HashCell`

use std::{array, cmp::Eq, fmt, hash::Hash, marker::PhantomData};

use rustc_hash::FxHashMap;

use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Cartesian;

use super::{PointUpdate, PointsNearBall, WithSearchRadius, vec_cell};

/// Bucket sort points into cubes with [`HashMap`]-backed storage
///
/// See [`VecCell`] for a complete description of the algorithm. Use [`VecCell`]
/// for dense, bounded collections of points. Use [`HashMap`] for sparse and/or
/// unbounded collections of points.
///
/// [`VecCell`]: crate::VecCell
/// [`HashMap`]: std::collections::HashMap
///
/// # Examples
///
/// The default [`HashCell`] set both the *nominal* and *maximum* search radii to 1.0.
/// ```
/// use hoomd_spatial::HashCell;
///
/// let hash_cell = HashCell::<usize, 3>::default();
/// ```
///
/// Use the builder API to set any or all parameters:
///
/// ```
/// use hoomd_spatial::HashCell;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let hash_cell = HashCell::<usize, 3>::builder()
///     .nominal_search_radius(2.5.try_into()?)
///     .maximum_search_radius(7.5)
///     .build();
/// # Ok(())
/// # }
/// ```
pub struct HashCell<K, const D: usize> {
    /// The width of each cell.
    cell_width: PositiveReal,

    /// A map from cell indices to cell contents.
    particle_indices: FxHashMap<[i64; D], Vec<K>>,

    /// A map from particle indices to cell indices.
    cell_index: FxHashMap<K, [i64; D]>,

    /// Pre-computed stencils.
    stencils: Vec<Vec<[i64; D]>>,
}

/// Construct a [`HashCell`] with given parameters.
///
/// # Example
///
/// ```
/// use hoomd_spatial::HashCell;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let hash_cell = HashCell::<usize, 3>::builder()
///     .nominal_search_radius(2.5.try_into()?)
///     .maximum_search_radius(7.5)
///     .build();
/// # Ok(())
/// # }
/// ```
pub struct HashCellBuilder<K, const D: usize> {
    /// Most commonly used search radius.
    nominal_search_radius: PositiveReal,

    /// Largest possible search radius.
    maximum_search_radius: f64,

    /// Track the key type.
    phantom_key: PhantomData<K>,
}

impl<K, const D: usize> HashCellBuilder<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Choose the most commonly used search radius.
    ///
    /// [`HashCell`] performs the best when searching for points within the
    /// *nominal search radius* of a given position.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::HashCell;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let hash_cell = HashCell::<usize, 3>::builder()
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
    /// *nominal search radius*. [`HashCell`] will panic when asked to search for
    /// points within a radius larger than the maximum.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::HashCell;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let hash_cell = HashCell::<usize, 3>::builder()
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

    /// Construct the [`HashCell`] with the chosen parameters.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::HashCell;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let hash_cell = HashCell::<usize, 3>::builder()
    ///     .nominal_search_radius(2.5.try_into()?)
    ///     .maximum_search_radius(7.5)
    ///     .build();
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn build(self) -> HashCell<K, D> {
        let maximum_stencil_radius =
            (self.maximum_search_radius / self.nominal_search_radius.get()).ceil() as u32;

        HashCell {
            cell_width: self.nominal_search_radius,
            particle_indices: FxHashMap::default(),
            cell_index: FxHashMap::default(),
            stencils: vec_cell::generate_all_stencils(maximum_stencil_radius.max(1)),
        }
    }
}

impl<K, const D: usize> Default for HashCell<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Construct a default [`HashCell`].
    ///
    /// The default sets both the *nominal* and *maximum* search radii to 1.0.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_spatial::HashCell;
    ///
    /// let hash_cell = HashCell::<usize, 3>::default();
    /// ```
    #[inline]
    fn default() -> Self {
        Self::builder().build()
    }
}

impl<K, const D: usize> WithSearchRadius for HashCell<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Construct a [`HashCell`] with the given search radius.
    ///
    /// Set both the *nominal* and *maximum* search radii to `radius`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_spatial::{HashCell, WithSearchRadius};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let hash_cell = HashCell::<usize, 3>::with_search_radius(2.5.try_into()?);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn with_search_radius(radius: PositiveReal) -> Self {
        Self::builder().nominal_search_radius(radius).build()
    }
}

impl<K, const D: usize> HashCell<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Compute the cell index given a position in space.
    #[inline]
    fn cell_index_from_position(&self, position: &Cartesian<D>) -> [i64; D] {
        std::array::from_fn(|j| (position.coordinates[j] / self.cell_width.get()).floor() as i64)
    }

    /// Remove excess capacity from dynamically allocated arrays.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.particle_indices.retain(|_, v| !v.is_empty());
        self.particle_indices.shrink_to_fit();
        self.cell_index.shrink_to_fit();
    }

    /// Construct a `HashCell` builder.
    ///
    /// Use the builder to set any or all parameters and construct a [`HashCell`].
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_spatial::HashCell;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let hash_cell = HashCell::<usize, 3>::builder()
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
    pub fn builder() -> HashCellBuilder<K, D> {
        HashCellBuilder {
            nominal_search_radius: 1.0
                .try_into()
                .expect("hard-coded constant is a positive real"),
            maximum_search_radius: 1.0,
            phantom_key: PhantomData,
        }
    }
}

impl<K, const D: usize> PointUpdate<Cartesian<D>, K> for HashCell<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Insert or update a point identified by a key.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::{HashCell, PointUpdate};
    ///
    /// let mut hash_cell = HashCell::default();
    ///
    /// hash_cell.insert(0, [1.25, 2.5].into());
    /// ```
    #[inline]
    fn insert(&mut self, key: K, position: Cartesian<D>) {
        let cell_idx = self.cell_index_from_position(&position);
        let old_cell_index = self.cell_index.insert(key, cell_idx);
        // This checks if old_cell_index is None or if it is different from the new cell index.
        if old_cell_index != Some(cell_idx) {
            // Add the particle index to the new cell index vector.
            self.particle_indices.entry(cell_idx).or_default().push(key);

            if let Some(old_cell_index) = old_cell_index {
                // If the particle was in a different cell, we need to remove it from the old cell.
                self.particle_indices
                    .entry(old_cell_index)
                    .and_modify(|particle_indices| {
                        if let Some(pos) = particle_indices.iter().position(|x| *x == key) {
                            particle_indices.swap_remove(pos);
                        }
                    });
            }
        }
    }

    /// Remove the point with the given key.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::{HashCell, PointUpdate};
    ///
    /// let mut hash_cell = HashCell::default();
    /// hash_cell.insert(0, [1.25, 2.5].into());
    ///
    /// hash_cell.remove(&0)
    /// ```
    #[inline]
    fn remove(&mut self, key: &K) {
        let cell_idx = self.cell_index.remove(key);
        if let Some(cell_idx) = cell_idx {
            // If the particle was found in the cell list, remove it from the particle indices.
            self.particle_indices
                .entry(cell_idx)
                .and_modify(|particle_indices| {
                    // Find the index of removed particle in the vector of particle indices.
                    if let Some(idx) = particle_indices.iter().position(|x| x == key) {
                        // Remove the particle index from the vector.
                        particle_indices.swap_remove(idx);
                    }
                });
        }
    }

    /// Get the number of points in the spatial data structure.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::{HashCell, PointUpdate};
    ///
    /// let mut hash_cell = HashCell::default();
    /// hash_cell.insert(0, [1.25, 2.5].into());
    ///
    /// assert_eq!(hash_cell.len(), 1)
    /// ```
    #[inline]
    fn len(&self) -> usize {
        self.cell_index.len()
    }

    /// Test if the spatial data structure is empty.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::{HashCell, PointUpdate};
    ///
    /// let mut hash_cell = HashCell::default();
    /// # hash_cell.insert(0, [1.25, 2.5].into());
    /// # hash_cell.remove(&0);
    ///
    /// assert!(hash_cell.is_empty());
    /// ```
    #[inline]
    fn is_empty(&self) -> bool {
        self.cell_index.is_empty()
    }

    /// Test if the spatial data structure contains a key.
    /// ```
    /// use hoomd_spatial::{HashCell, PointUpdate};
    ///
    /// let mut hash_cell = HashCell::default();
    /// hash_cell.insert(0, [1.25, 2.5].into());
    ///
    /// assert!(hash_cell.contains_key(&0));
    /// ```
    #[inline]
    fn contains_key(&self, key: &K) -> bool {
        self.cell_index.contains_key(key)
    }

    /// Remove all points.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::{HashCell, PointUpdate};
    ///
    /// let mut hash_cell = HashCell::default();
    /// hash_cell.insert(0, [1.25, 2.5].into());
    ///
    /// hash_cell.clear();
    /// ```
    #[inline]
    fn clear(&mut self) {
        self.cell_index.clear();
        self.particle_indices.clear();
    }
}

/// Iterate over keys in the cell list around a given center cell.
struct PointsIterator<'a, K, const D: usize> {
    /// Keys of the current cell iteration (None if the cell is empty)
    keys: Option<&'a Vec<K>>,

    /// The cell list we are iterating in.
    cell_list: &'a HashCell<K, D>,

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
            self.keys = self.cell_list.particle_indices.get(&cell_index);
        }
    }
}

impl<const D: usize, K> PointsNearBall<Cartesian<D>, K> for HashCell<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Find all the points that *might* be in the given ball.
    ///
    /// `points_near_ball` will iterate over all points in the given ball *and
    /// possibly others as well*. [`HashCell`] may iterate over the points in
    /// any order.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::{HashCell, PointUpdate, PointsNearBall};
    ///
    /// let mut hash_cell = HashCell::default();
    /// hash_cell.insert(0, [1.25, 0.0].into());
    /// hash_cell.insert(1, [3.25, 0.75].into());
    /// hash_cell.insert(2, [-10.0, 12.0].into());
    ///
    /// for key in hash_cell.points_near_ball(&[2.0, 0.0].into(), 1.0) {
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
    fn points_near_ball(&self, position: &Cartesian<D>, radius: f64) -> impl Iterator<Item = K> {
        let stencil_index = (radius / self.cell_width.get()).ceil() as usize - 1;
        assert!(
            stencil_index < self.stencils.len(),
            "search radius must be less than or equal to the maximum search radius"
        );

        let center = self.cell_index_from_position(position);
        let stencil = &self.stencils[stencil_index];

        PointsIterator {
            keys: self.particle_indices.get(&center),
            cell_list: self,
            index_in_current_cell: 0,
            current_stencil: 0,
            stencil,
            center,
        }
    }
}

impl<K, const D: usize> fmt::Display for HashCell<K, D> {
    /// Summarize the contents of the cell list.
    ///
    /// This is a slow operation. It is meant to be printed to logs only
    /// occasionally, such as at the end of a benchmark or simulation.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_spatial::HashCell;
    /// use log::info;
    ///
    /// let vec_cell = HashCell::<usize, 3>::default();
    ///
    /// info!("{vec_cell}");
    /// ```
    #[allow(
        clippy::missing_inline_in_public_items,
        reason = "no need to inline display"
    )]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let largest_cell_size = self
            .particle_indices
            .values()
            .map(Vec::len)
            .fold(0, usize::max);

        writeln!(f, "HashCell<K, {D}>:")?;
        writeln!(f, "- {} total cells.", self.particle_indices.len(),)?;
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
#[expect(
    clippy::used_underscore_binding,
    reason = "Used for const parameterization."
)]
#[cfg(test)]
mod tests {
    use assert2::{assert, check, let_assert};
    use rand::{
        Rng, SeedableRng,
        distr::{Distribution, Uniform},
        rngs::StdRng,
    };
    use rstest::*;

    use super::*;
    use hoomd_vector::{Metric, distribution::Ball};

    #[test]
    fn test_cell_index() {
        let cell_list = HashCell::<usize, 3>::builder()
            .nominal_search_radius(
                2.0.try_into()
                    .expect("hard-coded constant is a positive real"),
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
        let mut cell_list = HashCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));

        check!(cell_list.cell_index.get(&0) == Some(&[0, 0]));

        let keys = cell_list.particle_indices.get(&[0, 0]);
        let_assert!(Some(keys) = keys);
        check!(keys.len() == 1);
        check!(keys.contains(&0));
    }

    #[test]
    fn test_insert_many() {
        let mut cell_list = HashCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.995, 0.897]));
        cell_list.insert(2, Cartesian::from([-0.125, 3.25]));

        check!(cell_list.cell_index.get(&0) == Some(&[0, 0]));
        check!(cell_list.cell_index.get(&1) == Some(&[0, 0]));
        check!(cell_list.cell_index.get(&2) == Some(&[-1, 3]));

        let keys = cell_list.particle_indices.get(&[0, 0]);
        let_assert!(Some(keys) = keys);
        check!(keys.len() == 2);
        check!(keys.contains(&0));
        check!(keys.contains(&1));

        let keys = cell_list.particle_indices.get(&[-1, 3]);
        let_assert!(Some(keys) = keys);
        check!(keys.len() == 1);
        check!(keys.contains(&2));
    }

    #[test]
    fn test_insert_again_same() {
        let mut cell_list = HashCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(0, Cartesian::from([0.25, 0.5]));
        cell_list.insert(0, Cartesian::from([0.5, 0.75]));

        check!(cell_list.cell_index.get(&0) == Some(&[0, 0]));

        let keys = cell_list.particle_indices.get(&[0, 0]);
        let_assert!(Some(keys) = keys);
        check!(keys.len() == 1);
        check!(keys.contains(&0));
    }

    #[test]
    fn test_insert_again_different() {
        let mut cell_list = HashCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.25, 0.5]));
        cell_list.insert(1, Cartesian::from([-0.5, -0.75]));

        check!(cell_list.cell_index.get(&0) == Some(&[0, 0]));
        check!(cell_list.cell_index.get(&1) == Some(&[-1, -1]));

        let keys = cell_list.particle_indices.get(&[0, 0]);
        let_assert!(Some(keys) = keys);
        check!(keys.len() == 1);
        check!(keys.contains(&0));

        let keys = cell_list.particle_indices.get(&[-1, -1]);
        let_assert!(Some(keys) = keys);
        check!(keys.len() == 1);
        check!(keys.contains(&1));
    }

    #[test]
    fn test_remove() {
        let mut cell_list = HashCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.995, 0.897]));
        cell_list.insert(2, Cartesian::from([-0.125, 3.25]));

        cell_list.remove(&1);
        cell_list.remove(&2);

        check!(cell_list.cell_index.get(&0) == Some(&[0, 0]));
        check!(cell_list.cell_index.get(&1) == None);
        check!(cell_list.cell_index.get(&2) == None);

        let keys = cell_list.particle_indices.get(&[0, 0]);
        let_assert!(Some(keys) = keys);
        check!(keys.len() == 1);
        check!(keys.contains(&0));

        let keys = cell_list.particle_indices.get(&[-1, 3]);
        let_assert!(Some(keys) = keys);
        assert!(keys.len() == 0);
    }

    #[test]
    fn test_clear() {
        let mut cell_list = HashCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.995, 0.897]));
        cell_list.insert(2, Cartesian::from([-0.125, 3.25]));

        cell_list.clear();

        check!(cell_list.cell_index.len() == 0);
        check!(cell_list.particle_indices.len() == 0);
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut cell_list = HashCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.995, 0.897]));
        cell_list.insert(2, Cartesian::from([-0.125, 3.25]));

        cell_list.remove(&1);
        cell_list.remove(&2);

        cell_list.shrink_to_fit();
        check!(cell_list.particle_indices.len() == 1);

        let keys = cell_list.particle_indices.get(&[0, 0]);
        let_assert!(Some(keys) = keys);
        check!(keys.len() == 1);
        check!(keys.contains(&0));
    }

    #[test]
    fn consistency() {
        const N_STEPS: usize = 65_536;
        let mut rng = StdRng::seed_from_u64(0);
        let mut reference = FxHashMap::default();

        let cell_width = 0.5;
        let mut cell_list = HashCell::builder()
            .nominal_search_radius(
                cell_width
                    .try_into()
                    .expect("hard-coded value should be positive"),
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
        // particle_indices is consistent.
        assert!(cell_list.cell_index.len() == reference.len());
        for (reference_key, reference_value) in reference.drain() {
            let value = cell_list.cell_index.get(&reference_key);
            assert!(value == Some(&reference_value));

            let keys = cell_list.particle_indices.get(&reference_value);
            let_assert!(Some(keys) = keys);
            check!(keys.contains(&reference_key));
        }

        // Ensure that there are no extra values in particle_indices.
        let total = cell_list.particle_indices.values().map(Vec::len).sum();
        check!(cell_list.cell_index.len() == total);
        check!(total > 2000);
    }

    #[test]
    fn test_outside() {
        let mut cell_list = HashCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.995, 0.897]));
        cell_list.insert(2, Cartesian::from([8.125, 0.0]));

        let potential_neighbors: Vec<_> = cell_list
            .points_near_ball(&[9.125, 0.0].into(), 1.0)
            .collect();
        assert!(potential_neighbors.len() == 1);
        check!(potential_neighbors[0] == 2);
    }

    #[rstest]
    #[case::d_2(PhantomData::<HashCell<usize, 2>>)]
    #[case::d_3(PhantomData::<HashCell<usize, 3>>)]
    fn test_points_near_ball<const D: usize>(
        #[case] _d: PhantomData<HashCell<usize, D>>,
        #[values(1.0, 0.5, 0.25)] nominal_search_radius: f64,
    ) {
        let mut rng = StdRng::seed_from_u64(0);
        let mut reference = Vec::new();

        let cell_width = 1.0;
        let mut cell_list = HashCell::builder()
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
            let potential_neighbors: Vec<_> = cell_list.points_near_ball(p_i, cell_width).collect();

            for (j, p_j) in reference.iter().enumerate() {
                if p_i.distance(p_j) <= cell_width {
                    check!(potential_neighbors.contains(&j));
                    n_neighbors += 1;
                }
            }
        }
        check!(n_neighbors >= n * 2);
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::{array, iter, mem, cmp::Eq, hash::Hash};

use log::trace;
use rustc_hash::FxHashMap;

use hoomd_vector::Cartesian;

use super::{PointUpdate, PointsInBall};

pub struct VecCell<K, const D: usize> {
    /// The width of each cell.
    cell_width: f64,
    /// A map from cell indices to cell contents.
    keys_map: Vec<Vec<K>>,
    /// A map from particle indices to cell indices.
    cell_index: FxHashMap<K, [i64; D]>,
    /// Location of the 0,..,0 cell.
    origin: Cartesian<D>,
    /// The shape of `keys_map` is `(half_extent * 2 + 1).powi(D)`.
    half_extent: u32,
}

/// Increment a cell index.
///
/// Counts from `[-half_extent, -half_extent, ..., -half_extent]` to
/// `[half_extent, half_extent, ..., half_extent]`. Returns `None` when
/// the increment would count past the end point.
#[inline]
fn increment_cell_index<const D: usize>(mut cell_index: [i64; D], half_extent: u32) -> Option<[i64; D]> {
    cell_index[D-1] += 1;

    for i in (0..D).rev() {
        if cell_index[i] > half_extent.into() {
            if i == 0 {
                return None;
            }

            cell_index[i] = -(i64::from(half_extent));
            cell_index[i-1] += 1;
        }
    }

    Some(cell_index)
}

impl<K, const D: usize> VecCell<K, D>
{
    /// Compute the cell index given a position in space.
    #[inline]
    fn cell_index_from_position(&self, position: &Cartesian<D>) -> [i64; D] {
        let v = *position - self.origin;
        std::array::from_fn(|j| (v.coordinates[j] / self.cell_width).floor() as i64)
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
            let v: usize = (cell_index[i] + i64::from(half_extent)).try_into().expect("cell index should be in bounds");
        
            vec_index += v * width;
            width *= (half_extent * 2 + 1) as usize;
        }
        Some(vec_index)
    }

    /// Get the keys in a given cell index
    #[inline]
    fn get_keys(&self, cell_index: &[i64; D]) -> &[K] {
        let index = Self::map_index_from_cell(self.half_extent, cell_index).expect("cell_index should be in bounds");
        &self.keys_map[index]
    }
}

impl<K, const D: usize> VecCell<K, D> where
K: Copy + Eq + Hash
    {
    #[inline]
    #[must_use]
    pub fn new(cell_width: f64, half_extent: u32) -> Self {
        VecCell {
            cell_width,
            keys_map: iter::repeat_n(Vec::new(), (half_extent * 2 + 1).pow(D as u32) as usize).collect(),
            cell_index: FxHashMap::default(),
            origin: Cartesian::default(),
            half_extent,
        }
    }

    // #[inline]
    // #[must_use]
    // pub fn with_cell_width_and_origin(cell_width: f64, origin: Cartesian<D>) -> Self {
    //     HashCell {
    //         cell_width,
    //         particle_indices: FxHashMap::default(),
    //         cell_index: FxHashMap::default(),
    //         origin,
    //     }
    // }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        for keys in &mut self.keys_map {
            keys.shrink_to_fit();
        }
        self.keys_map.shrink_to_fit();
        self.cell_index.shrink_to_fit();
    }

    /// Double the number of cells stored along each axis until it includes the target.
    #[inline]
    fn expand_to(&mut self, target: u32) {
        if self.half_extent >= target {
            return;
        }

        let mut new_half_extent = self.half_extent * 2;

        while new_half_extent < target {
            new_half_extent *= 2;
        }

        trace!("Expanding to {}^{} cells", new_half_extent*2+1, D);

        let mut new_keys_map: Vec<Vec<K>> = iter::repeat_n(Vec::new(), (new_half_extent*2 + 1).pow(D as u32) as usize).collect();
        let old_half_extent = self.half_extent;
        let old_keys_map = &mut self.keys_map;

        let mut old_cell_index = [-i64::from(old_half_extent); D];
        loop {
            let old_vec_index = Self::map_index_from_cell(old_half_extent, &old_cell_index)
                .expect("cell_index should be consistent with keys_map");
            let new_vec_index = Self::map_index_from_cell(new_half_extent, &old_cell_index)
                .expect("old_cell_index should be inside the new keys_map");
            new_keys_map[new_vec_index] = mem::take(&mut old_keys_map[old_vec_index]);
            
            old_cell_index = match increment_cell_index(old_cell_index, old_half_extent) {
                Some(index) => index,
                None => { break }
            };
        }

        self.half_extent = new_half_extent;
        self.keys_map = new_keys_map;
    }
}

impl<K, const D: usize> PointUpdate<Cartesian<D>, K> for VecCell<K, D> where
K: Copy + Eq + Hash {
    #[inline]
    fn insert(&mut self, key: K, position: Cartesian<D>) {
        let cell_index = self.cell_index_from_position(&position);
        let old_cell_index = self.cell_index.insert(key, cell_index);
        let map_index = Self::map_index_from_cell(self.half_extent, &cell_index).unwrap_or_else(|| {
            let max_half_extent = cell_index.iter().map(|x| x.unsigned_abs()).reduce(u64::max).expect("D should be greater than 1");
            self.expand_to(max_half_extent.try_into().expect("max extent cannot exceed u32::MAX"));
            Self::map_index_from_cell(self.half_extent, &cell_index)
                .expect("cell_index should be in the expanded VecCell")
            });

        // This checks if old_cell_index is None or if it is different from the new cell index.
        if old_cell_index != Some(cell_index) {
            // Add the particle index to the new cell index vector.
            self.keys_map[map_index]
                .push(key);

            if let Some(old_cell_index) = old_cell_index {
                // If the particle was in a different cell, we need to remove it from the old cell.
                let old_map_index = Self::map_index_from_cell(self.half_extent, &old_cell_index).expect("cell_index and keys_map should agree");
                let old_keys = &mut self.keys_map[old_map_index];
                if let Some(pos) = old_keys.iter().position(|x| *x == key) {
                    old_keys.swap_remove(pos);
                }
            }
        }
    }

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

    #[inline]
    fn clear(&mut self) {
        self.cell_index.clear();
        for keys in &mut self.keys_map {
            keys.clear();
        }
    }
}

struct PointsIterator<'a, K, const D: usize> {
    keys: Option<&'a Vec<K>>,
    cell_list: &'a VecCell<K, D>,
    index_in_current_cell: usize,
    current_stencil: usize,
    stencil: &'a [[i64; D]],
    center: [i64; D],
    }

impl<'a, K, const D: usize> Iterator for PointsIterator<'a, K, D> {
    type Item=&'a K;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(keys) = self.keys && self.index_in_current_cell < keys.len() {
                let last_index = self.index_in_current_cell;
                self.index_in_current_cell += 1;
                return Some(&keys[last_index]);
            }

            self.index_in_current_cell = 0;
            self.current_stencil += 1;
            
            if self.current_stencil >= self.stencil.len() {
                return None;
            }

            let cell_index = array::from_fn(|i| self.center[i] + self.stencil[self.current_stencil][i]);
            let map_index = VecCell::<K,D>::map_index_from_cell(self.cell_list.half_extent, &cell_index);
            self.keys = map_index.map(|index| &self.cell_list.keys_map[index]);
        }
    }
}

        const STENCIL_2D: [[i64; 2]; 9] = [[0, 0],
                       [0, -1],
                       [0, 1],
                       [-1, 0],
                       [1, 0],
                       [-1, -1],
                       [-1, 1],
                       [1, 1],
                       [1, -1]];

        const STENCIL_3D: [[i64; 3]; 27] = [[0, 0, 0],
                       [0, 0, -1],
                       [0, 0, 1],
                       [0, -1, 0],
                       [0, 1, 0],
                       [-1, 0, 0],
                       [1, 0, 0],
                       [0, -1, -1],
                       [0, 1, -1],
                       [0, -1, 1],
                       [0, 1, 1],
                       [-1, -1, 0],
                       [-1, 1, 0],
                       [1, -1, 0],
                       [1, 1, 0],
                       [-1, 0, -1],
                       [-1, 0, 1],
                       [1, 0, -1],
                       [1, 0, 1],
                       [-1, -1, -1],
                       [-1, -1, 1],
                       [-1, 1, -1],
                       [-1, 1, 1],
                       [1, -1, -1],
                       [1, -1, 1],
                       [1, 1, -1],
                       [1, 1, 1],
                    ];

impl<K> PointsInBall<Cartesian<2>, K> for VecCell<K, 2> where
K: Copy + Eq + Hash
{
    #[inline]
    fn points_potentially_in_ball<'a>(&'a self, position: &Cartesian<2>, radius: f64) -> impl Iterator<Item=&'a K> where K: 'a {
        assert!(radius <= self.cell_width, "search radius must be less than or equal to the cell width");
        let center = self.cell_index_from_position(position);
        let map_index = Self::map_index_from_cell(self.half_extent,
            &array::from_fn(|i| center[i] + STENCIL_2D[0][i]));

        PointsIterator {
            keys: map_index.map(|index| &self.keys_map[index]),
            cell_list: self,
            index_in_current_cell: 0,
            current_stencil: 0,
            stencil: &STENCIL_2D,
            center,
        }
    }
}

impl<K> PointsInBall<Cartesian<3>, K> for VecCell<K, 3> where
K: Copy + Eq + Hash
{
    #[inline]
    fn points_potentially_in_ball<'a>(&'a self, position: &Cartesian<3>, radius: f64) -> impl Iterator<Item=&'a K> where K: 'a {
        assert!(radius <= self.cell_width, "search radius must be less than or equal to the cell width");
        let center = self.cell_index_from_position(position);
        let map_index = Self::map_index_from_cell(self.half_extent,
            &array::from_fn(|i| center[i] + STENCIL_3D[0][i]));

        PointsIterator {
            keys: map_index.map(|index| &self.keys_map[index]),
            cell_list: self,
            index_in_current_cell: 0,
            current_stencil: 0,
            stencil: &STENCIL_3D,
            center,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, distr::{Distribution, Uniform}, rngs::StdRng};
    use hoomd_vector::{distribution::Ball, Metric};

    #[test]
    fn test_increment_cell_index() {
        let cell_index = [-1, -1];
        let cell_index = increment_cell_index(cell_index, 1);
        assert_eq!(cell_index, Some([-1, 0]));
        let cell_index = increment_cell_index(cell_index.unwrap(), 1);
        assert_eq!(cell_index, Some([-1, 1]));
        let cell_index = increment_cell_index(cell_index.unwrap(), 1);
        assert_eq!(cell_index, Some([0, -1]));
        let cell_index = increment_cell_index(cell_index.unwrap(), 1);
        assert_eq!(cell_index, Some([0, 0]));
        let cell_index = increment_cell_index(cell_index.unwrap(), 1);
        assert_eq!(cell_index, Some([0, 1]));
        let cell_index = increment_cell_index(cell_index.unwrap(), 1);
        assert_eq!(cell_index, Some([1, -1]));
        let cell_index = increment_cell_index(cell_index.unwrap(), 1);
        assert_eq!(cell_index, Some([1, 0]));
        let cell_index = increment_cell_index(cell_index.unwrap(), 1);
        assert_eq!(cell_index, Some([1, 1]));
        assert_eq!(increment_cell_index(cell_index.unwrap(), 1), None);

        assert_eq!(increment_cell_index([1, 2, 2], 2), Some([2, -2, -2]));
        assert_eq!(increment_cell_index([0, 1, 2], 2), Some([0, 2, -2]));
        assert_eq!(increment_cell_index([0, 0, -2], 2), Some([0, 0, -1]));
        assert_eq!(increment_cell_index([2, 2, 2], 2), None);
    }

    // #[test]
    // fn test_cell_index() {
    //     let cell_list = HashCell::<usize, 3>::with_cell_width(2.0);
    //     assert_eq!(cell_list.cell_index_from_position(&[0.0, 0.0, 0.0].into()), [0, 0, 0]);
    //     assert_eq!(cell_list.cell_index_from_position(&[2.0, 0.0, 0.0].into()), [1, 0, 0]);
    //     assert_eq!(cell_list.cell_index_from_position(&[0.0, 2.0, 0.0].into()), [0, 1, 0]);
    //     assert_eq!(cell_list.cell_index_from_position(&[0.0, 0.0, 2.0].into()), [0, 0, 1]);
    //     assert_eq!(cell_list.cell_index_from_position(&[-41.5, 18.5, -0.125].into()), [-21, 9, -1]);

    //     let cell_list = HashCell::<usize, 3>::with_cell_width_and_origin(2.0, [-4.0, 2.0, 8.0].into());
    //     assert_eq!(cell_list.cell_index_from_position(&[0.0, 0.0, 0.0].into()), [2, -1, -4]);
    //     assert_eq!(cell_list.cell_index_from_position(&[2.0, 0.0, 0.0].into()), [3, -1, -4]);
    //     assert_eq!(cell_list.cell_index_from_position(&[0.0, 2.0, 0.0].into()), [2, 0, -4]);
    //     assert_eq!(cell_list.cell_index_from_position(&[0.0, 0.0, 2.0].into()), [2, -1, -3]);
    //     assert_eq!(cell_list.cell_index_from_position(&[-41.5, 18.5, -0.125].into()), [-19, 8, -5]);
    // }
    
    #[test]
    fn test_insert_one() {
        let cell_width = 1.0;
        let mut cell_list = VecCell::new(cell_width, 10);

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));

        assert_eq!(cell_list.cell_index.get(&0), Some(&[0, 0]));

        let keys = cell_list.get_keys(&[0, 0]);
        assert_eq!(keys.len(), 1);
        assert!(keys.contains(&0));
    }

    #[test]
    fn test_insert_many() {
        let cell_width = 1.0;
        let mut cell_list = VecCell::new(cell_width, 10);

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.995, 0.897]));
        cell_list.insert(2, Cartesian::from([-0.125, 3.25]));

        assert_eq!(cell_list.cell_index.get(&0), Some(&[0, 0]));
        assert_eq!(cell_list.cell_index.get(&1), Some(&[0, 0]));
        assert_eq!(cell_list.cell_index.get(&2), Some(&[-1, 3]));

        let keys = cell_list.get_keys(&[0, 0]);
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&0));
        assert!(keys.contains(&1));

        let keys = cell_list.get_keys(&[-1, 3]);
        assert_eq!(keys.len(), 1);
        assert!(keys.contains(&2));
    }

    #[test]
    fn test_insert_again_same() {
        let cell_width = 1.0;
        let mut cell_list = VecCell::new(cell_width, 10);

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(0, Cartesian::from([0.25, 0.5]));
        cell_list.insert(0, Cartesian::from([0.5, 0.75]));

        assert_eq!(cell_list.cell_index.get(&0), Some(&[0, 0]));

        let keys = cell_list.get_keys(&[0, 0]);
        assert_eq!(keys.len(), 1);
        assert!(keys.contains(&0));
    }

    #[test]
    fn test_insert_again_different() {
        let cell_width = 1.0;
        let mut cell_list = VecCell::new(cell_width, 10);

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.25, 0.5]));
        cell_list.insert(1, Cartesian::from([-0.5, -0.75]));

        assert_eq!(cell_list.cell_index.get(&0), Some(&[0, 0]));
        assert_eq!(cell_list.cell_index.get(&1), Some(&[-1, -1]));

        let keys = cell_list.get_keys(&[0, 0]);
        assert_eq!(keys.len(), 1);
        assert!(keys.contains(&0));

        let keys = cell_list.get_keys(&[-1, -1]);
        assert_eq!(keys.len(), 1);
        assert!(keys.contains(&1));
    }

    #[test]
    fn test_remove() {
        let cell_width = 1.0;
        let mut cell_list = VecCell::new(cell_width, 10);

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.995, 0.897]));
        cell_list.insert(2, Cartesian::from([-0.125, 3.25]));

        cell_list.remove(&1);
        cell_list.remove(&2);

        assert_eq!(cell_list.cell_index.get(&0), Some(&[0, 0]));
        assert_eq!(cell_list.cell_index.get(&1), None);
        assert_eq!(cell_list.cell_index.get(&2), None);

        let keys = cell_list.get_keys(&[0, 0]);
        assert_eq!(keys.len(), 1);
        assert!(keys.contains(&0));

        let keys = cell_list.get_keys(&[-1, 3]);
        assert_eq!(keys.len(), 0);
    }

    #[test]
    fn test_clear() {
        let cell_width = 1.0;
        let mut cell_list = VecCell::new(cell_width, 10);

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.995, 0.897]));
        cell_list.insert(2, Cartesian::from([-0.125, 3.25]));

        cell_list.clear();

        assert_eq!(cell_list.cell_index.len(), 0);
        // TODO: assert all vec lengths
    }

    #[test]
    fn test_shrink_to_fit() {
        let cell_width = 1.0;
        let mut cell_list = VecCell::new(cell_width, 10);

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.995, 0.897]));
        cell_list.insert(2, Cartesian::from([-0.125, 3.25]));

        cell_list.remove(&1);
        cell_list.remove(&2);

        cell_list.shrink_to_fit();

        // TODO: check vec capacities

        let keys = cell_list.get_keys(&[0, 0]);
        assert_eq!(keys.len(), 1);
        assert!(keys.contains(&0));
    }

    #[test]
    fn consistency() {
        const N_STEPS: usize = 65_536;
        let mut rng = StdRng::seed_from_u64(0);
        let mut reference = FxHashMap::default();

        let cell_width = 0.5;
        let mut cell_list = VecCell::new(cell_width, 82);
        let position_distribution = Ball { radius: 20.0.try_into().expect("hardcoded value should be positive") };
        let key_distribution = Uniform::new(0, N_STEPS/4).expect("hardcoded distribution should be valid");

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
        assert_eq!(cell_list.cell_index.len(), reference.len());
        for (reference_key,reference_value) in reference.drain() {
            let value = cell_list.cell_index.get(&reference_key);
            assert_eq!(value, Some(&reference_value));

            let keys = cell_list.get_keys(&reference_value);
            assert!(keys.contains(&reference_key));
        }

        // Ensure that there are no extra values in keys_map.
        let total = cell_list.keys_map.iter().map(Vec::len).sum();
        assert_eq!(cell_list.cell_index.len(), total);
    }

    // TODO: Test queries just outside the allocated space.

    #[test]
    fn points_in_ball_2d() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut reference = Vec::new();

        let cell_width = 1.0;
        let mut cell_list = VecCell::new(cell_width, 42);
        let position_distribution = Ball { radius: 20.0.try_into().expect("hardcoded value should be positive") };

        for key in 0..2048 {
            let position: Cartesian<2> = position_distribution.sample(&mut rng);

            cell_list.insert(key, position);
            reference.push(position);
        }

        for p_i in &reference {
            let potential_neighbors: Vec<_> = cell_list.points_potentially_in_ball(p_i, cell_width).copied().collect();

            for (j, p_j) in reference.iter().enumerate() {
                if p_i.distance(p_j) <= cell_width {
                    assert!(potential_neighbors.contains(&j));
                }
            }
        }
    }

    // #[test]
    // fn test_find_potential_neighbor_indices() {
    //     let cell_width = 1.0;

    //     // Create some sample 2D Cartesian positions.
    //     let p0 = Cartesian {
    //         coordinates: [0.2, 0.3],
    //     };
    //     let p1 = Cartesian {
    //         coordinates: [0.8, 1.3],
    //     };
    //     let p2 = Cartesian {
    //         coordinates: [1.2, 0.2],
    //     };
    //     let p3 = Cartesian {
    //         coordinates: [1.5, 1.5],
    //     };

    //     // Construct a vector of positions.
    //     let positions = vec![p0, p1, p2, p3];

    //     let indices = vec![0, 1, 2, 3]; // Particle indices corresponding to positions.

    //     // Build the CellList.
    //     let cell_list = CellList::<2>::new(cell_width, &positions, &indices);

    //     // Define a cutoff radius.
    //     let cutoff_radius = 10.5;

    //     // Use p0 ([0.2, 0.3] falls in cell [0,0]) as the query position.
    //     let potential_neighbor_indices = cell_list
    //         .find_potential_neighbor_indices(&0, &cutoff_radius)
    //         .collect::<Vec<_>>();

    //     // p0's index should appear.
    //     assert!(potential_neighbor_indices.contains(&0));
    //     assert!(potential_neighbor_indices.contains(&1));
    //     assert!(potential_neighbor_indices.contains(&2));
    //     assert!(potential_neighbor_indices.contains(&3));
    // }
}

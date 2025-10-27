// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::{array, cmp::Eq, hash::Hash, marker::PhantomData};

use rustc_hash::FxHashMap;

use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Cartesian;

use super::{PointUpdate, PointsInBall, WithSearchRadius, vec_cell};

/// Cell list is a spatial data structure used for efficient neighbor finding based on assigning particles to cell grids.
///
/// Use cell list in your MD simulation to speed up neighbor finding for evaluation of forces between particles.
/// The `CellList` also has a builder API associated with it (see `CellListBuilder`).
///
/// # Example
///
/// ```
/// use hoomd_spatial::{CellList, CellListBuilder};
/// use hoomd_vector::Cartesian;
/// // Create some sample 2D Cartesian positions.
/// # fn main() {
/// let positions = vec![
///     Cartesian {
///         coordinates: [0.2, 0.3],
///     },
///     Cartesian {
///         coordinates: [0.8, 1.3],
///     },
///     Cartesian {
///         coordinates: [8.5, 9.5],
///     },
/// ];
/// let indices = vec![0, 1, 2]; // Particle indices corresponding to positions.
/// // Define the cell width.
/// let cell_width = 2.0;
/// // Create a cell list object from the builder
/// let mut cell_list = CellListBuilder::<2>::new(cell_width)
///     .with_positions_and_indices(&positions, &indices)
///     .build();
/// // Add another particle to the cell list.
/// let new_position = Cartesian {
///     coordinates: [1.2, 1.3],
/// };
/// let new_index: usize = 3; // New particle index.
/// // Add particles to the cell list.
// cell_list.insert(&new_position, &new_index);
/// // Now delete the first particle from the cell list.
/// cell_list.remove(0);
/// // Shrink the cell list to fit its current capacity.
/// cell_list.shrink_to_fit();
/// // Print the cell indices of particle 2
/// println!("Cell index for particle 2: {:?}", cell_list.cell_index(2));
/// // Translate particle 2 to a new position.
/// let new_particle_position = Cartesian {
///     coordinates: [8.2, 9.3],
/// };
/// // TODO change based on fait of translate_particle function
/// cell_list.insert(&new_particle_position, &2);
/// // Get the cell index for the second particle.
/// println!("Cell index for particle 2: {:?}", cell_list.cell_index(2));
/// // Find potential neighbor indices for particle 2.
/// let cutoff_radius = 1.5;
/// // Find potential neighbor indices
/// let potential_neighbors = cell_list
///     .find_potential_neighbor_indices(&2, &cutoff_radius)
///     .collect::<Vec<_>>();
/// // Print the potential neighbor indices.
/// println!(
///     "Potential neighbor indices for particle 2: {:?}",
///     potential_neighbors
/// );
/// # }
/// ```
pub struct HashCell<K, const D: usize> {
    /// The width of each cell.
    cell_width: PositiveReal,
    
    /// A map from cell indices to cell contents.
    particle_indices: FxHashMap<[i64; D], Vec<K>>,
    
    /// A map from particle indices to cell indices.
    cell_index: FxHashMap<K, [i64; D]>,
    
    /// Location of the 0,..,0 cell.
    origin: Cartesian<D>,

    /// Pre-computed stencils.
    stencils: Vec<Vec<[i64; D]>>,
}

pub struct HashCellBuilder<K, const D: usize> {
    /// Most commonly used search radius.
    nominal_search_radius: PositiveReal,

    /// Largest possible search radius.
    maximum_search_radius: f64,

    /// Location of the 0,..,0 cell.
    origin: Cartesian<D>,

    /// Track the key type.
    phantom_key: PhantomData<K>,
}

impl<K, const D: usize> HashCellBuilder<K, D> where
        K: Copy + Eq + Hash {
    pub fn nominal_search_radius(mut self, nominal_search_radius: PositiveReal) -> Self {
        self.nominal_search_radius = nominal_search_radius;
        self
    }

    pub fn maximum_search_radius(mut self, maximum_search_radius: f64) -> Self {
        self.maximum_search_radius = maximum_search_radius;
        self
    }

    pub fn origin(mut self, origin: Cartesian<D>) -> Self {
        self.origin = origin;
        self
    }

    pub fn build(self) -> HashCell<K, D> 
    {
        let maximum_stencil_radius = (self.maximum_search_radius / self.nominal_search_radius.get()).ceil() as u32;
    
        HashCell {
            cell_width: self.nominal_search_radius,
            particle_indices: FxHashMap::default(),
            cell_index: FxHashMap::default(),
            origin: self.origin,
            stencils: vec_cell::generate_all_stencils(maximum_stencil_radius.min(1)),
        }
    }
}

impl<K, const D: usize> Default for HashCell<K, D> where
K: Copy + Eq + Hash
{
    fn default() -> Self {
         Self::builder().build()
    }
}

impl<K, const D: usize> WithSearchRadius for HashCell<K, D> where
K: Copy + Eq + Hash
{
    fn with_search_radius(radius: PositiveReal) -> Self {
         Self::builder()
            .nominal_search_radius(radius)
            .build()
    }
    }


impl<K, const D: usize> HashCell<K, D> where
K: Copy + Eq + Hash
{
    /// Compute the cell index given a position in space.
    #[inline]
    fn cell_index_from_position(&self, position: &Cartesian<D>) -> [i64; D] {
        let v = *position - self.origin;
        std::array::from_fn(|j| (v.coordinates[j] / self.cell_width.get()).floor() as i64)
    }

    /// Shrink both hashmaps in the cell list to fit their current capacity.
    ///
    /// This function cleans up (read deletes) any empty cells in the `particle_indices` hashmap
    /// and shrinks the capacity of both `particle_indices` and `cell_index` hashmaps
    /// to their current length. This is useful for reducing memory usage after many insertions
    /// and deletions, leaving many empty cells.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::CellList;
    /// use hoomd_vector::Cartesian;
    /// // Create some sample 2D Cartesian positions.
    /// let positions = vec![
    /// Cartesian { coordinates: [0.2, 0.3] },
    /// Cartesian { coordinates: [2.8, 2.3] },
    /// Cartesian { coordinates: [8.5, 9.5] },
    /// ];
    /// let indices = vec![0, 1, 2]; // Particle indices corresponding to positions.
    /// // Define the cell width.
    /// let cell_width = 1.0;
    /// // Build the cell list from positions.
    /// let mut cell_list = CellList::<2>::new(cell_width, &positions, &indices);
    /// // Remove the first particle from the cell list.
    /// cell_list.remove(0);
    /// // Now the cell list has an empty cell associated with cell particle 0 was in.
    /// println!("Before shrink_to_fit: {:?}", cell_list.particle_indices.capacity());
    /// // Call shrink_to_fit to clean up empty cells and reduce memory usage.
    /// cell_list.shrink_to_fit();
    /// println!("After shrink_to_fit: {:?}", cell_list.particle_indices.capacity());
    /// ```
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.particle_indices.retain(|_, v| !v.is_empty());
        self.particle_indices.shrink_to_fit();
        self.cell_index.shrink_to_fit();
    }

    #[inline]
    #[must_use]
    pub fn builder() -> HashCellBuilder<K, D> {
        HashCellBuilder {
            nominal_search_radius: 1.0.try_into().expect("hard-coded constant is a positive real"),
            maximum_search_radius: 1.0,
            origin: Cartesian::default(),
            phantom_key: PhantomData,
        }
    }
}

impl<K, const D: usize> PointUpdate<Cartesian<D>, K> for HashCell<K, D> where
K: Copy + Eq + Hash {
    /// Add particle to the cell list. If the particle is already in the cell list,
    /// it will update its position in the cell list.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_spatial::CellList;
    /// use hoomd_vector::Cartesian;
    ///
    /// // Create some sample 2D Cartesian positions.
    /// let positions = vec![
    ///     Cartesian {
    ///         coordinates: [0.2, 0.3],
    ///     },
    ///     Cartesian {
    ///         coordinates: [0.8, 1.3],
    ///     },
    ///     Cartesian {
    ///         coordinates: [8.5, 9.5],
    ///     },
    /// ];
    /// // Particle indices corresponding to positions.
    /// let indices = vec![0, 1, 2]; // Particle indices corresponding to positions.
    /// // Define the cell width.
    /// let cell_width = 1.0;
    /// // Build the cell list from positions.
    /// let mut cell_list = CellList::<2>::new(cell_width, &positions, &indices);
    ///
    /// // Add a new particle to the cell list.
    /// let new_position = Cartesian {
    ///     coordinates: [1.2, 1.3],
    /// };
    /// cell_list.insert(&new_position, &3);
    /// ```
    #[inline]
    fn insert(&mut self, key: K, position: Cartesian<D>) {
        let cell_idx = self.cell_index_from_position(&position);
        let old_cell_index = self.cell_index.insert(key, cell_idx);
        // This checks if old_cell_index is None or if it is different from the new cell index.
        if old_cell_index != Some(cell_idx) {
            // Add the particle index to the new cell index vector.
            self.particle_indices
                .entry(cell_idx)
                .or_default()
                .push(key);

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

    /// Remove particle from the cell list.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_spatial::CellList;
    /// use hoomd_vector::Cartesian;
    ///
    /// // Create some sample 2D Cartesian positions.
    /// let positions = vec![
    ///     Cartesian {
    ///         coordinates: [0.2, 0.3],
    ///     },
    ///     Cartesian {
    ///         coordinates: [0.8, 1.3],
    ///     },
    ///     Cartesian {
    ///         coordinates: [8.5, 9.5],
    ///     },
    /// ];
    /// // Particle indices corresponding to positions.
    /// let indices = vec![0, 1, 2]; // Particle indices corresponding to positions.
    /// // Define the cell width.
    /// let cell_width = 1.0;
    /// // Build the cell list from positions.
    /// let mut cell_list = CellList::<2>::new(cell_width, &positions, &indices);
    ///
    /// // Remove the first particle from the cell list.
    /// cell_list.remove(0);
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

    #[inline]
    fn clear(&mut self) {
        self.cell_index.clear();
        self.particle_indices.clear();
    }
}

struct PointsIterator<'a, K, const D: usize> {
    keys: Option<&'a Vec<K>>,
    cell_list: &'a HashCell<K, D>,
    index_in_current_cell: usize,
    current_stencil: usize,
    stencil: &'a [[i64; D]],
    center: [i64; D],
    }

impl<'a, K, const D: usize> Iterator for PointsIterator<'a, K, D>
where K: Copy
{
    type Item=K;

    // Required method
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(keys) = self.keys && self.index_in_current_cell < keys.len() {
                let last_index = self.index_in_current_cell;
                self.index_in_current_cell += 1;
                return Some(keys[last_index]);
            }

            self.index_in_current_cell = 0;
            self.current_stencil += 1;
            
            if self.current_stencil >= self.stencil.len() {
                return None;
            }

            let cell_index = array::from_fn(|i| self.center[i] + self.stencil[self.current_stencil][i]);
            self.keys = self.cell_list.particle_indices.get(&cell_index);
        }
    }
}

impl<const D: usize, K> PointsInBall<Cartesian<D>, K> for HashCell<K, D> where
K: Copy + Eq + Hash
{
    #[inline]
    fn points_potentially_in_ball(&self, position: &Cartesian<D>, radius: f64) -> impl Iterator<Item=K> {
        let stencil_index = (radius / self.cell_width.get()).ceil() as usize - 1;
        assert!(stencil_index < self.stencils.len(), "search radius must be less than or equal to the maximum search radius");

        let center = self.cell_index_from_position(position);
        let stencil = &self.stencils[stencil_index];
        
        PointsIterator {
            keys: self.particle_indices.get(&center),
            cell_list: self,
            index_in_current_cell: 0,
            current_stencil: 0,
            stencil: &stencil,
            center,
        }
    }
}

// TODO: Test HashCell<K,3>

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, distr::{Distribution, Uniform}, rngs::StdRng};
    use hoomd_vector::{distribution::Ball, Metric};

    #[test]
    fn test_cell_index() {
        let cell_list = HashCell::<usize, 3>::builder().nominal_search_radius(2.0.try_into().expect("hard-coded constant is a positive real")).build();
        assert_eq!(cell_list.cell_index_from_position(&[0.0, 0.0, 0.0].into()), [0, 0, 0]);
        assert_eq!(cell_list.cell_index_from_position(&[2.0, 0.0, 0.0].into()), [1, 0, 0]);
        assert_eq!(cell_list.cell_index_from_position(&[0.0, 2.0, 0.0].into()), [0, 1, 0]);
        assert_eq!(cell_list.cell_index_from_position(&[0.0, 0.0, 2.0].into()), [0, 0, 1]);
        assert_eq!(cell_list.cell_index_from_position(&[-41.5, 18.5, -0.125].into()), [-21, 9, -1]);

        let cell_list = HashCell::<usize, 3>::builder().nominal_search_radius(2.0.try_into().expect("hard-coded constant is a positive real")).origin([-4.0, 2.0, 8.0].into()).build();
        assert_eq!(cell_list.cell_index_from_position(&[0.0, 0.0, 0.0].into()), [2, -1, -4]);
        assert_eq!(cell_list.cell_index_from_position(&[2.0, 0.0, 0.0].into()), [3, -1, -4]);
        assert_eq!(cell_list.cell_index_from_position(&[0.0, 2.0, 0.0].into()), [2, 0, -4]);
        assert_eq!(cell_list.cell_index_from_position(&[0.0, 0.0, 2.0].into()), [2, -1, -3]);
        assert_eq!(cell_list.cell_index_from_position(&[-41.5, 18.5, -0.125].into()), [-19, 8, -5]);
    }
    
    #[test]
    fn test_insert_one() {
        let mut cell_list = HashCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));

        assert_eq!(cell_list.cell_index.get(&0), Some(&[0, 0]));

        let keys = cell_list.particle_indices.get(&[0, 0]);
        assert!(keys.is_some());
        if let Some(keys) = keys {
            assert_eq!(keys.len(), 1);
            assert!(keys.contains(&0));
        }
    }

    #[test]
    fn test_insert_many() {
        let mut cell_list = HashCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.995, 0.897]));
        cell_list.insert(2, Cartesian::from([-0.125, 3.25]));

        assert_eq!(cell_list.cell_index.get(&0), Some(&[0, 0]));
        assert_eq!(cell_list.cell_index.get(&1), Some(&[0, 0]));
        assert_eq!(cell_list.cell_index.get(&2), Some(&[-1, 3]));

        let keys = cell_list.particle_indices.get(&[0, 0]);
        assert!(keys.is_some());
        if let Some(keys) = keys {
            assert_eq!(keys.len(), 2);
            assert!(keys.contains(&0));
            assert!(keys.contains(&1));
        }

        let keys = cell_list.particle_indices.get(&[-1, 3]);
        assert!(keys.is_some());
        if let Some(keys) = keys {
            assert_eq!(keys.len(), 1);
            assert!(keys.contains(&2));
        }
    }

    #[test]
    fn test_insert_again_same() {
        let mut cell_list = HashCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(0, Cartesian::from([0.25, 0.5]));
        cell_list.insert(0, Cartesian::from([0.5, 0.75]));

        assert_eq!(cell_list.cell_index.get(&0), Some(&[0, 0]));

        let keys = cell_list.particle_indices.get(&[0, 0]);
        assert!(keys.is_some());
        if let Some(keys) = keys {
            assert_eq!(keys.len(), 1);
            assert!(keys.contains(&0));
        }
    }

    #[test]
    fn test_insert_again_different() {
        let mut cell_list = HashCell::default();
        
        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.25, 0.5]));
        cell_list.insert(1, Cartesian::from([-0.5, -0.75]));

        assert_eq!(cell_list.cell_index.get(&0), Some(&[0, 0]));
        assert_eq!(cell_list.cell_index.get(&1), Some(&[-1, -1]));

        let keys = cell_list.particle_indices.get(&[0, 0]);
        assert!(keys.is_some());
        if let Some(keys) = keys {
            assert_eq!(keys.len(), 1);
            assert!(keys.contains(&0));
        }

        let keys = cell_list.particle_indices.get(&[-1, -1]);
        assert!(keys.is_some());
        if let Some(keys) = keys {
            assert_eq!(keys.len(), 1);
            assert!(keys.contains(&1));
        }
    }

    #[test]
    fn test_remove() {
        let mut cell_list = HashCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.995, 0.897]));
        cell_list.insert(2, Cartesian::from([-0.125, 3.25]));

        cell_list.remove(&1);
        cell_list.remove(&2);

        assert_eq!(cell_list.cell_index.get(&0), Some(&[0, 0]));
        assert_eq!(cell_list.cell_index.get(&1), None);
        assert_eq!(cell_list.cell_index.get(&2), None);

        let keys = cell_list.particle_indices.get(&[0, 0]);
        assert!(keys.is_some());
        if let Some(keys) = keys {
            assert_eq!(keys.len(), 1);
            assert!(keys.contains(&0));
        }

        let keys = cell_list.particle_indices.get(&[-1, 3]);
        assert!(keys.is_some());
        if let Some(keys) = keys {
            assert_eq!(keys.len(), 0);
        }
    }

    #[test]
    fn test_clear() {
        let mut cell_list = HashCell::default();

        cell_list.insert(0, Cartesian::from([0.125, 0.25]));
        cell_list.insert(1, Cartesian::from([0.995, 0.897]));
        cell_list.insert(2, Cartesian::from([-0.125, 3.25]));

        cell_list.clear();

        assert_eq!(cell_list.cell_index.len(), 0);
        assert_eq!(cell_list.particle_indices.len(), 0);
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
        assert_eq!(cell_list.particle_indices.len(), 1);

        let keys = cell_list.particle_indices.get(&[0, 0]);
        assert!(keys.is_some());
        if let Some(keys) = keys {
            assert_eq!(keys.len(), 1);
            assert!(keys.contains(&0));
        }
    }

    #[test]
    fn consistency() {
        const N_STEPS: usize = 65_536;
        let mut rng = StdRng::seed_from_u64(0);
        let mut reference = FxHashMap::default();

        let cell_width = 0.5;
        let mut cell_list = HashCell::builder().nominal_search_radius(cell_width.try_into().expect("hard-coded value should be positive")).build();
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
        // particle_indices is consistent.
        assert_eq!(cell_list.cell_index.len(), reference.len());
        for (reference_key,reference_value) in reference.drain() {
            let value = cell_list.cell_index.get(&reference_key);
            assert_eq!(value, Some(&reference_value));

            let keys = cell_list.particle_indices.get(&reference_value);
            assert!(keys.is_some());
            if let Some(keys) = keys {
                assert!(keys.contains(&reference_key));
            }
        }

        // Ensure that there are no extra values in particle_indices.
        let total = cell_list.particle_indices.values().map(Vec::len).sum();
        assert_eq!(cell_list.cell_index.len(), total);
    }

    #[test]
    fn points_in_ball_2d() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut reference = Vec::new();

        let cell_width = 1.0;
        let mut cell_list = HashCell::default();
        let position_distribution = Ball { radius: 20.0.try_into().expect("hardcoded value should be positive") };

        for key in 0..2048 {
            let position: Cartesian<2> = position_distribution.sample(&mut rng);

            cell_list.insert(key, position);
            reference.push(position);
        }

        for p_i in &reference {
            let potential_neighbors: Vec<_> = cell_list.points_potentially_in_ball(p_i, cell_width).collect();

            for (j, p_j) in reference.iter().enumerate() {
                if p_i.distance(p_j) <= cell_width {
                    assert!(potential_neighbors.contains(&j));
                }
            }
        }
    }
}

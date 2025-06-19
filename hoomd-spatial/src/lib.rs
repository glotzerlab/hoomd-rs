// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
/*! Implements spatial data structures for efficient neighbor finding.
 */

use hoomd_vector::Cartesian;
use std::{collections::HashMap, path::Iter};

/** This enum represents the flags for particles in the cell list.
It is used to distinguish between real particles and ghost particles.
This is useful for simulations where ghost particles are used to handle periodic boundary conditions or other special cases.
**/
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// TODO: use enums for particle flags
pub enum ParticleFlag {
    /// Real particles
    Real = 0,
    /// Ghost particles - usually used for periodic boundary conditions or other special cases.
    Ghost = 1,
}

/** Cell list docs. */
pub struct CellList<const D: usize> {
    /// The width of each cell.
    pub cell_width: f64,
    /// A map from cell indices to particle indices.
    pub particle_indices: HashMap<[i32; D], Vec<usize>>,
    /// A map from particle indices to cell indices.
    pub cell_index: HashMap<usize, [i32; D]>,
}

pub struct CellListBuilder<const D: usize> {
    pub cell_width: f64,
    pub positions: Vec<Cartesian<D>>,
    pub indices: Vec<usize>,
}

impl<const D:usize> CellListBuilder<D>{
    pub fn new(cell_width: f64) -> Self {
        Self {
            cell_width,
            positions: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Optionally supply positions and indices to initialize the cell list.
    pub fn with_positions_and_indices(
        mut self,
        positions: &Vec<Cartesian<D>>,
        indices: &Vec<usize>,
    ) -> Self {
        self.positions = positions.clone();
        self.indices = indices.clone();
    }

    pub fn build(self) -> CellList<D> {
        CellList::new(self.cell_width, &self.positions, &self.indices)
    }
}

impl<const D: usize> Default for CellList<D> {
    // TODO How to do this with D? Hashmap doesnt depend on D yet.
    fn default() -> Self {
        // Default cell width is 1.0, and empty particle indices and cell index maps.
        CellList {
            cell_width: 1.0,
            particle_indices: HashMap::new(),
            cell_index: HashMap::new(),
        }
    }
}

//TODO think about providing shrink_to_fit() method to reduce memory usage after many
//insertions and deletions and we are left with many empty cells.
impl<const D: usize> CellList<D> {
    /** Builder API helper function.
     */
    pub fn builder() -> CellListBuilder<D>{
        CellListBuilder::<D>::default()
    }

    /** A helper function which converts given positions to cell indices.
    // To generalize this we will have to make it a Trait function
     */
    #[inline]
    fn cell_index_from_position(cell_width: f64, position: &Cartesian<D>) -> [i32; D] {
        std::array::from_fn(|j| (position.coordinates[j] / cell_width).floor() as i32) // TODO: instead We can have tryinto() here with expect. would need to test performance.
    }

    /** Create a new cell list from the given cell width and positions.

    # Example

    ```
    use hoomd_spatial::CellList;
    use hoomd_vector::Cartesian;

    // Create some sample 2D Cartesian positions.
    let positions = vec![
        Cartesian { coordinates: [0.2, 0.3] },
        Cartesian { coordinates: [0.8, 1.3] },
        Cartesian { coordinates: [8.5, 9.5] },
    ];
    let indices = vec![0, 1, 2]; // Particle indices corresponding to positions.
    // Define the cell width.
    let cell_width = 1.0;
    // Build the cell list from positions.
    let cell_list = CellList::<2>::new(cell_width, &positions, &indices);
    ```
    */
    #[inline]
    #[must_use]
    // TODO: Take a look into builder API and make positions optional. Keep new as is,
    // and make a default without positions.
    pub fn new(cell_width: f64, positions: &[Cartesian<D>], indices: &[usize]) -> Self {
        let mut instance = Self {
            cell_width,
            particle_indices: HashMap::new(),
            cell_index: HashMap::new(),
        };

        for (position, index) in positions.iter().zip(indices.iter()) {
            instance.insert(position, index);
        }

        instance
    }

    /** Create an empty cell list with the given cell width.
    This is useful for initializing a cell list
    that will be populated later.
     
    # Example
    ```
    use hoomd_spatial::CellList;
    // Create an empty 2D cell list with a cell width of 1.0.
    let cell_width = 1.0;
    let cell_list = CellList::<2>::empty(cell_width);
    ```
    */
    #[inline]
    #[must_use]
    pub fn empty(cell_width: f64) -> Self {
        Self {
            cell_width,
            particle_indices: HashMap::new(),
            cell_index: HashMap::new(),
        }
    }

    /** Returns a cell index (in form of a tuple) for a given particle index.

    # Example

    ```
    use hoomd_spatial::CellList;
    use hoomd_vector::Cartesian;

    // Create some sample 2D Cartesian positions.
    let positions = vec![
        Cartesian { coordinates: [0.2, 0.3] },
        Cartesian { coordinates: [0.8, 1.3] },
        Cartesian { coordinates: [8.5, 9.5] },
    ];
    // Indices of particles corresponding to positions.
    let indices = vec![0, 1, 2];
    // Define the cell width.
    let cell_width = 1.0;
    // Build the cell list from positions.
    let cell_list = CellList::<2>::new(cell_width, &positions, &indices);

    // Get the cell index for the first particle.
    let cell_index = cell_list.cell_index(0).unwrap();
    ```
     */
    #[inline]
    #[must_use]
    pub fn cell_index(&self, particle_index: usize) -> Option<&[i32; D]> {
        self.cell_index.get(&particle_index)
    }

    /** Add particle to the cell list. If the particle is already in the cell list,
    it will update its position in the cell list.

    # Example

    ```
    use hoomd_spatial::CellList;
    use hoomd_vector::Cartesian;

    // Create some sample 2D Cartesian positions.
    let positions = vec![
        Cartesian { coordinates: [0.2, 0.3] },
        Cartesian { coordinates: [0.8, 1.3] },
        Cartesian { coordinates: [8.5, 9.5] },
    ];
    // Particle indices corresponding to positions.
    let indices = vec![0, 1, 2]; // Particle indices corresponding to positions.
    // Define the cell width.
    let cell_width = 1.0;
    // Build the cell list from positions.
    let mut cell_list = CellList::<2>::new(cell_width, &positions, &indices);

    // Add a new particle to the cell list.
    let new_position = Cartesian { coordinates: [1.2, 1.3] };
    cell_list.insert(&new_position, 3);
    ```
    */
    #[inline]
    pub fn insert(&mut self, position: &Cartesian<D>, index: &usize) {
        let cell_idx = Self::cell_index_from_position(self.cell_width, position);
        let old_cell_index = self.cell_index.insert(*index, cell_idx);
        // This checks if old_cell_index is None or if it is different from the new cell index.
        if old_cell_index != Some(cell_idx) {
            // Add the particle index to the new cell index vector.
            self.particle_indices
                .entry(cell_idx)
                .or_default()
                .push(*index);
            if let Some(old_cell_index) = old_cell_index {
                // If the particle was in a different cell, we need to remove it from the old cell.
                self.particle_indices
                    .entry(old_cell_index)
                    .and_modify(|particle_indices| {
                        if let Some(pos) = particle_indices.iter().position(|&x| x == *index) {
                            particle_indices.swap_remove(pos);
                        }
                    });
            }
        }
    }

    /** Remove particle from the cell list.

    Note that removing a particle from the cell list will not change the maximum particle index.

    # Example

    ```
    use hoomd_spatial::CellList;
    use hoomd_vector::Cartesian;

    // Create some sample 2D Cartesian positions.
    let positions = vec![
        Cartesian { coordinates: [0.2, 0.3] },
        Cartesian { coordinates: [0.8, 1.3] },
        Cartesian { coordinates: [8.5, 9.5] },
    ];
    // Particle indices corresponding to positions.
    let indices = vec![0, 1, 2]; // Particle indices corresponding to positions.
    // Define the cell width.
    let cell_width = 1.0;
    // Build the cell list from positions.
    let mut cell_list = CellList::<2>::new(cell_width, &positions, &indices);

    // Remove the first particle from the cell list.
    cell_list.remove(0);
    ```
    */
    #[inline]
    pub fn remove(&mut self, particle_index: usize) {
        let cell_idx = self.cell_index.remove(&particle_index);
        if let Some(cell_idx) = cell_idx {
            // If the particle was found in the cell list, remove it from the particle indices.
            self.particle_indices
                .entry(cell_idx)
                .and_modify(|particle_indices| {
                    // Find the index of removed particle in the vector of particle indices.
                    if let Some(idx) = particle_indices
                        .iter()
                        .position(|&x| x == particle_index)
                    {
                        // Remove the particle index from the vector.
                        particle_indices.swap_remove(idx);
                    }
                });
        }
    }

    /** Translate particle to a new position.

    # Example

    ```
    use hoomd_spatial::CellList;
    use hoomd_vector::Cartesian;

    // Create some sample 2D Cartesian positions.
    let positions = vec![
        Cartesian { coordinates: [0.2, 0.3] },
        Cartesian { coordinates: [0.8, 1.3] },
        Cartesian { coordinates: [8.5, 9.5] },
    ];
    let cell_width = 1.0;
    // Build the cell list from positions.
    let mut cell_list = CellList::<2>::new(cell_width, &positions);

    // Translate the first particle to a new position.
    let new_position = Cartesian { coordinates: [1.2, 1.3] };
    cell_list.translate_particle(0, new_position);
    ```
    */
    #[inline]
    // TODO: Do I even need this function? It is the same as insert...
    pub fn translate_particle(
        &mut self,
        particle_index: usize,
        new_particle_position: Cartesian<D>,
    ) {
        self.insert(&new_particle_position, &particle_index);
    }

    /** Find potential neighbor indices.

    This function finds the POTENTIAL neighbor indices for a given position and cutoff radius.

    # Example

    ```
    use hoomd_spatial::CellList;
    use hoomd_vector::Cartesian;

    // Create some sample 2D Cartesian positions.
    let positions = vec![
        Cartesian { coordinates: [0.2, 0.3] },
        Cartesian { coordinates: [0.8, 1.3] },
        Cartesian { coordinates: [8.5, 9.5] },
    ];
    let cell_width = 1.0;
    // Build the cell list from positions.
    let cell_list = CellList::<2>::new(cell_width, &positions);

    // Choose a query position (for example, the first one).
    let query_position = &positions[0];
    // Define a cutoff radius.
    let cutoff_radius = 1.5;
    // Create a mutable vector to store potential neighbor indices.
    let mut potential_neighbor_indices = Vec::new();

    // Call the function to find potential neighbor indices.
    cell_list.find_potential_neighbor_indices(query_position, &cutoff_radius, &mut potential_neighbor_indices);

    // Print the resulting neighbor indices.
    println!("Potential neighbor indices: {:?}", potential_neighbor_indices);
    ```
     */
    #[inline]
    // TODO: Return an iterator instead of a mutable vector argument.
    // instead of recursion, loop over the number of iterations and use //
    pub fn find_potential_neighbor_indices(
        &self,
        position: &Cartesian<D>,
        cutoff_radius: &f64
    ) -> Iter {
        // implement later
        // This function will find the potential neighbor indices for a given position
        // and cutoff radius. It will return an iterator over the potential neighbor
        // indices.
        unimplemented!("find_potential_neighbor_indices is not yet implemented");

        // old code
      //  // Plan: use logic similar to collect into
      //  // return the result in neighbor indices -> probably first clear and then push
      //  // Check if the neighbor_indices (output argument) goes first or last
      //  // return potential index only - not actual neighbors.
      //  let cell_idx = Self::cell_index_from_position(self.cell_width, position);
      //  // clean neighbor_indices
      //  potential_neighbor_indices.clear();
      //  // calculate how many cell widths in ids need to be checked in each dimension,
      //  // this is a single integer number
      //  let max_cell_translations_to_check = (cutoff_radius / self.cell_width).ceil() as i32;
      //  let max_offset = max_cell_translations_to_check;

      //  let mut cells_translations_to_check: Vec<[i32; D]> = Vec::new();

      //  // Define a recursive helper function to generate all translation combinations.
      //  // For each dimension, it iterates from -max_offset to max_offset.
      //  // TODO figure out if there is a better way to do this. Is there an itertools
      //  // like functionality in std library? cartesian product?
      //  fn generate_translations<const D: usize>(
      //      dim: usize,
      //      current: &mut Vec<i32>,
      //      max_offset: i32,
      //      translations: &mut Vec<[i32; D]>,
      //  ) {
      //      if dim == D {
      //          // Convert the current vector to an array of length D.
      //          let arr: [i32; D] = current.clone().try_into().expect("Incorrect length");
      //          translations.push(arr);
      //          return;
      //      }
      //      for offset in -max_offset..=max_offset {
      //          current.push(offset);
      //          generate_translations(dim + 1, current, max_offset, translations);
      //          current.pop();
      //      }
      //  }

      //  let mut current = Vec::new();
      //  generate_translations::<D>(
      //      0,
      //      &mut current,
      //      max_offset,
      //      &mut cells_translations_to_check,
      //  );

      //  // For each cell translation, compute the neighbor cell index and add any particle indices.
      //  for cell_translation in cells_translations_to_check.iter() {
      //      let neighbor_cell_idx = std::array::from_fn(|i| cell_idx[i] + cell_translation[i]);
      //      if let Some(particle_indices) = self.particle_indices.get(&neighbor_cell_idx) {
      //          potential_neighbor_indices.extend(particle_indices);
      //      }
      //  }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

//    // test builder pattern
//    #[test]
//    fn test_cell_list_builder() {
//        let cell_width = 1.0;
//        let positions = vec![
//            Cartesian { coordinates: [0.2, 0.3] },
//            Cartesian { coordinates: [1.2, 1.3] },
//        ];
//        let indices = vec![0, 1]; // Particle indices corresponding to positions.
//
//        let cell_list = CellList::<2>::builder()
//            .with_positions_and_indices(&positions, &indices)
//            .build();
//
//        assert_eq!(cell_list.cell_width, cell_width);
//        assert_eq!(cell_list.particle_indices.len(), 2);
//        assert_eq!(cell_list.cell_index.len(), 2);
//    }

    #[test]
    fn test_add_particle() {
        let cell_width = 1.0;
        let positions = vec![Cartesian {
            coordinates: [0.2, 0.3],
        }];
        let indices = vec![0]; // Particle index corresponding to the position.
        let mut cell_list = CellList::<2>::new(cell_width, &positions, &indices);

        let new_position = Cartesian {
            coordinates: [1.2, 1.3],
        }; // cell index [1,1]
        let new_index: usize = 1; // New particle index.
        cell_list.insert(&new_position, &new_index);

        let cell_idx_new = cell_list.cell_index(new_index).unwrap();
        let expected_cell_idx = CellList::<2>::cell_index_from_position(cell_width, &new_position);
        assert_eq!(*cell_idx_new, expected_cell_idx);

        let idx_in_new_cell = cell_list.particle_indices.get(&expected_cell_idx).unwrap();
        assert!(idx_in_new_cell.contains(&new_index));
    }

    #[test]
    fn test_translate_particle() {
        let cell_width = 1.0;
        let positions = vec![
            Cartesian {
                coordinates: [0.2, 0.3],
            }, // initially in cell [0,0]
            Cartesian {
                coordinates: [1.2, 0.3],
            }, // in cell [1,0]
        ];
        let indices = vec![0, 1]; // Particle indices corresponding to positions.
        let mut cell_list = CellList::<2>::new(cell_width, &positions, &indices);

        // Translate first particle to a new position in a different cell.
        let new_position = Cartesian {
            coordinates: [1.1, 1.2],
        }; // expected cell [1,1]
        cell_list.translate_particle(0, new_position);

        let expected_cell_idx = CellList::<2>::cell_index_from_position(cell_width, &new_position);
        let cell_idx_after = cell_list.cell_index(0).unwrap();
        assert_eq!(*cell_idx_after, expected_cell_idx);
    }

    #[test]
    fn test_remove_particle() {
        let cell_width = 1.0;
        let positions = vec![
            Cartesian {
                coordinates: [0.2, 0.3],
            },
            Cartesian {
                coordinates: [1.2, 1.3],
            },
        ];
        let indices = vec![0, 1]; // Particle indices corresponding to positions.
        let mut cell_list = CellList::<2>::new(cell_width, &positions, &indices);

        // Cell id of particle to be removed
        let cell_idx = cell_list.cell_index(0).copied();

        // Remove the first particle (index 0).
        cell_list.remove(0);
        let removed_particle_cell_index = cell_list.cell_index(0);
        assert!(removed_particle_cell_index.is_none());

        // The cell corresponding to the removed particle should not hold the index.
        let particle_indices_in_old_cell = cell_list
            .particle_indices
            .get(&cell_idx.unwrap())
            .expect("The cell index should exist");
        assert!(
            particle_indices_in_old_cell.is_empty(),
            "Expected cell to be empty after removal"
        );
    }

    #[test]
    fn test_remove_nonexistent_particle() {
        let cell_width = 1.0;
        let positions = vec![Cartesian {
            coordinates: [0.2, 0.3],
        }];
        let indices = vec![0]; // Particle index corresponding to the position.
        let mut cell_list = CellList::<2>::new(cell_width, &positions, &indices);
        // Attempt to remove a particle that doesn't exist.
        cell_list.remove(42);
        // check if cell lists contains the original particle.
        let cell_idx = cell_list.cell_index(0).unwrap();
        let particle_indices_in_cell = cell_list
            .particle_indices
            .get(cell_idx)
            .expect("The cell index should exist");
        assert!(!particle_indices_in_cell.is_empty(), "Cell should not be empty");
        assert!(particle_indices_in_cell.contains(&0), "Cell should contain the original particle");
    }

    #[test]
    fn test_find_potential_neighbor_indices() {
        let cell_width = 1.0;

        // Create some sample 2D Cartesian positions.
        let p0 = Cartesian {
            coordinates: [0.2, 0.3],
        };
        let p1 = Cartesian {
            coordinates: [0.8, 1.3],
        };
        let p2 = Cartesian {
            coordinates: [1.2, 0.2],
        };
        let p3 = Cartesian {
            coordinates: [1.5, 1.5],
        };

        // Construct a vector of positions.
        let positions = vec![p0, p1, p2, p3];

        let indices = vec![0, 1, 2, 3]; // Particle indices corresponding to positions.

        // Build the CellList.
        let cell_list = CellList::<2>::new(cell_width, &positions, &indices);

        // Define a cutoff radius.
        let cutoff_radius = 10.5;

        // Use p0 ([0.2, 0.3] falls in cell [0,0]) as the query position.
        let it = cell_list.find_potential_neighbor_indices(
            &p0,
            &cutoff_radius,
        );

        // p0's index should appear.
        //assert!(potential_neighbor_indices.contains(&0));
        //assert!(potential_neighbor_indices.contains(&1));
        //assert!(potential_neighbor_indices.contains(&2));
        //assert!(potential_neighbor_indices.contains(&3));
    }
}

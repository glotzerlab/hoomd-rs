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
use std::collections::HashMap;

/** For ghost particles
**/
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParticleFlag {
    // Real particles
    Real = 0,
    // Ghost particles
    Ghost = 1,
}

/** Cell list docs. */
pub struct CellList<const N: usize> {
    /// The width of each cell.
    pub cell_width: f64,
    /// A map from cell indices to particle indices.
    pub cell_idx_to_particle_indices: HashMap<[isize; N], Vec<usize>>,
    /// A map from particle indices to cell indices.
    pub particle_idx_to_cell_index: HashMap<usize, [isize; N]>,
    /// The maximum particle index.
    pub particle_max_index: usize,
}

impl<const N: usize> CellList<N> {
    /** A helper function which converts given positions to cell indices.
     */
    #[inline]
    fn cell_index_from_position(cell_width: f64, position: &Cartesian<N>) -> [isize; N] {
        std::array::from_fn(|j| (position.coordinates[j] / cell_width).floor() as isize)
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
    let cell_width = 1.0;
    // Build the cell list from positions.
    let cell_list = CellList::<2>::new(cell_width, &positions);
    ```
    */
    #[inline]
    pub fn new(cell_width: f64, positions: &Vec<Cartesian<N>>) -> Self {
        let mut instance = Self {
            cell_width,
            cell_idx_to_particle_indices: HashMap::new(),
            particle_idx_to_cell_index: HashMap::new(),
            particle_max_index: 0,
        };

        for position in positions {
            instance.add_particle(position);
        }

        instance
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
    let cell_width = 1.0;
    // Build the cell list from positions.
    let cell_list = CellList::<2>::new(cell_width, &positions);

    // Get the cell index for the first particle.
    let cell_index = cell_list.cell_index_from_particle_index(0).unwrap();
    ```
     */
    #[inline]
    #[must_use]
    pub fn cell_index_from_particle_index(&self, particle_index: usize) -> Option<&[isize; N]> {
        self.particle_idx_to_cell_index.get(&particle_index)
    }

    /** Add particle to the cell list.

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

    // Add a new particle to the cell list.
    let new_position = Cartesian { coordinates: [1.2, 1.3] };
    cell_list.add_particle(&new_position);
    ```
    */
    #[inline]
    pub fn add_particle(&mut self, position: &Cartesian<N>) {
        let particle_index = self.particle_max_index;
        let cell_idx = Self::cell_index_from_position(self.cell_width, position);
        self.cell_idx_to_particle_indices
            .entry(cell_idx)
            .or_insert(Vec::new())
            .push(particle_index);
        self.particle_idx_to_cell_index
            .insert(particle_index, cell_idx);
        self.particle_max_index += 1;
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
    let cell_width = 1.0;
    // Build the cell list from positions.
    let mut cell_list = CellList::<2>::new(cell_width, &positions);

    // Remove the first particle from the cell list.
    cell_list.remove_particle(0);
    ```

    # Panics

    This function will panic if the particle index is not found in the cell list.
    */
    #[inline]
    pub fn remove_particle(&mut self, particle_index: usize) {
        //  unwrap will panic if particle index is not present
        // TODO think about using entry_and_modify to make this code cleaner
        if let Some(cell_idx) = self.particle_idx_to_cell_index.remove(&particle_index) {
            let particle_indices = self
                .cell_idx_to_particle_indices
                .get_mut(&cell_idx)
                .expect("Cell index found in the cell list.");
            let index = particle_indices
                .iter()
                .position(|&x| x == particle_index)
                .expect("Particle index is in the cell list.");
            particle_indices.swap_remove(index);
        } else {
            panic!("Particle index not found in cell list");
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

    # Panics

    This function will panic if the particle index is not found in the cell list.
    */
    #[inline]
    pub fn translate_particle(
        &mut self,
        particle_index: usize,
        new_particle_position: Cartesian<N>,
    ) {
        // TODO I have code repetition from add and remove particles, think about refactoring
        let current_cell_idx = self
            .particle_idx_to_cell_index
            .get(&particle_index)
            .expect("Particle index found in the cell list.");
        let new_cell_idx = Self::cell_index_from_position(self.cell_width, &new_particle_position);
        if *current_cell_idx != new_cell_idx {
            let particle_indices = self
                .cell_idx_to_particle_indices
                .get_mut(current_cell_idx)
                .expect("Cell index found in the cell list.");
            let index = particle_indices
                .iter()
                .position(|&x| x == particle_index)
                .expect("Particle index is in the cell list.");
            particle_indices.swap_remove(index);
            self.cell_idx_to_particle_indices
                .entry(new_cell_idx)
                .or_insert(Vec::new())
                .push(particle_index);
            self.particle_idx_to_cell_index
                .insert(particle_index, new_cell_idx);
        }
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
    pub fn find_potential_neighbor_indices(
        &self,
        position: &Cartesian<N>,
        cutoff_radius: &f64,
        potential_neighbor_indices: &mut Vec<usize>,
    ) {
        // use logic similar to collect into
        // return the result in neighbor indices -> probably first clear and then push
        // Check if the neighbor_indices (output argument) goes first or last
        // return potential index only - not actual neighbors.
        let cell_idx = Self::cell_index_from_position(self.cell_width, position);
        // clean neighbor_indices
        potential_neighbor_indices.clear();
        // calculate how many cell widths in ids need to be checked in each dimension,
        // this is a single integer number
        let max_cell_translations_to_check = (cutoff_radius / self.cell_width).ceil() as isize;
        let max_offset = max_cell_translations_to_check - 1;

        let mut cells_translations_to_check: Vec<[isize; N]> = Vec::new();

        // Define a recursive helper function to generate all translation combinations.
        // For each dimension, it iterates from -max_offset to max_offset.
        fn generate_translations<const N: usize>(
            dim: usize,
            current: &mut Vec<isize>,
            max_offset: isize,
            translations: &mut Vec<[isize; N]>,
        ) {
            if dim == N {
                // Convert the current vector to an array of length N.
                let arr: [isize; N] = current.clone().try_into().expect("Incorrect length");
                translations.push(arr);
                return;
            }
            for offset in -max_offset..=max_offset {
                current.push(offset);
                generate_translations(dim + 1, current, max_offset, translations);
                current.pop();
            }
        }

        let mut current = Vec::new();
        generate_translations::<N>(
            0,
            &mut current,
            max_offset,
            &mut cells_translations_to_check,
        );

        // For each cell translation, compute the neighbor cell index and add any particle indices.
        for cell_translation in cells_translations_to_check.iter() {
            let neighbor_cell_idx = std::array::from_fn(|i| cell_idx[i] + cell_translation[i]);
            if let Some(particle_indices) =
                self.cell_idx_to_particle_indices.get(&neighbor_cell_idx)
            {
                potential_neighbor_indices.extend(particle_indices);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_particle() {
        let cell_width = 1.0;
        let positions = vec![Cartesian {
            coordinates: [0.2, 0.3],
        }];
        let mut cell_list = CellList::<2>::new(cell_width, &positions);

        let new_position = Cartesian {
            coordinates: [1.2, 1.3],
        };
        cell_list.add_particle(&new_position);

        let cell_idx_new = cell_list
            .cell_index_from_particle_index(cell_list.particle_max_index - 1)
            .unwrap();
        let expected_cell_idx = CellList::<2>::cell_index_from_position(cell_width, &new_position);
        assert_eq!(*cell_idx_new, expected_cell_idx);

        let idx_in_new_cell = cell_list
            .cell_idx_to_particle_indices
            .get(&expected_cell_idx)
            .unwrap();
        assert!(idx_in_new_cell.contains(&(cell_list.particle_max_index - 1)));
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
        let mut cell_list = CellList::<2>::new(cell_width, &positions);

        // Translate first particle to a new position in a different cell.
        let new_position = Cartesian {
            coordinates: [1.1, 1.2],
        }; // expected cell [1,1]
        cell_list.translate_particle(0, new_position);

        let expected_cell_idx = CellList::<2>::cell_index_from_position(cell_width, &new_position);
        let cell_idx_after = cell_list.cell_index_from_particle_index(0).unwrap();
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
        let mut cell_list = CellList::<2>::new(cell_width, &positions);

        // Cell id of particle to be removed
        let cell_idx = cell_list.cell_index_from_particle_index(0).copied();

        // Remove the first particle (index 0).
        cell_list.remove_particle(0);
        let removed_particle_cell_index = cell_list.cell_index_from_particle_index(0);
        assert!(removed_particle_cell_index.is_none());

        // The cell corresponding to the removed particle should not hold the index.
        let particle_indices_in_old_cell = cell_list
            .cell_idx_to_particle_indices
            .get(&cell_idx.unwrap())
            .expect("The cell index should exist");
        assert!(
            particle_indices_in_old_cell.is_empty(),
            "Expected cell to be empty after removal"
        );
    }

    #[test]
    #[should_panic(expected = "Particle index not found in cell list")]
    fn test_remove_nonexistent_particle() {
        let cell_width = 1.0;
        let positions = vec![Cartesian {
            coordinates: [0.2, 0.3],
        }];
        let mut cell_list = CellList::<2>::new(cell_width, &positions);
        // Attempt to remove a particle that doesn't exist.
        cell_list.remove_particle(42);
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

        // Build the CellList.
        let cell_list = CellList::<2>::new(cell_width, &positions);

        // Define a cutoff radius.
        let cutoff_radius = 10.5;
        let mut potential_neighbor_indices = Vec::new();

        // Use p0 ([0.2, 0.3] falls in cell [0,0]) as the query position.
        cell_list.find_potential_neighbor_indices(
            &p0,
            &cutoff_radius,
            &mut potential_neighbor_indices,
        );

        // p0's index should appear.
        assert!(potential_neighbor_indices.contains(&0));
        assert!(potential_neighbor_indices.contains(&1));
        assert!(potential_neighbor_indices.contains(&2));
        assert!(potential_neighbor_indices.contains(&3));
    }
}

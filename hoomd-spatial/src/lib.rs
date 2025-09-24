// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
//! Implements spatial data structures for efficient neighbor finding.

use hoomd_vector::Cartesian;
use std::collections::HashMap;

/// This enum represents the flags for particles in the cell list.
/// It is used to distinguish between real particles and ghost particles.
/// This is useful for simulations where ghost particles are used to handle periodic boundary conditions or other special cases.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// TODO: use enums for particle flags
pub enum ParticleFlag {
    /// Real particles
    Real = 0,
    /// Ghost particles - usually used for periodic boundary conditions or other special cases.
    Ghost = 1,
}

/// Cell list is a spatial data structure used for efficient neighbor finding based on assigning particles to cell grids.
///
/// Use cell list in your MD simulation to speed up neighbor finding for evaluation of forces between particles.
/// The `CellList` also has a builder API associated with it (see `CellListBuilder`).
///
/// # Example
///
/// ```
/// use hoomd_spatial::CellList;
/// use hoomd_spatial::CellListBuilder;
/// use hoomd_vector::Cartesian;
/// // Create some sample 2D Cartesian positions.
/// let positions = vec![
/// Cartesian { coordinates: [0.2, 0.3] },
/// Cartesian { coordinates: [0.8, 1.3] },
/// Cartesian { coordinates: [8.5, 9.5] },
/// ];
/// let indices = vec![0, 1, 2]; // Particle indices corresponding to positions.
/// // Define the cell width.
/// let cell_width = 2.0;
/// // Create a cell list object from the builder
/// let mut cell_list = CellListBuilder::<2>::new(cell_width).with_positions_and_indices(&positions, &indices).build();
/// add another particle to the cell list.
/// let new_position = Cartesian { coordinates: [1.2, 1.3] };
/// let new_index: usize = 3; // New particle index.
/// // Add particles to the cell list.
/// cell_list.insert(&new_position, &new_index);
/// // Now delete the first particle from the cell list.
/// cell_list.remove(0);
/// // Shrink the cell list to fit its current capacity.
/// cell_list.shrink_to_fit();
/// print the cell indices of particle 2
/// println!("Cell index for particle 2: {:?}", cell_list.cell_index(2));
/// // Translate particle 2 to a new position.
/// let new_particle_position = Cartesian { coordinates: [8.2, 9.3] };
/// // TODO change based on fait of translate_particle function
/// cell_list.insert(&new_particle_position, &2);
/// // Get the cell index for the second particle.
/// println!("Cell index for particle 2: {:?}", cell_list.cell_index(2));
/// // Find potential neighbor indices for particle 2.
/// let cutoff_radius = 1.5;
/// // Find potential neighbor indices
/// let potential_neighbors = cell_list.find_potential_neighbor_indices(&2, &cutoff_radius).collect::<Vec<_>>();
/// // Print the potential neighbor indices.
/// println!("Potential neighbor indices for particle 2: {:?}", potential_neighbors);
/// ```
pub struct CellList<const D: usize> {
    /// The width of each cell.
    pub cell_width: f64,
    /// A map from cell indices to particle indices.
    pub particle_indices: HashMap<[i32; D], Vec<usize>>,
    /// A map from particle indices to cell indices.
    pub cell_index: HashMap<usize, [i32; D]>,
}

/// `CellListBuilder` is a builder for creating a `CellList`.
///
/// Each `CellList` must at least be given a cell width. Builder API allows for creation of empty `CellList` without any particles.
/// Particles can also be added by adding positions and indices to the `CellListBuilder`.
///
/// # Example constructing an empty `CellList` using the builder API.
///
/// ```
/// use hoomd_spatial::{CellList, CellListBuilder};
/// use hoomd_vector::Cartesian;
///
/// // Define the cell width.
/// let cell_width = 2.0;
/// // Create a cell list object from the builder
/// let cell_list = CellListBuilder::<2>::new(cell_width).build();
/// ```
///
/// # Example constructing a `CellList` with particles using the builder API.
/// ```
/// use hoomd_spatial::{CellList, CellListBuilder};
/// use hoomd_vector::Cartesian;
/// // Define the cell width.
/// let cell_width = 2.0;
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
/// let indices = vec![0, 1, 2]; // Particle indices corresponding to positions.
/// // Build a cell list with particles.
/// let cell_list = CellListBuilder::<2>::new(cell_width)
///     .with_positions_and_indices(&positions, &indices)
///     .build();
/// ```
pub struct CellListBuilder<const D: usize> {
    /// The width of each cell.
    pub cell_width: f64,
    /// The positions of the particles in the cell list.
    pub positions: Vec<Cartesian<D>>,
    /// The indices of the particles in the cell list.
    pub indices: Vec<usize>,
}

impl<const D: usize> CellListBuilder<D> {
    /// Create a new cell list builder from the given cell width and positions.
    ///
    /// This is usually used with the build command, to construct a `CellList`. Empty cell list without particles can be created as well using the builder API.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_spatial::{CellList, CellListBuilder};
    /// use hoomd_vector::Cartesian;
    /// // Define the cell width.
    /// let cell_width = 2.0;
    /// // Create a cell list object from the builder
    /// let cell_list = CellListBuilder::<2>::new(cell_width).build();
    /// ```
    #[inline]
    #[must_use]
    pub fn new(cell_width: f64) -> Self {
        Self {
            cell_width,
            positions: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Adds particles to the newly initialized cell list builder in `CellList` builder API.
    ///
    /// Builder API supports optional parameters which are particle positions and indices for populating the `CellList`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_spatial::{CellList, CellListBuilder};
    /// use hoomd_vector::Cartesian;
    /// // Define the cell width.
    /// let cell_width = 2.0;
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
    /// let indices = vec![0, 1, 2]; // Particle indices corresponding to positions.
    /// // Build a cell list with particles.
    /// let cell_list = CellListBuilder::new(cell_width)
    ///     .with_positions_and_indices(&positions, &indices)
    ///     .build();
    /// ```
    #[inline]
    #[must_use]
    pub fn with_positions_and_indices(
        mut self,
        positions: &Vec<Cartesian<D>>,
        indices: &Vec<usize>,
    ) -> Self {
        self.positions.clone_from(positions);
        self.indices.clone_from(indices);
        self
    }

    /// Create an actual `CellList` from `CellListBuilder`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_spatial::{CellList, CellListBuilder};
    /// use hoomd_vector::Cartesian;
    /// // Define the cell width.
    /// let cell_width = 2.0;
    /// // Create a builder object
    /// let cell_list_builder = CellListBuilder::<2>::new(cell_width);
    /// // Create a cell list object from the builder
    /// let cell_list = cell_list_builder.build();
    /// ```
    #[inline]
    #[must_use]
    pub fn build(self) -> CellList<D> {
        CellList::new(self.cell_width, &self.positions, &self.indices)
    }
}

impl<const D: usize> CellList<D> {
    /// Builder API helper function.
    #[inline]
    #[must_use]
    pub fn builder(cell_width: f64) -> CellListBuilder<D> {
        CellListBuilder::new(cell_width)
    }

    /// A helper function which converts given positions to cell indices.
    /// To generalize this we will have to make it a Trait function
    #[expect(clippy::cast_possible_truncation, reason = "Intentional truncation.")]
    #[inline]
    fn cell_index_from_position(cell_width: f64, position: &Cartesian<D>) -> [i32; D] {
        std::array::from_fn(|j| (position.coordinates[j] / cell_width).floor() as i32) // TODO: instead We can have tryinto() here with expect. would need to test performance.
    }

    /// Create a new cell list from the given cell width and positions.
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
    /// let indices = vec![0, 1, 2]; // Particle indices corresponding to positions.
    /// // Define the cell width.
    /// let cell_width = 1.0;
    /// // Build the cell list from positions.
    /// let cell_list = CellList::<2>::new(cell_width, &positions, &indices);
    /// ```
    #[inline]
    #[must_use]
    pub fn new(cell_width: f64, positions: &[Cartesian<D>], indices: &[usize]) -> Self {
        let mut instance = Self {
            cell_width,
            particle_indices: HashMap::with_capacity(positions.len()),
            cell_index: HashMap::with_capacity(indices.len()),
        };

        for (position, index) in positions.iter().zip(indices.iter()) {
            instance.insert(position, index);
        }

        instance
    }

    /// Create an empty cell list with the given cell width.
    /// This is useful for initializing a cell list
    /// that will be populated later.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::CellList;
    /// // Create an empty 2D cell list with a cell width of 1.0.
    /// let cell_width = 1.0;
    /// let cell_list = CellList::<2>::empty(cell_width);
    /// ```
    #[inline]
    #[must_use]
    pub fn empty(cell_width: f64) -> Self {
        Self {
            cell_width,
            particle_indices: HashMap::new(),
            cell_index: HashMap::new(),
        }
    }

    /// Returns a cell index (in form of a tuple) for a given particle index.
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
    /// // Indices of particles corresponding to positions.
    /// let indices = vec![0, 1, 2];
    /// // Define the cell width.
    /// let cell_width = 1.0;
    /// // Build the cell list from positions.
    /// let cell_list = CellList::<2>::new(cell_width, &positions, &indices);
    ///
    /// // Get the cell index for the first particle.
    /// let cell_index = cell_list.cell_index(0).unwrap();
    /// ```
    #[inline]
    #[must_use]
    pub fn cell_index(&self, particle_index: usize) -> Option<&[i32; D]> {
        self.cell_index.get(&particle_index)
    }

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

    /// Remove particle from the cell list.
    ///
    /// Note that removing a particle from the cell list will not change the maximum particle index.
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
    pub fn remove(&mut self, particle_index: usize) {
        let cell_idx = self.cell_index.remove(&particle_index);
        if let Some(cell_idx) = cell_idx {
            // If the particle was found in the cell list, remove it from the particle indices.
            self.particle_indices
                .entry(cell_idx)
                .and_modify(|particle_indices| {
                    // Find the index of removed particle in the vector of particle indices.
                    if let Some(idx) = particle_indices.iter().position(|&x| x == particle_index) {
                        // Remove the particle index from the vector.
                        particle_indices.swap_remove(idx);
                    }
                });
        }
    }

    /// Translate particle to a new position.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_spatial::CellList;
    /// use hoomd_vector::Cartesian;
    ///
    /// // Create some sample 2D Cartesian positions.
    /// let positions = vec![
    /// Cartesian { coordinates: [0.2, 0.3] },
    /// Cartesian { coordinates: [0.8, 1.3] },
    /// Cartesian { coordinates: [8.5, 9.5] },
    /// ];
    /// let cell_width = 1.0;
    /// let indices = vec![0, 1, 2]; // Particle indices corresponding to positions.
    /// // Build the cell list from positions.
    /// let//  mut cell_list = CellList::<2>::new(cell_width, &positions, &indices);
    ///
    ///
    /// // Translate the first particle to a new position.
    /// let new_position = Cartesian { coordinates: [1.2, 1.3] };
    /// cell_list.translate_particle(0, new_position);
    /// ```
    #[inline]
    // TODO: Do I even need this function? It is the same as insert...
    pub fn translate_particle(
        &mut self,
        particle_index: usize,
        new_particle_position: Cartesian<D>,
    ) {
        self.insert(&new_particle_position, &particle_index);
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
    /// println!("Before shrink_to_fit: {:?}", cell_list.particle_indices.size());
    /// // Call shrink_to_fit to clean up empty cells and reduce memory usage.
    /// cell_list.shrink_to_fit();
    /// println!("After shrink_to_fit: {:?}", cell_list.particle_indices.size());
    /// ```
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.particle_indices.retain(|_, v| !v.is_empty());
        self.particle_indices.shrink_to_fit();
        self.cell_index.shrink_to_fit();
    }

    /// Find potential neighbor indices.
    ///
    /// This function finds the POTENTIAL neighbor indices for a given position and cutoff radius.
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
    /// let cell_width = 1.0;
    /// let indices = vec![0, 1, 2]; // Particle indices corresponding to positions.
    /// // Build the cell list from positions.
    /// let cell_list = CellList::<2>::new(cell_width, &positions, &indices);
    ///
    /// // Choose a query position (for example, the first one).
    /// let query_position = &positions[0];
    /// // Define a cutoff radius.
    /// let cutoff_radius = 1.5;
    ///
    /// // Call the function to find potential neighbor indices.
    /// let potential_neighbor_indices = cell_list
    ///     .find_potential_neighbor_indices(&0, &cutoff_radius)
    ///     .collect::<Vec<_>>();
    ///
    /// // Print the resulting neighbor indices.
    /// println!(
    ///     "Potential neighbor indices: {:?}",
    ///     potential_neighbor_indices
    /// );
    /// ```
    #[expect(clippy::cast_possible_truncation, reason = "Intentional truncation.")]
    #[inline]
    // TODO: instead of recursion, loop over the number of iterations and use //
    pub fn find_potential_neighbor_indices<'a>(
        &'a self,
        particle_index: &usize,
        cutoff_radius: &f64,
    ) -> impl Iterator<Item = usize> + 'a {
        // Generate all D‑dimensional offsets in [-n..=n]^D
        // Define a recursive helper function to generate all translation combinations.
        // For each dimension, it iterates from -max_offset to max_offset.
        // TODO figure out if there is a better way to do this. Is there an itertools
        // like functionality in std library? cartesian product?
        fn generate_translations<const D: usize>(
            i: usize,
            n: i32,
            current: &mut [i32; D],
            translations: &mut Vec<[i32; D]>,
        ) {
            if i == D {
                translations.push(*current);
            } else {
                for offset in -n..=n {
                    current[i] = offset;
                    generate_translations(i + 1, n, current, translations);
                }
            }
        }
        let particle_cell_idx = self.cell_index[particle_index];
        let n = (cutoff_radius / self.cell_width).ceil() as i32; // TODO try try_into but check performance
        let mut translations = Vec::new();
        let mut current = [0; D];
        generate_translations(0, n, &mut current, &mut translations);
        translations.into_iter().flat_map(move |delta| {
            // compute neighbor cell coords
            let mut c = particle_cell_idx;
            for i in 0..D {
                c[i] += delta[i];
            }
            // yield occupants or empty vec
            self.particle_indices
                .get(&c)
                .cloned()
                .unwrap_or_default()
                .into_iter()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_new_has_empty_state() {
        let cell_width = 1.0;
        let builder = CellList::<2>::builder(cell_width);
        assert_eq!(builder.cell_width, cell_width); // from Default
        assert!(builder.positions.is_empty());
        assert!(builder.indices.is_empty());
    }

    #[test]
    fn builder_with_positions_and_indices_works() {
        let positions = vec![
            Cartesian {
                coordinates: [0.0, 0.0],
            },
            Cartesian {
                coordinates: [1.0, 1.0],
            },
        ];
        let indices = vec![0, 1];
        let builder = CellList::<2>::builder(1.0).with_positions_and_indices(&positions, &indices);

        assert_eq!(builder.positions, positions);
        assert_eq!(builder.indices, indices);
    }

    #[test]
    fn builder_build_creates_celllist() {
        let positions = vec![
            Cartesian {
                coordinates: [0.5, 0.5],
            },
            Cartesian {
                coordinates: [1.5, 1.5],
            },
        ];
        let indices = vec![0, 1];

        let cell_list = CellList::<2>::builder(1.0)
            .with_positions_and_indices(&positions, &indices)
            .build();

        assert_eq!(cell_list.cell_width, 1.0);
        assert_eq!(cell_list.cell_index.len(), 2);
        assert_eq!(cell_list.particle_indices.len(), 2);
    }

    #[test]
    fn builder_can_chain_methods() {
        let cell_list = CellList::<2>::builder(1.0)
            .with_positions_and_indices(
                &vec![
                    Cartesian {
                        coordinates: [0.5, 0.5],
                    },
                    Cartesian {
                        coordinates: [2.1, 2.9],
                    },
                ],
                &vec![5, 9],
            )
            .build();

        assert_eq!(cell_list.cell_index.get(&5).unwrap(), &[0, 0]);
        assert_eq!(cell_list.cell_index.get(&9).unwrap(), &[2, 2]);
    }

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
        assert!(
            !particle_indices_in_cell.is_empty(),
            "Cell should not be empty"
        );
        assert!(
            particle_indices_in_cell.contains(&0),
            "Cell should contain the original particle"
        );
    }

    #[test]
    fn test_shrink_to_fit() {
        let cell_width = 1.0;
        let positions = vec![
            Cartesian {
                coordinates: [0.2, 0.3],
            },
            Cartesian {
                coordinates: [2.2, 2.3],
            },
        ];
        let indices = vec![0, 1]; // Particle indices corresponding to positions.
        let mut cell_list = CellList::<2>::new(cell_width, &positions, &indices);
        // check the size of the particle indices before removing any particles
        assert_eq!(cell_list.particle_indices.len(), 2);
        // Remove the first particle.
        cell_list.remove(0);
        // Check the size of the particle indices after removing a particle.
        assert_eq!(cell_list.particle_indices.len(), 2);
        // Now shrink to fit.
        cell_list.shrink_to_fit();
        // After shrinking, the size should be 1, as one particle was removed.
        assert_eq!(cell_list.particle_indices.len(), 1);
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
        let potential_neighbor_indices = cell_list
            .find_potential_neighbor_indices(&0, &cutoff_radius)
            .collect::<Vec<_>>();

        // p0's index should appear.
        assert!(potential_neighbor_indices.contains(&0));
        assert!(potential_neighbor_indices.contains(&1));
        assert!(potential_neighbor_indices.contains(&2));
        assert!(potential_neighbor_indices.contains(&3));
    }
}

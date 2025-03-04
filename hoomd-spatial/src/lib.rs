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
        let current_cell_idx = self
            .particle_idx_to_cell_index
            .get(&particle_index)
            .expect("Particle index found in the cell list.");
        let new_cell_idx = Self::cell_index_from_position(self.cell_width, &new_particle_position);
        if *current_cell_idx != new_cell_idx {
            self.remove_particle(particle_index);
            self.add_particle(&new_particle_position);
        }
    }

}
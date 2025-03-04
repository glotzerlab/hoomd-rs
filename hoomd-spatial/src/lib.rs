// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
/*! TODO DOCS */

use hoomd_vector::Cartesian;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParticleFlag {
    Real,
    Ghost,
}

pub struct CellList<const N: usize> {
    pub cell_width: f64,
    pub cell_idx_to_particle_indices: HashMap<[usize; N], Vec<usize>>,
    pub particle_idx_to_cell_index: HashMap<usize, [usize; N]>,
    pub particle_max_index: usize,
}

impl<const N: usize> CellList<N> {
    #[inline]
    fn cell_index_from_position(cell_width: f64, position: &Cartesian<N>) -> [usize; N] {
        std::array::from_fn(|j| (position.coordinates[j] / cell_width).floor() as usize)
    }

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

    pub fn cell_index_from_particle_index(&self, particle_index: usize) -> Option<&[usize; N]> {
        self.particle_idx_to_cell_index.get(&particle_index)
    }

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
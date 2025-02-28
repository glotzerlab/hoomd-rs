// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
/*! TODO DOCS */

use std::collections::HashMap;
use hoomd_vector::Cartesian;


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParticleFlag {
    Real,
    Ghost,
}


pub struct CellList<const N: usize> {
    pub cell_width: f64,
    pub cell_idx_to_particle_indices: HashMap<[i32; N], Vec<i32>>,
    pub particle_idx_to_cell_index: HashMap<i32, [i32; N]>,
    pub particle_max_index: i32,
}

impl<const N:usize> CellList<N> {
    pub fn new(cell_width: f64, positions: &Vec<Cartesian<N>>) -> Self {
        let mut particle_idx_to_cell_index = HashMap::new();
        let mut cell_idx_to_particle_indices = HashMap::new();
        for (i, position) in positions.iter().enumerate() {
            // Create the cell index for each particle.
            let cell_idx = std::array::from_fn(|j| (position.coordinates[j] / cell_width).floor() as i32);
            let particle_index: i32 = i as i32;
            cell_idx_to_particle_indices
                .entry(cell_idx)
                .or_insert(Vec::new())
                .push(particle_index);
            particle_idx_to_cell_index.insert(particle_index, cell_idx);
        }
        Self {
            cell_width,
            cell_idx_to_particle_indices,
            particle_idx_to_cell_index,
            particle_max_index: positions.len() as i32,
        }
    }

    #[inline]
    pub fn cell_index_from_position(&self, position: &Cartesian<N>) -> [i32; N] {
        std::array::from_fn(|j| (position.coordinates[j] / self.cell_width).floor() as i32)
    }

    pub fn cell_index_from_particle_index(&self, particle_index: i32) -> Option<[i32; N]> {
        self.particle_idx_to_cell_index.get(&particle_index).copied()
    }

    pub fn add_particle(&mut self, position: &Cartesian<N>) {
        self.particle_max_index += 1;
        let particle_index = self.particle_max_index;
        let cell_idx = self.cell_index_from_position(position);
        self.cell_idx_to_particle_indices
            .entry(cell_idx)
            .or_insert(Vec::new())
            .push(particle_index);
        self.particle_idx_to_cell_index.insert(particle_index, cell_idx);
    }

    pub fn remove_particle(&mut self, particle_index: i32) {
        // check if particle idx is smaller then particle max index
        assert!(particle_index <= self.particle_max_index);
        let cell_idx = self.particle_idx_to_cell_index.remove(&particle_index).unwrap();
        let particle_indices = self.cell_idx_to_particle_indices.get_mut(&cell_idx).unwrap();
        let index = particle_indices.iter().position(|&x| x == particle_index).unwrap();
        particle_indices.swap_remove(index);
    }

    pub fn translate_particle(&mut self, particle_index: i32, new_particle_position: Cartesian<N>){
        let current_cell_idx = self.particle_idx_to_cell_index.get(&particle_index).unwrap();
        let new_cell_idx = self.cell_index_from_position(&new_particle_position);
        if current_cell_idx != &new_cell_idx {
            self.remove_particle(particle_index);
            self.add_particle(&new_particle_position);
        }
    }

}
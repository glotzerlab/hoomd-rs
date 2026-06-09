// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Methods that compute properties of microstates.

use hoomd_microstate::{Body, Tagged};

mod translational_kinetic_energy;
mod rotational_kinetic_energy;

pub trait TranslationalKineticEnergy<B, S> {
    #[inline]
    fn translational_kinetic_energy(&self) -> (f64, usize) {
        self.translational_kinetic_energy_with_filter(|_| true)
    }

    fn translational_kinetic_energy_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(&self,
        should_sum_body: F) -> (f64, usize);
}

pub trait RotationalKineticEnergy<B, S> {
    #[inline]
    fn rotational_kinetic_energy(&self) -> (f64, usize) {
        self.rotational_kinetic_energy_with_filter(|_| true)
    }

    fn rotational_kinetic_energy_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(&self,
        should_sum_body: F) -> (f64, usize);
}

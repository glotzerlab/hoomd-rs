// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! TODO

#![allow(
    clippy::missing_inline_in_public_items,
    reason = "No need to inline macros"
)]

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

mod delta_energy_insert;
mod delta_energy_one;
mod delta_energy_remove;
mod maximum_interaction_range;
mod orientation;
mod position;
mod total_energy;

/// TODO
#[proc_macro_derive(DeltaEnergyInsert)]
pub fn delta_energy_insert_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    delta_energy_insert::delta_energy_insert(input).into()
}

/// TODO
#[proc_macro_derive(DeltaEnergyOne)]
pub fn delta_energy_one_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    delta_energy_one::delta_energy_one(input).into()
}

/// TODO
#[proc_macro_derive(DeltaEnergyRemove)]
pub fn delta_energy_remove_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    delta_energy_remove::delta_energy_remove(input).into()
}

/// TODO
#[proc_macro_derive(MaximumInteractionRange)]
pub fn maximum_interaction_range_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    maximum_interaction_range::maximum_interaction_range(input).into()
}

/// TODO
#[proc_macro_derive(Orientation)]
pub fn orientation_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    orientation::orientation(input)
}

/// TODO
#[proc_macro_derive(Position)]
pub fn position_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    position::position(input)
}

/// TODO
#[proc_macro_derive(TotalEnergy)]
pub fn total_energy_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    total_energy::total_energy(input).into()
}

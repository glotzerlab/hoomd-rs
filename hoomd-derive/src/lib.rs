// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Derive macros for traits from a variety of hoomd-rs crates.
//!
//! # Complete documentation
//!
//! `hoomd-derive` is is a part of *hoomd-rs*. Read the [complete documentation]
//! for more information.
//!
//! [complete documentation]: https://hoomd-rs.readthedocs.io

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
mod net_site_force;
mod orientation;
mod position;
mod site_pair_energy;
mod total_energy;

/// Automatically implement the `hoomd_interaction::DeltaEnergyInsert` trait.
///
/// The implemented `delta_energy_insert` sums the result of `delta_energy_insert`
/// over all fields. The implementation returns early when any one field returns
/// infinity.
///
/// Valid on:
/// * Structs with named fields.
/// * Tuple structs.
#[proc_macro_derive(DeltaEnergyInsert)]
pub fn delta_energy_insert_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    delta_energy_insert::delta_energy_insert(input).into()
}

/// Automatically implement the `hoomd_interaction::DeltaEnergyOne` trait.
///
/// The implemented `delta_energy_one` sums the result of `delta_energy_one`
/// over all fields. The implementation returns early when any one field returns
/// infinity.
///
/// Valid on:
/// * Structs with named fields.
/// * Tuple structs.
#[proc_macro_derive(DeltaEnergyOne)]
pub fn delta_energy_one_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    delta_energy_one::delta_energy_one(input).into()
}

/// Automatically implement the `hoomd_interaction::DeltaEnergyRemove` trait.
///
/// The implemented `delta_energy_remove` sums the result of `delta_energy_remove`
/// over all fields. The implementation returns early when any one field returns
/// infinity.
///
/// Valid on:
/// * Structs with named fields.
/// * Tuple structs.
#[proc_macro_derive(DeltaEnergyRemove)]
pub fn delta_energy_remove_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    delta_energy_remove::delta_energy_remove(input).into()
}

/// Automatically implement the `hoomd_interaction::MaximumInteractionRange` trait.
///
/// If the type has a `maximum_interaction_range` field, the derived implementation
/// returns it. If the type does not, the derived implementation returns the
/// maximum of `maximum_interaction_range()` of each field.
///
/// Valid on:
/// * Structs with named fields.
/// * Tuple structs.
#[proc_macro_derive(MaximumInteractionRange)]
pub fn maximum_interaction_range_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    maximum_interaction_range::maximum_interaction_range(input).into()
}

/// Automatically implement the `hoomd_interaction::NetSiteForce` trait.
///
/// The implemented `net_site_force` sums the result of `net_site_force`
/// over all fields.
///
/// Valid on:
/// * Structs with named fields.
/// * Tuple structs.
#[proc_macro_derive(NetSiteForce)]
pub fn net_site_force_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    net_site_force::net_site_force(input).into()
}

/// Automatically implement the `hoomd_microstate::property::Orientation` trait.
///
/// The derived implementation returns a reference to the structure's `orientation`
/// field.
///
/// Valid on structs with named fields.
#[proc_macro_derive(Orientation)]
pub fn orientation_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    orientation::orientation(input)
}

/// Automatically implement the `hoomd_microstate::property::Position` trait.
///
/// The derived implementation returns a reference to the structure's `position`
/// field.
///
/// Valid on structs with named fields.
#[proc_macro_derive(Position)]
pub fn position_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    position::position(input)
}

/// Automatically implement the `hoomd_interaction::SitePairEnergy` trait.
///
/// The implemented `site_pair_energy` sums the result of `site_pair_energy`
/// over all fields. The implementation returns early when any one field returns
/// infinity. The implemented `site_pair_energy_initial` behaves similarly.
/// The derived `is_only_infinite_or_zero` returns true only when all fields
/// also return true for the same method.
///
/// Valid on:
/// * Structs with named fields.
/// * Tuple structs.
#[proc_macro_derive(SitePairEnergy)]
pub fn site_pair_energy_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    site_pair_energy::site_pair_energy(input).into()
}

/// Automatically implement the `hoomd_interaction::TotalEnergy` trait.
///
/// The implemented `total_energy` sums the result of `total_energy`
/// over all fields. The implementation returns early when any one field returns
/// infinity.
///
/// Valid on:
/// * Structs with named fields.
/// * Tuple structs.
#[proc_macro_derive(TotalEnergy)]
pub fn total_energy_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    total_energy::total_energy(input).into()
}

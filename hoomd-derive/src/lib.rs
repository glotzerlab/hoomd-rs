// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! TODO

#![allow(
    clippy::missing_inline_in_public_items,
    reason = "No need to inline macros"
)]

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

mod maximum_interaction_range;
mod orientation;
mod position;

/// TODO
#[proc_macro_derive(MaximumInteractionRange)]
pub fn maximum_interaction_range_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    maximum_interaction_range::maximum_interaction_range(&input)
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

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! TODO: Overview and examples
//! # Tips
//!
//! When using a microstate for both simulation and analysis, set the maximum
//! interaction range to the larger of the model's maximum interaction range and
//! the largest neighbor distance you will use when computing order parameters.
//! You may get better performance setting `nominal_search_radius` to the
//! model's interaction range (when more time is spent in evaluating the model)
//! or the overall maximum range (when more time is spent evaluating order
//! parameters).

pub mod math;

mod k_atic_psi;
pub use k_atic_psi::k_atic_psi;

mod steinhardt;
pub use steinhardt::Steinhardt;

mod sites_in_ball;
pub use sites_in_ball::SitesInBall;

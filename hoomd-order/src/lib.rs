// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
// TODO: Documentation

mod local;
pub mod meshless_voro; 

use hoomd_microstate::Microstate;
use hoomd_vector::Cartesian;

pub use {
    local::{NeighborList, PowerDiagramCenters},
    meshless_voro::{Voronoi}
};

/** Computes the set of circle centers and radii whose power diagram gives the Voronoi tesselation
*/
pub trait PowerDiagram {
    /// Get the power diagram of a set of points
    fn power_diagram(&self) -> PowerDiagramCenters;
}

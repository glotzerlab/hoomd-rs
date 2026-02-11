// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Graphical elements that depict the simulation microstate.
//!
//! The `representation` module contains types that you can use to visually represent
//! your sites, bodies, or calculated properties of the simulation state. Each type
//! has implementation dependent details, although most include a `setup` method
//! that you need to add to the `Startup` schedule and helper methods that you can call
//! in the `Update` schedule to synchronize the state.
//!
//! Most of the types in `representation` are opaque primitives intended to
//! represent sites or bodies. Some, such as those ending in `Boundary` are thin
//! outlines intended to represent the simulation boundaries.

pub mod disk;
pub mod ellipse;
pub mod surface_mesh;
pub mod plane_mesh;

pub(crate) mod rectangular_boundary;
pub use rectangular_boundary::RectangularBoundary;

pub(crate) mod hyperbolic_disk;
pub use hyperbolic_disk::{HyperbolicDisk, HyperbolicDiskAssets, HyperbolicDiskMaterial};

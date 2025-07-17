// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Graphical elements that depict the simulation microstate.

The `representation` module contains types that you can use to visually represent
your sites, bodies, or calculated properties of the simulation state. Each type
has implementation dependent details, although most include a `setup` method
that you need to add to the `Startup` schedule and helper methods that you can call
in the `Update` schedule to synchronize the state.

Most of the types in `representation` are opaque primitives intended to
represent sites or bodies. Some, such as those ending in `Boundary` are thin
outlines intended to represent the simulation boundaries.
 */

pub(crate) mod disk;
pub use disk::{Disk, DiskAssets};

pub(crate) mod rectangular_boundary;
pub use rectangular_boundary::RectangularBoundary;

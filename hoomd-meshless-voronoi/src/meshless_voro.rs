// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! [`Meshless-voro`] package
 */

#![allow(clippy::all)]
#![allow(clippy::pedantic)]

#[allow(clippy::all)]
#[allow(clippy::pedantic)]
mod bounding_sphere;
#[allow(clippy::all)]
#[allow(clippy::pedantic)]
mod geometry;
#[allow(clippy::all)]
#[allow(clippy::pedantic)]
mod part;
#[allow(clippy::all)]
#[allow(clippy::pedantic)]
mod rtree_nn;
#[allow(clippy::all)]
#[allow(clippy::pedantic)]
mod simple_cycle;

#[allow(dead_code)]
#[allow(clippy::all)]
#[allow(clippy::pedantic)]
mod space;
#[allow(dead_code)]
#[allow(clippy::all)]
#[allow(clippy::pedantic)]
mod util;
#[allow(private_bounds)]
#[allow(clippy::all)]
#[allow(clippy::pedantic)]
mod voronoi;

#[allow(clippy::all)]
#[allow(clippy::pedantic)]
pub use voronoi::{
    convex_cell::Vertex, half_space::HalfSpace, integrals, ConvexCell, Dimensionality, Voronoi,
    VoronoiCell, VoronoiFace, VoronoiIntegrator,
};

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Geometric representations of common shapes in N-dimensional space.
//!
//! Geometric primitives defined in this package are designed to be lightweight
//! representations of geometry, independent of a global reference. This design
//! makes struct suitible for use with simulation code, and ensures shapes are
//! constructible from minimal information.
//!
//! For shapes with parameterizable dimension, a `const N: usize` generic parameter
//! encodes the dimensionality.

mod capsule;
pub use capsule::Capsule;

mod convex_polytope;
pub use convex_polytope::{ConvexPolygon, ConvexPolyhedron, ConvexPolytope};

mod cuboid;
pub use cuboid::{Cuboid, Rectangle};

mod parallelepiped;
pub use parallelepiped::{Hyperparallelepiped, Parallelepiped, Parallelogram};

mod cylinder;
pub use cylinder::Cylinder;

mod hyperellipsoid;
pub use hyperellipsoid::{Ellipse, Ellipsoid, Hyperellipsoid};

mod simplex3;
pub use simplex3::Simplex3;

mod sphere;
pub use sphere::{Circle, Hypersphere, Sphere};

mod sphero;
pub use sphero::Sphero;

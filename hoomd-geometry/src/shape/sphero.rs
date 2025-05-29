// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Sphero`] */

/** Round a `Shape` with some radius.
*/
pub struct Sphero<S> {
    /// The struct to be rounded. This is typically a `Shape`, but does not have to be
    pub shape: S,
    /// The radius of the rounding sphere
    pub rounding_radius: f64,
}

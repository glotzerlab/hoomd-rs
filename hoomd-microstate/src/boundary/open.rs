// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Open
*/

use super::Boundary;

/** Allow bodies and sites to exist anywhere in space.

Every point lies inside `Open` boundary conditions.
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Open;

impl<V, B, S> Boundary<V, B, S> for Open {
    #[inline]
    fn is_inside(&self, _point: &V) -> bool {
        true
    }
}

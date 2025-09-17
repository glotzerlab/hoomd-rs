// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Open
 */

use tinyvec::ArrayVec;

use super::{Error, GenerateGhosts, MAX_GHOSTS, Wrap};

/** Allow bodies and sites to exist anywhere in space.

Every point lies inside `Open` boundary conditions, bodies and sites
are never wrapped, and there are no ghost sites.
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Open;

impl<P> Wrap<P> for Open {
    #[inline]
    fn wrap(&self, properties: P) -> Result<P, Error> {
        Ok(properties)
    }
}

impl<S> GenerateGhosts<S> for Open
where
    S: Default,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        f64::INFINITY
    }

    #[inline]
    fn generate_ghosts(&self, _site_properties: &S) -> ArrayVec<[S; MAX_GHOSTS]> {
        ArrayVec::new()
    }
}

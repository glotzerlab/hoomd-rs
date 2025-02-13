// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! TODO
*/

use super::Particle;

/** Point
*/
#[derive(Clone, Copy)]
pub struct Point<V> {
    position: V,
}

impl<V: Copy> Particle<V> for Point<V> {
    #[inline]
    fn position(&self) -> &V {
        &self.position
    }

    #[inline]
    fn position_mut(&mut self) -> &mut V {
        &mut self.position
    }
}

impl<V: Default> Default for Point<V> {
    #[inline]
    fn default() -> Self {
        Self {
            position: V::default(),
        }
    }
}

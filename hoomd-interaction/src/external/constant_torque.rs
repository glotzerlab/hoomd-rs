// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`ConstantTorque`]

use hoomd_vector::Wedge;

use crate::SiteForceAndTorque;

/// Apply the same torque to every site, independent of the site's properties.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstantTorque<V: Wedge> {
    /// Torque $`[\mathrm{energy}]`$.
    pub torque: V::Bivector,
}

impl<S, V> SiteForceAndTorque<S> for ConstantTorque<V> where
V: Default + Wedge,
V::Bivector: Copy + Default,
{
    type Force = V;

    #[inline]
    fn site_force_and_torque(&self, _site_properties: &S) -> (V, V::Bivector) {
        (V::default(), self.torque)
    }
}

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`ConstantTorque`]
use serde::{Deserialize, Serialize};

use hoomd_vector::{Outer, Wedge};

use crate::SiteForceVirialAndTorque;

/// Apply the same torque to every site, independent of the site's properties.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstantTorque<V: Wedge> {
    /// Torque $`[\mathrm{energy}]`$.
    pub torque: V::Bivector,
}

impl<S, V> SiteForceVirialAndTorque<S> for ConstantTorque<V>
where
    V: Default + Wedge + Outer,
    V::Bivector: Copy + Default,
    V::Tensor: Default,
{
    type Force = V;

    #[inline]
    fn site_force_virial_and_torque(&self, _site_properties: &S) -> (V, V::Tensor, V::Bivector) {
        (V::default(), V::Tensor::default(), self.torque)
    }
}

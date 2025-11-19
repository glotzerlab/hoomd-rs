// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`ConstantTorque`]

use std::ops::Mul;

use hoomd_vector::{Unit, WedgeProduct};

use crate::SiteForceAndTorque;
use hoomd_microstate::{Microstate, Site};

/// Constant torque potential.
/// TODO: more documentation
#[derive(Clone, Debug, PartialEq)]
pub struct ConstantTorque<V: WedgeProduct> {
    /// Interaction strength *(\[energy\])*.
    pub alpha: f64,
    /// Direction of the torque vector *(unitless)*.
    /// TODO: figure out how to use Unit
    pub direction: V::Bivector,
}

impl<V, B, S, C> SiteForceAndTorque<V, B, S, C> for ConstantTorque<V>
where
    V: WedgeProduct + Default,
    V::Bivector: Default + Mul<f64, Output = V::Bivector> + Clone
{
    #[inline]
    fn net_force_and_torque_on_site(&self, _microstate: &Microstate<B, S, C>, _site: &Site<S>) -> (V, V::Bivector) {
        let force = V::default();
        let torque = self.direction.clone() * self.alpha;
        (force, torque)
    }
}

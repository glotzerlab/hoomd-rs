// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`ConstantTorque`]

use std::ops::Mul;

use hoomd_vector::WedgeProduct;

use crate::SiteForceAndTorque;
use hoomd_microstate::{Microstate, Site};

/// Constant torque potential.
/// 
/// `a` is the strength of the constant torque and
/// `direction` is the direction to apply the 
/// constant torque.
/// 
/// Apply a constant torque on each 
/// [`Site`](hoomd_microstate::Site) in the system
/// with zero force.
/// 
/// The force and torque are
/// 
/// ```math
/// \begin{align}
///     &\mathbf{f}_\alpha = 0 \\
///     &\boldsymbol{\tau}_\alpha = a \mathbf{d}
/// \end{align}
/// ```
/// Where $`a`$ is the strength of the constant torque and
/// $`\mathbf{d}`$ is the direction to apply the 
/// constant torque.
/// 
/// # Note
/// In two-dimension, the `direction` is a scalar
/// either +1 or -1.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstantTorque<V: WedgeProduct> {
    /// Interaction strength *(\[energy\])*.
    pub alpha: f64,
    /// Direction of the torque vector *(unitless)*.
    /// TODO: figure out how to use Unit
    pub direction: V::Bivector,
}

impl<V, B, S, X, C> SiteForceAndTorque<V, B, S, X, C> for ConstantTorque<V>
where
    V: WedgeProduct + Default,
    V::Bivector: Default + Mul<f64, Output = V::Bivector> + Clone
{
    /// Calculate the force and torque.
    #[inline]
    fn net_force_and_torque_on_site(&self, _microstate: &Microstate<B, S, X, C>, _site: &Site<S>) -> (V, V::Bivector) {
        let force = V::default();
        let torque = self.direction.clone() * self.alpha;
        (force, torque)
    }
}

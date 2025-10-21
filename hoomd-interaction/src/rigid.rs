// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Rigid intrabody interactions.
//!
//! This module provides the Rigid type, which handles intrabody summation of
//! forces and torques, and also provides logic for converting between site
//! forces and body torques.
//! 
//! TODO: Expand documentation.

use std::ops::AddAssign;

use hoomd_microstate::{property::{Orientation, Position}, Microstate, Transform};
use hoomd_vector::{Angle, Cartesian, Cross, Rotate, Rotation, RotationMatrix, Vector, Versor, WedgeProduct};

use crate::{NetBodyForce, NetBodyTorque, SiteForce, SiteTorque};

pub struct Rigid<E>(pub E);

impl<V, B, S, C, E> NetBodyForce<V, B, S, C> for Rigid<E>
where
    V: Vector + Default,
    B: Transform<S>,
    S: Position<Position = V>,
    E: SiteForce<V, B, S, C>,
{
    /// Sum the forces on the sites to get the net force on a body.
    #[inline]
    fn net_force_on_body(&self, microstate: &Microstate<B, S, C>, body_index: usize) -> V {
        let mut total = V::default();
        for site in microstate.iter_body_sites(body_index) {
            total += self.0.net_force_on_site(microstate, site);
        }
        total
    }
    // TODO: when doing this calculation for cutoff pairwise forces, consider
    // whether to track which body/sites have been calculated already, which
    // would prevent double-calculations.
}


impl<const N: usize, V, B, S, C, E, R> NetBodyTorque<N, V, B, S, C> for Rigid<E>
where
    V: Vector + WedgeProduct,
    B: Transform<S> + Orientation<Rotation = R>,
    S: Position<Position = V>,
    E: SiteForce<V, B, S, C>, // TODO: do we need + SiteTorque<V, B, S, C> ?
    R: Rotate<V>,
    RotationMatrix<N>: From<R>,
    V::Bivector: Default + AddAssign,
{
    #[inline]
    fn net_torque_on_body(&self, microstate: &Microstate<B, S, C>, body_index: usize) -> V::Bivector {
        let mut total = V::Bivector::default();

        let q = microstate.bodies()[body_index].item.properties.orientation();  // the body's orientation in the system frame
        // let q = RotationMatrix::from(*q);    // TODO: add a "to" method (microoptimization)
        
        // Torque based on forces on all sites around the center of mass
        for (site_index, site) in microstate.iter_body_sites(body_index).enumerate() {
            // Get relevant quantities
            let site_body_frame = &microstate.bodies()[body_index].item.sites[site_index];
            let r_body_frame = site_body_frame.position();                                  // the site's position in the body frame (which we need in order to not have wrapping issues)
            let r = q.rotate(r_body_frame);                                      // the moment arm in the system frame
            let f = self.0.net_force_on_site(microstate, site);                     // the force on the site in the system frame

            // Calculate Torque in the system frame
            let t = f.wedge_product(&r);

            // Add to the total
            total += t;
        }

        total
    }
}
// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Rigid intrabody interactions.
//!
//! This module provides the Rigid type, which handles intrabody summation of
//! forces and torques, and also provides logic for converting between site
//! forces and body torques.
//!
//! TODO: Expand documentation.

use std::ops::{AddAssign, Sub};

use crate::{
    NetBodyForce, NetBodyForceAndTorque, NetBodyForceAndVirial, NetBodyTorque, SiteForce,
    SiteForceAndVirial,
};
use hoomd_microstate::{
    Microstate, Transform,
    property::{Orientation, Position},
};
use hoomd_vector::{
    Angle, Cartesian, Cross, Rotate, Rotation, RotationMatrix, TensorProduct, Vector, Versor,
    WedgeProduct,
};
use hoomd_linear_algebra::GeneralMatrix;

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
    fn net_torque_on_body(
        &self,
        microstate: &Microstate<B, S, C>,
        body_index: usize,
    ) -> V::Bivector {
        let mut total = V::Bivector::default();

        let q = microstate.bodies()[body_index]
            .item
            .properties
            .orientation(); // the body's orientation in the system frame
        // let q = RotationMatrix::from(*q);    // TODO: add a "to" method (microoptimization)

        // Torque based on forces on all sites around the center of mass
        for (site_index, site) in microstate.iter_body_sites(body_index).enumerate() {
            // Get relevant quantities
            let site_body_frame = &microstate.bodies()[body_index].item.sites[site_index];
            let r_body_frame = site_body_frame.position(); // the site's position in the body frame (which we need in order to not have wrapping issues)
            let r = q.rotate(r_body_frame); // the moment arm in the system frame
            let f = self.0.net_force_on_site(microstate, site); // the force on the site in the system frame

            // Calculate Torque in the system frame
            let t = r.wedge_product(&f);

            // Add to the total
            total += t;
        }

        total
    }
}

impl<const N: usize, V, B, S, C, E, R> NetBodyForceAndTorque<N, V, B, S, C> for Rigid<E>
where
    V: Vector + WedgeProduct + Default,
    B: Transform<S> + Orientation<Rotation = R>,
    S: Position<Position = V>,
    E: SiteForce<V, B, S, C>, // TODO: do we need + SiteTorque<V, B, S, C> ?
    R: Rotate<V>,
    RotationMatrix<N>: From<R>,
    V::Bivector: Default + AddAssign,
{
    #[inline]
    fn net_force_and_torque_on_body(
        &self,
        microstate: &Microstate<B, S, C>,
        body_index: usize,
    ) -> (V, <V as WedgeProduct>::Bivector) {
        let mut total_force = V::default();
        let mut total_torque = V::Bivector::default();

        let q = microstate.bodies()[body_index]
            .item
            .properties
            .orientation(); // the body's orientation in the system frame
        // let q = RotationMatrix::from(*q);    // TODO: add a "to" method (microoptimization)

        // Torque based on forces on all sites around the center of mass
        for (site_index, site) in microstate.iter_body_sites(body_index).enumerate() {
            // Get relevant quantities
            let site_body_frame = &microstate.bodies()[body_index].item.sites[site_index];
            let r_body_frame = site_body_frame.position(); // the site's position in the body frame (which we need in order to not have wrapping issues)
            let r = q.rotate(r_body_frame); // the moment arm in the system frame
            let f = self.0.net_force_on_site(microstate, site); // the force on the site in the system frame

            // Calculate Torque in the system frame
            let t = r.wedge_product(&f);

            // Add to the total
            total_force += f;
            total_torque += t;
        }

        (total_force, total_torque)
    }
}

impl<V, B, S, C, E, R> NetBodyForceAndVirial<V, B, S, C> for Rigid<E>
where
    V: Vector + Default + TensorProduct,
    B: Transform<S> + Orientation<Rotation = R>,
    S: Position<Position = V>,
    E: SiteForceAndVirial<V, B, S, C>,
    R: Rotate<V>,
    V::Tensor: GeneralMatrix + AddAssign + Sub<Output = V::Tensor>
{
    /// Sum the forces on the sites to get the net force on a body.
    #[inline]
    fn net_force_and_virial_on_body(
        &self,
        microstate: &Microstate<B, S, C>,
        body_index: usize,
    ) -> (V, V::Tensor) {
        let mut total_force = V::default();
        let mut total_virial = V::Tensor::zeros();
        let q = microstate.bodies()[body_index]
            .item
            .properties
            .orientation(); // the body's orientation in the system frame

        for (site_index, site) in microstate.iter_body_sites(body_index).enumerate() {
            let site_body_frame = &microstate.bodies()[body_index].item.sites[site_index];
            let r_body_frame = site_body_frame.position(); // the site's position in the body frame (which we need in order to not have wrapping issues)
            let r = q.rotate(r_body_frame); // the moment arm in the system frame
            let (force, virial) = self.0.net_force_and_virial_on_site(microstate, site);

            // calculate the virial correction due to rigid body constraint.
            let virial_correction = force.tensor_product(&r);

            total_force += force;
            total_virial += virial - virial_correction;
        }
        (total_force, total_virial)
    }
}

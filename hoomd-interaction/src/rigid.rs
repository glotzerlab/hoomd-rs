// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Rigid.

use std::ops::{Add, AddAssign};
use serde::{Deserialize, Serialize};

use crate::{
    DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, MaximumInteractionRange, NetBodyForce, NetBodyForceAndTorque, NetSiteForce, NetSiteForceAndTorque, TotalEnergy
};
use hoomd_microstate::{
    Body, Microstate, Transform, property::{Orientation, Position}
};
use hoomd_vector::{Rotate, Vector, Wedge};

/// Rigid body interactions.
///
/// The [`Rigid`] newtype implements [`NetBodyForce`]  for wrapped force
/// interaction model types that implement [`NetSiteForce`]. It also implements
/// [`NetBodyForceAndTorque`] for interaction model types that implement
/// [`NetSiteForceAndTorque`].
///
/// [`Rigid`] computes the net force and torque on a rigid body that results
/// from the forces/torques on all of its sites:
/// ```math
/// \vec{F}_\mathrm{body} = \sum_{i \in \mathrm{body}} \vec{F}_{i}
/// ```
/// ```math
/// \vec{\tau}_\mathrm{body} = \sum_{i \in \mathrm{body}} (\mathbf{q}_\mathrm{body} \cdot \vec{r}_{\mathrm{body},i} \cdot \mathbf{q}_\mathrm{body}^*) \wedge \vec{F}_i + \vec{\tau}_{i}
/// ```
///
/// The generic type names are:
/// * `F`: The evaluator that implements [`NetSiteForce`] and/or [`NetSiteForceAndTorque`].
///
/// # Example
///
/// ```
/// use hoomd_interaction::{
///     Rigid, PairwiseCutoff, pairwise::Isotropic, univariate::LennardJones,
/// };
///
/// let lennard_jones: LennardJones = LennardJones {
///     epsilon: 1.0,
///     sigma: 1.0,
/// };
/// let evaluator = Isotropic{ interaction: lennard_jones, r_cut: 2.5};
/// let rigid = Rigid(PairwiseCutoff(evaluator));
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rigid<F>(pub F);

impl<V, B, S, X, C, F> NetBodyForce<B, S, X, C> for Rigid<F>
where
    V: Vector + Default,
    B: Transform<S>,
    S: Position<Position = V>,
    F: NetSiteForce<B, S, X, C, Force = V>,
{
    type Force = V;
    
    /// Compute the net force on a body in the microstate.
    ///
    /// The net force on a body is the sum of the net site forces for all sites
    /// in the body:
    /// ```math
    /// \vec{F}_\mathrm{body} = \sum_{i \in \mathrm{body}} \vec{F}_{i}
    /// ```
    /// where the net site force $` \vec{F}_i `$ is given by `F`'s implementation of
    /// [`NetSiteForce`].
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    ///
    /// use hoomd_interaction::{
    ///     Rigid, PairwiseCutoff, pairwise::Isotropic, univariate::LennardJones, NetBodyForce
    /// };
    /// use hoomd_microstate::{
    ///     Body, Microstate,
    ///     boundary::Open,
    ///     property::{OrientedPoint, Point},
    /// };
    /// use hoomd_vector::{Cartesian, Versor};
    ///
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::single_site(OrientedPoint {
    ///             position: Cartesian::from([0.0, 0.0, 0.0]),
    ///             orientation: Versor::default(),
    ///         },
    ///         Point::new(Cartesian::<3>::default()),
    ///         ),
    ///     Body::single_site(OrientedPoint {
    ///             position: Cartesian::from([1.0, 0.0, 0.0]),
    ///             orientation: Versor::default(),
    ///         },
    ///         Point::new(Cartesian::<3>::default()),
    ///         ),
    /// ])?;
    ///
    /// let lennard_jones: LennardJones = LennardJones {
    ///             epsilon: 1.0,
    ///             sigma: 1.0};
    ///
    /// let force_interaction_model = PairwiseCutoff(
    ///     Isotropic{ 
    ///         interaction: lennard_jones,
    ///         r_cut: 2.5,
    /// });
    /// let rigid = Rigid(force_interaction_model);
    ///
    /// let body_force_0 = rigid.net_body_force(&microstate, 0);
    /// let body_force_1 = rigid.net_body_force(&microstate, 1);
    ///
    /// assert_relative_eq!(body_force_0, Cartesian::from([-24.0, 0.0, 0.0]));
    /// assert_relative_eq!(body_force_1, Cartesian::from([24.0, 0.0, 0.0]));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn net_body_force(&self, microstate: &Microstate<B, S, X, C>, body_index: usize) -> V {
        let mut total = V::default();
        for site_index in microstate.iter_body_site_indices(body_index) {
            total += self.0.net_site_force(microstate, site_index);
       }
        total
    }
}

impl<V, B, S, X, C, F, R> NetBodyForceAndTorque<B, S, X, C> for Rigid<F>
where
    V: Vector + Wedge + Default,
    B: Transform<S> + Orientation<Rotation = R>,
    S: Position<Position = V>,
    F: NetSiteForceAndTorque<B, S, X, C, Force = V>,
    R: Rotate<V>,
    V::Bivector: Default + Add<Output = V::Bivector> + AddAssign,
{
    type Force = V;
    
    /// Compute the net force and torque on a body in the microstate.
    ///
    /// The net force on a body is the sum of the net site forces for all sites
    /// in the body, and the net torque is the sum of the torques resulting from those
    /// forces *and* intrinsic torques applied to the sites:
    /// ```math
    /// \vec{F}_\mathrm{body} = \sum_{i \in \mathrm{body}} \vec{F}_{i}
    /// ```
    /// ```math
    /// \vec{\tau}_\mathrm{body} = \sum_{i \in \mathrm{body}} (\mathbf{q}_\mathrm{body} \cdot \vec{r}_{\mathrm{body},i} \cdot \mathbf{q}_\mathrm{body}^*) \wedge \vec{F}_i + \vec{\tau}_{i}
    /// ```
    /// where $` \mathbf{q}_\mathrm{body} `$ is the body's orientation,
    /// $` \vec{r}_{\mathrm{body},i} `$ is the position of site *i* in the body
    /// frame, and $` \vec{F}_i `$ and $` \vec{\tau}_i `$ are the net site force and torque
    /// given by `F`'s implementation of [`NetSiteForceAndTorque`].
    ///
    /// The symbol $` \wedge `$ denotes the [`Wedge`] product. The resulting torque
    /// $` \vec{\tau}_\mathrm{body} `$ is in the system frame.
    ///
    /// # Example
    /// ```
    /// use hoomd_interaction::{
    ///     Rigid, PairwiseCutoff, pairwise::Isotropic, univariate::LennardJones, NetBodyForceAndTorque
    /// };
    ///
    /// use hoomd_microstate::{
    ///     Body, Microstate,
    ///     boundary::Open,
    ///     property::{OrientedPoint, Point},
    /// };
    /// use hoomd_vector::{Cartesian, Versor};
    ///
    /// use approxim::assert_relative_eq;
    ///
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::single_site(OrientedPoint {
    ///             position: Cartesian::from([0.0, 2.0, 0.0]),
    ///             orientation: Versor::default(),
    ///         },
    ///         Point::new(Cartesian::from([0.0, -2.0, 0.0])),
    ///         ),
    ///     Body::single_site(OrientedPoint {
    ///             position: Cartesian::from([1.0, 0.0, 0.0]),
    ///             orientation: Versor::default(),
    ///         },
    ///         Point::new(Cartesian::<3>::default())
    ///         ),
    /// ])?;
    ///
    /// let lennard_jones: LennardJones = LennardJones {
    ///             epsilon: 1.0,
    ///             sigma: 1.0};
    ///
    /// let force_interaction_model = PairwiseCutoff(
    ///     Isotropic{ 
    ///         interaction: lennard_jones,
    ///         r_cut: 2.5,
    /// });
    /// let rigid = Rigid(force_interaction_model);
    ///
    /// let (body_force, body_torque) = rigid.net_body_force_and_torque(&microstate, 0);
    ///
    /// assert_relative_eq!(body_force, Cartesian::from([-24.0, 0.0, 0.0]));
    /// assert_relative_eq!(body_torque, Cartesian::from([0.0, 0.0, -48.0]));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn net_body_force_and_torque(
        &self,
        microstate: &Microstate<B, S, X, C>,
        body_index: usize,
    ) -> (V, V::Bivector) {
        let mut total_force = V::default();
        let mut total_torque = V::Bivector::default();

        let q = microstate.bodies()[body_index]
            .item
            .properties
            .orientation();

        for (body_site_index, microstate_site_index) in microstate.iter_body_site_indices(body_index).enumerate() {
            let site_body_frame = &microstate.bodies()[body_index].item.sites[body_site_index];
            let r_body_frame = site_body_frame.position();
            let r = q.rotate(r_body_frame);
            let (site_force, site_torque) = self.0.net_site_force_and_torque(microstate, microstate_site_index);

            total_force += site_force;
            total_torque += r.wedge(&site_force) + site_torque;
        }

        (total_force, total_torque)
    }
}

impl<F> MaximumInteractionRange for Rigid<F>
where F: MaximumInteractionRange
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.0.maximum_interaction_range()
    }
}

impl<M, F> TotalEnergy<M> for Rigid<F>
where F: TotalEnergy<M>
{
    #[inline]
    fn total_energy(&self, microstate: &M) -> f64 {
        self.0.total_energy(microstate)
    }

    #[inline]
    fn delta_energy_total(&self, initial_microstate: &M,
        final_microstate: &M,
    ) -> f64 { self.0.delta_energy_total(initial_microstate, final_microstate) }
}

impl <B, S, X, C, F> DeltaEnergyOne<B, S, X, C> for Rigid<F>
where F: DeltaEnergyOne<B, S, X, C>
{
    #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
    self.0.delta_energy_one(initial_microstate, body_index, final_body)
    }
}

impl <B, S, X, C, F> DeltaEnergyInsert<B, S, X, C> for Rigid<F>
where F: DeltaEnergyInsert<B, S, X, C>
{
    #[inline]
    fn delta_energy_insert(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        new_body: &Body<B, S>,
    ) -> f64 {
    self.0.delta_energy_insert(initial_microstate, new_body)
    }
}

impl <B, S, X, C, F> DeltaEnergyRemove<B, S, X, C> for Rigid<F>
where F: DeltaEnergyRemove<B, S, X, C>
{
    #[inline]
    fn delta_energy_remove(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        body_index: usize,
    ) -> f64 {
    self.0.delta_energy_remove(initial_microstate, body_index)
    }
}

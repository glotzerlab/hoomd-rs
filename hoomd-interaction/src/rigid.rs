// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Rigid.

use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Sub};

use crate::{
    DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, MaximumInteractionRange,
    NetBodyForceAndVirial, NetBodyForceVirialAndTorque, NetSiteForceAndVirial,
    NetSiteForceVirialAndTorque, TotalEnergy,
};
use hoomd_microstate::{
    Body, Microstate, Transform,
    property::{Orientation, Position},
};
use hoomd_vector::{Outer, Rotate, Vector, Wedge};

/// Rigid body interactions.
///
/// The [`Rigid`] newtype implements [`NetBodyForceAndVirial`]  for wrapped force
/// interaction model types that implement [`NetSiteForceAndVirial`]. It also implements
/// [`NetBodyForceVirialAndTorque`] for interaction model types that implement
/// [`NetSiteForceVirialAndTorque`].
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
/// * `F`: The evaluator that implements [`NetSiteForceAndVirial`] and/or [`NetSiteForceVirialAndTorque`].
///
/// # Example
///
/// ```
/// use hoomd_interaction::{
///     PairwiseCutoff, Rigid, pairwise::Isotropic, univariate::LennardJones,
/// };
///
/// let lennard_jones: LennardJones = LennardJones {
///     epsilon: 1.0,
///     sigma: 1.0,
/// };
/// let evaluator = Isotropic {
///     interaction: lennard_jones,
///     r_cut: 2.5,
/// };
/// let rigid = Rigid(PairwiseCutoff(evaluator));
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rigid<F>(pub F);

impl<V, B, S, X, C, F> NetBodyForceAndVirial<B, S, X, C> for Rigid<F>
where
    V: Vector + Default + Outer,
    B: Transform<S> + Position<Position = V>,
    S: Position<Position = V>,
    F: NetSiteForceAndVirial<B, S, X, C, Force = V>,
    V::Tensor: Default + AddAssign + Sub<Output = V::Tensor>,
{
    type Force = V;

    /// Compute the net force and virial on a body in the microstate.
    ///
    /// The net force and virial on a body are the sums of the net forces and
    /// virials on all sites in the body:
    ///
    /// ```math
    /// \begin{align*}
    /// \vec{F}_\mathrm{body} &= \sum_{\mathrm{site} \in \mathrm{body}} \vec{F}_\mathrm{site} \\
    /// \mathbf{W}_\mathrm{body} &= \sum_{\mathrm{site} \in \mathrm{body}} \mathbf{W}_\mathrm{site} - \mathbf{F}_\mathrm{site} \otimes \left( \vec{r}_\mathrm{site}^\mathrm{global} - \vec{r}_{body}^\mathrm{global} \right) \\
    /// \end{align*}
    /// ```
    ///
    /// where the net site forces and virials are given by `F`'s implementation of
    /// [`NetSiteForceAndVirial`], and $`\vec{r}_\mathrm{site}^\mathrm{global}`$ and $`\vec{r}_\mathrm{body}^\mathrm{global}`$
    /// are the positions in the global frame of the site and body, respectively.
    ///
    /// The second term in the virial summation is required to correct for
    /// the centripetal forces implicit in the rigid body constraint. For more
    /// information, see [Glaser et al. 2020](https://doi.org/10.1016/j.commatsci.2019.109430),
    /// especially equations 23 and 24 and algorithm 2.
    ///
    /// # Example
    /// ```
    /// use approxim::assert_relative_eq;
    ///
    /// use hoomd_interaction::{
    ///     NetBodyForceAndVirial, PairwiseCutoff, Rigid, pairwise::Isotropic,
    ///     univariate::LennardJones,
    /// };
    /// use hoomd_linear_algebra::matrix::Matrix;
    /// use hoomd_microstate::{
    ///     Body, Microstate,
    ///     boundary::Open,
    ///     property::{OrientedPoint, Point},
    /// };
    /// use hoomd_vector::{Cartesian, Versor};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::single_site(
    ///         OrientedPoint {
    ///             position: Cartesian::from([0.0, 0.0, 0.0]),
    ///             orientation: Versor::default(),
    ///         },
    ///         Point::new(Cartesian::<3>::default()),
    ///     ),
    ///     Body::single_site(
    ///         OrientedPoint {
    ///             position: Cartesian::from([1.0, 0.0, 0.0]),
    ///             orientation: Versor::default(),
    ///         },
    ///         Point::new(Cartesian::<3>::default()),
    ///     ),
    /// ])?;
    ///
    /// let lennard_jones: LennardJones = LennardJones {
    ///     epsilon: 1.0,
    ///     sigma: 1.0,
    /// };
    ///
    /// let force_interaction_model = PairwiseCutoff(Isotropic {
    ///     interaction: lennard_jones,
    ///     r_cut: 2.5,
    /// });
    /// let rigid = Rigid(force_interaction_model);
    ///
    /// let (body_force_0, body_virial_0) =
    ///     rigid.net_body_force_and_virial(&microstate, 0);
    /// let (body_force_1, body_virial_1) =
    ///     rigid.net_body_force_and_virial(&microstate, 1);
    ///
    /// assert_relative_eq!(body_force_0, Cartesian::from([-24.0, 0.0, 0.0]));
    /// assert_eq!(
    ///     body_virial_0,
    ///     Matrix {
    ///         rows: [[12.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ///     }
    /// );
    ///
    /// assert_relative_eq!(body_force_1, Cartesian::from([24.0, 0.0, 0.0]));
    /// assert_eq!(
    ///     body_virial_1,
    ///     Matrix {
    ///         rows: [[12.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ///     }
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn net_body_force_and_virial(
        &self,
        microstate: &Microstate<B, S, X, C>,
        body_index: usize,
    ) -> (V, V::Tensor) {
        let body_position_global = microstate.bodies()[body_index].item.properties.position();

        let mut total_force = V::default();
        let mut total_virial = V::Tensor::default();

        for (body_site_index, microstate_site_index) in
            microstate.iter_body_site_indices(body_index).enumerate()
        {
            let (site_force, site_virial) = self
                .0
                .net_site_force_and_virial(microstate, microstate_site_index);

            // NetBodyForceAndVirial is implemented with as few assumptions as possible.
            // Use Transform to discover the relative position of the site in the global
            // frame and then subtract to find it relative to the body.
            let body = &microstate.bodies()[body_index];
            let site_position_global = *body
                .item
                .properties
                .transform(&body.item.sites[body_site_index])
                .position();

            let virial_correction =
                site_force.outer(&(site_position_global - *body_position_global));

            total_force += site_force;
            total_virial += site_virial - virial_correction;
        }
        (total_force, total_virial)
    }
}

impl<V, B, S, X, C, F, R> NetBodyForceVirialAndTorque<B, S, X, C> for Rigid<F>
where
    V: Vector + Wedge + Default + Outer,
    B: Transform<S> + Orientation<Rotation = R> + Position<Position = V>,
    S: Position<Position = V>,
    F: NetSiteForceVirialAndTorque<B, S, X, C, Force = V>,
    R: Rotate<V>,
    V::Bivector: Default + Add<Output = V::Bivector> + AddAssign,
    V::Tensor: Default + AddAssign + Sub<Output = V::Tensor>,
{
    type Force = V;

    /// Compute the net force, virial, and torque on a body in the microstate.
    ///
    /// The net force and virial on a body are the sums of the net forces and
    /// virials on all sites in the body, and the net torque is the sum of the
    /// torques resulting from those forces *and* intrinsic torques applied to
    /// the sites:
    ///
    /// ```math
    /// \begin{align*}
    /// \vec{F}_\mathrm{body} &= \sum_{\mathrm{site} \in \mathrm{body}} \vec{F}_\mathrm{site} \\
    /// \mathbf{W}_\mathrm{body} &= \sum_{\mathrm{site} \in \mathrm{body}} \mathbf{W}_\mathrm{site} - \mathbf{F}_\mathrm{site} \otimes \left( \vec{r}_\mathrm{site}^\mathrm{global} - \vec{r}_\mathrm{body}^\mathrm{global} \right) \\
    /// \vec{\tau}_\mathrm{body} &= \sum_{\mathrm{site} \in \mathrm{body}} (\mathbf{q}_{body} \cdot \vec{r}_{body,site} \cdot \mathbf{q}_{body}^*) \wedge \vec{F}_\mathrm{site} + \vec{\tau}_\mathrm{site} \\
    /// \end{align*}
    /// ```
    ///
    /// where $` \mathbf{q}_{body} `$ is the body's orientation,
    /// $` \vec{r}_{body,site} `$ is the position of site *i* in the body
    /// frame, and the net site forces, virials, and torques are given by `F`'s
    /// implementation of [`NetSiteForceVirialAndTorque`].
    ///
    /// The symbol $` \wedge `$ denotes the [`Wedge`] product. The resulting torque
    /// $` \vec{\tau}_{body} `$ is in the system frame.
    ///
    /// The second term in the virial summation is required to correct for
    /// the centripetal forces implicit in the rigid body constraint. For more
    /// information, see [Glaser et al. 2020](https://doi.org/10.1016/j.commatsci.2019.109430),
    /// especially equations 23 and 24 and algorithm 2.
    ///
    /// # Example
    /// ```
    /// use hoomd_interaction::{
    ///     NetBodyForceVirialAndTorque, PairwiseCutoff, Rigid,
    ///     pairwise::Isotropic, univariate::LennardJones,
    /// };
    ///
    /// use hoomd_linear_algebra::matrix::Matrix;
    /// use hoomd_microstate::{
    ///     Body, Microstate,
    ///     boundary::Open,
    ///     property::{OrientedPoint, Point},
    /// };
    /// use hoomd_vector::{Cartesian, Versor};
    ///
    /// use approxim::assert_relative_eq;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body::single_site(
    ///         OrientedPoint {
    ///             position: Cartesian::from([0.0, 2.0, 0.0]),
    ///             orientation: Versor::default(),
    ///         },
    ///         Point::new(Cartesian::from([0.0, -2.0, 0.0])),
    ///     ),
    ///     Body::single_site(
    ///         OrientedPoint {
    ///             position: Cartesian::from([1.0, 0.0, 0.0]),
    ///             orientation: Versor::default(),
    ///         },
    ///         Point::new(Cartesian::<3>::default()),
    ///     ),
    /// ])?;
    ///
    /// let lennard_jones: LennardJones = LennardJones {
    ///     epsilon: 1.0,
    ///     sigma: 1.0,
    /// };
    ///
    /// let force_interaction_model = PairwiseCutoff(Isotropic {
    ///     interaction: lennard_jones,
    ///     r_cut: 2.5,
    /// });
    /// let rigid = Rigid(force_interaction_model);
    ///
    /// let (body_force, body_virial, body_torque) =
    ///     rigid.net_body_force_virial_and_torque(&microstate, 0);
    ///
    /// assert_relative_eq!(body_force, Cartesian::from([-24.0, 0.0, 0.0]));
    /// assert_relative_eq!(body_torque, Cartesian::from([0.0, 0.0, -48.0]));
    /// assert_eq!(
    ///     body_virial,
    ///     Matrix {
    ///         rows: [[12.0, -48.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ///     }
    /// );
    ///
    /// let (body_force, body_virial, body_torque) =
    ///     rigid.net_body_force_virial_and_torque(&microstate, 1);
    ///
    /// assert_relative_eq!(body_force, Cartesian::from([24.0, 0.0, 0.0]));
    /// assert_relative_eq!(body_torque, Cartesian::from([0.0, 0.0, 0.0]));
    /// assert_eq!(
    ///     body_virial,
    ///     Matrix {
    ///         rows: [[12.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ///     }
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn net_body_force_virial_and_torque(
        &self,
        microstate: &Microstate<B, S, X, C>,
        body_index: usize,
    ) -> (V, V::Tensor, V::Bivector) {
        let mut total_force = V::default();
        let mut total_virial = V::Tensor::default();
        let mut total_torque = V::Bivector::default();

        let q = microstate.bodies()[body_index]
            .item
            .properties
            .orientation();

        for (body_site_index, microstate_site_index) in
            microstate.iter_body_site_indices(body_index).enumerate()
        {
            let site_body_frame = &microstate.bodies()[body_index].item.sites[body_site_index];
            let r_body_frame = site_body_frame.position();
            let r = q.rotate(r_body_frame);
            let (site_force, site_virial, site_torque) = self
                .0
                .net_site_force_virial_and_torque(microstate, microstate_site_index);

            let virial_correction = site_force.outer(&r);

            total_force += site_force;
            total_virial += site_virial - virial_correction;
            total_torque += r.wedge(&site_force) + site_torque;
        }

        (total_force, total_virial, total_torque)
    }
}

impl<F> MaximumInteractionRange for Rigid<F>
where
    F: MaximumInteractionRange,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.0.maximum_interaction_range()
    }
}

impl<M, F> TotalEnergy<M> for Rigid<F>
where
    F: TotalEnergy<M>,
{
    #[inline]
    fn total_energy(&self, microstate: &M) -> f64 {
        self.0.total_energy(microstate)
    }

    #[inline]
    fn delta_energy_total(&self, initial_microstate: &M, final_microstate: &M) -> f64 {
        self.0
            .delta_energy_total(initial_microstate, final_microstate)
    }
}

impl<B, S, X, C, F> DeltaEnergyOne<B, S, X, C> for Rigid<F>
where
    F: DeltaEnergyOne<B, S, X, C>,
{
    #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
        self.0
            .delta_energy_one(initial_microstate, body_index, final_body)
    }
}

impl<B, S, X, C, F> DeltaEnergyInsert<B, S, X, C> for Rigid<F>
where
    F: DeltaEnergyInsert<B, S, X, C>,
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

impl<B, S, X, C, F> DeltaEnergyRemove<B, S, X, C> for Rigid<F>
where
    F: DeltaEnergyRemove<B, S, X, C>,
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

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

use crate::{
    NetBodyForce, NetBodyForceAndTorque, NetBodyTorque, SiteForceAndTorque,
};
use hoomd_microstate::{
    Microstate, Transform,
    property::{Orientation, Position},
};
use hoomd_vector::{Rotate, RotationMatrix, Vector, WedgeProduct};

/// Rigid body interactions.
///
/// The generic type names are:
/// * `E`: The evaluator that implements [`SiteForceAndTorque`].
/// 
/// Given an evaluator,
/// [`Rigid`] provides methods for summing the forces and torques on every 
/// [`Site`](hoomd_microstate::Site) to determine net forces, 
/// and torques on every [`Body`](hoomd_microstate::Body).
///
/// Use [`Rigid`] by wrapping it around [`CutoffPair`](crate::cutoff_pair::CutoffPair), 
/// [`External`](crate::External) or your own custom type.
///
/// # Example
///
/// ```
/// use hoomd_interaction::{
///     rigid::Rigid, PairwiseCutoff, pairwise::Isotropic, univariate::LennardJones,
/// };
///
/// let lennard_jones: LennardJones = LennardJones {
///     epsilon: 1.5,
///     sigma: 2.0,
/// };
/// let evaluator = Isotropic{ interaction: lennard_jones, r_cut: 2.0*6.0};
/// let rigid = Rigid(PairwiseCutoff(evaluator));
/// ```
pub struct Rigid<E>(pub E);

impl<V, B, S, X, C, E> NetBodyForce<V, B, S, X, C> for Rigid<E>
where
    V: Vector + Default + WedgeProduct,
    B: Transform<S>,
    S: Position<Position = V>,
    E: SiteForceAndTorque<V, B, S, X, C>,
{
    /// Compute the net force.
    ///
    /// `microstate` describes the system configuration and `body_index` specifies
    /// the body with index $`i`$ within the system for which the net force
    /// $`\mathbf{f}_i`$ is calculated.
    ///
    /// First, the net force acting on each constituent [`Site`](hoomd_microstate::Site)
    /// $`\alpha`$ are calculated in [`CutoffPair::net_force_and_torque_on_site`](crate::cutoff_pair::CutoffPair)
    /// and [`External::net_force_and_torque_on_site`](crate::External).
    ///
    /// Then, the net force acting on the [`Body`](hoomd_microstate::Body)
    /// $`i`$ are calculated
    ///
    /// ```math
    /// \begin{align}
    ///     &\mathbf{f}_{i} = \sum_{\alpha} \mathbf{f}_{i, \alpha} \\
    /// \end{align}
    /// ```
    ///
    /// # Example
    /// ```
    /// use hoomd_interaction::{
    ///     rigid::Rigid, PairwiseCutoff, pairwise::Isotropic, univariate::LennardJones, NetBodyForce
    /// };
    /// use hoomd_linear_algebra::{
    ///     GeneralMatrix,
    ///     matrix::Matrix,
    /// };
    ///
    /// use hoomd_microstate::{
    ///     Body, Microstate,
    ///     boundary::Open,
    ///     property::{OrientedPoint, Point},
    /// };
    /// use hoomd_vector::{Cartesian, Versor};
    ///
    /// use approxim::assert_abs_diff_eq;
    ///
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body {
    ///         properties: OrientedPoint {
    ///             position: Cartesian::from([0.0, 0.0, 0.0]),
    ///             orientation: Versor::default(),
    ///         },
    ///         sites: vec![Point {
    ///             position: Cartesian::<3>::default(),
    ///         }],
    ///         },
    ///     Body {
    ///         properties: OrientedPoint {
    ///             position: Cartesian::from([1.0, 0.0, 0.0]),
    ///             orientation: Versor::default(),
    ///         },
    ///         sites: vec![Point {
    ///             position: Cartesian::<3>::default(),
    ///         }],
    ///         },
    /// ])?;
    ///
    /// let force = Rigid(PairwiseCutoff(
    ///     Isotropic{ 
    ///         interaction: LennardJones::<12, 6> {
    ///                 epsilon: 1.0,
    ///                 sigma: 2.0_f64.powf(-1.0 / 6.0),
    ///         }, 
    ///         r_cut: 6.0,
    /// }));
    ///
    ///    
    /// let net_force = force.net_force_on_body(&microstate, 0);
    ///
    /// assert_abs_diff_eq!(net_force, Cartesian::from([0.0, 0.0, 0.0]), epsilon = 1e-14);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn net_force_on_body(&self, microstate: &Microstate<B, S, X, C>, body_index: usize) -> V {
        let mut total = V::default();
        for site in microstate.iter_body_sites(body_index) {
            let (f_on_site, _) = self.0.net_force_and_torque_on_site(microstate, site);
            total += f_on_site;
        }
        total
    }
}

impl<const N: usize, V, B, S, X, C, E, R> NetBodyTorque<N, V, B, S, X, C> for Rigid<E>
where
    V: Vector + WedgeProduct,
    B: Transform<S> + Orientation<Rotation = R>,
    S: Position<Position = V>,
    E: SiteForceAndTorque<V, B, S, X, C>,
    R: Rotate<V>,
    RotationMatrix<N>: From<R>,
    V::Bivector: Default + AddAssign,
{
    /// Compute the net torque.
    ///
    /// `microstate` describes the system configuration and `body_index` specifies
    /// the body with index $`i`$ within the system for which the net torque
    /// $`\boldsymbol{\tau}_i`$ is calculated.
    ///
    /// First, the net force and torque acting on each constituent [`Site`](hoomd_microstate::Site)
    /// $`\alpha`$ are calculated in [`CutoffPair::net_force_and_torque_on_site`](crate::cutoff_pair::CutoffPair).
    /// and [`External::net_force_and_torque_on_site`](crate::External).
    ///
    /// Then, the net force and torque acting on the [`Body`](hoomd_microstate::Body)
    /// $`i`$ are calculated
    ///
    /// ```math
    /// \begin{align}
    ///     &\boldsymbol{\tau}_{i} = \sum_{\alpha} q_i\mathbf{r}_{\mathrm{body}, \alpha}q_i^* \wedge \mathbf{f}_{i, \alpha} + \boldsymbol{\tau}_{i, \alpha}
    /// \end{align}
    /// ```
    /// Where $`q_i`$ is the orientation of the [`Body`](hoomd_microstate::Body)
    /// $`i`$ and $`\mathbf{r}_{\mathrm{body}, \alpha}`$
    /// is the position of the constituent [`Site`](hoomd_microstate::Site)
    /// $`\alpha`$ in the body frame. The symbol $`\wedge`$ represents
    /// the [WedgeProduct], equivalent to [Cross](hoomd_vector::Cross) in three-dimension.
    ///
    /// # Example
    /// ```
    /// use hoomd_interaction::{
    ///     rigid::Rigid, PairwiseCutoff, pairwise::Isotropic, univariate::LennardJones, NetBodyTorque
    /// };
    ///
    /// use hoomd_microstate::{
    ///     Body, Microstate,
    ///     boundary::Open,
    ///     property::{OrientedPoint, Point},
    /// };
    /// use hoomd_vector::{Cartesian, Versor};
    ///
    /// use approxim::assert_abs_diff_eq;
    ///
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body {
    ///         properties: OrientedPoint {
    ///             position: Cartesian::from([0.0, 0.0, 0.0]),
    ///             orientation: Versor::default(),
    ///         },
    ///         sites: vec![Point {
    ///             position: Cartesian::from([0.0, 3.0_f64.sqrt() / 2.0, 0.0]),
    ///         }],
    ///         },
    ///     Body {
    ///         properties: OrientedPoint {
    ///             position: Cartesian::from([0.5, 0.0, 0.0]),
    ///             orientation: Versor::default(),
    ///         },
    ///         sites: vec![Point {
    ///             position: Cartesian::<3>::default(),
    ///         }],
    ///         },
    /// ])?;
    ///
    /// let force = Rigid(PairwiseCutoff(
    ///     Isotropic{ 
    ///         interaction: LennardJones::<12, 6> {
    ///                 epsilon: 1.0,
    ///                 sigma: 2.0_f64.powf(-1.0 / 6.0),
    ///         }, 
    ///         r_cut: 6.0,
    /// }));
    /// 
    /// let net_torque = force.net_torque_on_body(&microstate, 0);
    ///
    /// assert_abs_diff_eq!(net_torque, Cartesian::from([0.0, 0.0, 0.0]), epsilon = 1e-14);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Note
    ///
    /// The current implementation assumes the pure torque $`\boldsymbol{\tau}_{i, \alpha}`$ acting on the
    /// constituent [`Site`](hoomd_microstate::Site) $`\alpha`$
    /// (not contributed by the force) solely
    /// results from the external field in [`External`](crate::External).
    #[inline]
    fn net_torque_on_body(
        &self,
        microstate: &Microstate<B, S, X, C>,
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

            let (f_on_site, t_on_site) = self.0.net_force_and_torque_on_site(microstate, site); // the force on the site is in the system frame

            // Calculate Torque in the system frame
            let t_from_f_on_site = r.wedge_product(&f_on_site);

            // Add to the total
            total += t_from_f_on_site;

            // Torque from torques directly on the sites
            total += t_on_site;
        }

        total
    }
}

impl<const N: usize, V, B, S, X, C, E, R> NetBodyForceAndTorque<N, V, B, S, X, C> for Rigid<E>
where
    V: Vector + WedgeProduct + Default,
    B: Transform<S> + Orientation<Rotation = R>,
    S: Position<Position = V>,
    E: SiteForceAndTorque<V, B, S, X, C>,
    R: Rotate<V>,
    RotationMatrix<N>: From<R>,
    V::Bivector: Default + AddAssign,
{
    /// Compute the net force and torque.
    ///
    /// The force that is associate with the torque calculation will be reused
    /// in this function to reduce the costs.
    ///
    /// `microstate` describes the system configuration and `body_index` specifies
    /// the body with index $`i`$ within the system for which the net force and torque
    /// $`\mathbf{f}_i`$, $`\boldsymbol{\tau}_i`$ are calculated.
    ///
    /// First, the net force acting on each constituent [`Site`](hoomd_microstate::Site)
    /// $`\alpha`$ are calculated in [`CutoffPair::net_force_and_torque_on_site`](crate::cutoff_pair::CutoffPair).
    /// and [`External::net_force_and_torque_on_site`](crate::External).
    ///
    /// Then, the net force and torque acting on the [`Body`](hoomd_microstate::Body)
    /// $`i`$ are calculated
    ///
    /// ```math
    /// \begin{align}
    ///     &\mathbf{f}_{i} = \sum_{\alpha} \mathbf{f}_{i, \alpha} \\
    ///     &\boldsymbol{\tau}_{i} = \sum_{\alpha} q_i\mathbf{r}_{\mathrm{body}, \alpha}q_i^* \wedge \mathbf{f}_{i, \alpha} + \boldsymbol{\tau}_{i, \alpha}
    /// \end{align}
    /// ```
    /// Where $`q_i`$ is the orientation of the [`Body`](hoomd_microstate::Body)
    /// $`i`$ and $`\mathbf{r}_{\mathrm{body}, \alpha}`$
    /// is the position of the constituent [`Site`](hoomd_microstate::Site)
    /// $`\alpha`$ in the body frame. The symbol $`\wedge`$ represents
    /// the [WedgeProduct], equivalent to [Cross](hoomd_vector::Cross) in three-dimension.
    ///
    /// # Example
    /// ```
    /// use hoomd_interaction::{
    ///     rigid::Rigid, PairwiseCutoff, pairwise::Isotropic, univariate::LennardJones, NetBodyForceAndTorque
    /// };
    ///
    /// use hoomd_microstate::{
    ///     Body, Microstate,
    ///     boundary::Open,
    ///     property::{OrientedPoint, Point},
    /// };
    /// use hoomd_vector::{Cartesian, Versor};
    ///
    /// use approxim::assert_abs_diff_eq;
    ///
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut microstate = Microstate::new();
    /// microstate.extend_bodies([
    ///     Body {
    ///         properties: OrientedPoint {
    ///             position: Cartesian::from([0.0, 0.0, 0.0]),
    ///             orientation: Versor::default(),
    ///         },
    ///         sites: vec![Point {
    ///             position: Cartesian::from([0.0, 3.0_f64.sqrt() / 2.0, 0.0]),
    ///         }],
    ///         },
    ///     Body {
    ///         properties: OrientedPoint {
    ///             position: Cartesian::from([0.5, 0.0, 0.0]),
    ///             orientation: Versor::default(),
    ///         },
    ///         sites: vec![Point {
    ///             position: Cartesian::<3>::default(),
    ///         }],
    ///         },
    /// ])?;
    ///
    /// let force = Rigid(PairwiseCutoff(
    ///     Isotropic{ 
    ///         interaction: LennardJones::<12, 6> {
    ///                 epsilon: 1.0,
    ///                 sigma: 2.0_f64.powf(-1.0 / 6.0),
    ///         }, 
    ///         r_cut: 6.0,
    /// }));
    ///
    /// let (net_force, net_torque) = force.net_force_and_torque_on_body(&microstate, 0);
    ///
    /// assert_abs_diff_eq!(net_force, Cartesian::from([0.0, 0.0, 0.0]), epsilon = 1e-13);
    /// assert_abs_diff_eq!(net_torque, Cartesian::from([0.0, 0.0, 0.0]), epsilon = 1e-14);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Note
    ///
    /// The current implementation assumes the pure torque $`\boldsymbol{\tau}_{i, \alpha}`$ acting on the
    /// constituent [`Site`](hoomd_microstate::Site) $`\alpha`$
    /// (not contributed by the force) solely
    /// results from the external field in [`External`](crate::External).
    #[inline]
    fn net_force_and_torque_on_body(
        &self,
        microstate: &Microstate<B, S, X, C>,
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
            let (f_on_site, t_on_site) = self.0.net_force_and_torque_on_site(microstate, site); // the force on the site in the system frame

            // Calculate Torque in the system frame
            let t_from_f_on_site = r.wedge_product(&f_on_site);

            // Add to the total
            total_force += f_on_site;
            total_torque += t_from_f_on_site;

            // Torque from torques directly on the sites
            total_torque += t_on_site;
        }

        (total_force, total_torque)
    }
}


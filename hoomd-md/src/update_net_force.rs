// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Define `UpdateNetForce`

use rayon::prelude::*;

use hoomd_interaction::{NetBodyForceAndVirial, NetBodyForceVirialAndTorque};
use hoomd_microstate::{
    Microstate,
    property::{NetForce, NetTorque, NetVirial},
};
use hoomd_vector::{Outer, Vector, Wedge};

/// Compute the net force and virial given by an interaction model and apply it
/// to each body in the microstate.
///
/// Given an interaction model that implements [`NetBodyForceAndVirial`],
/// [`UpdateNetForceAndVirial`] sets the [`NetForce`] and [`NetVirial`]
/// properties of each body in the microstate to the one computed by the
/// interaction model.
///
/// [`NetBodyForceAndVirial`]: hoomd_interaction::NetBodyForceAndVirial
/// [`NetForce`]: hoomd_microstate::property::NetForce
/// [`NetVirial`]: hoomd_microstate::property::NetVirial
///
/// # Example
/// ```
/// use hoomd_interaction::{
///     PairwiseCutoff, Rigid, pairwise::Isotropic, univariate::LennardJones,
/// };
/// use hoomd_md::UpdateNetForceAndVirial;
/// use hoomd_microstate::{
///     Body, Microstate,
///     property::{DynamicPoint, Point},
/// };
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::builder()
///     .bodies([
///         Body::single_site(DynamicPoint::default(), Point::default()),
///         Body::single_site(
///             DynamicPoint {
///                 position: Cartesian::<2>::from([2.0, 0.0]),
///                 ..Default::default()
///             },
///             Point::default(),
///         ),
///     ])
///     .try_build()?;
///
/// let lennard_jones = LennardJones::<12, 6>::default();
/// let pairwise_cutoff = PairwiseCutoff(Isotropic {
///     interaction: lennard_jones,
///     r_cut: 2.5,
/// });
/// let rigid = Rigid(pairwise_cutoff);
///
/// microstate.update_net_force_and_virial(&rigid);
/// #   Ok(())
/// # }
/// ```
pub trait UpdateNetForceAndVirial<E> {
    /// Compute and set the net force and virial on each body.
    fn update_net_force_and_virial(&mut self, interaction_model: &E);
}

/// Compute the net force, virial, and torque given by an interaction model and
/// apply them to each body in the microstate.
///
/// Given an interaction model that implements [`NetBodyForceVirialAndTorque`],
/// [`UpdateNetForceVirialAndTorque`] sets the [`NetForce`], [`NetVirial`] and
/// [`NetTorque`] properties of each body in the microstate to the ones computed
/// by the interaction model.
///
/// [`NetBodyForceVirialAndTorque`]: hoomd_interaction::NetBodyForceVirialAndTorque
/// [`NetForce`]: hoomd_microstate::property::NetForce
/// [`NetVirial`]: hoomd_microstate::property::NetVirial
/// [`NetTorque`]: hoomd_microstate::property::NetTorque
///
/// # Example
///
/// ```
/// use hoomd_interaction::{
///     PairwiseCutoff, Rigid, pairwise::Isotropic, univariate::LennardJones,
/// };
/// use hoomd_md::UpdateNetForceVirialAndTorque;
/// use hoomd_microstate::{
///     Body, Microstate,
///     property::{DynamicOrientedPoint, Point},
/// };
/// use hoomd_vector::{Angle, Cartesian};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate: Microstate<
///     DynamicOrientedPoint<Cartesian<2>, Angle>,
///     Point<Cartesian<2>>,
///     _,
///     _,
/// > = Microstate::builder()
///     .bodies([
///         Body::single_site(
///             DynamicOrientedPoint {
///                 position: Cartesian::<2>::from([0.0, -1.0]),
///                 ..Default::default()
///             },
///             Point::new([0.0, 1.0].into()),
///         ),
///         Body::single_site(
///             DynamicOrientedPoint {
///                 position: Cartesian::<2>::from([2.0, -2.0]),
///                 ..Default::default()
///             },
///             Point::new([0.0, 2.0].into()),
///         ),
///     ])
///     .try_build()?;
///
/// let lennard_jones = LennardJones::<12, 6>::default();
/// let pairwise_cutoff = PairwiseCutoff(Isotropic {
///     interaction: lennard_jones,
///     r_cut: 2.5,
/// });
/// let rigid = Rigid(pairwise_cutoff);
///
/// microstate.update_net_force_virial_and_torque(&rigid);
/// #   Ok(())
/// # }
/// ```
pub trait UpdateNetForceVirialAndTorque<E> {
    /// Compute and set the net force, virial, and torque on each body.
    fn update_net_force_virial_and_torque(&mut self, interaction_model: &E);
}

impl<V, B, S, X, C, E> UpdateNetForceAndVirial<E> for Microstate<B, S, X, C>
where
    V: Default + Vector + Outer + Send,
    V::Tensor: Copy + Send,
    B: NetForce<NetForce = V> + NetVirial<NetVirial = V::Tensor> + Sync,
    S: Sync,
    X: Sync,
    C: Sync,
    E: NetBodyForceAndVirial<B, S, X, C, Force = V> + Sync,
{
    #[inline]
    fn update_net_force_and_virial(&mut self, interaction_model: &E) {
        let mut net_force_and_virial_tmp = Vec::new();

        (0..self.bodies().len())
            .into_par_iter()
            .map(|body_index| interaction_model.net_body_force_and_virial(self, body_index))
            .collect_into_vec(&mut net_force_and_virial_tmp);

        for (body_index, (net_force, net_virial)) in net_force_and_virial_tmp.iter().enumerate() {
            self.set_body_net_force(body_index, *net_force);
            self.set_body_net_virial(body_index, *net_virial);
        }
    }
}

impl<V, B, S, X, C, E> UpdateNetForceVirialAndTorque<E> for Microstate<B, S, X, C>
where
    V: Default + Vector + Wedge + Outer + Send,
    V::Tensor: Copy + Send,
    V::Bivector: Copy + Send,
    B: NetForce<NetForce = V>
        + NetVirial<NetVirial = V::Tensor>
        + NetTorque<NetTorque = V::Bivector>
        + Sync,
    S: Sync,
    X: Sync,
    C: Sync,
    E: NetBodyForceVirialAndTorque<B, S, X, C, Force = V> + Sync,
{
    #[inline]
    fn update_net_force_virial_and_torque(&mut self, interaction_model: &E) {
        let mut net_force_virial_and_torque_tmp = Vec::new();

        (0..self.bodies().len())
            .into_par_iter()
            .map(|body_index| interaction_model.net_body_force_virial_and_torque(self, body_index))
            .collect_into_vec(&mut net_force_virial_and_torque_tmp);

        for (body_index, (net_force, net_virial, net_torque)) in
            net_force_virial_and_torque_tmp.iter().enumerate()
        {
            self.set_body_net_force(body_index, *net_force);
            self.set_body_net_virial(body_index, *net_virial);
            self.set_body_net_torque(body_index, *net_torque);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approxim::assert_relative_eq;
    use assert2::check;

    use hoomd_interaction::{PairwiseCutoff, Rigid, pairwise::Isotropic, univariate::LennardJones};
    use hoomd_microstate::{
        Body,
        property::{DynamicOrientedPoint, DynamicPoint, Point},
    };
    use hoomd_vector::{Angle, Cartesian, Versor};

    // TODO: add virial tests

    #[test]
    fn net_force_2d() -> anyhow::Result<()> {
        let mut microstate = Microstate::builder()
            .bodies([
                Body::single_site(DynamicPoint::default(), Point::default()),
                Body::single_site(
                    DynamicPoint {
                        position: Cartesian::<2>::from([2.0, 0.0]),
                        ..Default::default()
                    },
                    Point::default(),
                ),
            ])
            .try_build()?;

        let lennard_jones = LennardJones::<12, 6>::default();
        let pairwise_cutoff = PairwiseCutoff(Isotropic {
            interaction: lennard_jones,
            r_cut: 2.5,
        });
        let rigid = Rigid(pairwise_cutoff);

        check!(microstate.bodies()[0].item.properties.net_force == [0.0, 0.0].into());
        check!(microstate.bodies()[1].item.properties.net_force == [0.0, 0.0].into());

        microstate.update_net_force_and_virial(&rigid);

        assert_relative_eq!(
            microstate.bodies()[0].item.properties.net_force,
            [93.0 / 512.0, 0.0].into()
        );
        assert_relative_eq!(
            microstate.bodies()[1].item.properties.net_force,
            [-93.0 / 512.0, 0.0].into()
        );

        Ok(())
    }

    #[test]
    fn net_force_3d() -> anyhow::Result<()> {
        let mut microstate = Microstate::builder()
            .bodies([
                Body::single_site(DynamicPoint::default(), Point::default()),
                Body::single_site(
                    DynamicPoint {
                        position: Cartesian::<3>::from([2.0, 0.0, 0.0]),
                        ..Default::default()
                    },
                    Point::default(),
                ),
            ])
            .try_build()?;

        let lennard_jones = LennardJones::<12, 6>::default();
        let pairwise_cutoff = PairwiseCutoff(Isotropic {
            interaction: lennard_jones,
            r_cut: 2.5,
        });
        let rigid = Rigid(pairwise_cutoff);

        check!(microstate.bodies()[0].item.properties.net_force == [0.0, 0.0, 0.0].into());
        check!(microstate.bodies()[1].item.properties.net_force == [0.0, 0.0, 0.0].into());

        microstate.update_net_force_and_virial(&rigid);

        assert_relative_eq!(
            microstate.bodies()[0].item.properties.net_force,
            [93.0 / 512.0, 0.0, 0.0].into()
        );
        assert_relative_eq!(
            microstate.bodies()[1].item.properties.net_force,
            [-93.0 / 512.0, 0.0, 0.0].into()
        );

        Ok(())
    }

    #[test]
    fn net_force_and_torque_2d() -> anyhow::Result<()> {
        let mut microstate: Microstate<
            DynamicOrientedPoint<Cartesian<2>, Angle>,
            Point<Cartesian<2>>,
            _,
            _,
        > = Microstate::builder()
            .bodies([
                Body::single_site(
                    DynamicOrientedPoint {
                        position: Cartesian::<2>::from([0.0, -1.0]),
                        ..Default::default()
                    },
                    Point::new([0.0, 1.0].into()),
                ),
                Body::single_site(
                    DynamicOrientedPoint {
                        position: Cartesian::<2>::from([2.0, -2.0]),
                        ..Default::default()
                    },
                    Point::new([0.0, 2.0].into()),
                ),
            ])
            .try_build()?;

        let lennard_jones = LennardJones::<12, 6>::default();
        let pairwise_cutoff = PairwiseCutoff(Isotropic {
            interaction: lennard_jones,
            r_cut: 2.5,
        });
        let rigid = Rigid(pairwise_cutoff);

        check!(microstate.bodies()[0].item.properties.net_force == [0.0, 0.0].into());
        check!(microstate.bodies()[1].item.properties.net_force == [0.0, 0.0].into());
        check!(microstate.bodies()[0].item.properties.net_torque == 0.0);
        check!(microstate.bodies()[1].item.properties.net_torque == 0.0);

        microstate.update_net_force_virial_and_torque(&rigid);

        assert_relative_eq!(
            microstate.bodies()[0].item.properties.net_force,
            [93.0 / 512.0, 0.0].into()
        );
        assert_relative_eq!(
            microstate.bodies()[1].item.properties.net_force,
            [-93.0 / 512.0, 0.0].into()
        );

        assert_relative_eq!(
            microstate.bodies()[0].item.properties.net_torque,
            -93.0 / 512.0
        );
        assert_relative_eq!(
            microstate.bodies()[1].item.properties.net_torque,
            2.0 * 93.0 / 512.0
        );

        Ok(())
    }

    #[test]
    fn net_force_and_torque_3d() -> anyhow::Result<()> {
        let mut microstate: Microstate<
            DynamicOrientedPoint<Cartesian<3>, Versor>,
            Point<Cartesian<3>>,
            _,
            _,
        > = Microstate::builder()
            .bodies([
                Body::single_site(
                    DynamicOrientedPoint {
                        position: Cartesian::<3>::from([0.0, -1.0, 0.0]),
                        ..Default::default()
                    },
                    Point::new([0.0, 1.0, 0.0].into()),
                ),
                Body::single_site(
                    DynamicOrientedPoint {
                        position: Cartesian::<3>::from([2.0, -2.0, 0.0]),
                        ..Default::default()
                    },
                    Point::new([0.0, 2.0, 0.0].into()),
                ),
            ])
            .try_build()?;

        let lennard_jones = LennardJones::<12, 6>::default();
        let pairwise_cutoff = PairwiseCutoff(Isotropic {
            interaction: lennard_jones,
            r_cut: 2.5,
        });
        let rigid = Rigid(pairwise_cutoff);

        check!(microstate.bodies()[0].item.properties.net_force == [0.0, 0.0, 0.0].into());
        check!(microstate.bodies()[1].item.properties.net_force == [0.0, 0.0, 0.0].into());
        check!(microstate.bodies()[0].item.properties.net_torque == [0.0, 0.0, 0.0].into());
        check!(microstate.bodies()[1].item.properties.net_torque == [0.0, 0.0, 0.0].into());

        microstate.update_net_force_virial_and_torque(&rigid);

        assert_relative_eq!(
            microstate.bodies()[0].item.properties.net_force,
            [93.0 / 512.0, 0.0, 0.0].into()
        );
        assert_relative_eq!(
            microstate.bodies()[1].item.properties.net_force,
            [-93.0 / 512.0, 0.0, 0.0].into()
        );

        assert_relative_eq!(
            microstate.bodies()[0].item.properties.net_torque,
            [0.0, 0.0, -93.0 / 512.0].into()
        );
        assert_relative_eq!(
            microstate.bodies()[1].item.properties.net_torque,
            [0.0, 0.0, 2.0 * 93.0 / 512.0].into()
        );

        Ok(())
    }
}

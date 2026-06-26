//! Define `UpdateNetForce`

use hoomd_interaction::{NetBodyForce, NetBodyForceAndTorque};
use hoomd_microstate::{Microstate, property::{NetForce, NetTorque}};
use hoomd_vector::{Vector, Wedge};

/// Compute the net force given by an interaction model and apply it to each body in the
/// microstate.
///
/// Given an interaction model that implements [`NetBodyForce`], [`UpdateNetForce`]
/// sets the [`NetForce`] property of each body in the microstate to the one computed
/// by the interaction model.
///
/// [`NetBodyForce`]: hoomd_interaction::NetBodyForce
/// [`NetForce`]: hoomd_microstate::property::NetForce
///
/// # Example
/// ```
/// use hoomd_interaction::{PairwiseCutoff, Rigid, pairwise::Isotropic, univariate::LennardJones};
/// use hoomd_microstate::{Microstate, Body, property::{DynamicPoint, Point}};
/// use hoomd_vector::{Cartesian};
/// use hoomd_md::UpdateNetForce;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut microstate = Microstate::builder().
///         bodies([
///         Body::single_site(DynamicPoint::default(), Point::default()),
///         Body::single_site(DynamicPoint { position: Cartesian::<2>::from([2.0, 0.0]), .. Default::default()}, Point::default()),
///     ]).try_build()?;
///
///     let lennard_jones =  LennardJones::<12,6>::default();
///     let pairwise_cutoff = PairwiseCutoff(Isotropic {
///         interaction: lennard_jones,
///         r_cut: 2.5,
///     });
///     let rigid = Rigid(pairwise_cutoff);
///
///
///     microstate.update_net_force(&rigid);
/// #   Ok(())
/// # }
/// ```
pub trait UpdateNetForce<E> {
    /// Compute and set the net force on each body.
    fn update_net_force(&mut self, interaction_model: &E);
}

/// Compute the net force and torque given by an interaction model and apply them
/// to each body in the microstate.
///
/// Given an interaction model that implements [`NetBodyForceAndTorque`], [`UpdateNetForceAndTorque`]
/// sets the [`NetForce`] and [`NetTorque`] properties of each body in the microstate to
/// the ones computed by the interaction model.
///
/// [`NetBodyForceAndTorque`]: hoomd_interaction::NetBodyForceAndTorque
/// [`NetForce`]: hoomd_microstate::property::NetForce
/// [`NetTorque`]: hoomd_microstate::property::NetTorque
///
/// # Example
///
/// ```
/// use hoomd_interaction::{PairwiseCutoff, Rigid, pairwise::Isotropic, univariate::LennardJones};
/// use hoomd_microstate::{Microstate, Body, property::{DynamicOrientedPoint, Point}};
/// use hoomd_vector::{Angle, Cartesian};
/// use hoomd_md::UpdateNetForceAndTorque;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut microstate: Microstate<DynamicOrientedPoint<Cartesian<2>, Angle>, Point<Cartesian<2>>, _, _> = Microstate::builder().
///         bodies([
///         Body::single_site(DynamicOrientedPoint { position: Cartesian::<2>::from([0.0, -1.0]), .. Default::default()}, Point::new([0.0, 1.0].into())),
///         Body::single_site(DynamicOrientedPoint { position: Cartesian::<2>::from([2.0, -2.0]), .. Default::default()}, Point::new([0.0, 2.0].into())),
///     ]).try_build()?;
///
///     let lennard_jones =  LennardJones::<12,6>::default();
///     let pairwise_cutoff = PairwiseCutoff(Isotropic {
///         interaction: lennard_jones,
///         r_cut: 2.5,
///     });
///     let rigid = Rigid(pairwise_cutoff);
///
///     microstate.update_net_force_and_torque(&rigid);
/// #   Ok(())
/// # }
/// ```
pub trait UpdateNetForceAndTorque<E> {
    /// Compute and set the net force and torque on each body.
    fn update_net_force_and_torque(&mut self, interaction_model: &E);
}

impl<V, B, S, X, C, E> UpdateNetForce<E> for Microstate<B, S, X, C>
where
    V: Default + Vector,
    B: NetForce<NetForce = V>,
    E: NetBodyForce<B, S, X, C, Force=V>
{
    #[inline]
    fn update_net_force(&mut self, interaction_model: &E) {
        for body_index in 0..self.bodies().len() {
            let net_force = interaction_model.net_body_force(self, body_index);
            self.set_body_net_force(body_index, net_force);
        }
    }
}

impl<V, B, S, X, C, E> UpdateNetForceAndTorque<E> for Microstate<B, S, X, C>
where
    V: Default + Vector + Wedge,
    B: NetForce<NetForce = V> + NetTorque<NetTorque = V::Bivector>,
    E: NetBodyForceAndTorque<B, S, X, C, Force=V>
{
    #[inline]
    fn update_net_force_and_torque(&mut self, interaction_model: &E) {
        for body_index in 0..self.bodies().len() {
            let (net_force, net_torque) = interaction_model.net_body_force_and_torque(self, body_index);
            self.set_body_net_force(body_index, net_force);
            self.set_body_net_torque(body_index, net_torque);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use assert2::check;
    use approxim::assert_relative_eq;

    use hoomd_interaction::{PairwiseCutoff, Rigid, pairwise::Isotropic, univariate::LennardJones};
    use hoomd_microstate::{Body, property::{DynamicOrientedPoint, DynamicPoint, Point}};
    use hoomd_vector::{Angle, Cartesian, Versor};

    #[test]
    fn net_force_2d() -> anyhow::Result<()> {
        let mut microstate = Microstate::builder().
            bodies([
            Body::single_site(DynamicPoint::default(), Point::default()),
            Body::single_site(DynamicPoint { position: Cartesian::<2>::from([2.0, 0.0]), .. Default::default()}, Point::default()),
        ]).try_build()?;

        let lennard_jones =  LennardJones::<12,6>::default();
        let pairwise_cutoff = PairwiseCutoff(Isotropic {
            interaction: lennard_jones,
            r_cut: 2.5,
        });
        let rigid = Rigid(pairwise_cutoff);

        check!(microstate.bodies()[0].item.properties.net_force == [0.0, 0.0].into());
        check!(microstate.bodies()[1].item.properties.net_force == [0.0, 0.0].into());

        microstate.update_net_force(&rigid);

        assert_relative_eq!(microstate.bodies()[0].item.properties.net_force, [93.0 / 512.0, 0.0].into());
        assert_relative_eq!(microstate.bodies()[1].item.properties.net_force, [-93.0 / 512.0, 0.0].into());
    
        Ok(())
    }

    #[test]
    fn net_force_3d() -> anyhow::Result<()> {
        let mut microstate = Microstate::builder().
            bodies([
            Body::single_site(DynamicPoint::default(), Point::default()),
            Body::single_site(DynamicPoint { position: Cartesian::<3>::from([2.0, 0.0, 0.0]), .. Default::default()}, Point::default()),
        ]).try_build()?;

        let lennard_jones =  LennardJones::<12,6>::default();
        let pairwise_cutoff = PairwiseCutoff(Isotropic {
            interaction: lennard_jones,
            r_cut: 2.5,
        });
        let rigid = Rigid(pairwise_cutoff);

        check!(microstate.bodies()[0].item.properties.net_force == [0.0, 0.0, 0.0].into());
        check!(microstate.bodies()[1].item.properties.net_force == [0.0, 0.0, 0.0].into());

        microstate.update_net_force(&rigid);

        assert_relative_eq!(microstate.bodies()[0].item.properties.net_force, [93.0 / 512.0, 0.0, 0.0].into());
        assert_relative_eq!(microstate.bodies()[1].item.properties.net_force, [-93.0 / 512.0, 0.0, 0.0].into());
    
        Ok(())
    }

    #[test]
    fn net_force_and_torque_2d() -> anyhow::Result<()> {
        let mut microstate: Microstate<DynamicOrientedPoint<Cartesian<2>, Angle>, Point<Cartesian<2>>, _, _> = Microstate::builder().
            bodies([
            Body::single_site(DynamicOrientedPoint { position: Cartesian::<2>::from([0.0, -1.0]), .. Default::default()}, Point::new([0.0, 1.0].into())),
            Body::single_site(DynamicOrientedPoint { position: Cartesian::<2>::from([2.0, -2.0]), .. Default::default()}, Point::new([0.0, 2.0].into())),
        ]).try_build()?;

        let lennard_jones =  LennardJones::<12,6>::default();
        let pairwise_cutoff = PairwiseCutoff(Isotropic {
            interaction: lennard_jones,
            r_cut: 2.5,
        });
        let rigid = Rigid(pairwise_cutoff);

        check!(microstate.bodies()[0].item.properties.net_force == [0.0, 0.0].into());
        check!(microstate.bodies()[1].item.properties.net_force == [0.0, 0.0].into());
        check!(microstate.bodies()[0].item.properties.net_torque == 0.0);
        check!(microstate.bodies()[1].item.properties.net_torque == 0.0);

        microstate.update_net_force_and_torque(&rigid);

        assert_relative_eq!(microstate.bodies()[0].item.properties.net_force, [93.0 / 512.0, 0.0].into());
        assert_relative_eq!(microstate.bodies()[1].item.properties.net_force, [-93.0 / 512.0, 0.0].into());

        assert_relative_eq!(microstate.bodies()[0].item.properties.net_torque, -93.0 / 512.0);
        assert_relative_eq!(microstate.bodies()[1].item.properties.net_torque, 2.0 * 93.0 / 512.0);
    
        Ok(())
    }

    #[test]
    fn net_force_and_torque_3d() -> anyhow::Result<()> {
        let mut microstate: Microstate<DynamicOrientedPoint<Cartesian<3>, Versor>, Point<Cartesian<3>>, _, _> = Microstate::builder().
            bodies([
            Body::single_site(DynamicOrientedPoint { position: Cartesian::<3>::from([0.0, -1.0, 0.0]), .. Default::default()}, Point::new([0.0, 1.0, 0.0].into())),
            Body::single_site(DynamicOrientedPoint { position: Cartesian::<3>::from([2.0, -2.0, 0.0]), .. Default::default()}, Point::new([0.0, 2.0, 0.0].into())),
        ]).try_build()?;

        let lennard_jones =  LennardJones::<12,6>::default();
        let pairwise_cutoff = PairwiseCutoff(Isotropic {
            interaction: lennard_jones,
            r_cut: 2.5,
        });
        let rigid = Rigid(pairwise_cutoff);

        check!(microstate.bodies()[0].item.properties.net_force == [0.0, 0.0, 0.0].into());
        check!(microstate.bodies()[1].item.properties.net_force == [0.0, 0.0, 0.0].into());
        check!(microstate.bodies()[0].item.properties.net_torque == [0.0, 0.0, 0.0].into());
        check!(microstate.bodies()[1].item.properties.net_torque == [0.0, 0.0, 0.0].into());

        microstate.update_net_force_and_torque(&rigid);

        assert_relative_eq!(microstate.bodies()[0].item.properties.net_force, [93.0 / 512.0, 0.0, 0.0].into());
        assert_relative_eq!(microstate.bodies()[1].item.properties.net_force, [-93.0 / 512.0, 0.0, 0.0].into());

        assert_relative_eq!(microstate.bodies()[0].item.properties.net_torque, [0.0, 0.0, -93.0 / 512.0].into());
        assert_relative_eq!(microstate.bodies()[1].item.properties.net_torque, [0.0, 0.0, 2.0 * 93.0 / 512.0].into());
    
        Ok(())
    }
}

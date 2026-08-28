// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `RotationalKineticEnergy`

use super::RotationalKineticEnergy;
use hoomd_microstate::{
    Body,
    Microstate,
    Tagged,
    property::{
        AngularMomentum,
        MomentOfInertia,
        Orientation,
        RotationalMotionTypes
    }
};
use hoomd_vector::{Angle, Versor};

/// Aggregate rotational kinetic energy (and degrees of freedom).
/// 
/// This trait binds the aggregation scheme to the type that represents
/// orientation. Implement this trait on a type that represents body orientation
/// to make a [`Microstate`] containing such bodies compatible with
/// [`RotationalKineticEnergy`].
/// 
/// [`Microstate`]: hoomd_microstate::Microstate
pub trait AggregateEnergyRotation: RotationalMotionTypes
{
    /// Return the contribution of a single body to the system's rotational kinetic energy (and degrees of freedom).
    fn energy_and_dof_per_body(
        moment_of_inertia: &Self::MomentOfInertia,
        angular_momentum: &Self::AngularMomentum,
    ) -> (f64, usize);
}

///  Aggregation for bodies in 2-dimensional cartesian space.
impl AggregateEnergyRotation for Angle {
    /// Return the contribution of a single body to the system's rotational kinetic energy (and degrees of freedom).
    /// 
    /// In 2-dimensional cartesian space, each body has either 0 rotational
    /// degrees of freedom (when $` I = 0 `$) or 1 (when $` I \ne 0 `$). The
    /// total number of rotational degrees of freedom is given by
    /// 
    /// ```math
    /// N_{rot} = \sum_{i \in \mathrm{selection}} \left| I_i \ne 0 \right|
    /// ```
    /// 
    /// where $` \left| \right| `$ is the [Iverson bracket].
    /// 
    /// [Iverson bracket]: https://en.wikipedia.org/wiki/Iverson_bracket
    ///
    /// The total rotational kinetic energy is given by
    /// 
    /// ```math
    /// K_{rot} = \sum_{i \in \mathrm{selection}} \frac{L_i^2}{2 I_i}
    /// ```
    /// 
    /// ignoring bodies for which $` I_i = 0 `$.
    fn energy_and_dof_per_body(
        moment_of_inertia: &Self::MomentOfInertia,
        angular_momentum: &Self::AngularMomentum,
    ) -> (f64, usize) {
        if *moment_of_inertia == 0.0 {
            (0.0, 0)
        } else {
            (angular_momentum.powi(2) / (2.0 * moment_of_inertia), 1)
        }
    }
}

///  Aggregation for bodies in 3-dimensional cartesian space.
impl AggregateEnergyRotation for Versor {
    /// Return the contribution of a single body to the system's rotational kinetic energy (and degrees of freedom).
    /// 
    /// In 3-dimensional cartesian space, each body has 0 to 3 rotational
    /// degrees of freedom, depending on the number of non-zero components of
    /// the body's diagonalized moment of inertia. The total number of 
    /// rotational degrees of freedom is given by
    /// 
    /// ```math
    /// N_{rot} = \sum_{i \in \mathrm{selection}} \left| I_{xx,i} \ne 0 \right| + \left| I_{yy,i} \ne 0 \right| + \left| I_{zz,i} \ne 0 \right|
    /// ```
    ///
    /// The total rotational kinetic energy is given by
    /// 
    /// ```math
    /// K_{rot} = \sum_{i \in \mathrm{selection}}\frac{L_{x,i}^2}{2I_{xx,i}} + \frac{L_{y,i}^2}{2I_{yy,i}} + \frac{L_{z,i}^2}{2I_{zz,i}}
    /// ```
    /// 
    /// ignoring terms for which $` I_{jj,i} = 0 `$.
    fn energy_and_dof_per_body(
        moment_of_inertia: &Self::MomentOfInertia,
        angular_momentum: &Self::AngularMomentum,
    ) -> (f64, usize) {
        let (mut energy_total, mut dof_count) = (0.0, 0);

        for (momentum, inertia) in
            angular_momentum.coordinates.iter().zip(moment_of_inertia)
        {
            if *inertia != 0.0 {
                energy_total += momentum.powi(2) / (2.0 * inertia);
                dof_count += 1;
            }
        }

        (energy_total, dof_count)
    }
}

impl<B, S, X, C> RotationalKineticEnergy<B, S> for Microstate<B, S, X, C>
where
    B: Orientation<Rotation: AggregateEnergyRotation>
        + MomentOfInertia<MomentOfInertia = <B::Rotation as RotationalMotionTypes>::MomentOfInertia>
        + AngularMomentum<AngularMomentum = <B::Rotation as RotationalMotionTypes>::AngularMomentum>
{
    #[inline]
    fn rotational_kinetic_energy_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &self,
        should_sum_body: F,
    ) -> (f64, usize) {
        self.bodies()
            .iter()
            .filter(|&body| should_sum_body(body))
            .fold((0.0, 0), |(cum_total, cum_count), body| {
                let (total, count) = <B::Rotation as AggregateEnergyRotation>::energy_and_dof_per_body(
                    body.item.properties.moment_of_inertia(),
                    body.item.properties.angular_momentum()
                );

                (cum_total + total, cum_count + count)
            })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approxim::assert_relative_eq;
    use assert2::check;

    use hoomd_microstate::{
        Body,
        property::{DynamicOrientedPoint, Point},
    };
    use hoomd_vector::{Angle, Cartesian, Versor};

    #[test]
    fn kinetic_energy_2d() -> anyhow::Result<()> {
        let microstate: Microstate<DynamicOrientedPoint<Cartesian<2>, Angle>, _, _, _> =
            Microstate::builder()
                .bodies([
                    Body::single_site(DynamicOrientedPoint::default(), Point::default()),
                    Body::single_site(
                        DynamicOrientedPoint {
                            moment_of_inertia: 0.0,
                            ..Default::default()
                        },
                        Point::default(),
                    ),
                    Body::single_site(
                        DynamicOrientedPoint {
                            moment_of_inertia: 2.0,
                            angular_momentum: 8.0,
                            ..Default::default()
                        },
                        Point::default(),
                    ),
                    Body::single_site(
                        DynamicOrientedPoint {
                            moment_of_inertia: 4.0,
                            angular_momentum: 3.0,
                            ..Default::default()
                        },
                        Point::default(),
                    ),
                    Body::single_site(
                        DynamicOrientedPoint {
                            moment_of_inertia: 3.0,
                            angular_momentum: 2.0,
                            ..Default::default()
                        },
                        Point::default(),
                    ),
                ])
                .try_build()?;

        let (total_kinetic_energy, total_degrees_of_freedom) =
            microstate.rotational_kinetic_energy();
        check!(total_degrees_of_freedom == 4);
        assert_relative_eq!(total_kinetic_energy, 64.0 / 4.0 + 9.0 / 8.0 + 4.0 / 6.0);

        let (filtered_kinetic_energy, filtered_degrees_of_freedom) =
            microstate.rotational_kinetic_energy_with_filter(|b| b.tag <= 2);
        check!(filtered_degrees_of_freedom == 2);
        assert_relative_eq!(filtered_kinetic_energy, 64.0 / 4.0);

        Ok(())
    }

    #[test]
    fn kinetic_energy_3d() -> anyhow::Result<()> {
        let microstate: Microstate<DynamicOrientedPoint<Cartesian<3>, Versor>, _, _, _> =
            Microstate::builder()
                .bodies([
                    Body::single_site(DynamicOrientedPoint::default(), Point::default()),
                    Body::single_site(
                        DynamicOrientedPoint {
                            moment_of_inertia: [0.0, 0.0, 0.0],
                            angular_momentum: [1.0, 1.0, 1.0].into(),
                            ..Default::default()
                        },
                        Point::default(),
                    ),
                    Body::single_site(
                        DynamicOrientedPoint {
                            moment_of_inertia: [2.0, 0.0, 0.0],
                            angular_momentum: [8.0, 1.0, 1.0].into(),
                            ..Default::default()
                        },
                        Point::default(),
                    ),
                    Body::single_site(
                        DynamicOrientedPoint {
                            moment_of_inertia: [0.0, 6.0, 0.0],
                            angular_momentum: [1.0, 3.0, 1.0].into(),
                            ..Default::default()
                        },
                        Point::default(),
                    ),
                    Body::single_site(
                        DynamicOrientedPoint {
                            moment_of_inertia: [0.0, 0.0, 3.0],
                            angular_momentum: [1.0, 1.0, -4.0].into(),
                            ..Default::default()
                        },
                        Point::default(),
                    ),
                    Body::single_site(
                        DynamicOrientedPoint {
                            moment_of_inertia: [2.0, 4.0, 6.0],
                            angular_momentum: [3.0, 2.0, -4.0].into(),
                            ..Default::default()
                        },
                        Point::default(),
                    ),
                ])
                .try_build()?;

        let (total_kinetic_energy, total_degrees_of_freedom) =
            microstate.rotational_kinetic_energy();
        check!(total_degrees_of_freedom == 9);
        assert_relative_eq!(
            total_kinetic_energy,
            64.0 / 4.0 + 9.0 / 12.0 + 16.0 / 6.0 + 9.0 / 4.0 + 4.0 / 8.0 + 16.0 / 12.0
        );

        let (filtered_kinetic_energy, filtered_degrees_of_freedom) =
            microstate.rotational_kinetic_energy_with_filter(|b| b.tag <= 2);
        check!(filtered_degrees_of_freedom == 4);
        assert_relative_eq!(filtered_kinetic_energy, 64.0 / 4.0);

        Ok(())
    }
}

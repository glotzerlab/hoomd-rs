// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `RotationalKineticEnergy`

use super::RotationalKineticEnergy;
use hoomd_microstate::{Body, Microstate, Tagged, property::DynamicOrientedPoint};
use hoomd_vector::{Angle, Versor, Wedge};

impl<P, S, X, C> RotationalKineticEnergy<DynamicOrientedPoint<P, Angle>, S>
    for Microstate<DynamicOrientedPoint<P, Angle>, S, X, C>
where
    P: Wedge,
{
    #[inline]
    fn rotational_kinetic_energy_with_filter<
        F: Fn(&Tagged<Body<DynamicOrientedPoint<P, Angle>, S>>) -> bool,
    >(
        &self,
        should_sum_body: F,
    ) -> (f64, usize) {
        self.bodies()
            .iter()
            .filter(|&body| should_sum_body(body))
            .fold((0.0, 0), |(total, count), body| {
                let moment_of_inertia = body.item.properties.moment_of_inertia;
                let angular_momentum = body.item.properties.angular_momentum;

                if moment_of_inertia != 0.0 {
                    (
                        total + angular_momentum.powi(2) / (2.0 * moment_of_inertia),
                        count + 1,
                    )
                } else {
                    (total, count)
                }
            })
    }
}

impl<P, S, X, C> RotationalKineticEnergy<DynamicOrientedPoint<P, Versor>, S>
    for Microstate<DynamicOrientedPoint<P, Versor>, S, X, C>
where
    P: Wedge,
{
    #[inline]
    fn rotational_kinetic_energy_with_filter<
        F: Fn(&Tagged<Body<DynamicOrientedPoint<P, Versor>, S>>) -> bool,
    >(
        &self,
        should_sum_body: F,
    ) -> (f64, usize) {
        self.bodies()
            .iter()
            .filter(|&body| should_sum_body(body))
            .fold((0.0, 0), |(mut total, mut count), body| {
                let moment_of_inertia = body.item.properties.moment_of_inertia;
                let angular_momentum = body.item.properties.angular_momentum;

                for (momentum, inertia) in angular_momentum.coordinates.iter().zip(moment_of_inertia) {
                    if inertia != 0.0 {
                        total += momentum.powi(2) / (2.0 * inertia);
                        count += 1;
                    }
                }

                (total, count)
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
        assert_relative_eq!(total_kinetic_energy, 64.0 / 4.0 + 9.0 / 12.0 + 16.0 / 6.0 + 9.0 / 4.0 + 4.0 / 8.0 + 16.0 / 12.0);

        let (filtered_kinetic_energy, filtered_degrees_of_freedom) =
            microstate.rotational_kinetic_energy_with_filter(|b| b.tag <= 2);
        check!(filtered_degrees_of_freedom == 4);
        assert_relative_eq!(filtered_kinetic_energy, 64.0 / 4.0);

        Ok(())
    }

}

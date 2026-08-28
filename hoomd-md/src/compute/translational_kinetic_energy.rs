// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `TranslationalKineticEnergy`

use super::TranslationalKineticEnergy;
use hoomd_microstate::{
    Body, Microstate, Tagged,
    property::{Mass, Momentum},
};
use hoomd_vector::InnerProduct;

impl<V, B, S, X, C> TranslationalKineticEnergy<B, S> for Microstate<B, S, X, C>
where
    V: InnerProduct,
    B: Momentum<Momentum = V> + Mass,
{
    /// Compute the total translational kinetic energy (and degrees of freedom) over selected bodies in the microstate.
    /// 
    /// In [`InnerProduct`] vector space, each body has $`D`$ translational
    /// degrees of freedom, where $`D`$ is the number of dimensions. The total
    /// number of translational degrees of freedom is given by
    /// 
    /// ```math
    /// N_{trans} = \sum_{i \in \mathrm{selection}} D
    /// ```
    /// 
    /// [`InnerProduct`]: hoomd_vector::InnerProduct
    /// 
    /// The total translational kinetic energy is given by
    /// 
    /// ```math
    /// K_{trans} = \sum_{i \in \mathrm{selection}} \frac{\vec{p}_i \cdot \vec{p}_i}{2m_i}
    /// ```
    /// 
    /// assuming that all selected bodies have $` m_i \ne 0 `$.
    #[inline]
    fn translational_kinetic_energy_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &self,
        should_sum_body: F,
    ) -> (f64, usize) {
        self.bodies()
            .iter()
            .filter(|&body| should_sum_body(body))
            .fold((0.0, 0), |(total, count), body| {
                let p = body.item.properties.momentum();
                (
                    total + p.norm_squared() / (2.0 * body.item.properties.mass()),
                    count + V::n_dimensions(),
                )
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
        property::{DynamicPoint, Point},
    };
    use hoomd_vector::Cartesian;

    #[test]
    fn kinetic_energy_2d() -> anyhow::Result<()> {
        let microstate = Microstate::builder()
            .bodies([
                Body::single_site(DynamicPoint::default(), Point::default()),
                Body::single_site(
                    DynamicPoint {
                        mass: 2.0,
                        momentum: Cartesian::<2>::from([2.0, 0.0]),
                        ..Default::default()
                    },
                    Point::default(),
                ),
                Body::single_site(
                    DynamicPoint {
                        mass: 4.0,
                        momentum: Cartesian::<2>::from([1.0, 1.0]),
                        ..Default::default()
                    },
                    Point::default(),
                ),
                Body::single_site(
                    DynamicPoint {
                        mass: 3.0,
                        momentum: Cartesian::<2>::from([-4.0, -2.0]),
                        ..Default::default()
                    },
                    Point::default(),
                ),
            ])
            .try_build()?;

        let (total_kinetic_energy, total_degrees_of_freedom) =
            microstate.translational_kinetic_energy();
        check!(total_degrees_of_freedom == 8);
        assert_relative_eq!(total_kinetic_energy, 1.0 + 2.0 / 8.0 + 20.0 / 6.0);

        let (filtered_kinetic_energy, filtered_degrees_of_freedom) =
            microstate.translational_kinetic_energy_with_filter(|b| b.tag <= 1);
        check!(filtered_degrees_of_freedom == 4);
        assert_relative_eq!(filtered_kinetic_energy, 1.0);

        Ok(())
    }

    #[test]
    fn kinetic_energy_3d() -> anyhow::Result<()> {
        let microstate = Microstate::builder()
            .bodies([
                Body::single_site(DynamicPoint::default(), Point::default()),
                Body::single_site(
                    DynamicPoint {
                        mass: 2.0,
                        momentum: Cartesian::<3>::from([2.0, 0.0, 0.0]),
                        ..Default::default()
                    },
                    Point::default(),
                ),
                Body::single_site(
                    DynamicPoint {
                        mass: 4.0,
                        momentum: Cartesian::<3>::from([1.0, 1.0, 1.0]),
                        ..Default::default()
                    },
                    Point::default(),
                ),
                Body::single_site(
                    DynamicPoint {
                        mass: 3.0,
                        momentum: Cartesian::<3>::from([-4.0, -2.0, 1.0]),
                        ..Default::default()
                    },
                    Point::default(),
                ),
            ])
            .try_build()?;

        let (total_kinetic_energy, total_degrees_of_freedom) =
            microstate.translational_kinetic_energy();
        check!(total_degrees_of_freedom == 12);
        assert_relative_eq!(total_kinetic_energy, 1.0 + 3.0 / 8.0 + 21.0 / 6.0);

        let (filtered_kinetic_energy, filtered_degrees_of_freedom) =
            microstate.translational_kinetic_energy_with_filter(|b| b.tag <= 1);
        check!(filtered_degrees_of_freedom == 6);
        assert_relative_eq!(filtered_kinetic_energy, 1.0);

        Ok(())
    }
}

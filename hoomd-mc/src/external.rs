// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement `DeltaEnergy*` for external potentials.
*/

use super::DeltaEnergyOne;
use hoomd_interaction::{Single, SiteEnergy};
use hoomd_microstate::{Body, Microstate, Transform, boundary::Boundary, property::Position};

impl<V, B, S, C, E> DeltaEnergyOne<V, B, S, C> for Single<E>
where
    E: SiteEnergy<S>,
    B: Transform<S>,
    S: Position<V>,
    C: Boundary<V, B, S>,
{
    #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<V, B, S, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
        let mut energy_final = 0.0;
        for s in &final_body.sites {
            match initial_microstate
                .boundary()
                .wrap_site(final_body.properties.transform(s))
            {
                Ok(wrapped_site) => energy_final += self.site_energy(&wrapped_site),
                Err(_) => return f64::INFINITY,
            }
        }

        let energy_initial = initial_microstate
            .iter_body_sites(body_index)
            .fold(0.0, |total, s| total + self.site_energy(&s.properties));

        energy_final - energy_initial
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_microstate::{MicrostateBuilder, boundary::Square, property::Point};
    use hoomd_vector::Cartesian;

    struct Zero;

    impl SiteEnergy<Point<Cartesian<2>>> for Zero {
        fn site_energy(&self, _site_properties: &Point<Cartesian<2>>) -> f64 {
            0.0
        }
    }

    #[test]
    fn site_outside() {
        let square = Square {
            l: 4.0
                .try_into()
                .expect("hard-coded constant should be positive"),
        };

        let body = Body {
            properties: Point::new(Cartesian::from([0.0, 0.0])),
            sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
        };
        let mut final_body = body.clone();
        final_body.properties.position[0] = 1.0;

        let microstate = MicrostateBuilder::with_boundary(square)
            .bodies([body])
            .try_build()
            .expect("the hard-coded bodies should be in the boundary");

        let energy = Single(Zero);

        assert_eq!(
            energy.delta_energy_one(&microstate, 0, &final_body),
            f64::INFINITY
        );
    }
}

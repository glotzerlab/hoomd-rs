// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement `Single`
*/

use crate::{DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, SiteEnergy, TotalEnergy};
use hoomd_microstate::{Body, Microstate, Transform, boundary::Wrap, property::Position};

/** Compute system properties from external fields.

Given an inner type that implements [`SiteEnergy`], [`Single`] represents:

```math
U_\mathrm{total} = \sum_{i=0}^{N-1} U\left( s_i \right)
```
where $`s_i`$ is the full set of site properties for site i.

For the inner type, use one from [`external`] or your own custom type.

[`external`]: crate::external

# Example

```
use hoomd_interaction::{Single, TotalEnergy, external::Linear};
use hoomd_microstate::{Microstate, Body, property::Point};
use hoomd_vector::Cartesian;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = Microstate::new();
microstate.extend_bodies([Body::point(Cartesian::from([1.0, 0.0])),
                          Body::point(Cartesian::from([-1.0, 2.0]))])?;

let linear = Single(Linear{ alpha: 1.0,
    plane_origin: Cartesian::default(),
    plane_normal: [0.0, 1.0].try_into()? });

let total_energy = linear.total_energy(&microstate);
assert_eq!(total_energy, 2.0);
# Ok(())
# }
```
*/
pub struct Single<E>(pub E);

impl<B, S, C, E> TotalEnergy<Microstate<B, S, C>> for Single<E>
where
    E: SiteEnergy<S>,
{
    /** Compute the total energy of the microstate contributed by functions of a single site.

    The sum over sites differs from HOOMD-blue where external energies are
    evaluated only at the body centers. In general, hoomd-rs interactions apply
    to sites. Use a custom implementation to compute energies over body centers.

    # Example

    ```
    use hoomd_interaction::{Single, TotalEnergy, external::Linear};
    use hoomd_microstate::{Microstate, Body, property::Point};
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut microstate = Microstate::new();
    microstate.extend_bodies([Body::point(Cartesian::from([1.0, 0.0])),
                              Body::point(Cartesian::from([-1.0, 2.0]))])?;

    let linear = Single(Linear{ alpha: 1.0,
        plane_origin: Cartesian::default(),
        plane_normal: [0.0, 1.0].try_into()? });

    let total_energy = linear.total_energy(&microstate);
    assert_eq!(total_energy, 2.0);
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn total_energy(&self, microstate: &Microstate<B, S, C>) -> f64 {
        microstate
            .sites()
            .iter()
            .fold(0.0, |total, s| total + self.0.site_energy(&s.properties))
    }
}

/** Evaluate the change in energy contributed by `Single` when a single body is updated.

# Example

```
use hoomd_interaction::{DeltaEnergyOne, Single, external::Linear};
use hoomd_microstate::{Microstate, Body, property::Point};
use hoomd_vector::Cartesian;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = Microstate::new();
microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])))?;

let linear = Single(Linear{ alpha: 1.0,
    plane_origin: Cartesian::default(),
    plane_normal: [0.0, 1.0].try_into()? });

let delta_energy = linear.delta_energy_one(&microstate, 0,
    &Body::point([0.0, -1.0].into()));
assert_eq!(delta_energy, -1.0);
# Ok(())
# }
```
*/
impl<V, B, S, C, E> DeltaEnergyOne<B, S, C> for Single<E>
where
    E: SiteEnergy<S>,
    B: Transform<S>,
    S: Position<Vector = V>,
    C: Wrap<B> + Wrap<S>,
{
    #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
        let mut energy_final = 0.0;
        for s in &final_body.sites {
            match initial_microstate
                .boundary()
                .wrap(final_body.properties.transform(s))
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

/** Evaluate the change in energy contributed by `Single` when a single body is inserted.

# Example

```
use hoomd_interaction::{DeltaEnergyInsert, Single, external::Linear};
use hoomd_microstate::{Microstate, Body, property::Point};
use hoomd_vector::Cartesian;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = Microstate::new();
microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])))?;

let linear = Single(Linear{ alpha: 1.0,
    plane_origin: Cartesian::default(),
    plane_normal: [0.0, 1.0].try_into()? });

let delta_energy = linear.delta_energy_insert(&microstate,
    &Body::point([0.0, -1.0].into()));
assert_eq!(delta_energy, -1.0);
# Ok(())
# }
```
*/
impl<V, B, S, C, E> DeltaEnergyInsert<B, S, C> for Single<E>
where
    E: SiteEnergy<S>,
    B: Transform<S>,
    S: Position<Vector = V>,
    C: Wrap<B> + Wrap<S>,
{
    #[inline]
    fn delta_energy_insert(
        &self,
        initial_microstate: &Microstate<B, S, C>,
        new_body: &Body<B, S>,
    ) -> f64 {
        let mut energy_final = 0.0;
        for s in &new_body.sites {
            match initial_microstate
                .boundary()
                .wrap(new_body.properties.transform(s))
            {
                Ok(wrapped_site) => energy_final += self.site_energy(&wrapped_site),
                Err(_) => return f64::INFINITY,
            }
        }

        energy_final
    }
}

/** Evaluate the change in energy contributed by `Single` when a single body is removed.

# Example

```
use hoomd_interaction::{DeltaEnergyRemove, Single, external::Linear};
use hoomd_microstate::{Microstate, Body, property::Point};
use hoomd_vector::Cartesian;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = Microstate::new();
microstate.add_body(Body::point(Cartesian::from([0.0, 1.0])))?;

let linear = Single(Linear{ alpha: 1.0,
    plane_origin: Cartesian::default(),
    plane_normal: [0.0, 1.0].try_into()? });

let delta_energy = linear.delta_energy_remove(&microstate, 0);
assert_eq!(delta_energy, -1.0);
# Ok(())
# }
```
*/
impl<B, S, C, E> DeltaEnergyRemove<B, S, C> for Single<E>
where
    E: SiteEnergy<S>,
{
    #[inline]
    fn delta_energy_remove(
        &self,
        initial_microstate: &Microstate<B, S, C>,
        body_index: usize,
    ) -> f64 {
        let energy_initial = initial_microstate
            .iter_body_sites(body_index)
            .fold(0.0, |total, s| total + self.site_energy(&s.properties));

        -energy_initial
    }
}

impl<E, S> SiteEnergy<S> for Single<E>
where
    E: SiteEnergy<S>,
{
    #[inline]
    fn site_energy(&self, site_properties: &S) -> f64 {
        self.0.site_energy(site_properties)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hoomd_geometry::shape::Cuboid;
    use hoomd_microstate::boundary::{Closed, Open};
    use hoomd_microstate::property::{Point, Position};
    use hoomd_microstate::{Body, Microstate, MicrostateBuilder};
    use hoomd_vector::Cartesian;
    use rstest::*;

    struct TestSE;

    impl<S> SiteEnergy<S> for TestSE
    where
        S: Position<Vector = Cartesian<2>>,
    {
        fn site_energy(&self, site_properties: &S) -> f64 {
            site_properties.position()[0] + site_properties.position()[1]
        }
    }

    #[fixture]
    fn microstate() -> Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open> {
        let mut microstate = Microstate::new();
        microstate
            .extend_bodies([
                Body::point(Cartesian::from([1.0, 0.0])),
                Body::point(Cartesian::from([-1.0, 3.0])),
            ])
            .expect("hard-coded bodies should be in the boundary");
        microstate
    }

    #[rstest]
    fn single_total(microstate: Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open>) {
        let test_se = TestSE;
        let single = Single(test_se);

        assert_eq!(single.total_energy(&microstate), 3.0);
    }

    #[rstest]
    fn single_site(microstate: Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open>) {
        let test_se = TestSE;
        let single = Single(test_se);

        assert_eq!(single.site_energy(&microstate.sites()[0].properties), 1.0);
        assert_eq!(single.site_energy(&microstate.sites()[1].properties), 2.0);
    }

    mod delta_energy_one {
        use super::*;

        struct Zero;

        impl SiteEnergy<Point<Cartesian<2>>> for Zero {
            fn site_energy(&self, _site_properties: &Point<Cartesian<2>>) -> f64 {
                0.0
            }
        }

        #[test]
        fn site_outside() {
            let cuboid = Cuboid {
                edge_lengths: [
                    4.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    4.0.try_into()
                        .expect("hard-coded constant should be positive"),
                ],
            };
            let square = Closed(cuboid);

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

        // TODO: Test DeltaEnergyOne delta E
    }

    // TODO: Test DeltaEnergyInsert
    // TODO: TestDeltaEnergyRemove
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Sweep

use super::{Count, LocalTrial, Trial};
use hoomd_interaction::DeltaEnergyOne;
use hoomd_microstate::{
    boundary::{GenerateGhosts, Wrap}, property::Position, Body, Microstate, SiteKey, Transform
};
use hoomd_simulation::macrostate::Temperature;
use hoomd_spatial::PointUpdate;

use rand::Rng;

/// Apply a local trial move to each body in the microstate.
///
/// Each trial move is accepted when:
/// ```math
/// r < \exp\left(\frac{-\Delta H}{kT}\right)
/// ```
/// where `r` is a random value uniformly distributed in `[0,1)`, $`\Delta H`$ is
/// the change in energy computed by the given `hamiltonian` and $`kT`$ is the
/// `temperature` given in `macrostate`.
///
/// # Example
///
/// ```
/// use hoomd_interaction::Zero;
/// use hoomd_mc::{Sweep, Translate, Trial};
/// use hoomd_microstate::{Body, Microstate, property::Position};
/// use hoomd_simulation::macrostate::Isothermal;
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::new();
/// microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])));
/// let d = 0.1;
/// let translate = Translate::with_maximum_distance(d.try_into()?);
/// let translate_sweep = Sweep(translate);
///
/// let hamiltonian = Zero;
/// let macrostate = Isothermal { temperature: 1.0 };
///
/// for _ in 0..1_000 {
///     translate_sweep.apply(&mut microstate, &hamiltonian, &macrostate);
///     microstate.increment_step();
/// }
/// # Ok(())
/// # }
/// ```
pub struct Sweep<L>(pub L);

impl<P, B, S, X, C, L, H, MA> Trial<Microstate<B, S, X, C>, H, MA> for Sweep<L>
where
    P: Copy,
    B: Copy + Default + Transform<S> + Position<Position = P>,
    S: Copy + Default + Position<Position = P>,
    X: PointUpdate<P, SiteKey>,
    L: LocalTrial<B>,
    H: DeltaEnergyOne<B, S, X, C>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    MA: Temperature,
{
    type Count = Count;

    #[inline]
    fn apply(
        &self,
        microstate: &mut Microstate<B, S, X, C>,
        hamiltonian: &H,
        macrostate: &MA,
    ) -> Self::Count {
        let kt = macrostate.temperature();
        let mut rng = microstate.counter().make_rng();
        let mut count = Self::Count::default();
        let mut trial = Body::<B, S>::default();

        // For loop over a range instead of bodies().iter() as the latter holds an immutable borrow.
        // The call to `update_body_properties` makes a mutable borrow of microstate.
        for body_index in 0..microstate.bodies().len() {
            trial.clone_from(&microstate.bodies()[body_index].item);

            // Wrap the body position here. The site positions will be wrapped
            // by the delta_energy methods and again by update_body_properties.
            // We could early reject if we checked the site properties first. If
            // performance becomes an issue, we could also wrap site properties
            // once here and pass those to unchecked variants of the delta
            // energy and update methods.
            match microstate
                .boundary()
                .wrap(self.0.propose(&mut rng, trial.properties))
            {
                Ok(new_properties) => {
                    trial.properties = new_properties;

                    let delta_h = hamiltonian.delta_energy_one(microstate, body_index, &trial);
                    if delta_h != f64::INFINITY &&
                        rng.random::<f64>() < (-delta_h / kt).exp()
                        && microstate
                            .update_body_properties(body_index, trial.properties)
                            .is_ok()
                    {
                        count.accepted += 1;
                    } else {
                        count.rejected += 1;
                    }
                }
                Err(_) => count.rejected += 1,
            }
        }

        microstate.increment_substep();
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Translate;
    use approxim::assert_relative_eq;
    use hoomd_geometry::shape::Hypercuboid;
    use hoomd_interaction::{External, SiteEnergy, TotalEnergy, Zero};
    use hoomd_microstate::{MicrostateBuilder, boundary::Closed, property::Point};
    use hoomd_simulation::macrostate::Isothermal;
    use hoomd_vector::{Cartesian, InnerProduct};
    use rstest::*;

    const K: f64 = 2.0;
    const N_STEPS: u64 = 200_000;

    struct Harmonic(Cartesian<2>);

    impl SiteEnergy<Point<Cartesian<2>>> for Harmonic {
        fn site_energy(&self, site_properties: &Point<Cartesian<2>>) -> f64 {
            1.0 / 2.0 * K * (site_properties.position - self.0).norm_squared()
        }
    }

    struct Right;
    impl LocalTrial<Point<Cartesian<2>>> for Right {
        fn propose<R: Rng>(
            &self,
            _rng: &mut R,
            body_properties: Point<Cartesian<2>>,
        ) -> Point<Cartesian<2>> {
            let mut trial = body_properties;
            trial.position_mut()[0] += 1.0;
            trial
        }
    }

    #[rstest]
    fn harmonic_oscillators(#[values(1.0, 2.5)] kt: f64) {
        // Model a harmonic oscillator and validate the average position and energy distribution.
        // Check with a relatively large tolerance because N_STEPS is relatively
        // small to keep the test run short.
        const EPSILON: f64 = 0.3;

        let origin = Cartesian::from([1.0, -2.0]);

        let mut microstate = Microstate::new();
        microstate
            .add_body(Body::point(origin))
            .expect("the hard-coded body should be inside the boundary");
        let hamiltonian = External(Harmonic(origin));
        let macrostate = Isothermal { temperature: kt };

        let d = 0.1;
        let translate = Translate::with_maximum_distance(
            d.try_into()
                .expect("hard-coded constant should be positive"),
        );
        let translate_sweep = Sweep(translate);

        let mut position_accumulator = Cartesian::default();
        let mut energy_accumulator = 0.0;

        for _ in 0..N_STEPS {
            translate_sweep.apply(&mut microstate, &hamiltonian, &macrostate);

            position_accumulator += microstate.bodies()[0].item.properties.position;
            energy_accumulator += hamiltonian.total_energy(&microstate);

            microstate.increment_step();
        }

        let position_average = position_accumulator / (N_STEPS as f64);
        assert_relative_eq!(position_average[0], origin[0], epsilon = EPSILON);
        assert_relative_eq!(position_average[1], origin[1], epsilon = EPSILON);

        let energy_average = energy_accumulator / (N_STEPS as f64);
        assert_relative_eq!(energy_average, kt, epsilon = EPSILON);
    }

    #[test]
    fn reject_boundary_body() {
        let cuboid = Hypercuboid {
            edge_lengths: [
                4.0.try_into()
                    .expect("hard-coded constant should be positive"),
                4.0.try_into()
                    .expect("hard-coded constant should be positive"),
            ],
        };
        let square = Closed(cuboid);

        let mut microstate = MicrostateBuilder::with_boundary(square)
            .bodies([Body::point([0.0, 0.0].into())])
            .try_build()
            .expect("the hard-coded bodies should be in the boundary");
        let hamiltonian = Zero;
        let translate = Right;
        let translate_sweep = Sweep(translate);
        let macrostate = Isothermal { temperature: 1.0 };

        // The first move to the right ends in the boundary and should be accepted.
        let counter = translate_sweep.apply(&mut microstate, &hamiltonian, &macrostate);
        assert_eq!(counter.accepted, 1);
        assert_eq!(counter.rejected, 0);

        // The second move to the right places the body just on the boundary and should be
        // rejected.
        let counter = translate_sweep.apply(&mut microstate, &hamiltonian, &macrostate);
        assert_eq!(counter.accepted, 0);
        assert_eq!(counter.rejected, 1);
    }

    #[test]
    fn reject_boundary_site() {
        let body = Body {
            properties: Point::new(Cartesian::from([0.0, 0.0])),
            sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
        };

        let cuboid = Hypercuboid {
            edge_lengths: [
                6.0.try_into()
                    .expect("hard-coded constant should be positive"),
                6.0.try_into()
                    .expect("hard-coded constant should be positive"),
            ],
        };
        let square = Closed(cuboid);
        let mut microstate = MicrostateBuilder::with_boundary(square)
            .bodies([body])
            .try_build()
            .expect("the hard-coded bodies should be in the boundary");
        let hamiltonian = Zero;
        let translate = Right;
        let translate_sweep = Sweep(translate);
        let macrostate = Isothermal { temperature: 1.0 };

        // The first move to the right ends in the boundary and should be accepted.
        let counter = translate_sweep.apply(&mut microstate, &hamiltonian, &macrostate);
        assert_eq!(counter.accepted, 1);
        assert_eq!(counter.rejected, 0);

        // The second move to the right places the body just on the boundary and should be
        // rejected.
        let counter = translate_sweep.apply(&mut microstate, &hamiltonian, &macrostate);
        assert_eq!(counter.accepted, 0);
        assert_eq!(counter.rejected, 1);
    }
}

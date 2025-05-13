// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Sweep
*/

use super::{Count, DeltaEnergyOne, LocalTrial, Trial};
use hoomd_microstate::boundary::Boundary;
use hoomd_microstate::property::Position;
use hoomd_microstate::{Body, Microstate, Transform};

use rand::Rng;
use std::marker::PhantomData;

/** Apply a local trial move to every body in the microstate.

Each trial move is accepted when:
<!-- r < \exp\left(\frac{-\Delta H}{kT}\right) -->
<math display="block" class="tml-display" style="display:block math;"><mrow><mi>r</mi><mo>&lt;</mo><mrow><mi>exp</mi><mo>⁡</mo></mrow><mrow><mo fence="true" form="prefix">(</mo><mfrac><mrow><mo lspace="0em" rspace="0em">−</mo><mpadded lspace="0"><mi mathvariant="normal">Δ</mi></mpadded><mi>H</mi></mrow><mrow><mi>k</mi><mi>T</mi></mrow></mfrac><mo fence="true" form="postfix">)</mo></mrow></mrow></math>
where `r` is a random value uniformly distributed in `[0,1)`, `\Delta H` is
the change in energy computed by the given `hamiltonian` and `kT` is the given
`state` value (the last argument to `apply`).


# Example

```
use hoomd_mc::{Sweep, Translate, Trial, Zero};
use hoomd_microstate::property::Position;
use hoomd_microstate::{Body, Microstate};
use hoomd_vector::Cartesian;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = Microstate::new();
microstate.add_body(Body::point(Cartesian::from([0.0, 0.0])));
let d = 0.1;
let translate_sweep = Sweep::new(Translate::new(d.try_into()?));

let hamiltonian = Zero;
let kt = 1.0;

for _ in 0..1_000 {
    translate_sweep.apply(&mut microstate, &hamiltonian, &kt);
    microstate.increment_step();
}
# Ok(())
# }
```
*/
#[expect(
    clippy::partial_pub_fields,
    reason = "Users do not need to be aware of phantom data."
)]
pub struct Sweep<L, V> {
    /// The local trial to apply.
    pub local: L,

    /// Make sweep depend on the vector type V even though it doesn't store a vector.
    vector_type: PhantomData<V>,
}

impl<L, V> Sweep<L, V> {
    /// Construct a new `Sweep` with the given local trial.
    #[inline]
    #[must_use]
    pub fn new(local: L) -> Self {
        Self {
            local,
            vector_type: PhantomData,
        }
    }
}

impl<V, B, S, C, L, H> Trial<Microstate<V, B, S, C>, H> for Sweep<L, V>
where
    B: Copy + Clone + Default + Transform<S> + Position<V>,
    S: Clone + Default + Position<V>,
    L: LocalTrial<B>,
    H: DeltaEnergyOne<V, B, S, C>,
    C: Boundary<V, B, S>,
{
    type Count = Count;
    type Macrostate = f64;

    #[inline]
    fn apply(
        &self,
        microstate: &mut Microstate<V, B, S, C>,
        hamiltonian: &H,
        state: &Self::Macrostate,
    ) -> Self::Count where {
        let kt = state;
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
                .wrap_body(self.local.propose(&mut rng, trial.properties))
            {
                Ok(new_properties) => {
                    trial.properties = new_properties;

                    let delta_h = hamiltonian.delta_energy_one(microstate, body_index, &trial);
                    if rng.random::<f64>() < (-delta_h / kt).exp()
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
    use crate::{Translate, Zero};
    use ::approx::assert_relative_eq;
    use hoomd_interaction::{Single, SiteEnergy, TotalEnergy};
    use hoomd_microstate::{MicrostateBuilder, boundary::Square, property::Point};
    use hoomd_vector::{Cartesian, Vector};
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
            .expect("valid body");
        let hamiltonian = Single(Harmonic(origin));

        let d = 0.1;
        let translate = Translate::new(d.try_into().expect("positive real"));
        let translate_sweep = Sweep::new(translate);

        let mut position_accumulator = Cartesian::default();
        let mut energy_accumulator = 0.0;

        for _ in 0..N_STEPS {
            translate_sweep.apply(&mut microstate, &hamiltonian, &kt);

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
        let square = Square { l: 4.0 };
        let mut microstate = MicrostateBuilder::with_boundary(square)
            .bodies([Body::point([0.0, 0.0].into())])
            .try_build()
            .expect("the hard-coded bodies should be in the boundary.");
        let hamiltonian = Zero;
        let translate = Right;
        let translate_sweep = Sweep::new(translate);

        // The first move to the right ends in the boundary and should be accepted.
        let counter = translate_sweep.apply(&mut microstate, &hamiltonian, &1.0);
        assert_eq!(counter.accepted, 1);
        assert_eq!(counter.rejected, 0);

        // The second move to the right places the body just on the boundary and should be
        // rejected.
        let counter = translate_sweep.apply(&mut microstate, &hamiltonian, &1.0);
        assert_eq!(counter.accepted, 0);
        assert_eq!(counter.rejected, 1);
    }

    #[test]
    fn reject_boundary_site() {
        let body = Body {
            properties: Point::new(Cartesian::from([0.0, 0.0])),
            sites: [Point::new(Cartesian::from([1.0, 0.0]))].into(),
        };

        let square = Square { l: 6.0 };
        let mut microstate = MicrostateBuilder::with_boundary(square)
            .bodies([body])
            .try_build()
            .expect("the hard-coded bodies should be in the boundary.");
        let hamiltonian = Zero;
        let translate = Right;
        let translate_sweep = Sweep::new(translate);

        // The first move to the right ends in the boundary and should be accepted.
        let counter = translate_sweep.apply(&mut microstate, &hamiltonian, &1.0);
        assert_eq!(counter.accepted, 1);
        assert_eq!(counter.rejected, 0);

        // The second move to the right places the body just on the boundary and should be
        // rejected.
        let counter = translate_sweep.apply(&mut microstate, &hamiltonian, &1.0);
        assert_eq!(counter.accepted, 0);
        assert_eq!(counter.rejected, 1);
    }
}

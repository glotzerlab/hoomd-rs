// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Closed
*/

use rand::{Rng, distr::Distribution};
use tinyvec::ArrayVec;

use super::{Error, GenerateGhosts, MAX_GHOSTS, Wrap};
use crate::property::Position;
use hoomd_geometry::IsPointInside;

/** Restrict points to the inside of a shape.

[`Closed`] is a newtype that wraps a shape. It prevents bodies and sites from
existing outside the shape. Bodies and sites are never wrapped, and there are no
ghost sites.
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Closed<T>(pub T);

impl<P, T, M> Wrap<P> for Closed<T>
where
    P: Position<Metric = M>,
    T: IsPointInside<M>,
{
    #[inline]
    fn wrap(&self, properties: P) -> Result<P, Error> {
        if self.0.is_point_inside(properties.position()) {
            Ok(properties)
        } else {
            Err(Error::CannotWrapProperties)
        }
    }
}

impl<S, T> GenerateGhosts<S> for Closed<T>
where
    S: Default,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        f64::INFINITY
    }

    #[inline]
    fn generate_ghosts(&self, _site_properties: &S) -> ArrayVec<[S; MAX_GHOSTS]> {
        ArrayVec::new()
    }
}

impl<T, V> Distribution<V> for Closed<T>
where
    T: Distribution<V>,
{
    /** Generate points uniformly distributed in the wrapped shape.

    # Example

    ```
    use hoomd_geometry::{IsPointInside, shape::Sphere};
    use hoomd_microstate::boundary::Closed;
    use rand::{SeedableRng, rngs::StdRng, distr::Distribution};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sphere = Closed(Sphere { radius: 5.0.try_into()? });
    let mut rng = StdRng::seed_from_u64(1);

    let point = sphere.sample(&mut rng);
    assert!(sphere.0.is_point_inside(&point));
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> V {
        self.0.sample(rng)
    }
}

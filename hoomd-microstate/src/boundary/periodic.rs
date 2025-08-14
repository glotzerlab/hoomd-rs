// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement periodic boundary conditions. */

use rand::{Rng, distr::Distribution};

use super::{Error, MaximumAllowableInteractionRange};

mod cuboid;

/** Describe a simulation space that repeats in one or more directions.

[`Periodic`] is a newtype that wraps a shape. Use it to set the `boundary`
for a [`Microstate`].

When bodies or sites exit the shape through one of the periodic sides of the
shape, they are wrapped to the other side. Similarly, sites that are within
the interaction range of one of the periodic sides appear as ghost sites just
outside the opposite side. Depending on the shape type `T`, `Periodic<T>` might
implement fully periodic boundaries or ones that are periodic in some directions
and closed in others.

[`Periodic`] is implemented for the following shapes:
* [`Cuboid<2>`] also known as [`Rectangle`]
* [`Cuboid<3>`]

[`Cuboid<2>`]: hoomd_geometry::shape::Cuboid
[`Cuboid<3>`]: hoomd_geometry::shape::Cuboid
[`Rectangle`]: hoomd_geometry::shape::Rectangle
[`Microstate`]: crate::Microstate

# Example

```
use hoomd_geometry::shape::Rectangle;
use hoomd_microstate::boundary::Periodic;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let periodic = Periodic::new(2.5, Rectangle::with_equal_edges(10.0.try_into()?))?;
# Ok(())
# }
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Periodic<T> {
    /// The largest interaction distance between two sites.
    maximum_interaction_range: f64,

    /// Bound the points that belong to the primary image.
    shape: T,
}

impl<T> Periodic<T>
where
    T: MaximumAllowableInteractionRange,
{
    /** Construct a new periodic boundary condition.

    # Errors

    [`Error::InteractionRangeTooLarge`] when `maximum_interaction_range` is
    larger than the maximum allowable interaction range by the shape.

    # Example

    ```
    use hoomd_geometry::shape::Rectangle;
    use hoomd_microstate::boundary::Periodic;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let periodic = Periodic::new(2.5, Rectangle::with_equal_edges(10.0.try_into()?))?;
    # Ok(())
    # }
    ```
    */
    #[inline]
    pub fn new(maximum_interaction_range: f64, shape: T) -> Result<Self, Error> {
        if maximum_interaction_range > shape.maximum_allowable_interaction_range() {
            return Err(Error::InteractionRangeTooLarge(
                maximum_interaction_range,
                shape.maximum_allowable_interaction_range(),
            ));
        }

        Ok(Self {
            maximum_interaction_range,
            shape,
        })
    }
}

impl<T> Periodic<T> {

    /** Access the boundary's shape.
    
    # Example

    ```
    use hoomd_geometry::shape::Rectangle;
    use hoomd_microstate::boundary::Periodic;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let periodic = Periodic::new(2.5, Rectangle::with_equal_edges(10.0.try_into()?))?;

    assert_eq!(periodic.shape().edge_lengths[0].get(), 10.0);
    # Ok(())
    # }
    ```
    */
    #[inline]
    pub fn shape(&self) -> &T {
        &self.shape
    }
}

impl<T, V> Distribution<V> for Periodic<T>
where
    T: Distribution<V>,
{
    /** Generate points uniformly distributed in the wrapped shape.

    # Example

    ```
    use rand::{SeedableRng, rngs::StdRng, distr::Distribution};
    
    use hoomd_geometry::{IsPointInside, shape::Cuboid};
    use hoomd_microstate::boundary::Periodic;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cuboid = Cuboid { edge_lengths: [6.0.try_into()?, 8.0.try_into()?] };
    let periodic = Periodic::new(2.5, cuboid)?;
    let mut rng = StdRng::seed_from_u64(1);

    let point = periodic.sample(&mut rng);
    assert!(periodic.shape().is_point_inside(&point));
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> V {
        self.shape.sample(rng)
    }
}

// TODO: Test that Periodic::new errors correctly

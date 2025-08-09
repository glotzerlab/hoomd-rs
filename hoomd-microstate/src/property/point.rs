// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Point */

use super::Position;
use crate::Transform;
use hoomd_vector::Metric;

/** A position in space and nothing more.

Use [`Point`] as a [`Body`](crate::Body) or [`Site`](crate::Site) property type.

# Example

```
use hoomd_vector::Cartesian;
use hoomd_microstate::property::Point;

let point = Point::new(Cartesian::from([1.0, -2.0, 3.0]));
```
*/
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point<M> {
    /// The location of the point in space.
    pub position: M,
}

impl<M> Point<M> {
    /** Construct a new point at the given position.

    # Example

    ```
    use hoomd_vector::Cartesian;
    use hoomd_microstate::property::Point;

    let point = Point::new(Cartesian::from([1.0, -2.0, 3.0]));
    ```
    */
    #[inline]
    #[must_use]
    pub fn new(position: M) -> Self {
        Self { position }
    }
}

/** Move [`Point`] properties from the local body frame to the system frame.
*/
impl<M> Transform<Point<M>> for Point<M>
where
    M: Metric,
{
    /** Points transform by vector addition.

    ```math
    \vec{r} = \vec{r}_\mathrm{body} + \vec{r}_\mathrm{site}
    ```

    ```
    use hoomd_vector::Cartesian;
    use hoomd_microstate::{property::Point, Transform};

    let body_properties = Point::new(Cartesian::from([1.0, -2.0, 3.0]));
    let site_properties = Point::new(Cartesian::from([-3.0, 2.0, 1.0]));

    let system_site = body_properties.transform(&site_properties);
    assert_eq!(system_site.position, [-2.0, 0.0, 4.0].into());
    ```
    */
    #[inline]
    fn transform(&self, site_properties: &Point<M>) -> Point<M> {
        Point {
            position: Metric::site_to_system(&self.position, &site_properties.position),
        }
    }
}

impl<M> Position for Point<M> {
    type Metric = M;

    #[inline]
    fn position(&self) -> &M {
        &self.position
    }

    #[inline]
    fn position_mut(&mut self) -> &mut M {
        &mut self.position
    }
}

// TODO: tests.

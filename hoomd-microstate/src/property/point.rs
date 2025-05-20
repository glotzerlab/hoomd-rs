// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Point */

use super::Position;
use crate::Transform;
use hoomd_vector::Vector;

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
pub struct Point<V> {
    /// The location of the point in space.
    pub position: V,
}

impl<V> Point<V> {
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
    pub fn new(position: V) -> Self {
        Self { position }
    }
}

/** Move [`Point`] properties from the local body frame to the system frame.
*/
impl<V> Transform<Point<V>> for Point<V>
where
    V: Vector,
{
    /** Points transform by vector addition.

    <!-- \vec{r} = \vec{r}_\mathrm{body} + \vec{r}_\mathrm{site} -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><mover><mi>r</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo>=</mo><msub><mover><mi>r</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mpadded lspace="0"><mi>body</mi></mpadded></msub><mo>+</mo><msub><mover><mi>r</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mpadded lspace="0"><mi>site</mi></mpadded></msub></mrow></math>

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
    fn transform(&self, site_properties: &Point<V>) -> Point<V> {
        Point {
            position: self.position + site_properties.position,
        }
    }
}

impl<V> Position for Point<V> {
    type Vector = V;

    #[inline]
    fn position(&self) -> &V {
        &self.position
    }

    #[inline]
    fn position_mut(&mut self) -> &mut V {
        &mut self.position
    }
}

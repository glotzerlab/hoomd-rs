// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Point */

use super::{Orientation, Point, Position};
use crate::Transform;
use hoomd_vector::{Rotate, Vector};

/** The position and orientation of an extended body.

Use [`OrientedPoint`] as a [`Body`](crate::Body) or [`Site`](crate::Site) property type.

# Example

```
use hoomd_microstate::property::OrientedPoint;
use hoomd_vector::{Angle, Cartesian};

let point = OrientedPoint { position: Cartesian::from([1.0, -3.0]),
    orientation: Angle::from(1.2),
};
```
*/
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OrientedPoint<V, R> {
    /// The location of the extended body in space.
    pub position: V,
    /// Rotate from the body's reference frame into another frame.
    pub orientation: R,
}

/** Move [`Point`] properties from the local body frame to the system frame.
*/
impl<V, R> Transform<Point<V>> for OrientedPoint<V, R>
where
    V: Vector,
    R: Rotate<V>,
{
    /** Rotate the point first, then translate.

    ```math
    \vec{r} = \vec{r}_\mathrm{body} + R(\vec{r}_\mathrm{site})
    ```

    ```
    use hoomd_vector::{Angle, Cartesian};
    use hoomd_microstate::{property::{OrientedPoint, Point}, Transform};
    use std::f64::consts::PI;
    use approx::assert_relative_eq;

    let body_properties = OrientedPoint {
        position: Cartesian::from([1.0, -2.0]),
        orientation: Angle::from(PI/2.0),
    };
    let site_properties = Point::new(Cartesian::from([-1.0, 0.0]));

    let system_site = body_properties.transform(&site_properties);
    assert_relative_eq!(system_site.position, [1.0, -3.0].into());
    ```
    */
    #[inline]
    fn transform(&self, site_properties: &Point<V>) -> Point<V> {
        Point {
            position: self.position + self.orientation.rotate(&site_properties.position),
        }
    }
}

impl<V, R> Position for OrientedPoint<V, R> {
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

impl<V, R> Orientation for OrientedPoint<V, R> {
    type Rotation = R;

    #[inline]
    fn orientation(&self) -> &R {
        &self.orientation
    }

    #[inline]
    fn orientation_mut(&mut self) -> &mut R {
        &mut self.orientation
    }
}

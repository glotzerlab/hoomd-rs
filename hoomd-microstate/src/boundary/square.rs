// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Square
*/

use super::Boundary;
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Cartesian;

use rand::{Rng, distr::{Distribution, Uniform}};
use std::array;

/** Restrict bodies and sites to the inside of a square.

The square covers the points:
* `-l/2 <= x < l/2`
* `-l/2 <= y < l/2`

# Example

```
use hoomd_microstate::boundary::Square;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let square = Square { l: 10.0.try_into()? };

assert_eq!(square.l.get(), 10.0);
# Ok(())
# }
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Square {
    /// Side length *(\[length\])*.
    pub l: PositiveReal,
}

impl Square {
    /** Get the maximum x coordinate in the square (exclusive).

    # Example

    ```
    use hoomd_microstate::boundary::Square;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let square = Square { l: 10.0.try_into()? };

    assert_eq!(square.maximum_x(), 5.0);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn maximum_x(&self) -> f64 {
        self.l.get()/2.0
    }

    /** Get the maximum y coordinate in the square (exclusive).

    # Example

    ```
    use hoomd_microstate::boundary::Square;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let square = Square { l: 10.0.try_into()? };

    assert_eq!(square.maximum_y(), 5.0);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn maximum_y(&self) -> f64 {
        self.l.get()/2.0
    }
    
    /** Get the minimum x coordinate in the square (inclusive).

    # Example

    ```
    use hoomd_microstate::boundary::Square;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let square = Square { l: 10.0.try_into()? };

    assert_eq!(square.minimum_x(), -5.0);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn minimum_x(&self) -> f64 {
        -self.l.get()/2.0
    }

    /** Get the minimum y coordinate in the square (inclusive).

    # Example

    ```
    use hoomd_microstate::boundary::Square;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let square = Square { l: 10.0.try_into()? };

    assert_eq!(square.minimum_y(), -5.0);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn minimum_y(&self) -> f64 {
        -self.l.get()/2.0
    }
}

impl<B, S> Boundary<Cartesian<2>, B, S> for Square {
    #[inline]
    fn is_inside(&self, point: &Cartesian<2>) -> bool {
        let l = self.l.get();
        point[0] >= -l / 2.0 && point[1] >= -l / 2.0 && point[0] < l / 2.0 && point[1] < l / 2.0
    }
}

impl Distribution<Cartesian<2>> for Square {
    /** Generate points uniformly distributed in the square.

    TODO: Example
    */
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Cartesian<2> {
        let uniform = Uniform::new(self.minimum_x(), self.minimum_y())
            .expect("square should always have real valued extents where the minimum is less than the maximum");

        array::from_fn(|_| uniform.sample(rng)).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::Point;
    use crate::{Body, Transform, boundary::Error};

    use rstest::*;

    const TOP: f64 = 1.0_f64.next_down();
    const OUTSIDE_BOTTOM: f64 = (-1.0_f64).next_down();
    const OUTSIDE_TOP: f64 = 1.0;

    #[rstest]
    fn valid_points(#[values([0.0, 0.0], [-1.0, -1.0], [-1.0, 0.0], [TOP, TOP])] v: [f64; 2]) {
        let square = Square {
            l: 2.0
                .try_into()
                .expect("hard-coded constant should be positive"),
        };
        let v = v.into();

        assert!(<Square as Boundary<
            Cartesian<2>,
            Point<Cartesian<2>>,
            Point<Cartesian<2>>,
        >>::is_inside(&square, &v));

        let body = Body::point(v);
        assert_eq!(
            <Square as Boundary<Cartesian<2>, Point<Cartesian<2>>, Point<Cartesian<2>>>>::wrap_body(
                &square,
                body.properties
            ),
            Ok(body.properties)
        );

        let site = body.properties.transform(&body.sites[0]);
        assert_eq!(
            <Square as Boundary<Cartesian<2>, Point<Cartesian<2>>, Point<Cartesian<2>>>>::wrap_site(
                &square, site
            ),
            Ok(site)
        );
    }

    #[rstest]
    fn invalid_points(
        #[values([-10.0, 10.0], [OUTSIDE_BOTTOM, 0.0], [-1.0, OUTSIDE_BOTTOM], [OUTSIDE_BOTTOM, 0.0], [OUTSIDE_TOP, OUTSIDE_TOP])]
        v: [f64; 2],
    ) {
        let square = Square {
            l: 2.0
                .try_into()
                .expect("hard-coded constant should be positive"),
        };
        let v = v.into();

        assert!(!<Square as Boundary<
            Cartesian<2>,
            Point<Cartesian<2>>,
            Point<Cartesian<2>>,
        >>::is_inside(&square, &v));

        let body = Body::point(v);
        assert_eq!(
            <Square as Boundary<Cartesian<2>, Point<Cartesian<2>>, Point<Cartesian<2>>>>::wrap_body(
                &square,
                body.properties
            ),
            Err(Error::CannotWrapBodyProperties)
        );

        let site = body.properties.transform(&body.sites[0]);
        assert_eq!(
            <Square as Boundary<Cartesian<2>, Point<Cartesian<2>>, Point<Cartesian<2>>>>::wrap_site(
                &square, site
            ),
            Err(Error::CannotWrapSiteProperties)
        );
    }
}

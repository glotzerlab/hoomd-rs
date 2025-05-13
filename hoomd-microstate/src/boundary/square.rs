// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Square
*/

use super::Boundary;
use crate::property::Point;
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Cartesian;

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
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Square {
    /// Side length *(\[length\])*.
    pub l: PositiveReal,
}

impl Boundary<Cartesian<2>, Point<Cartesian<2>>, Point<Cartesian<2>>> for Square {
    #[inline]
    fn is_inside(&self, point: &Cartesian<2>) -> bool {
        let l = self.l.get();
        point[0] >= -l / 2.0 && point[1] >= -l / 2.0 && point[0] < l / 2.0 && point[1] < l / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Body, Error, Transform};

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

        assert!(square.is_inside(&v));

        let body = Body::point(v);
        assert_eq!(square.wrap_body(body.properties), Ok(body.properties));

        let site = body.properties.transform(&body.sites[0]);
        assert_eq!(square.wrap_site(site), Ok(site));
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

        assert!(!square.is_inside(&v));

        let body = Body::point(v);
        assert_eq!(
            square.wrap_body(body.properties),
            Err(Error::CannotWrapProperties)
        );

        let site = body.properties.transform(&body.sites[0]);
        assert_eq!(square.wrap_site(site), Err(Error::CannotWrapProperties));
    }
}

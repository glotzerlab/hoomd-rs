// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement `PeriodicSquare`
*/

use tinyvec::ArrayVec;

use crate::property::Position;
use super::{Boundary, Error};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Cartesian;

/** Tile the plane with squares.

The primary image covers the points:
* `-side_length/2 <= x < side_length/2`
* `-side_length/2 <= y < side_length/2`

Any point outside the primary image can be wrapped.

# Example

```
use hoomd_microstate::boundary::PeriodicSquare;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let periodic_square = PeriodicSquare::try_new(10.0.try_into()?, 2.5)?;

assert_eq!(periodic_square.side_length().get(), 10.0);
# Ok(())
# }
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeriodicSquare {
    /// The length of one side of the square *(\[length\])*.
    side_length: PositiveReal,

    /// Maximum interaction range.
    maximum_interaction_range: f64,
}

impl PeriodicSquare {
    #[inline]
    pub fn try_new(side_length: PositiveReal, maximum_interaction_range: f64) -> Result<Self, Error> {

        Ok(Self {side_length, maximum_interaction_range})
    }

    /** Get the length of one side of the square.

    # Example

    ```
    use hoomd_microstate::boundary::PeriodicSquare;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let periodic_square = PeriodicSquare::try_new(10.0.try_into()?, 2.5)?;

    assert_eq!(periodic_square.side_length().get(), 10.0);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn side_length(&self) -> PositiveReal {
        self.side_length
    }

    /// Wrap a position vector into the primary image.
    #[inline]
    fn wrap_position(&self, position: &Cartesian<2>) -> Cartesian<2> {
        todo!();
    }
}

impl<B, S> Boundary<Cartesian<2>, B, S> for PeriodicSquare {
    #[inline]
    fn is_inside(&self, point: &Cartesian<2>) -> bool {
        let l = self.side_length.get();
        point[0] >= -l / 2.0 && point[1] >= -l / 2.0 && point[0] < l / 2.0 && point[1] < l / 2.0
    }

    #[inline]
    fn wrap_body(&self, body_properties: B) -> Result<B, Error>
    where
        B: Position<Vector = Cartesian<2>>,
    {
        let mut wrapped = body_properties;
        *wrapped.position_mut() = self.wrap_position(wrapped.position());
        Ok(wrapped)
    }

    #[inline]
    fn wrap_site(&self, site_properties: S) -> Result<S, Error>
    where
        S: Position<Vector = Cartesian<2>>,
    {
        let mut wrapped = site_properties;
        *wrapped.position_mut() = self.wrap_position(wrapped.position());
        Ok(wrapped)
    }

    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }

    #[inline]
    fn generate_ghosts(&self, site_properties: &S) -> ArrayVec<[S; super::MAX_GHOSTS]>
    where
        S: Default,
        S: Position<Vector = Cartesian<2>>,
    {
        ArrayVec::new()
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

    fn test_primary_point<T>(point: Cartesian<2>, boundary: &T)
        where T: Boundary<Cartesian<2>, Point<Cartesian<2>>, Point<Cartesian<2>>> {

        assert!(boundary.is_inside(&point));

        let body = Body::point(point);
        assert_eq!(boundary.wrap_body(
                body.properties
            ),
            Ok(body.properties)
        );

        let site = body.properties.transform(&body.sites[0]);
        assert_eq!(
            boundary.wrap_site(site
            ),
            Ok(site)
        );

        }

    #[rstest]
    fn primary_points(#[values([0.0, 0.0], [-1.0, -1.0], [-1.0, 0.0], [TOP, TOP])] v: [f64; 2]) {
        let periodic_square = PeriodicSquare::try_new(
            2.0
                .try_into()
                .expect("hard-coded constant should be positive"),
            0.0).expect("hard-coded boundary should be valid");
        let v = v.into();
        test_primary_point(v, &periodic_square);
    }

    #[rstest]
    fn invalid_points(
        #[values([-10.0, 10.0], [OUTSIDE_BOTTOM, 0.0], [-1.0, OUTSIDE_BOTTOM], [OUTSIDE_BOTTOM, 0.0], [OUTSIDE_TOP, OUTSIDE_TOP])]
        v: [f64; 2],
    ) {
        let periodic_square = PeriodicSquare::try_new(
            2.0
                .try_into()
                .expect("hard-coded constant should be positive"),
            0.0).expect("hard-coded boundary should be valid");
        let v = v.into();

        assert!(!<PeriodicSquare as Boundary<
            Cartesian<2>,
            Point<Cartesian<2>>,
            Point<Cartesian<2>>,
        >>::is_inside(&periodic_square, &v));

        let body = Body::point(v);
        assert_eq!(
            <PeriodicSquare as Boundary<Cartesian<2>, Point<Cartesian<2>>, Point<Cartesian<2>>>>::wrap_body(
                &periodic_square,
                body.properties
            ),
            Err(Error::CannotWrapBodyProperties)
        );

        let site = body.properties.transform(&body.sites[0]);
        assert_eq!(
            <PeriodicSquare as Boundary<Cartesian<2>, Point<Cartesian<2>>, Point<Cartesian<2>>>>::wrap_site(
                &periodic_square, site
            ),
            Err(Error::CannotWrapSiteProperties)
        );
    }
}

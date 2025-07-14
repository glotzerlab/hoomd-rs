// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Cuboid`] */

use crate::{BoundingSphereRadius, SupportMapping, Volume};
use hoomd_vector::Cartesian;
use itertools::multizip;
use std::array;

/** A shape with with all perpendicular angles made from axis-aligned edges.

A [`Cuboid`] is the N-dimensional analog of a rectangle, and is defined by
its edge lengths. Each perpendicular edge of the cuboid is aligned along the
corresponding Cartesian axis. The Cuboid is placed with its centroid at the
origin.

# Example

Construction and basic methods:
```
use hoomd_geometry::shape::Cuboid;
use hoomd_geometry::Volume;

let unit_cube = Cuboid {edge_lengths: [1.0; 3]};
assert_eq!(unit_cube.volume(), 1.0);

let min_extents = unit_cube.minimal_extents();
let max_extents = unit_cube.maximal_extents();
assert_eq!(min_extents, [-0.5; 3]);
assert_eq!(max_extents, [0.5; 3]);

let rectangular_prism = Cuboid {edge_lengths: [1.0, 1.0, 9.0]};

assert_eq!(rectangular_prism.volume(), 9.0);
```

Perform a fast AABB intersection tests:
```
use hoomd_geometry::shape::Cuboid;

let unit_cube = Cuboid {edge_lengths: [1.0; 3]};
let rectangular_prism = Cuboid {edge_lengths: [1.0, 1.0, 9.0]};

assert_eq!(unit_cube.intersects_aligned(&rectangular_prism, &[1.0; 3].into()), true);
assert_eq!(unit_cube.intersects_aligned(&rectangular_prism, &[1.1; 3].into()), false);
```

Wrap with [`Convex`](crate::Convex) to check intersections of oriented cuboids:

```
use hoomd_geometry::{Convex, IntersectsAt, shape::Rectangle};
use hoomd_vector::Angle;
use std::f64::consts::PI;

let square = Convex(Rectangle {edge_lengths: [1.0; 2]});

assert_eq!(square.intersects_at(&square, &[1.1, 0.0].into(), &Angle::default()), false);
assert_eq!(square.intersects_at(&square, &[1.1, 0.0].into(), &Angle::from(PI/4.0)), true);
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cuboid<const N: usize> {
    /// The lengths of each edge of the cuboid.
    pub edge_lengths: [f64; N],
}

/** An axis-aligned rectangle.

# Examples

Basic construction and methods:
```
use hoomd_geometry::shape::Rectangle;
use hoomd_geometry::Volume;

let rectangle = Rectangle {edge_lengths: [2.0, 4.0]};
assert_eq!(rectangle.volume(), 8.0);
```

Intersection tests:
```
use hoomd_geometry::{Convex, IntersectsAt, shape::Rectangle};
use hoomd_vector::{Cartesian, Angle};
use std::f64::consts::PI;

# fn main() -> Result<(), hoomd_geometry::Error> {
let rectangle = Rectangle { edge_lengths: [4.0, 2.0] };
let rectangle = Convex(rectangle);

assert_eq!(rectangle.intersects_at(&rectangle, &[0.0, 2.1].into(), &Angle::default()), false);
assert_eq!(rectangle.intersects_at(&rectangle, &[0.0, 2.1].into(), &Angle::from(PI/2.0)), true);
# Ok(())
# }
```
*/
pub type Rectangle = Cuboid<2>;

impl Cuboid<3> {
    /// Length of the `Cuboid` edge along the x axis
    #[inline]
    #[must_use]
    pub fn a(&self) -> f64 {
        self.edge_lengths[0]
    }
    /// Length of the `Cuboid` edge along the y axis
    #[inline]
    #[must_use]
    pub fn b(&self) -> f64 {
        self.edge_lengths[1]
    }
    /// Length of the `Cuboid` edge along the z axis
    #[inline]
    #[must_use]
    pub fn c(&self) -> f64 {
        self.edge_lengths[2]
    }
}

impl<const N: usize> Cuboid<N> {
    /** Test for intersections between two *axis-aligned* cuboids.

    This test is much faster than a general oriented cuboid (OBB) intersection, which
    can be achieved by wrapping with the [`Convex`](crate::Convex) newtype.

    # Example

    ```
    use hoomd_geometry::shape::Cuboid;

    let unit_cube = Cuboid {edge_lengths: [1.0; 3]};
    let rectangular_prism = Cuboid {edge_lengths: [1.0, 1.0, 9.0]};

    assert_eq!(unit_cube.intersects_aligned(&rectangular_prism, &[1.0; 3].into()), true);
    assert_eq!(unit_cube.intersects_aligned(&rectangular_prism, &[1.1; 3].into()), false);
    ```
    */
    #[must_use]
    #[inline]
    pub fn intersects_aligned(&self, other: &Cuboid<N>, v_ij: &Cartesian<N>) -> bool {
        let b_mins = Cartesian::from(other.minimal_extents()) + *v_ij;
        let b_maxs = Cartesian::from(other.maximal_extents()) + *v_ij;
        multizip((
            self.minimal_extents(),
            b_maxs,
            self.maximal_extents(),
            b_mins,
        ))
        .all(|(a_min, b_max, a_max, b_min)| (a_min <= b_max) && (a_max >= b_min))
    }
}

impl<const N: usize> Volume for Cuboid<N> {
    #[inline]
    fn volume(&self) -> f64 {
        self.edge_lengths
            .into_iter()
            .reduce(|acc, x| acc * x)
            .expect("N should be >= 1")
    }
}

impl<const N: usize> BoundingSphereRadius for Cuboid<N> {
    #[inline]
    fn bounding_sphere_radius(&self) -> f64 {
        f64::sqrt(
            self.edge_lengths
                .into_iter()
                .map(|x| (x / 2.0).powi(2))
                .sum(),
        )
    }
}

impl<const N: usize> SupportMapping<Cartesian<N>> for Cuboid<N> {
    #[inline]
    fn support_mapping(&self, n: &Cartesian<N>) -> Cartesian<N> {
        let mut iter = n
            .into_iter()
            .zip(self.edge_lengths)
            .map(|(n_i, l_i)| l_i / 2.0 * n_i.signum());
        array::from_fn(|_| iter.next().unwrap_or_default()).into()
    }
}

impl<const N: usize> Cuboid<N> {
    #[inline]
    #[must_use]
    /** Determine the maximal extents of the cuboid along each Cartesian axis.

    # Example

    ```
    use hoomd_geometry::shape::Cuboid;

    let unit_cube = Cuboid {edge_lengths: [1.0; 3].into()};

    let max_extents = unit_cube.maximal_extents();
    assert_eq!(max_extents, [0.5; 3]);
    ```
    */
    pub fn maximal_extents(&self) -> [f64; N] {
        array::from_fn(|i| self.edge_lengths[i] / 2.0)
    }

    #[inline]
    #[must_use]
    /** Determine the minimal extents of the cuboid along each Cartesian axis.

    # Example

    ```
    use hoomd_geometry::shape::Cuboid;

    let unit_cube = Cuboid {edge_lengths: [1.0; 3].into()};

    let min_extents = unit_cube.minimal_extents();
    assert_eq!(min_extents, [-0.5; 3]);
    ```
    */
    pub fn minimal_extents(&self) -> [f64; N] {
        array::from_fn(|i| -self.edge_lengths[i] / 2.0)
    }
}

#[cfg(test)]
#[expect(clippy::used_underscore_binding, reason = "Required for const tests.")]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rstest::*;
    use std::marker::PhantomData;

    #[rstest(
        edges0 => [[2.0, 2.0, 2.0]],
        edges1 => [[1.0, 1.0, 1.0]],
    )]
    fn test_box_intersections_aligned(edges0: [f64; 3], edges1: [f64; 3]) {
        let (s0, s1) = (
            Cuboid {
                edge_lengths: edges0,
            },
            Cuboid {
                edge_lengths: edges1,
            },
        );
        // Should all be false (no intersection), which we invert to true
        assert!(!s0.intersects_aligned(&s1, &[10.0, 10.0, 10.0].into()));
        // Boundaries are aligned
        assert!(s0.intersects_aligned(&s1, &[1.5, 1.5, 1.5].into()));
        // Both at origin - will always intersect for any cuboids
        assert!(s0.intersects_aligned(&s1, &[0.0, 0.0, 0.0].into()));
    }
    #[rstest(
        edges0 => [[2.0, 2.0]],
        edges1 => [[1.0, 1.0]],
    )]
    fn test_box_intersections_2d_aligned(edges0: [f64; 2], edges1: [f64; 2]) {
        let (c0, c1) = (
            Cuboid {
                edge_lengths: edges0,
            },
            Cuboid {
                edge_lengths: edges1,
            },
        );
        // Should all be false (no intersection), which we invert to true
        assert!(!c0.intersects_aligned(&c1, &[10.0, 10.0].into()));
        // Boundaries are aligned
        assert!(c0.intersects_aligned(&c1, &[1.5, 1.5].into()));
        // Both at origin - will always intersect for any cuboids
        assert!(c0.intersects_aligned(&c1, &[0.0, 0.0].into()));
    }

    #[rstest(
        _n => [
            PhantomData::<Cuboid<1>>,
            PhantomData::<Cuboid<2>>,
            PhantomData::<Cuboid<3>>,
            PhantomData::<Cuboid<4>>
        ],
        l => [1e-6, 1.0, 3.456, 99_999_999.9],
    )]
    fn test_box_extents<const N: usize>(_n: PhantomData<Cuboid<N>>, l: f64) {
        let c = Cuboid {
            edge_lengths: [l; N],
        };
        assert_eq!(c.maximal_extents(), [l / 2.0; N]);
        assert_eq!(c.minimal_extents(), [-l / 2.0; N]);
    }

    #[rstest(
        _n => [
            PhantomData::<Cuboid<1>>,
            PhantomData::<Cuboid<2>>,
            PhantomData::<Cuboid<3>>,
            PhantomData::<Cuboid<4>>
        ],
        l => [1e-6, 1.0, 3.456, 99_999_999.9],
    )]
    fn test_box_volume<const N: usize>(_n: PhantomData<Cuboid<N>>, l: f64) {
        let c = Cuboid {
            edge_lengths: [l; N],
        };
        assert_relative_eq!(
            c.volume(),
            if N != 0 {
                l.powi(i32::try_from(N).unwrap())
            } else {
                0.0
            }
        );
    }

    #[rstest(
        l => [1e-6, 1.0, 3.456, 99_999_999.9],
    )]
    fn test_box_abc(l: f64) {
        let c = Cuboid {
            edge_lengths: [l; 3],
        };
        assert_eq!([c.a(), c.b(), c.c()], [l; 3]);
    }

    #[test]
    fn bounding_sphere_radius_2d() {
        let cuboid = Cuboid {
            edge_lengths: [1.0, 1.0],
        };
        assert_relative_eq!(cuboid.bounding_sphere_radius(), 2.0_f64.sqrt() / 2.0);

        let cuboid = Cuboid {
            edge_lengths: [2.0, 2.0],
        };
        assert_relative_eq!(cuboid.bounding_sphere_radius(), 2.0_f64.sqrt());

        let cuboid = Cuboid {
            edge_lengths: [6.0, 8.0],
        };
        assert_relative_eq!(cuboid.bounding_sphere_radius(), 5.0);
    }

    #[test]
    fn bounding_sphere_radius_3d() {
        let cuboid = Cuboid {
            edge_lengths: [1.0, 1.0, 1.0],
        };
        assert_relative_eq!(cuboid.bounding_sphere_radius(), 3.0_f64.sqrt() / 2.0);

        let cuboid = Cuboid {
            edge_lengths: [2.0, 2.0, 2.0],
        };
        assert_relative_eq!(cuboid.bounding_sphere_radius(), 3.0_f64.sqrt());

        let cuboid = Cuboid {
            edge_lengths: [0.0, 6.0, 0.0],
        };
        assert_relative_eq!(cuboid.bounding_sphere_radius(), 3.0);

        let cuboid = Cuboid {
            edge_lengths: [6.0, 8.0, 0.0],
        };
        assert_relative_eq!(cuboid.bounding_sphere_radius(), 5.0);

        let cuboid = Cuboid {
            edge_lengths: [2.0, 4.0, 6.0],
        };
        assert_relative_eq!(cuboid.bounding_sphere_radius(), 14.0_f64.sqrt());
    }

    #[test]
    fn support_mapping_2d() {
        let cuboid = Cuboid {
            edge_lengths: [2.0, 4.0],
        };

        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([1.0, 0.1])),
            [1.0, 2.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([1.0, -0.1])),
            [1.0, -2.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([-0.1, 1.0])),
            [-1.0, 2.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([-0.1, -1.0])),
            [-1.0, -2.0].into()
        );
    }

    #[test]
    fn support_mapping_3d() {
        let cuboid = Cuboid {
            edge_lengths: [2.0, 4.0, 6.0],
        };

        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([1.0, 0.1, 0.1])),
            [1.0, 2.0, 3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([1.0, 0.1, -0.1])),
            [1.0, 2.0, -3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([1.0, -0.1, 0.1])),
            [1.0, -2.0, 3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([1.0, -0.1, -0.1])),
            [1.0, -2.0, -3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([-1.0, 0.1, 0.1])),
            [-1.0, 2.0, 3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([-1.0, 0.1, -0.1])),
            [-1.0, 2.0, -3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([-1.0, -0.1, 0.1])),
            [-1.0, -2.0, 3.0].into()
        );
        assert_relative_eq!(
            cuboid.support_mapping(&Cartesian::from([-1.0, -0.1, -0.1])),
            [-1.0, -2.0, -3.0].into()
        );
    }
}

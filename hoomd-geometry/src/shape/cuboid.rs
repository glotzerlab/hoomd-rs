// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Hypercuboid`]

use itertools::multizip;
use rand::{
    Rng,
    distr::{Distribution, Uniform},
};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::{array, ops::Mul};

use crate::{BoundingSphereRadius, Error, IsPointInside, MapPoint, Scale, SupportMapping, Volume};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Cartesian;

/// A shape with with all perpendicular angles made from axis-aligned edges.
///
/// A [`Hypercuboid`] is the N-dimensional analog of a rectangle, and is defined by
/// its edge lengths. Each perpendicular edge of the cuboid is aligned along the
/// corresponding Cartesian axis. The hypercuboid is placed with its centroid at the
/// origin.
///
/// # Example
///
/// Construction and basic methods:
/// ```
/// use hoomd_geometry::{Volume, shape::Hypercuboid};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let unit_cube = Hypercuboid {
///     edge_lengths: [1.0.try_into()?; 3],
/// };
/// assert_eq!(unit_cube.volume(), 1.0);
///
/// let min_extents = unit_cube.minimal_extents();
/// let max_extents = unit_cube.maximal_extents();
/// assert_eq!(min_extents, [-0.5; 3]);
/// assert_eq!(max_extents, [0.5; 3]);
///
/// let rectangular_prism = Hypercuboid {
///     edge_lengths: [1.0.try_into()?, 1.0.try_into()?, 9.0.try_into()?],
/// };
///
/// assert_eq!(rectangular_prism.volume(), 9.0);
/// # Ok(())
/// # }
/// ```
///
/// Perform a fast AABB intersection tests:
/// ```
/// use hoomd_geometry::shape::Hypercuboid;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let unit_cube = Hypercuboid {
///     edge_lengths: [1.0.try_into()?; 3],
/// };
/// let rectangular_prism = Hypercuboid {
///     edge_lengths: [1.0.try_into()?, 1.0.try_into()?, 9.0.try_into()?],
/// };
///
/// assert!(unit_cube.intersects_aligned(&rectangular_prism, &[1.0; 3].into()));
/// assert!(
///     !unit_cube.intersects_aligned(&rectangular_prism, &[1.1; 3].into())
/// );
/// # Ok(())
/// # }
/// ```
///
/// Wrap with [`Convex`](crate::Convex) to check intersections of oriented cuboids:
///
/// ```
/// use hoomd_geometry::{Convex, IntersectsAt, shape::Rectangle};
/// use hoomd_vector::Angle;
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let square = Convex(Rectangle {
///     edge_lengths: [1.0.try_into()?; 2],
/// });
///
/// assert!(!square.intersects_at(
///     &square,
///     &[1.1, 0.0].into(),
///     &Angle::default()
/// ));
/// assert!(square.intersects_at(
///     &square,
///     &[1.1, 0.0].into(),
///     &Angle::from(PI / 4.0)
/// ));
/// # Ok(())
/// # }
/// ```
#[serde_as]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hypercuboid<const N: usize> {
    /// The lengths of each edge of the cuboid.
    #[serde_as(as = "[_; N]")]
    pub edge_lengths: [PositiveReal; N],
}

/// An axis-aligned rectangle.
///
/// # Examples
///
/// Basic construction and methods:
/// ```
/// use hoomd_geometry::{Volume, shape::Rectangle};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let rectangle = Rectangle {
///     edge_lengths: [2.0.try_into()?, 4.0.try_into()?],
/// };
/// assert_eq!(rectangle.volume(), 8.0);
/// # Ok(())
/// # }
/// ```
///
/// Intersection tests:
/// ```
/// use hoomd_geometry::{Convex, IntersectsAt, shape::Rectangle};
/// use hoomd_vector::{Angle, Cartesian};
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let rectangle = Rectangle {
///     edge_lengths: [4.0.try_into()?, 2.0.try_into()?],
/// };
/// let rectangle = Convex(rectangle);
///
/// assert!(!rectangle.intersects_at(
///     &rectangle,
///     &[0.0, 2.1].into(),
///     &Angle::default()
/// ));
/// assert!(rectangle.intersects_at(
///     &rectangle,
///     &[0.0, 2.1].into(),
///     &Angle::from(PI / 2.0)
/// ));
/// # Ok(())
/// # }
/// ```
pub type Rectangle = Hypercuboid<2>;

/// An axis-aligned cuboid.
///
/// # Examples
///
/// Basic construction and methods:
/// ```
/// use hoomd_geometry::{Volume, shape::Cuboid};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cuboid = Cuboid {
///     edge_lengths: [2.0.try_into()?, 4.0.try_into()?, 7.0.try_into()?],
/// };
/// assert_eq!(cuboid.volume(), 56.0);
/// # Ok(())
/// # }
/// ```
///
/// Intersection tests:
/// ```
/// use hoomd_geometry::{Convex, IntersectsAt, shape::Cuboid};
/// use hoomd_vector::{Cartesian, Versor};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cuboid = Cuboid {
///     edge_lengths: [4.0.try_into()?, 2.0.try_into()?, 7.0.try_into()?],
/// };
/// let cuboid = Convex(cuboid);
///
/// assert!(!cuboid.intersects_at(
///     &cuboid,
///     &[0.0, 2.1, 0.0].into(),
///     &Versor::default()
/// ));
/// assert!(cuboid.intersects_at(
///     &cuboid,
///     &[2.0, 1.0, 6.5].into(),
///     &Versor::default()
/// ));
/// # Ok(())
/// # }
/// ```
pub type Cuboid = Hypercuboid<3>;

impl Hypercuboid<3> {
    /// Length of the `Hypercuboid` edge along the x axis
    #[inline]
    #[must_use]
    pub fn a(&self) -> PositiveReal {
        self.edge_lengths[0]
    }
    /// Length of the `Hypercuboid` edge along the y axis
    #[inline]
    #[must_use]
    pub fn b(&self) -> PositiveReal {
        self.edge_lengths[1]
    }
    /// Length of the `Hypercuboid` edge along the z axis
    #[inline]
    #[must_use]
    pub fn c(&self) -> PositiveReal {
        self.edge_lengths[2]
    }
}

impl<const N: usize> Hypercuboid<N> {
    /// Construct a cuboid with all edge lengths equal.
    ///
    /// # Example
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let square = Rectangle::with_equal_edges(10.0.try_into()?);
    /// # Ok(())
    /// # }
    #[inline]
    #[must_use]
    pub fn with_equal_edges(l: PositiveReal) -> Self {
        Self {
            edge_lengths: [l; N],
        }
    }

    /// Test for intersections between two *axis-aligned* cuboids.
    ///
    /// This test is much faster than a general oriented cuboid (OBB) intersection, which
    /// can be achieved by wrapping with the [`Convex`](crate::Convex) newtype.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Hypercuboid;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let unit_cube = Hypercuboid {
    ///     edge_lengths: [1.0.try_into()?; 3],
    /// };
    /// let rectangular_prism = Hypercuboid {
    ///     edge_lengths: [1.0.try_into()?, 1.0.try_into()?, 9.0.try_into()?],
    /// };
    ///
    /// assert!(unit_cube.intersects_aligned(&rectangular_prism, &[1.0; 3].into()));
    /// assert!(
    ///     !unit_cube.intersects_aligned(&rectangular_prism, &[1.1; 3].into())
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn intersects_aligned(&self, other: &Hypercuboid<N>, v_ij: &Cartesian<N>) -> bool {
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

impl<const N: usize> Volume for Hypercuboid<N> {
    #[inline]
    fn volume(&self) -> f64 {
        self.edge_lengths
            .iter()
            .map(PositiveReal::get)
            .reduce(f64::mul)
            .expect("N should be >= 1")
    }
}

impl<const N: usize> BoundingSphereRadius for Hypercuboid<N> {
    #[inline]
    fn bounding_sphere_radius(&self) -> PositiveReal {
        f64::sqrt(
            self.edge_lengths
                .iter()
                .map(PositiveReal::get)
                .map(|x| (x / 2.0).powi(2))
                .sum(),
        )
        .try_into()
        .expect("expression evaluates to a positive real")
    }
}

impl<const N: usize> SupportMapping<Cartesian<N>> for Hypercuboid<N> {
    #[inline]
    fn support_mapping(&self, n: &Cartesian<N>) -> Cartesian<N> {
        let mut iter = n
            .into_iter()
            .zip(self.edge_lengths)
            .map(|(n_i, l_i)| l_i.get() / 2.0 * n_i.signum());
        array::from_fn(|_| iter.next().unwrap_or_default()).into()
    }
}

impl<const N: usize> Hypercuboid<N> {
    #[inline]
    #[must_use]
    /// Determine the maximal extents of the cuboid along each Cartesian axis.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Hypercuboid;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let unit_cube = Hypercuboid {
    ///     edge_lengths: [1.0.try_into()?; 3],
    /// };
    ///
    /// let max_extents = unit_cube.maximal_extents();
    /// assert_eq!(max_extents, [0.5; 3]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn maximal_extents(&self) -> [f64; N] {
        array::from_fn(|i| self.edge_lengths[i].get() / 2.0)
    }

    #[inline]
    #[must_use]
    /// Determine the minimal extents of the cuboid along each Cartesian axis.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Hypercuboid;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let unit_cube = Hypercuboid {
    ///     edge_lengths: [1.0.try_into()?; 3],
    /// };
    ///
    /// let min_extents = unit_cube.minimal_extents();
    /// assert_eq!(min_extents, [-0.5; 3]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn minimal_extents(&self) -> [f64; N] {
        array::from_fn(|i| -self.edge_lengths[i].get() / 2.0)
    }
}

impl Hypercuboid<2> {
    /// Represent the shape in the GSD box format.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rectangle = Rectangle {
    ///     edge_lengths: [10.0.try_into()?, 15.0.try_into()?],
    /// };
    ///
    /// let gsd_box = rectangle.to_gsd_box();
    /// assert_eq!(gsd_box, [10.0, 15.0, 0.0, 0.0, 0.0, 0.0]);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn to_gsd_box(&self) -> [f64; 6] {
        [
            self.edge_lengths[0].get(),
            self.edge_lengths[1].get(),
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    }
}

impl Hypercuboid<3> {
    /// Represent the shape in the GSD box format.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Cuboid;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rectangle = Cuboid {
    ///     edge_lengths: [10.0.try_into()?, 15.0.try_into()?, 20.0.try_into()?],
    /// };
    ///
    /// let gsd_box = rectangle.to_gsd_box();
    /// assert_eq!(gsd_box, [10.0, 15.0, 20.0, 0.0, 0.0, 0.0]);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn to_gsd_box(&self) -> [f64; 6] {
        [
            self.edge_lengths[0].get(),
            self.edge_lengths[1].get(),
            self.edge_lengths[2].get(),
            0.0,
            0.0,
            0.0,
        ]
    }
}

impl<const N: usize> IsPointInside<Cartesian<N>> for Hypercuboid<N> {
    /// Check if a cartesian vector is inside a cuboid.
    ///
    /// By conventions typically used in periodic boundary conditions, points
    /// exactly at the minimal extent are inside the shape but points exactly
    /// on the maximal extent are not:
    /// ```math
    /// -\frac{L_x}{2} \le x \lt \frac{L_x}{2}
    /// ```
    /// ```math
    /// -\frac{L_y}{2} \le y \lt \frac{L_y}{2}
    /// ```
    /// ... and so on
    ///
    /// ```
    /// use hoomd_geometry::{IsPointInside, shape::Hypercuboid};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let cuboid = Hypercuboid {
    ///     edge_lengths: [6.0.try_into()?, 8.0.try_into()?],
    /// };
    ///
    /// assert!(cuboid.is_point_inside(&[2.5, -3.5].into()));
    /// assert!(!cuboid.is_point_inside(&[4.0, -3.5].into()));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn is_point_inside(&self, point: &Cartesian<N>) -> bool {
        point
            .into_iter()
            .zip(&self.edge_lengths)
            .all(|(x, l)| -l.get() / 2.0 <= x && x < l.get() / 2.0)
    }
}

impl<const N: usize> Scale for Hypercuboid<N> {
    /// Construct a scaled hypercuboid.
    ///
    /// The resulting hypercuboid's edge lengths $` L_\mathrm{new} `$ are
    /// the original's $` L `$ scaled by $` v `$:
    /// ```math
    /// L_\mathrm{new} = v L
    /// ```
    ///
    /// The centroid remains at the origin.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::Rectangle};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rectangle = Rectangle::with_equal_edges(10.0.try_into()?);
    ///
    /// let scaled_rectangle = rectangle.scale_length(0.5.try_into()?);
    ///
    /// assert_eq!(scaled_rectangle.edge_lengths[0].get(), 5.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_length(&self, v: PositiveReal) -> Self {
        Self {
            edge_lengths: array::from_fn(|i| self.edge_lengths[i] * v),
        }
    }

    /// Construct a scaled hypercuboid.
    ///
    /// The resulting hypercuboid's edge lengths $` L_\mathrm{new} `$ are
    /// the original's $` L `$ scaled by $` v^\frac{1}{N} `$:
    /// ```math
    /// L_\mathrm{new} = v^\frac{1}{N} L
    /// ```
    ///
    /// The centroid remains at the origin.
    ///
    /// # Examples
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::Rectangle};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rectangle = Rectangle::with_equal_edges(10.0.try_into()?);
    ///
    /// let scaled_rectangle = rectangle.scale_volume(4.0.try_into()?);
    ///
    /// assert_eq!(scaled_rectangle.edge_lengths[0].get(), 20.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_volume(&self, v: PositiveReal) -> Self {
        let v = v.get().powf(1.0 / N as f64);
        self.scale_length(v.try_into().expect("v^{1/N} should be a positive real"))
    }
}

impl<const N: usize> MapPoint<Cartesian<N>> for Hypercuboid<N> {
    /// Map a point from one hypercuboid to another.
    ///
    /// Given a point P *inside `self`*, map it to the other shape
    /// by scaling.
    ///
    /// # Errors
    ///
    /// Returns [`Error::PointOutsideShape`] when `point` is outside the shape
    /// `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{MapPoint, shape::Rectangle};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rectangle_a = Rectangle::with_equal_edges(10.0.try_into()?);
    /// let rectangle_b = Rectangle::with_equal_edges(20.0.try_into()?);
    ///
    /// let mapped_point =
    ///     rectangle_a.map_point(Cartesian::from([-1.0, 1.0]), &rectangle_b);
    ///
    /// assert_eq!(mapped_point, Ok(Cartesian::from([-2.0, 2.0])));
    /// assert_eq!(
    ///     rectangle_a.map_point(Cartesian::from([-100.0, 1.0]), &rectangle_b),
    ///     Err(hoomd_geometry::Error::PointOutsideShape)
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn map_point(&self, point: Cartesian<N>, other: &Self) -> Result<Cartesian<N>, Error> {
        if !self.is_point_inside(&point) {
            return Err(Error::PointOutsideShape);
        }

        let scale: [_; N] = array::from_fn(|i| other.edge_lengths[i] / self.edge_lengths[i]);
        Ok(Cartesian::from(array::from_fn(|i| {
            (scale[i].get() * point[i]).clamp(
                -other.edge_lengths[i].get() / 2.0,
                (other.edge_lengths[i].get() / 2.0).next_down(),
            )
        })))
    }
}

impl<const N: usize> Distribution<Cartesian<N>> for Hypercuboid<N> {
    /// Generate points uniformly distributed in the cuboid.
    ///
    /// # Example
    ///
    /// ```
    /// use rand::{SeedableRng, distr::Distribution, rngs::StdRng};
    ///
    /// use hoomd_geometry::{IsPointInside, shape::Hypercuboid};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let cuboid = Hypercuboid {
    ///     edge_lengths: [6.0.try_into()?, 8.0.try_into()?],
    /// };
    /// let mut rng = StdRng::seed_from_u64(1);
    ///
    /// let point = cuboid.sample(&mut rng);
    /// assert!(cuboid.is_point_inside(&point));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Cartesian<N> {
        let minimal_extents = self.minimal_extents();
        let maximal_extents = self.maximal_extents();

        array::from_fn(|i| {

            let uniform = Uniform::new(minimal_extents[i], maximal_extents[i])
                .expect("cuboid should always have real valued extents where the minimum is less than the maximum");
            uniform.sample(rng)}).into()
    }
}

#[cfg(test)]
#[expect(clippy::used_underscore_binding, reason = "Required for const tests.")]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use assert2::check;
    use rand::{SeedableRng, distr::Distribution, rngs::StdRng};
    use rstest::*;
    use std::marker::PhantomData;

    /// Number of random samples to test.
    const N: usize = 1024;

    #[rstest(
        edges0 => [[2.0.try_into().expect("test value is a positive real"), 2.0.try_into().expect("test value is a positive real"), 2.0.try_into().expect("test value is a positive real")]],
        edges1 => [[1.0.try_into().expect("test value is a positive real"), 1.0.try_into().expect("test value is a positive real"), 1.0.try_into().expect("test value is a positive real")]],
    )]
    fn test_box_intersections_aligned(edges0: [PositiveReal; 3], edges1: [PositiveReal; 3]) {
        let (s0, s1) = (
            Hypercuboid {
                edge_lengths: edges0,
            },
            Hypercuboid {
                edge_lengths: edges1,
            },
        );
        // Should all be false (no intersection), which we invert to true
        check!(!s0.intersects_aligned(&s1, &[10.0, 10.0, 10.0].into()));
        // Boundaries are aligned
        check!(s0.intersects_aligned(&s1, &[1.5, 1.5, 1.5].into()));
        // Both at origin - will always intersect for any cuboids
        check!(s0.intersects_aligned(&s1, &[0.0, 0.0, 0.0].into()));
    }
    #[rstest(
        edges0 => [[2.0.try_into().expect("test value is a positive real"), 2.0.try_into().expect("test value is a positive real")]],
        edges1 => [[1.0.try_into().expect("test value is a positive real"), 1.0.try_into().expect("test value is a positive real")]],
    )]
    fn test_box_intersections_2d_aligned(edges0: [PositiveReal; 2], edges1: [PositiveReal; 2]) {
        let (c0, c1) = (
            Hypercuboid {
                edge_lengths: edges0,
            },
            Hypercuboid {
                edge_lengths: edges1,
            },
        );
        // Should all be false (no intersection), which we invert to true
        check!(!c0.intersects_aligned(&c1, &[10.0, 10.0].into()));
        // Boundaries are aligned
        check!(c0.intersects_aligned(&c1, &[1.5, 1.5].into()));
        // Both at origin - will always intersect for any cuboids
        check!(c0.intersects_aligned(&c1, &[0.0, 0.0].into()));
    }

    #[rstest(
        _n => [
            PhantomData::<Hypercuboid<1>>,
            PhantomData::<Hypercuboid<2>>,
            PhantomData::<Hypercuboid<3>>,
            PhantomData::<Hypercuboid<4>>
        ],
        l => [1e-6, 1.0, 3.456, 99_999_999.9],
    )]
    fn test_box_extents<const N: usize>(_n: PhantomData<Hypercuboid<N>>, l: f64) {
        let c = Hypercuboid {
            edge_lengths: [l.try_into().expect("test value is a positive real"); N],
        };
        check!(c.maximal_extents() == [l / 2.0; N]);
        check!(c.minimal_extents() == [-l / 2.0; N]);
    }

    #[rstest(
        _n => [
            PhantomData::<Hypercuboid<1>>,
            PhantomData::<Hypercuboid<2>>,
            PhantomData::<Hypercuboid<3>>,
            PhantomData::<Hypercuboid<4>>
        ],
        l => [1e-6, 1.0, 3.456, 99_999_999.9],
    )]
    fn test_box_volume<const N: usize>(
        _n: PhantomData<Hypercuboid<N>>,
        l: f64,
    ) -> anyhow::Result<()> {
        let c = Hypercuboid {
            edge_lengths: [l.try_into()?; N],
        };
        assert_relative_eq!(
            c.volume(),
            if N != 0 {
                l.powi(i32::try_from(N).unwrap())
            } else {
                0.0
            }
        );

        Ok(())
    }

    #[rstest(
        l => [1e-6, 1.0, 3.456, 99_999_999.9],
    )]
    fn test_box_abc(l: f64) -> anyhow::Result<()> {
        let c = Hypercuboid {
            edge_lengths: [l.try_into()?; 3],
        };
        check!([c.a(), c.b(), c.c()] == [l.try_into()?; 3]);

        Ok(())
    }

    #[test]
    fn bounding_sphere_radius_2d() -> anyhow::Result<()> {
        let cuboid = Hypercuboid {
            edge_lengths: [1.0.try_into()?, 1.0.try_into()?],
        };
        assert_relative_eq!(cuboid.bounding_sphere_radius().get(), 2.0_f64.sqrt() / 2.0);

        let cuboid = Hypercuboid {
            edge_lengths: [2.0.try_into()?, 2.0.try_into()?],
        };
        assert_relative_eq!(cuboid.bounding_sphere_radius().get(), 2.0_f64.sqrt());

        let cuboid = Hypercuboid {
            edge_lengths: [6.0.try_into()?, 8.0.try_into()?],
        };
        assert_relative_eq!(cuboid.bounding_sphere_radius().get(), 5.0);

        Ok(())
    }

    #[test]
    fn bounding_sphere_radius_3d() -> anyhow::Result<()> {
        let cuboid = Hypercuboid {
            edge_lengths: [1.0.try_into()?, 1.0.try_into()?, 1.0.try_into()?],
        };
        assert_relative_eq!(cuboid.bounding_sphere_radius().get(), 3.0_f64.sqrt() / 2.0);

        let cuboid = Hypercuboid {
            edge_lengths: [2.0.try_into()?, 2.0.try_into()?, 2.0.try_into()?],
        };
        assert_relative_eq!(cuboid.bounding_sphere_radius().get(), 3.0_f64.sqrt());

        let cuboid = Hypercuboid {
            edge_lengths: [2.0.try_into()?, 4.0.try_into()?, 6.0.try_into()?],
        };
        assert_relative_eq!(cuboid.bounding_sphere_radius().get(), 14.0_f64.sqrt());

        Ok(())
    }

    #[test]
    fn support_mapping_2d() -> anyhow::Result<()> {
        let cuboid = Hypercuboid {
            edge_lengths: [2.0.try_into()?, 4.0.try_into()?],
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

        Ok(())
    }

    #[test]
    fn support_mapping_3d() -> anyhow::Result<()> {
        let cuboid = Hypercuboid {
            edge_lengths: [2.0.try_into()?, 4.0.try_into()?, 6.0.try_into()?],
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

        Ok(())
    }

    #[test]
    fn is_point_inside() -> anyhow::Result<()> {
        let cuboid = Hypercuboid {
            edge_lengths: [2.0.try_into()?, 4.0.try_into()?],
        };

        check!(cuboid.is_point_inside(&Cartesian::from([0.0, 0.0])));
        check!(cuboid.is_point_inside(&Cartesian::from([-1.0, 0.0])));
        check!(cuboid.is_point_inside(&Cartesian::from([0.0, -2.0])));
        check!(cuboid.is_point_inside(&Cartesian::from([-1.0, -2.0])));
        check!(cuboid.is_point_inside(&Cartesian::from([0.5, -1.0])));

        check!(!cuboid.is_point_inside(&Cartesian::from([1.0, 0.0])));
        check!(!cuboid.is_point_inside(&Cartesian::from([0.0, 2.0])));
        check!(!cuboid.is_point_inside(&Cartesian::from([1.0, 2.0])));
        check!(!cuboid.is_point_inside(&Cartesian::from([10.0, -20.0])));

        Ok(())
    }

    #[test]
    fn distribution() -> anyhow::Result<()> {
        let cuboid = Hypercuboid {
            edge_lengths: [6.0.try_into()?, 10.0.try_into()?],
        };
        let mut rng = StdRng::seed_from_u64(3);

        let points: Vec<_> = (&cuboid).sample_iter(&mut rng).take(N).collect();
        check!(&points.iter().all(|p| cuboid.is_point_inside(p)));
        check!(&points.iter().any(|p| p[0] < -2.8));
        check!(&points.iter().any(|p| p[0] > 2.8));
        check!(&points.iter().any(|p| p[1] < -4.8));
        check!(&points.iter().any(|p| p[1] > 4.8));

        Ok(())
    }

    #[test]
    fn test_scale_length() -> anyhow::Result<()> {
        let cuboid = Hypercuboid {
            edge_lengths: [6.0.try_into()?, 10.0.try_into()?],
        };

        let scaled_cuboid = cuboid.scale_length(2.0.try_into()?);
        check!(scaled_cuboid.edge_lengths[0].get() == 12.0);
        check!(scaled_cuboid.edge_lengths[1].get() == 20.0);

        let scaled_cuboid = cuboid.scale_length(0.5.try_into()?);
        check!(scaled_cuboid.edge_lengths[0].get() == 3.0);
        check!(scaled_cuboid.edge_lengths[1].get() == 5.0);

        Ok(())
    }

    #[test]
    fn test_scale_volume() -> anyhow::Result<()> {
        let cuboid = Hypercuboid {
            edge_lengths: [6.0.try_into()?, 10.0.try_into()?],
        };

        let scaled_cuboid = cuboid.scale_volume(4.0.try_into()?);
        check!(scaled_cuboid.edge_lengths[0].get() == 12.0);
        check!(scaled_cuboid.edge_lengths[1].get() == 20.0);

        let scaled_cuboid = cuboid.scale_volume(0.25.try_into()?);
        check!(scaled_cuboid.edge_lengths[0].get() == 3.0);
        check!(scaled_cuboid.edge_lengths[1].get() == 5.0);

        Ok(())
    }

    #[test]
    fn test_map_basic() -> anyhow::Result<()> {
        let cuboid_a = Hypercuboid {
            edge_lengths: [6.0.try_into()?, 10.0.try_into()?],
        };

        let cuboid_b = Hypercuboid {
            edge_lengths: [12.0.try_into()?, 5.0.try_into()?],
        };

        check!(
            cuboid_a.map_point(Cartesian::from([0.0, 0.0]), &cuboid_b)
                == Ok(Cartesian::from([0.0, 0.0]))
        );
        check!(
            cuboid_b.map_point(Cartesian::from([0.0, 0.0]), &cuboid_a)
                == Ok(Cartesian::from([0.0, 0.0]))
        );

        check!(
            cuboid_a.map_point(Cartesian::from([100.0, 0.0]), &cuboid_b)
                == Err(Error::PointOutsideShape)
        );
        check!(
            cuboid_b.map_point(Cartesian::from([0.0, -200.0]), &cuboid_a)
                == Err(Error::PointOutsideShape)
        );

        check!(
            cuboid_a.map_point(Cartesian::from([2.0, 1.0]), &cuboid_b)
                == Ok(Cartesian::from([4.0, 0.5]))
        );
        check!(
            cuboid_b.map_point(Cartesian::from([-4.0, 0.5]), &cuboid_a)
                == Ok(Cartesian::from([-2.0, 1.0]))
        );

        check!(
            cuboid_a.map_point(Cartesian::from([-3.0, -5.0]), &cuboid_b)
                == Ok(Cartesian::from([-6.0, -2.5]))
        );
        check!(
            cuboid_b.map_point(Cartesian::from([-6.0, -2.5]), &cuboid_a)
                == Ok(Cartesian::from([-3.0, -5.0]))
        );

        Ok(())
    }

    #[test]
    fn test_map_corner() -> anyhow::Result<()> {
        let mut rng = StdRng::seed_from_u64(3);
        let uniform = Uniform::new(1.0, 1000.0)?;

        for _ in 0..65536 {
            let a = uniform.sample(&mut rng);
            let b = uniform.sample(&mut rng);
            let cuboid_a = Hypercuboid::<2>::with_equal_edges(a.try_into()?);
            let cuboid_b = Hypercuboid::<2>::with_equal_edges(b.try_into()?);

            // Test that points right on the boundary of one shape remain inside
            // the other shape. If not implemented correctly, map_point might
            // round  and place a point just outside the shape. This test fails
            // when the `.clamp` call in `map_point` is commented out.

            let lower_left =
                cuboid_a.map_point(Cartesian::from([-a / 2.0, -a / 2.0]), &cuboid_b)?;
            check!(
                cuboid_b.is_point_inside(&lower_left),
                "{lower_left:?} should be inside {cuboid_b:?}"
            );

            let upper_right = cuboid_a.map_point(
                Cartesian::from([(a / 2.0).next_down(), (a / 2.0).next_down()]),
                &cuboid_b,
            )?;
            check!(
                cuboid_b.is_point_inside(&upper_right),
                "{upper_right:?} should be inside {cuboid_b:?}"
            );
        }

        Ok(())
    }
}

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement periodic boundary conditions for cuboids in cartesian space.

use arrayvec::ArrayVec;

use crate::{
    boundary::{
        Error, GenerateGhosts, MAX_GHOSTS, MaximumAllowableInteractionRange, Periodic, Wrap,
    },
    property::Position,
};
use hoomd_geometry::{IsPointInside, shape::Hypercuboid};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Cartesian;

impl<const N: usize> MaximumAllowableInteractionRange for Hypercuboid<N> {
    /// The largest value that the maximum interaction range can take.
    ///
    /// For a cuboid, the maximum is
    /// ```math
    /// \frac{L_\mathrm{min}}{2}
    /// ```
    /// where $`L_\mathrm{min}`$ is the smallest edge length.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Hypercuboid;
    /// use hoomd_microstate::boundary::MaximumAllowableInteractionRange;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rectangular_prism = Hypercuboid {
    ///     edge_lengths: [2.0.try_into()?, 3.0.try_into()?, 9.0.try_into()?],
    /// };
    ///
    /// assert_eq!(rectangular_prism.maximum_allowable_interaction_range(), 1.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn maximum_allowable_interaction_range(&self) -> f64 {
        let minimum_l = self
            .edge_lengths
            .iter()
            .map(PositiveReal::get)
            .reduce(f64::min)
            .expect("cuboid should have dimension 1 or greater");
        minimum_l / 2.0
    }
}

impl<const N: usize, P> Wrap<P> for Periodic<Hypercuboid<N>>
where
    P: Position<Position = Cartesian<N>>,
{
    /// Wrap any cartesian vector to the inside of the given cuboid.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_microstate::{
    ///     boundary::{Periodic, Wrap},
    ///     property::Point,
    /// };
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let periodic =
    ///     Periodic::new(2.5, Rectangle::with_equal_edges(10.0.try_into()?))?;
    /// let point = Point::new(Cartesian::from([6.0, -15.0]));
    ///
    /// let wrapped_point = periodic.wrap(point)?;
    /// assert_eq!(wrapped_point.position, [-4.0, -5.0].into());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn wrap(&self, properties: P) -> Result<P, Error> {
        let mut properties = properties;
        let r = properties.position_mut();

        for (coordinate, edge_length) in r.coordinates.iter_mut().zip(self.shape.edge_lengths) {
            let edge_length = edge_length.get();
            let lambda = *coordinate / edge_length;
            let lambda = lambda - lambda.round();
            let lambda = if lambda == 0.5 { -0.5 } else { lambda };
            *coordinate = lambda * edge_length;
        }
        Ok(properties)
    }
}

impl<S> GenerateGhosts<S> for Periodic<Hypercuboid<2>>
where
    S: Position<Position = Cartesian<2>> + Copy + Default,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }

    /// Place periodic images of sites near the edge of the periodic boundary.
    ///
    /// For 2D cuboids, `generate_ghosts` places ghosts near the 4 edges and 4
    /// vertices.
    #[inline]
    fn generate_ghosts(&self, site_properties: &S) -> ArrayVec<S, MAX_GHOSTS> {
        let mut result = ArrayVec::new();

        let r = site_properties.position();
        let max = self.shape.maximal_extents();
        let min = self.shape.minimal_extents();

        if !self.shape.is_point_inside(r) {
            return result;
        }

        let new_site = |x, y| {
            let mut new_site = *site_properties;
            new_site.position_mut()[0] += x * self.shape.edge_lengths[0].get();
            new_site.position_mut()[1] += y * self.shape.edge_lengths[1].get();
            new_site
        };

        let near_left = r[0] < min[0] + self.maximum_interaction_range;
        let near_right = r[0] > max[0] - self.maximum_interaction_range;
        let near_top = r[1] > max[1] - self.maximum_interaction_range;
        let near_bottom = r[1] < min[1] + self.maximum_interaction_range;

        if near_right {
            result.push(new_site(-1.0, 0.0));
        }
        if near_left {
            result.push(new_site(1.0, 0.0));
        }
        if near_top {
            result.push(new_site(0.0, -1.0));
        }
        if near_bottom {
            result.push(new_site(0.0, 1.0));
        }
        if near_right && near_top {
            result.push(new_site(-1.0, -1.0));
        }
        if near_right && near_bottom {
            result.push(new_site(-1.0, 1.0));
        }
        if near_left && near_top {
            result.push(new_site(1.0, -1.0));
        }
        if near_left && near_bottom {
            result.push(new_site(1.0, 1.0));
        }

        result
    }
}

impl<S> GenerateGhosts<S> for Periodic<Hypercuboid<3>>
where
    S: Position<Position = Cartesian<3>> + Copy + Default,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }

    /// Place periodic images of sites near the edge of the periodic boundary.
    ///
    /// For 3D cuboids, `generate_ghosts` places ghosts near the 6 faces, 12 edges,
    /// and 8 vertices.
    #[inline]
    fn generate_ghosts(&self, site_properties: &S) -> ArrayVec<S, MAX_GHOSTS> {
        let mut result = ArrayVec::new();

        let r = site_properties.position();
        let max = self.shape.maximal_extents();
        let min = self.shape.minimal_extents();

        if !self.shape.is_point_inside(r) {
            return result;
        }

        let new_site = |x, y, z| {
            let mut new_site = *site_properties;
            new_site.position_mut()[0] += x * self.shape.edge_lengths[0].get();
            new_site.position_mut()[1] += y * self.shape.edge_lengths[1].get();
            new_site.position_mut()[2] += z * self.shape.edge_lengths[2].get();
            new_site
        };

        let near_left = r[0] < min[0] + self.maximum_interaction_range;
        let near_right = r[0] > max[0] - self.maximum_interaction_range;
        let near_top = r[1] > max[1] - self.maximum_interaction_range;
        let near_bottom = r[1] < min[1] + self.maximum_interaction_range;
        let near_front = r[2] > max[2] - self.maximum_interaction_range;
        let near_back = r[2] < min[2] + self.maximum_interaction_range;

        if near_right {
            result.push(new_site(-1.0, 0.0, 0.0));
        }
        if near_left {
            result.push(new_site(1.0, 0.0, 0.0));
        }
        if near_top {
            result.push(new_site(0.0, -1.0, 0.0));
        }
        if near_bottom {
            result.push(new_site(0.0, 1.0, 0.0));
        }
        if near_front {
            result.push(new_site(0.0, 0.0, -1.0));
        }
        if near_back {
            result.push(new_site(0.0, 0.0, 1.0));
        }

        if near_right && near_top {
            result.push(new_site(-1.0, -1.0, 0.0));
        }
        if near_right && near_bottom {
            result.push(new_site(-1.0, 1.0, 0.0));
        }
        if near_right && near_front {
            result.push(new_site(-1.0, 0.0, -1.0));
        }
        if near_right && near_back {
            result.push(new_site(-1.0, 0.0, 1.0));
        }
        if near_left && near_top {
            result.push(new_site(1.0, -1.0, 0.0));
        }
        if near_left && near_bottom {
            result.push(new_site(1.0, 1.0, 0.0));
        }
        if near_left && near_front {
            result.push(new_site(1.0, 0.0, -1.0));
        }
        if near_left && near_back {
            result.push(new_site(1.0, 0.0, 1.0));
        }

        if near_top && near_front {
            result.push(new_site(0.0, -1.0, -1.0));
        }
        if near_bottom && near_front {
            result.push(new_site(0.0, 1.0, -1.0));
        }
        if near_top && near_back {
            result.push(new_site(0.0, -1.0, 1.0));
        }
        if near_bottom && near_back {
            result.push(new_site(0.0, 1.0, 1.0));
        }

        if near_right && near_top && near_front {
            result.push(new_site(-1.0, -1.0, -1.0));
        }
        if near_right && near_top && near_back {
            result.push(new_site(-1.0, -1.0, 1.0));
        }
        if near_right && near_bottom && near_front {
            result.push(new_site(-1.0, 1.0, -1.0));
        }
        if near_right && near_bottom && near_back {
            result.push(new_site(-1.0, 1.0, 1.0));
        }
        if near_left && near_top && near_front {
            result.push(new_site(1.0, -1.0, -1.0));
        }
        if near_left && near_top && near_back {
            result.push(new_site(1.0, -1.0, 1.0));
        }
        if near_left && near_bottom && near_front {
            result.push(new_site(1.0, 1.0, -1.0));
        }
        if near_left && near_bottom && near_back {
            result.push(new_site(1.0, 1.0, 1.0));
        }

        result
    }
}

impl<S> GenerateGhosts<S> for Periodic<Hypercuboid<4>>
where
    S: Position<Position = Cartesian<4>> + Copy + Default,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }

    /// Place periodic images of sites near the edge of the periodic boundary.
    ///
    /// For 4D cuboids, `generate_ghosts` places ghosts near the 8 cells (3D faces),
    /// 24 faces (2D), 32 edges (1D), and 16 vertices (0D).
    #[inline]
    fn generate_ghosts(&self, site_properties: &S) -> ArrayVec<S, MAX_GHOSTS> {
        let mut result = ArrayVec::new();

        let r = site_properties.position();
        let max = self.shape.maximal_extents();
        let min = self.shape.minimal_extents();

        if !self.shape.is_point_inside(r) {
            return result;
        }

        let mut near_mask = 0u32;
        let mut dim_offset = [0.0_f64; 4];
        for i in 0..4 {
            if r[i] > max[i] - self.maximum_interaction_range {
                near_mask |= 1 << i;
                dim_offset[i] = -self.shape.edge_lengths[i].get();
            } else if r[i] < min[i] + self.maximum_interaction_range {
                near_mask |= 1 << i;
                dim_offset[i] = self.shape.edge_lengths[i].get();
            }
        }

        let mut subset = near_mask;
        while subset != 0 {
            let mut ghost = *site_properties;
            let pos = ghost.position_mut();
            for i in 0..4 {
                if subset & (1 << i) != 0 {
                    pos[i] += dim_offset[i];
                }
            }
            result.push(ghost);
            subset = (subset - 1) & near_mask;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::Point;

    use approxim::assert_relative_eq;
    use rand::{SeedableRng, distr::Distribution, rngs::StdRng};

    const N_SAMPLES: usize = 1024;

    mod cuboid_2 {
        use super::*;

        #[test]
        fn maximum_allowable() {
            let cuboid = Hypercuboid {
                edge_lengths: [
                    10.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    6.0.try_into()
                        .expect("hard-coded constant should be positive"),
                ],
            };

            assert_eq!(cuboid.maximum_allowable_interaction_range(), 3.0);

            let cuboid = Hypercuboid {
                edge_lengths: [
                    4.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    6.0.try_into()
                        .expect("hard-coded constant should be positive"),
                ],
            };

            assert_eq!(cuboid.maximum_allowable_interaction_range(), 2.0);

            let cuboid = Hypercuboid {
                edge_lengths: [
                    100.0
                        .try_into()
                        .expect("hard-coded constant should be positive"),
                    18.0.try_into()
                        .expect("hard-coded constant should be positive"),
                ],
            };

            assert_eq!(cuboid.maximum_allowable_interaction_range(), 9.0);
        }

        #[test]
        fn wrap() {
            let cuboid = Hypercuboid {
                edge_lengths: [
                    20.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    20.0.try_into()
                        .expect("hard-coded constant should be positive"),
                ],
            };

            let periodic = Periodic::new(0.0, cuboid).expect("hard-coded range should be valid");

            let point = Point::new([5.0, 3.0].into());
            assert_eq!(periodic.wrap(point), Ok(point));

            let point = Point::new([-10.0, -10.0].into());
            assert_eq!(periodic.wrap(point), Ok(point));

            let point = Point::new([10.0_f64.next_down(), 10.0_f64.next_down()].into());
            assert_eq!(periodic.wrap(point), Ok(point));

            let point = Point::new([10.0, 10.0].into());
            assert_eq!(periodic.wrap(point), Ok(Point::new([-10.0, -10.0].into())));

            let point = Point::new([20.0, 20.0].into());
            assert_eq!(periodic.wrap(point), Ok(Point::new([0.0, 0.0].into())));

            let point = Point::new([30.0, 30.0].into());
            assert_eq!(periodic.wrap(point), Ok(Point::new([-10.0, -10.0].into())));

            let point = Point::new([25.0, -35.0].into());
            assert_eq!(periodic.wrap(point), Ok(Point::new([5.0, 5.0].into())));
        }

        #[test]
        fn no_ghosts() {
            let cuboid = Hypercuboid {
                edge_lengths: [
                    20.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    10.0.try_into()
                        .expect("hard-coded constant should be positive"),
                ],
            };

            let periodic = Periodic::new(1.0, cuboid).expect("hard-coded range should be valid");

            let inner = Hypercuboid {
                edge_lengths: [
                    18.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    8.0.try_into()
                        .expect("hard-coded constant should be positive"),
                ],
            };

            let mut rng = StdRng::seed_from_u64(1);
            for _ in 0..N_SAMPLES {
                let point = inner.sample(&mut rng);
                let ghosts = periodic.generate_ghosts(&Point::new(point));
                assert!(ghosts.is_empty());
            }
        }

        #[test]
        fn ghosts() {
            let cuboid = Hypercuboid {
                edge_lengths: [
                    20.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    10.0.try_into()
                        .expect("hard-coded constant should be positive"),
                ],
            };

            let periodic = Periodic::new(1.0, cuboid).expect("hard-coded range should be valid");

            // no ghosts for points outside the boundary
            let ghosts = periodic.generate_ghosts(&Point::new([10.5, 0.0].into()));
            assert!(ghosts.is_empty());
            let ghosts = periodic.generate_ghosts(&Point::new([0.0, 5.5].into()));
            assert!(ghosts.is_empty());

            // edges
            let ghosts = periodic.generate_ghosts(&Point::new([9.5, 0.0].into()));
            assert_eq!(ghosts.len(), 1);
            assert_relative_eq!(ghosts[0].position, [-10.5, 0.0].into());

            let ghosts = periodic.generate_ghosts(&Point::new([-9.5, 0.0].into()));
            assert_eq!(ghosts.len(), 1);
            assert_relative_eq!(ghosts[0].position, [10.5, 0.0].into());

            let ghosts = periodic.generate_ghosts(&Point::new([0.0, 4.5].into()));
            assert_eq!(ghosts.len(), 1);
            assert_relative_eq!(ghosts[0].position, [0.0, -5.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([0.0, -4.5].into()));
            assert_eq!(ghosts.len(), 1);
            assert_relative_eq!(ghosts[0].position, [0.0, 5.5].into());

            // vertices
            let ghosts = periodic.generate_ghosts(&Point::new([9.5, 4.5].into()));
            assert_eq!(ghosts.len(), 3);
            assert_relative_eq!(ghosts[0].position, [-10.5, 4.5].into());
            assert_relative_eq!(ghosts[1].position, [9.5, -5.5].into());
            assert_relative_eq!(ghosts[2].position, [-10.5, -5.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([9.5, -4.5].into()));
            assert_eq!(ghosts.len(), 3);
            assert_relative_eq!(ghosts[0].position, [-10.5, -4.5].into());
            assert_relative_eq!(ghosts[1].position, [9.5, 5.5].into());
            assert_relative_eq!(ghosts[2].position, [-10.5, 5.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([-9.5, 4.5].into()));
            assert_eq!(ghosts.len(), 3);
            assert_relative_eq!(ghosts[0].position, [10.5, 4.5].into());
            assert_relative_eq!(ghosts[1].position, [-9.5, -5.5].into());
            assert_relative_eq!(ghosts[2].position, [10.5, -5.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([-9.5, -4.5].into()));
            assert_eq!(ghosts.len(), 3);
            assert_relative_eq!(ghosts[0].position, [10.5, -4.5].into());
            assert_relative_eq!(ghosts[1].position, [-9.5, 5.5].into());
            assert_relative_eq!(ghosts[2].position, [10.5, 5.5].into());
        }
    }

    mod cuboid_3 {
        use super::*;

        #[test]
        fn maximum_allowable() {
            let cuboid = Hypercuboid {
                edge_lengths: [
                    10.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    6.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    4.0.try_into()
                        .expect("hard-coded constant should be positive"),
                ],
            };

            assert_eq!(cuboid.maximum_allowable_interaction_range(), 2.0);

            let cuboid = Hypercuboid {
                edge_lengths: [
                    6.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    8.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    10.0.try_into()
                        .expect("hard-coded constant should be positive"),
                ],
            };

            assert_eq!(cuboid.maximum_allowable_interaction_range(), 3.0);

            let cuboid = Hypercuboid {
                edge_lengths: [
                    18.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    8.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    10.0.try_into()
                        .expect("hard-coded constant should be positive"),
                ],
            };

            assert_eq!(cuboid.maximum_allowable_interaction_range(), 4.0);
        }

        #[test]
        fn wrap() {
            let cuboid = Hypercuboid {
                edge_lengths: [
                    20.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    20.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    20.0.try_into()
                        .expect("hard-coded constant should be positive"),
                ],
            };

            let periodic = Periodic::new(0.0, cuboid).expect("hard-coded range should be valid");

            let point = Point::new([5.0, 3.0, 8.0].into());
            assert_eq!(periodic.wrap(point), Ok(point));

            let point = Point::new([-10.0, -10.0, -10.0].into());
            assert_eq!(periodic.wrap(point), Ok(point));

            let point = Point::new(
                [
                    10.0_f64.next_down(),
                    10.0_f64.next_down(),
                    10.0_f64.next_down(),
                ]
                .into(),
            );
            assert_eq!(periodic.wrap(point), Ok(point));

            let point = Point::new([10.0, 10.0, 10.0].into());
            assert_eq!(
                periodic.wrap(point),
                Ok(Point::new([-10.0, -10.0, -10.0].into()))
            );

            let point = Point::new([20.0, 20.0, 20.0].into());
            assert_eq!(periodic.wrap(point), Ok(Point::new([0.0, 0.0, 0.0].into())));

            let point = Point::new([30.0, 30.0, 30.0].into());
            assert_eq!(
                periodic.wrap(point),
                Ok(Point::new([-10.0, -10.0, -10.0].into()))
            );

            let point = Point::new([25.0, -35.0, 55.0].into());
            assert_eq!(
                periodic.wrap(point),
                Ok(Point::new([5.0, 5.0, -5.0].into()))
            );
        }

        #[test]
        fn no_ghosts() {
            let cuboid = Hypercuboid {
                edge_lengths: [
                    40.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    20.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    10.0.try_into()
                        .expect("hard-coded constant should be positive"),
                ],
            };

            let periodic = Periodic::new(1.0, cuboid).expect("hard-coded range should be valid");

            let inner = Hypercuboid {
                edge_lengths: [
                    38.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    18.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    8.0.try_into()
                        .expect("hard-coded constant should be positive"),
                ],
            };

            let mut rng = StdRng::seed_from_u64(1);
            for _ in 0..N_SAMPLES {
                let point = inner.sample(&mut rng);
                let ghosts = periodic.generate_ghosts(&Point::new(point));
                assert!(ghosts.is_empty());
            }
        }

        #[expect(clippy::too_many_lines, reason = "There are many cases to test.")]
        #[test]
        fn ghosts() {
            let cuboid = Hypercuboid {
                edge_lengths: [
                    20.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    10.0.try_into()
                        .expect("hard-coded constant should be positive"),
                    40.0.try_into()
                        .expect("hard-coded constant should be positive"),
                ],
            };

            let periodic = Periodic::new(1.0, cuboid).expect("hard-coded range should be valid");

            // no ghosts for points outside the boundary
            let ghosts = periodic.generate_ghosts(&Point::new([10.5, 0.0, 0.0].into()));
            assert!(ghosts.is_empty());
            let ghosts = periodic.generate_ghosts(&Point::new([0.0, 5.5, 0.0].into()));
            assert!(ghosts.is_empty());

            // faces
            let ghosts = periodic.generate_ghosts(&Point::new([9.5, 0.0, 0.0].into()));
            assert_eq!(ghosts.len(), 1);
            assert_relative_eq!(ghosts[0].position, [-10.5, 0.0, 0.0].into());

            let ghosts = periodic.generate_ghosts(&Point::new([-9.5, 0.0, 0.0].into()));
            assert_eq!(ghosts.len(), 1);
            assert_relative_eq!(ghosts[0].position, [10.5, 0.0, 0.0].into());

            let ghosts = periodic.generate_ghosts(&Point::new([0.0, 4.5, 0.0].into()));
            assert_eq!(ghosts.len(), 1);
            assert_relative_eq!(ghosts[0].position, [0.0, -5.5, 0.0].into());

            let ghosts = periodic.generate_ghosts(&Point::new([0.0, -4.5, 0.0].into()));
            assert_eq!(ghosts.len(), 1);
            assert_relative_eq!(ghosts[0].position, [0.0, 5.5, 0.0].into());

            let ghosts = periodic.generate_ghosts(&Point::new([0.0, 0.0, 19.5].into()));
            assert_eq!(ghosts.len(), 1);
            assert_relative_eq!(ghosts[0].position, [0.0, 0.0, -20.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([0.0, 0.0, -19.5].into()));
            assert_eq!(ghosts.len(), 1);
            assert_relative_eq!(ghosts[0].position, [0.0, 0.0, 20.5].into());

            // edges
            let ghosts = periodic.generate_ghosts(&Point::new([9.5, 4.5, 0.0].into()));
            assert_eq!(ghosts.len(), 3);
            assert_relative_eq!(ghosts[0].position, [-10.5, 4.5, 0.0].into());
            assert_relative_eq!(ghosts[1].position, [9.5, -5.5, 0.0].into());
            assert_relative_eq!(ghosts[2].position, [-10.5, -5.5, 0.0].into());

            let ghosts = periodic.generate_ghosts(&Point::new([9.5, -4.5, 0.0].into()));
            assert_eq!(ghosts.len(), 3);
            assert_relative_eq!(ghosts[0].position, [-10.5, -4.5, 0.0].into());
            assert_relative_eq!(ghosts[1].position, [9.5, 5.5, 0.0].into());
            assert_relative_eq!(ghosts[2].position, [-10.5, 5.5, 0.0].into());

            let ghosts = periodic.generate_ghosts(&Point::new([-9.5, 4.5, 0.0].into()));
            assert_eq!(ghosts.len(), 3);
            assert_relative_eq!(ghosts[0].position, [10.5, 4.5, 0.0].into());
            assert_relative_eq!(ghosts[1].position, [-9.5, -5.5, 0.0].into());
            assert_relative_eq!(ghosts[2].position, [10.5, -5.5, 0.0].into());

            let ghosts = periodic.generate_ghosts(&Point::new([-9.5, -4.5, 0.0].into()));
            assert_eq!(ghosts.len(), 3);
            assert_relative_eq!(ghosts[0].position, [10.5, -4.5, 0.0].into());
            assert_relative_eq!(ghosts[1].position, [-9.5, 5.5, 0.0].into());
            assert_relative_eq!(ghosts[2].position, [10.5, 5.5, 0.0].into());

            let ghosts = periodic.generate_ghosts(&Point::new([9.5, 0.0, 19.5].into()));
            assert_eq!(ghosts.len(), 3);
            assert_relative_eq!(ghosts[0].position, [-10.5, 0.0, 19.5].into());
            assert_relative_eq!(ghosts[1].position, [9.5, 0.0, -20.5].into());
            assert_relative_eq!(ghosts[2].position, [-10.5, 0.0, -20.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([-9.5, 0.0, 19.5].into()));
            assert_eq!(ghosts.len(), 3);
            assert_relative_eq!(ghosts[0].position, [10.5, 0.0, 19.5].into());
            assert_relative_eq!(ghosts[1].position, [-9.5, 0.0, -20.5].into());
            assert_relative_eq!(ghosts[2].position, [10.5, 0.0, -20.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([0.0, 4.5, 19.5].into()));
            assert_eq!(ghosts.len(), 3);
            assert_relative_eq!(ghosts[0].position, [0.0, -5.5, 19.5].into());
            assert_relative_eq!(ghosts[1].position, [0.0, 4.5, -20.5].into());
            assert_relative_eq!(ghosts[2].position, [0.0, -5.5, -20.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([0.0, -4.5, 19.5].into()));
            assert_eq!(ghosts.len(), 3);
            assert_relative_eq!(ghosts[0].position, [0.0, 5.5, 19.5].into());
            assert_relative_eq!(ghosts[1].position, [0.0, -4.5, -20.5].into());
            assert_relative_eq!(ghosts[2].position, [0.0, 5.5, -20.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([9.5, 0.0, -19.5].into()));
            assert_eq!(ghosts.len(), 3);
            assert_relative_eq!(ghosts[0].position, [-10.5, 0.0, -19.5].into());
            assert_relative_eq!(ghosts[1].position, [9.5, 0.0, 20.5].into());
            assert_relative_eq!(ghosts[2].position, [-10.5, 0.0, 20.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([-9.5, 0.0, -19.5].into()));
            assert_eq!(ghosts.len(), 3);
            assert_relative_eq!(ghosts[0].position, [10.5, 0.0, -19.5].into());
            assert_relative_eq!(ghosts[1].position, [-9.5, 0.0, 20.5].into());
            assert_relative_eq!(ghosts[2].position, [10.5, 0.0, 20.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([0.0, 4.5, -19.5].into()));
            assert_eq!(ghosts.len(), 3);
            assert_relative_eq!(ghosts[0].position, [0.0, -5.5, -19.5].into());
            assert_relative_eq!(ghosts[1].position, [0.0, 4.5, 20.5].into());
            assert_relative_eq!(ghosts[2].position, [0.0, -5.5, 20.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([0.0, -4.5, -19.5].into()));
            assert_eq!(ghosts.len(), 3);
            assert_relative_eq!(ghosts[0].position, [0.0, 5.5, -19.5].into());
            assert_relative_eq!(ghosts[1].position, [0.0, -4.5, 20.5].into());
            assert_relative_eq!(ghosts[2].position, [0.0, 5.5, 20.5].into());

            // vertices
            let ghosts = periodic.generate_ghosts(&Point::new([9.5, 4.5, 19.5].into()));
            assert_eq!(ghosts.len(), 7);
            assert_relative_eq!(ghosts[0].position, [-10.5, 4.5, 19.5].into());
            assert_relative_eq!(ghosts[1].position, [9.5, -5.5, 19.5].into());
            assert_relative_eq!(ghosts[2].position, [9.5, 4.5, -20.5].into());
            assert_relative_eq!(ghosts[3].position, [-10.5, -5.5, 19.5].into());
            assert_relative_eq!(ghosts[4].position, [-10.5, 4.5, -20.5].into());
            assert_relative_eq!(ghosts[5].position, [9.5, -5.5, -20.5].into());
            assert_relative_eq!(ghosts[6].position, [-10.5, -5.5, -20.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([9.5, 4.5, -19.5].into()));
            assert_eq!(ghosts.len(), 7);
            assert_relative_eq!(ghosts[0].position, [-10.5, 4.5, -19.5].into());
            assert_relative_eq!(ghosts[1].position, [9.5, -5.5, -19.5].into());
            assert_relative_eq!(ghosts[2].position, [9.5, 4.5, 20.5].into());
            assert_relative_eq!(ghosts[3].position, [-10.5, -5.5, -19.5].into());
            assert_relative_eq!(ghosts[4].position, [-10.5, 4.5, 20.5].into());
            assert_relative_eq!(ghosts[5].position, [9.5, -5.5, 20.5].into());
            assert_relative_eq!(ghosts[6].position, [-10.5, -5.5, 20.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([9.5, -4.5, 19.5].into()));
            assert_eq!(ghosts.len(), 7);
            assert_relative_eq!(ghosts[0].position, [-10.5, -4.5, 19.5].into());
            assert_relative_eq!(ghosts[1].position, [9.5, 5.5, 19.5].into());
            assert_relative_eq!(ghosts[2].position, [9.5, -4.5, -20.5].into());
            assert_relative_eq!(ghosts[3].position, [-10.5, 5.5, 19.5].into());
            assert_relative_eq!(ghosts[4].position, [-10.5, -4.5, -20.5].into());
            assert_relative_eq!(ghosts[5].position, [9.5, 5.5, -20.5].into());
            assert_relative_eq!(ghosts[6].position, [-10.5, 5.5, -20.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([9.5, -4.5, -19.5].into()));
            assert_eq!(ghosts.len(), 7);
            assert_relative_eq!(ghosts[0].position, [-10.5, -4.5, -19.5].into());
            assert_relative_eq!(ghosts[1].position, [9.5, 5.5, -19.5].into());
            assert_relative_eq!(ghosts[2].position, [9.5, -4.5, 20.5].into());
            assert_relative_eq!(ghosts[3].position, [-10.5, 5.5, -19.5].into());
            assert_relative_eq!(ghosts[4].position, [-10.5, -4.5, 20.5].into());
            assert_relative_eq!(ghosts[5].position, [9.5, 5.5, 20.5].into());
            assert_relative_eq!(ghosts[6].position, [-10.5, 5.5, 20.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([-9.5, 4.5, 19.5].into()));
            assert_eq!(ghosts.len(), 7);
            assert_relative_eq!(ghosts[0].position, [10.5, 4.5, 19.5].into());
            assert_relative_eq!(ghosts[1].position, [-9.5, -5.5, 19.5].into());
            assert_relative_eq!(ghosts[2].position, [-9.5, 4.5, -20.5].into());
            assert_relative_eq!(ghosts[3].position, [10.5, -5.5, 19.5].into());
            assert_relative_eq!(ghosts[4].position, [10.5, 4.5, -20.5].into());
            assert_relative_eq!(ghosts[5].position, [-9.5, -5.5, -20.5].into());
            assert_relative_eq!(ghosts[6].position, [10.5, -5.5, -20.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([-9.5, 4.5, -19.5].into()));
            assert_eq!(ghosts.len(), 7);
            assert_relative_eq!(ghosts[0].position, [10.5, 4.5, -19.5].into());
            assert_relative_eq!(ghosts[1].position, [-9.5, -5.5, -19.5].into());
            assert_relative_eq!(ghosts[2].position, [-9.5, 4.5, 20.5].into());
            assert_relative_eq!(ghosts[3].position, [10.5, -5.5, -19.5].into());
            assert_relative_eq!(ghosts[4].position, [10.5, 4.5, 20.5].into());
            assert_relative_eq!(ghosts[5].position, [-9.5, -5.5, 20.5].into());
            assert_relative_eq!(ghosts[6].position, [10.5, -5.5, 20.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([-9.5, -4.5, 19.5].into()));
            assert_eq!(ghosts.len(), 7);
            assert_relative_eq!(ghosts[0].position, [10.5, -4.5, 19.5].into());
            assert_relative_eq!(ghosts[1].position, [-9.5, 5.5, 19.5].into());
            assert_relative_eq!(ghosts[2].position, [-9.5, -4.5, -20.5].into());
            assert_relative_eq!(ghosts[3].position, [10.5, 5.5, 19.5].into());
            assert_relative_eq!(ghosts[4].position, [10.5, -4.5, -20.5].into());
            assert_relative_eq!(ghosts[5].position, [-9.5, 5.5, -20.5].into());
            assert_relative_eq!(ghosts[6].position, [10.5, 5.5, -20.5].into());

            let ghosts = periodic.generate_ghosts(&Point::new([-9.5, -4.5, -19.5].into()));
            assert_eq!(ghosts.len(), 7);
            assert_relative_eq!(ghosts[0].position, [10.5, -4.5, -19.5].into());
            assert_relative_eq!(ghosts[1].position, [-9.5, 5.5, -19.5].into());
            assert_relative_eq!(ghosts[2].position, [-9.5, -4.5, 20.5].into());
            assert_relative_eq!(ghosts[3].position, [10.5, 5.5, -19.5].into());
            assert_relative_eq!(ghosts[4].position, [10.5, -4.5, 20.5].into());
            assert_relative_eq!(ghosts[5].position, [-9.5, 5.5, 20.5].into());
            assert_relative_eq!(ghosts[6].position, [10.5, 5.5, 20.5].into());
        }
    }

    mod cuboid_4 {
        use super::*;
        use rstest::rstest;

        fn pos(value: f64) -> PositiveReal {
            value
                .try_into()
                .expect("hard-coded constant should be positive")
        }

        #[rstest]
        #[case::min_edge_is_last(
            [pos(10.0), pos(6.0), pos(4.0), pos(2.0)],
            1.0,
        )]
        #[case::min_edge_is_third(
            [pos(20.0), pos(10.0), pos(6.0), pos(8.0)],
            3.0,
        )]
        fn maximum_allowable(#[case] edge_lengths: [PositiveReal; 4], #[case] expected: f64) {
            let cuboid = Hypercuboid { edge_lengths };
            assert_eq!(cuboid.maximum_allowable_interaction_range(), expected);
        }

        #[rstest]
        #[case::interior(
            [5.0, 3.0, 8.0, -1.0],
            [5.0, 3.0, 8.0, -1.0],
        )]
        #[case::lower_bound(
            [-10.0, -10.0, -10.0, -10.0],
            [-10.0, -10.0, -10.0, -10.0],
        )]
        #[case::just_below_upper(
            [10.0_f64.next_down(), 10.0_f64.next_down(), 10.0_f64.next_down(), 10.0_f64.next_down()],
            [10.0_f64.next_down(), 10.0_f64.next_down(), 10.0_f64.next_down(), 10.0_f64.next_down()],
        )]
        #[case::at_upper(
            [10.0, 10.0, 10.0, 10.0],
            [-10.0, -10.0, -10.0, -10.0],
        )]
        #[case::wrap_around(
            [25.0, -35.0, 55.0, -15.0],
            [5.0, 5.0, -5.0, 5.0],
        )]
        fn wrap(#[case] input: [f64; 4], #[case] expected: [f64; 4]) {
            let cuboid = Hypercuboid {
                edge_lengths: [pos(20.0); 4],
            };
            let periodic = Periodic::new(0.0, cuboid).expect("hard-coded range should be valid");
            assert_eq!(
                periodic.wrap(Point::new(input.into())),
                Ok(Point::new(expected.into())),
            );
        }

        #[test]
        fn no_ghosts() {
            let cuboid = Hypercuboid {
                edge_lengths: [pos(40.0), pos(20.0), pos(10.0), pos(8.0)],
            };
            let periodic = Periodic::new(1.0, cuboid).expect("hard-coded range should be valid");

            let inner = Hypercuboid {
                edge_lengths: [pos(38.0), pos(18.0), pos(8.0), pos(6.0)],
            };

            let mut rng = StdRng::seed_from_u64(1);
            for _ in 0..N_SAMPLES {
                let point = inner.sample(&mut rng);
                let ghosts = periodic.generate_ghosts(&Point::new(point));
                assert!(ghosts.is_empty());
            }
        }

        /// lx=20, ly=10, lz=40, lw=8 with interaction range 1.0.
        fn unequal_box() -> Periodic<Hypercuboid<4>> {
            let cuboid = Hypercuboid {
                edge_lengths: [pos(20.0), pos(10.0), pos(40.0), pos(8.0)],
            };
            Periodic::new(1.0, cuboid).expect("hard-coded range should be valid")
        }

        #[rstest]
        #[case::outside_x([10.5, 0.0, 0.0, 0.0], 0)]
        #[case::outside_w([0.0, 0.0, 0.0, 4.5], 0)]
        #[case::face_pos_x([9.5, 0.0, 0.0, 0.0], 1)]
        #[case::face_neg_x([-9.5, 0.0, 0.0, 0.0], 1)]
        #[case::face_pos_y([0.0, 4.5, 0.0, 0.0], 1)]
        #[case::face_pos_w([0.0, 0.0, 0.0, 3.5], 1)]
        #[case::face_neg_w([0.0, 0.0, 0.0, -3.5], 1)]
        #[case::edge_pos_x_pos_y([9.5, 4.5, 0.0, 0.0], 3)]
        #[case::edge_pos_x_neg_y_pos_z([9.5, -4.5, 19.5, 0.0], 7)]
        fn ghosts(#[case] input: [f64; 4], #[case] expected_count: usize) {
            let periodic = unequal_box();
            let ghosts = periodic.generate_ghosts(&Point::new(input.into()));
            assert_eq!(ghosts.len(), expected_count);
        }

        #[test]
        fn ghosts_corner_all_pos() {
            if MAX_GHOSTS < 15 {
                return;
            }
            let periodic = unequal_box();
            let ghosts = periodic.generate_ghosts(&Point::new([9.5, 4.5, 19.5, 3.5].into()));
            assert_eq!(ghosts.len(), 15);
        }

        #[rstest]
        #[case::face_pos_x(
            [9.5, 0.0, 0.0, 0.0],
            [[-10.5, 0.0, 0.0, 0.0]].to_vec(),
        )]
        #[case::face_neg_x(
            [-9.5, 0.0, 0.0, 0.0],
            [[10.5, 0.0, 0.0, 0.0]].to_vec(),
        )]
        #[case::face_pos_y(
            [0.0, 4.5, 0.0, 0.0],
            [[0.0, -5.5, 0.0, 0.0]].to_vec(),
        )]
        #[case::face_pos_w(
            [0.0, 0.0, 0.0, 3.5],
            [[0.0, 0.0, 0.0, -4.5]].to_vec(),
        )]
        #[case::face_neg_w(
            [0.0, 0.0, 0.0, -3.5],
            [[0.0, 0.0, 0.0, 4.5]].to_vec(),
        )]
        #[case::edge_pos_x_pos_y(
            [9.5, 4.5, 0.0, 0.0],
            [[-10.5, -5.5, 0.0, 0.0], [9.5, -5.5, 0.0, 0.0], [-10.5, 4.5, 0.0, 0.0]].to_vec(),
        )]
        #[case::edge_pos_x_neg_y_pos_z(
            [9.5, -4.5, 19.5, 0.0],
            [
                [-10.5, 5.5, -20.5, 0.0],
                [9.5, 5.5, -20.5, 0.0],
                [-10.5, -4.5, -20.5, 0.0],
                [9.5, -4.5, -20.5, 0.0],
                [-10.5, 5.5, 19.5, 0.0],
                [9.5, 5.5, 19.5, 0.0],
                [-10.5, -4.5, 19.5, 0.0],
            ].to_vec(),
        )]
        fn ghost_positions(#[case] input: [f64; 4], #[case] expected: Vec<[f64; 4]>) {
            let periodic = unequal_box();
            let ghosts = periodic.generate_ghosts(&Point::new(input.into()));
            assert_eq!(ghosts.len(), expected.len());
            for (ghost, expected_pos) in ghosts.iter().zip(expected.iter()) {
                assert_relative_eq!(ghost.position, (*expected_pos).into());
            }
        }
    }
}

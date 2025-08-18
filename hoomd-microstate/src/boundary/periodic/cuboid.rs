// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement periodic boundary conditions for cuboids in cartesian space. */

use tinyvec::ArrayVec;

use crate::{
    boundary::{
        Error, GenerateGhosts, MAX_GHOSTS, MaximumAllowableInteractionRange, Periodic, Wrap,
    },
    property::Position,
};
use hoomd_geometry::{IsPointInside, shape::Cuboid};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Cartesian;

impl<const N: usize> MaximumAllowableInteractionRange for Cuboid<N> {
    /** The largest value that the maximum interaction range can take.

    For a cuboid, the maximum is
    ```math
    \frac{L_\mathrm{min}}{2}
    ```
    where $`L_\mathrm{min}`$ is the smallest edge length.

    # Example

    ```
    use hoomd_geometry::shape::Cuboid;
    use hoomd_microstate::boundary::MaximumAllowableInteractionRange;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rectangular_prism = Cuboid {edge_lengths: [2.0.try_into()?, 3.0.try_into()?, 9.0.try_into()?]};

    assert_eq!(rectangular_prism.maximum_allowable_interaction_range(), 1.0);
    # Ok(())
    # }
    ```
    */
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

impl<const N: usize, P> Wrap<P> for Periodic<Cuboid<N>>
where
    P: Position<Vector = Cartesian<N>>,
{
    /** Wrap any cartesian vector to the inside of the given cuboid.

    # Example

    ```
    use hoomd_geometry::shape::Rectangle;
    use hoomd_microstate::{boundary::{Periodic, Wrap}, property::Point};
    use hoomd_vector::Cartesian;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let periodic = Periodic::new(2.5, Rectangle::with_equal_edges(10.0.try_into()?))?;
    let point = Point::new(Cartesian::from([6.0, -15.0]));

    let wrapped_point = periodic.wrap(point)?;
    assert_eq!(wrapped_point.position, [-4.0, -5.0].into());
    # Ok(())
    # }
    ```
    */
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

impl<S> GenerateGhosts<S> for Periodic<Cuboid<2>>
where
    S: Position<Vector = Cartesian<2>> + Copy + Default,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }

    /** Place periodic images of sites near the edge of the periodic boundary.

    For 2D cuboids, `generate_ghosts` places ghosts near the 4 edges and 4
    vertices.
    */
    #[inline]
    fn generate_ghosts(&self, site_properties: &S) -> ArrayVec<[S; MAX_GHOSTS]> {
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

impl<S> GenerateGhosts<S> for Periodic<Cuboid<3>>
where
    S: Position<Vector = Cartesian<3>> + Copy + Default,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }

    /** Place periodic images of sites near the edge of the periodic boundary.

    For 3D cuboids, `generate_ghosts` places ghosts near the 6 faces, 12 edges,
    and 8 vertices.
    */
    #[inline]
    fn generate_ghosts(&self, site_properties: &S) -> ArrayVec<[S; MAX_GHOSTS]> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::Point;

    use approx::assert_relative_eq;
    use rand::{SeedableRng, rngs::StdRng, distr::Distribution};

    const N_SAMPLES: usize = 1024;

    mod cuboid_2 {
        use super::*;

        #[test]
        fn maximum_allowable() {
            let cuboid = Cuboid { edge_lengths: [
                10.0.try_into().expect("hard-coded constant should be positive"),
                6.0.try_into().expect("hard-coded constant should be positive")], };

            assert_eq!(cuboid.maximum_allowable_interaction_range(), 3.0);

            let cuboid = Cuboid { edge_lengths: [
                4.0.try_into().expect("hard-coded constant should be positive"),
                6.0.try_into().expect("hard-coded constant should be positive")], };

            assert_eq!(cuboid.maximum_allowable_interaction_range(), 2.0);

            let cuboid = Cuboid { edge_lengths: [
                100.0.try_into().expect("hard-coded constant should be positive"),
                18.0.try_into().expect("hard-coded constant should be positive")], };

            assert_eq!(cuboid.maximum_allowable_interaction_range(), 9.0);
        }

        #[test]
        fn wrap() {
            let cuboid = Cuboid { edge_lengths: [
                20.0.try_into().expect("hard-coded constant should be positive"),
                20.0.try_into().expect("hard-coded constant should be positive")], };

            let periodic = Periodic::new(0.0, cuboid).expect("hard-coded range should be valid");

            let point = Point::new([5.0, 3.0].into());
            assert_eq!(periodic.wrap(point), Ok(point));

            let point = Point::new([-10.0, -10.0].into());
            assert_eq!(periodic.wrap(point), Ok(point));

            let point = Point::new([10.0f64.next_down(), 10.0f64.next_down()].into());
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
            let cuboid = Cuboid { edge_lengths: [
                20.0.try_into().expect("hard-coded constant should be positive"),
                10.0.try_into().expect("hard-coded constant should be positive")], };

            let periodic = Periodic::new(1.0, cuboid).expect("hard-coded range should be valid");

            let inner = Cuboid { edge_lengths: [
                18.0.try_into().expect("hard-coded constant should be positive"),
                8.0.try_into().expect("hard-coded constant should be positive")], };

            let mut rng = StdRng::seed_from_u64(1);
            for _ in 0..N_SAMPLES {
                let point = inner.sample(&mut rng);
                let ghosts = periodic.generate_ghosts(&Point::new(point));
                assert!(ghosts.is_empty());
            }
        }

        #[test]
        fn ghosts() {
            let cuboid = Cuboid { edge_lengths: [
                20.0.try_into().expect("hard-coded constant should be positive"),
                10.0.try_into().expect("hard-coded constant should be positive")], };

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
            let cuboid = Cuboid { edge_lengths: [
                10.0.try_into().expect("hard-coded constant should be positive"),
                6.0.try_into().expect("hard-coded constant should be positive"),
                4.0.try_into().expect("hard-coded constant should be positive")], };

            assert_eq!(cuboid.maximum_allowable_interaction_range(), 2.0);

            let cuboid = Cuboid { edge_lengths: [
                6.0.try_into().expect("hard-coded constant should be positive"),
                8.0.try_into().expect("hard-coded constant should be positive"),
                10.0.try_into().expect("hard-coded constant should be positive"),
                ], };

            assert_eq!(cuboid.maximum_allowable_interaction_range(), 3.0);

            let cuboid = Cuboid { edge_lengths: [
                18.0.try_into().expect("hard-coded constant should be positive"),
                8.0.try_into().expect("hard-coded constant should be positive"),
                10.0.try_into().expect("hard-coded constant should be positive"),
                ], };

            assert_eq!(cuboid.maximum_allowable_interaction_range(), 4.0);
        }

        #[test]
        fn wrap() {
            let cuboid = Cuboid { edge_lengths: [
                20.0.try_into().expect("hard-coded constant should be positive"),
                20.0.try_into().expect("hard-coded constant should be positive"),
                20.0.try_into().expect("hard-coded constant should be positive"),
                ], };

            let periodic = Periodic::new(0.0, cuboid).expect("hard-coded range should be valid");

            let point = Point::new([5.0, 3.0, 8.0].into());
            assert_eq!(periodic.wrap(point), Ok(point));

            let point = Point::new([-10.0, -10.0, -10.0].into());
            assert_eq!(periodic.wrap(point), Ok(point));

            let point = Point::new([10.0f64.next_down(), 10.0f64.next_down(), 10.0f64.next_down()].into());
            assert_eq!(periodic.wrap(point), Ok(point));

            let point = Point::new([10.0, 10.0, 10.0].into());
            assert_eq!(periodic.wrap(point), Ok(Point::new([-10.0, -10.0, -10.0].into())));

            let point = Point::new([20.0, 20.0, 20.0].into());
            assert_eq!(periodic.wrap(point), Ok(Point::new([0.0, 0.0, 0.0].into())));

            let point = Point::new([30.0, 30.0, 30.0].into());
            assert_eq!(periodic.wrap(point), Ok(Point::new([-10.0, -10.0, -10.0].into())));

            let point = Point::new([25.0, -35.0, 55.0].into());
            assert_eq!(periodic.wrap(point), Ok(Point::new([5.0, 5.0, -5.0].into())));
        }

        #[test]
        fn no_ghosts() {
            let cuboid = Cuboid { edge_lengths: [
                40.0.try_into().expect("hard-coded constant should be positive"),
                20.0.try_into().expect("hard-coded constant should be positive"),
                10.0.try_into().expect("hard-coded constant should be positive")], };

            let periodic = Periodic::new(1.0, cuboid).expect("hard-coded range should be valid");

            let inner = Cuboid { edge_lengths: [
                38.0.try_into().expect("hard-coded constant should be positive"),
                18.0.try_into().expect("hard-coded constant should be positive"),
                8.0.try_into().expect("hard-coded constant should be positive")], };

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
            let cuboid = Cuboid { edge_lengths: [
                20.0.try_into().expect("hard-coded constant should be positive"),
                10.0.try_into().expect("hard-coded constant should be positive"),
                40.0.try_into().expect("hard-coded constant should be positive")], };

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
}

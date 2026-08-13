// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement periodic boundary conditions for triclinic boxes in cartesian space.

use std::array;

use arrayvec::ArrayVec;
use hoomd_spatial::PointUpdate;
use hoomd_utility::valid::PositiveReal;

use crate::{
    Body, Microstate, Replicate, SiteKey, Transform,
    boundary::{
        Error, GenerateGhosts, MAX_GHOSTS, MaximumAllowableInteractionRange, Periodic, Wrap,
    },
    property::Position,
};
use hoomd_geometry::{IsPointInside, shape::Triclinic};

use hoomd_vector::Cartesian;

impl MaximumAllowableInteractionRange for Triclinic {
    /// The largest value that the maximum interaction range can take.
    ///
    /// For a triclinic box, the maximum allowable interaction range is
    /// half the minimum distance to the nearest parallel planes:
    /// ```math
    /// r_\mathrm{max} = \frac{1}{2} \min(d_x, d_y, d_z)
    /// ```
    /// where $`d_i`$ are the distances to the nearest parallel planes
    /// in each direction, accounting for tilt factors.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Triclinic;
    /// use hoomd_microstate::boundary::MaximumAllowableInteractionRange;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic = Triclinic {
    ///     extents: [10.0.try_into()?, 10.0.try_into()?, 10.0.try_into()?],
    ///     tilt_factors: [0.0, 0.0, 0.0],
    /// };
    ///
    /// assert_eq!(triclinic.maximum_allowable_interaction_range(), 5.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn maximum_allowable_interaction_range(&self) -> f64 {
        let plane_distances = self.nearest_plane_distance();
        plane_distances
            .iter()
            .map(|x| x.get() * 0.5)
            .fold(f64::INFINITY, f64::min)
    }
}

impl<P> Wrap<P> for Periodic<Triclinic>
where
    P: Position<Position = Cartesian<3>>,
{
    /// Wrap any cartesian vector to the inside of the triclinic box. Wrapping is
    /// performed by converting to fractional coordinates to determine how many
    /// boxes away the position is, scaling the box displacement vector to real space,
    /// and subtracting the resultant vector from the position. That is,
    /// ```math
    /// \vec{r}'=\vec{r}-\mathbf{A}\text{ROUND}(\mathbf{A}^{-1}\vec{r})
    /// ```
    ///
    /// Points are wrapped using fractional coordinates that account for
    /// the triclinic tilt factors. The wrapping ensures particles remain
    /// within the periodic boundaries.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Triclinic;
    /// use hoomd_microstate::{
    ///     boundary::{Periodic, Wrap},
    ///     property::Point,
    /// };
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic = Triclinic {
    ///     extents: [10.0.try_into()?, 10.0.try_into()?, 10.0.try_into()?],
    ///     tilt_factors: [0.0, 0.0, 0.0],
    /// };
    /// let periodic = Periodic::new(5.0, triclinic)?;
    ///
    /// let point = Point::new(Cartesian::from([6.0, 0.0, 0.0]));
    /// let wrapped = periodic.wrap(point)?;
    /// assert_eq!(wrapped.position, Cartesian::from([-4.0, 0.0, 0.0]));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn wrap(&self, mut properties: P) -> Result<P, Error> {
        let r = properties.position_mut();
        let mut fractional = self.shape.fractional(r);
        for i in 0..3 {
            fractional[i] -= fractional[i].round();
            fractional[i] = if fractional[i] == 0.5 {
                -0.5
            } else {
                fractional[i]
            };
        }
        *r = self.shape.absolute(&fractional);
        Ok(properties)
    }
}

impl<S> GenerateGhosts<S> for Periodic<Triclinic>
where
    S: Position<Position = Cartesian<3>> + Copy + Default,
{
    /// In a triclinic box, we set the maximum interaction to 1/2 the shortest
    /// distance between parallel faces. This choice is made to avoid the possibility of
    /// generating more than one ghost particle along a given axis.
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }

    /// Generate periodic images of sites near the edge of the periodic boundary. Ghosts
    /// are generated based on a `maximum_interaction_range`, the particle position, and
    /// box geometry.
    ///
    /// A ghost needs to be generated if it is closer than the maximum interaction range
    /// to one of the walls of the simulation box. The boundary of this zone (the set of
    /// all points exactly `maximum_interaction_range` away) is a plane parallel to the
    /// wall. In fractional coordinates, this boundary plane will be of the
    /// form $`s_i =\pm 0.5 \mp \frac{r_{\text{max}}}{r_{\perp}}`$, where
    /// $`r_{\text{max}}`$ is the `maximum_interaction_range` and $`r_{\perp}`$ is the
    /// perpendicular distance between the faces of the box. Ghosts are generated by
    /// comparing the fractional coordinate to these planes.
    ///
    /// For triclinic boxes, `generate_ghosts` places ghosts near the 6 faces, 12 edges,
    /// and 8 vertices of the box.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Triclinic;
    /// use hoomd_microstate::{
    ///     boundary::{GenerateGhosts, Periodic},
    ///     property::Point,
    /// };
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic = Triclinic {
    ///     extents: [4.0.try_into()?, 4.0.try_into()?, 4.0.try_into()?],
    ///     tilt_factors: [0.0, 0.0, 0.0],
    /// };
    /// let periodic = Periodic::new(1.0, triclinic)?;
    ///
    /// // Point near the x-face
    /// let point = Point::new(Cartesian::from([1.9, 0.0, 0.0]));
    /// let ghosts = periodic.generate_ghosts(&point);
    ///
    /// // Should generate ghost on the opposite side
    /// assert!(!ghosts.is_empty());
    /// assert_eq!(ghosts[0].position[0], -2.1); // wrapped around
    /// //
    /// # Ok(())
    /// # }
    /// ```
    #[expect(
        clippy::too_many_lines,
        reason = "There are many (literal) corner cases."
    )]
    #[inline]
    fn generate_ghosts(&self, site_properties: &S) -> ArrayVec<S, MAX_GHOSTS> {
        let mut result = ArrayVec::new();
        let r: &Cartesian<3> = site_properties.position();
        if !self.shape.is_point_inside(r) {
            return result;
        }
        let edge_vectors = self.shape.edge_vectors();
        let new_site = |x, y, z| {
            let mut new_site = *site_properties;
            *new_site.position_mut() += x * edge_vectors[0];
            *new_site.position_mut() += y * edge_vectors[1];
            *new_site.position_mut() += z * edge_vectors[2];
            new_site
        };

        let plane_distances = self.shape.nearest_plane_distance();
        let frac = self.shape.fractional(r);

        let near_right = frac[0] > 0.5 - self.maximum_interaction_range / plane_distances[0].get();
        let near_left = frac[0] < -0.5 + self.maximum_interaction_range / plane_distances[0].get();
        let near_top = frac[1] > 0.5 - self.maximum_interaction_range / plane_distances[1].get();
        let near_bottom =
            frac[1] < -0.5 + self.maximum_interaction_range / plane_distances[1].get();
        let near_front = frac[2] > 0.5 - self.maximum_interaction_range / plane_distances[2].get();
        let near_back = frac[2] < -0.5 + self.maximum_interaction_range / plane_distances[2].get();

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

impl<B, S, X> Replicate<3, B, S, X, Periodic<Triclinic>>
    for Microstate<B, S, X, Periodic<Triclinic>>
where
    B: Transform<S> + Position<Position = Cartesian<3>>,
    S: Position<Position = Cartesian<3>> + Default,
    Body<B, S>: Clone,
    Periodic<Triclinic>: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    X: PointUpdate<Cartesian<3>, SiteKey> + Clone,
{
    /// Replicate the bodies in self `counts[0]` x `counts[1]` x `counts[2]` times and
    /// expand the periodic boundary accordingly.
    ///
    /// The new microstate is built with the same step, seed, and spatial data
    /// structure, as if it were cloned. The new microstate's boundary sets the
    /// given interaction range and is extended by `counts[i]` along each
    /// edge vector.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Triclinic;
    /// use hoomd_microstate::{Body, Microstate, Replicate, boundary::Periodic};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic = Triclinic {
    ///     extents: [10.0.try_into()?, 20.0.try_into()?, 30.0.try_into()?],
    ///     tilt_factors: [1.0, 1.0, 1.0],
    /// };
    ///
    /// let periodic = Periodic::new(0.0, triclinic)?;
    /// let microstate = Microstate::builder()
    ///     .boundary(periodic)
    ///     .bodies([Body::point(Cartesian::from([0.0, 0.0, 0.0]))])
    ///     .try_build()?;
    ///
    /// let replicated =
    ///     microstate.replicate_with_maximum_interaction_range([2, 2, 2], 1.0)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// * [`Error::NoReplication`] when any of the counts is 0.
    #[inline]
    fn replicate_with_maximum_interaction_range(
        &self,
        counts: [usize; 3],
        maximum_interaction_range: f64,
    ) -> Result<Microstate<B, S, X, Periodic<Triclinic>>, crate::Error> {
        // try_from_fn would be a cleaner way to write this, but it is not stable:
        // https://doc.rust-lang.org/std/array/fn.try_from_fn.html
        let mut checked_counts = [PositiveReal::default(); 3];
        for i in 0..3 {
            checked_counts[i] =
                PositiveReal::try_from(counts[i] as f64).map_err(crate::Error::NoReplication)?;
        }

        let old_extents = array::from_fn::<_, 3, _>(|i| self.boundary().shape.extents[i]);
        let new_boundary = Periodic::new(
            maximum_interaction_range,
            Triclinic {
                extents: array::from_fn(|i| old_extents[i] * checked_counts[i]),
                ..self.boundary().shape
            },
        )
        .expect("replicated boxes should always satisfy the maximum interaction range");

        let basis_vectors = self.boundary().shape.edge_vectors();
        let base_offset: Cartesian<3> = basis_vectors
            .iter()
            .enumerate()
            .map(|(i, &v)| -v / 2.0 * (checked_counts[i].get() - 1.0))
            .sum();

        self.build_replicate(counts, new_boundary, basis_vectors, base_offset)
    }

    /// Calls [`replicate_with_maximum_interaction_range`] with the current boundary's
    /// maximum interaction range.
    #[inline]
    fn replicate(
        &self,
        counts: [usize; 3],
    ) -> Result<Microstate<B, S, X, Periodic<Triclinic>>, crate::Error> {
        self.replicate_with_maximum_interaction_range(
            counts,
            self.boundary().maximum_interaction_range(),
        )
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::property::Point;

    use approxim::assert_relative_eq;

    use rstest::{fixture, rstest};

    #[fixture]
    fn sheared_triclinic() -> Triclinic {
        let two = 2.0.try_into().expect("2 should be positive");
        Triclinic {
            extents: [two, two, two],
            tilt_factors: [f64::sqrt(2.), f64::sqrt(2.), f64::sqrt(2.)],
        }
    }

    #[rstest]
    fn coordinate_conversion_roundtrip(sheared_triclinic: Triclinic) {
        // Test that converting to fractional and back gives the original position
        let periodic =
            Periodic::new(0.0, sheared_triclinic).expect("hard-coded range should be valid");

        let test_frac_positions = vec![
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5],
            [0.9, 0.8, 0.9],
            [-0.9, -0.8, -0.9],
        ];

        for frac_array in test_frac_positions {
            let frac = Cartesian::<3>::from(frac_array);
            let pos = periodic.shape.absolute(&frac);
            let frac_back = periodic.shape.fractional(&pos);
            assert_relative_eq!(frac, frac_back, epsilon = 1e-8);
        }
    }

    #[test]
    fn maximum_allowable_orthogonal() {
        // Test with orthogonal box (no tilt)
        let triclinic = Triclinic {
            extents: [
                20.0.try_into().unwrap(),
                10.0.try_into().unwrap(),
                40.0.try_into().unwrap(),
            ],
            tilt_factors: [0.0, 0.0, 0.0],
        };
        assert_eq!(triclinic.maximum_allowable_interaction_range(), 5.0);
    }

    #[rstest]
    fn maximum_allowable_tilted(sheared_triclinic: Triclinic) {
        // Test with tilted box
        let max_range = sheared_triclinic.maximum_allowable_interaction_range();
        assert_eq!(max_range, 1.0 / (9.0_f64 - 4.0 * 2.0_f64.sqrt()).sqrt());
    }

    #[test]
    fn wrap_orthogonal() {
        // Test wrapping with orthogonal box (no tilt)
        let triclinic = Triclinic {
            extents: [
                20.0.try_into().unwrap(),
                20.0.try_into().unwrap(),
                20.0.try_into().unwrap(),
            ],
            tilt_factors: [0.0, 0.0, 0.0],
        };
        let periodic = Periodic::new(0.0, triclinic).expect("hard-coded range should be valid");

        let point = Point::new([5.0, 3.0, 8.0].into());
        assert_eq!(periodic.wrap(point), Ok(point));

        let point = Point::new([-10.0, -10.0, -10.0].into());
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

    #[rstest]
    fn wrap_tilted(sheared_triclinic: Triclinic) {
        // Test wrapping with tilted box
        let periodic =
            Periodic::new(0.0, sheared_triclinic).expect("hard-coded range should be valid");

        // Point inside box should not change
        let point = Point::new([0.5, 0.5, 0.5].into());
        assert_relative_eq!(
            periodic.wrap(point).unwrap().position,
            point.position,
            epsilon = 1e-8
        );

        // Point at center should wrap
        let frac_point = [1.0, 1.0, 1.0].into();
        let abs_point = Point::new(periodic.shape.absolute(&frac_point));
        let wrapped = periodic.wrap(abs_point).expect("wrap should succeed");
        // Verify it's back inside the box
        assert_relative_eq!(wrapped.position, [0.0, 0.0, 0.0].into(), epsilon = 1e-8);
    }

    #[rstest]
    fn no_ghosts_interior(sheared_triclinic: Triclinic) {
        // Test that interior points generate no ghosts and boundary points do
        let periodic = Periodic::new(0.01, sheared_triclinic.clone())
            .expect("hard-coded range should be valid");

        let _edge_vectors = sheared_triclinic.edge_vectors();

        // Test interior point (not at origin)
        let mut interior_pos = Cartesian::from([0.2, 0.2, 0.2]);
        interior_pos = periodic.shape.absolute(&interior_pos);

        let ghosts = periodic.generate_ghosts(&Point::new(interior_pos));
        assert!(
            ghosts.is_empty(),
            "Interior point should not generate ghosts: {interior_pos:?}"
        );
    }

    #[rstest]
    fn ghosts_face_centers(sheared_triclinic: Triclinic) {
        // Test that points near face centers generate appropriate ghosts
        let periodic = Periodic::new(0.3, sheared_triclinic.clone())
            .expect("hard-coded range should be valid");

        let _edge_vectors = sheared_triclinic.edge_vectors();

        // Point near the x-face (at maximum x)
        let frac_pos = Cartesian::<3>::from([0.49, 0.0, 0.0]);
        let abs_point = Point::new(periodic.shape.absolute(&frac_pos));

        let ghosts = periodic.generate_ghosts(&abs_point);
        // Should generate at least 1 ghost (one for the face)
        assert!(!ghosts.is_empty(), "Should generate ghosts near face");
    }

    #[test]
    fn ghosts_orthogonal_faces() {
        // Comprehensive test with orthogonal box to validate ghost generation
        let triclinic = Triclinic {
            extents: [
                20.0.try_into().unwrap(),
                10.0.try_into().unwrap(),
                40.0.try_into().unwrap(),
            ],
            tilt_factors: [0.0, 0.0, 0.0],
        };
        let periodic = Periodic::new(1.0, triclinic).expect("hard-coded range should be valid");

        // no ghosts for points outside the boundary
        let ghosts = periodic.generate_ghosts(&Point::new(Cartesian::from([10.5, 0.0, 0.0])));
        assert!(ghosts.is_empty());

        let ghosts = periodic.generate_ghosts(&Point::new(Cartesian::from([0.0, 5.5, 0.0])));
        assert!(ghosts.is_empty());

        // Face: x-direction
        let ghosts = periodic.generate_ghosts(&Point::new(Cartesian::from([9.5, 0.0, 0.0])));
        assert_eq!(ghosts.len(), 1);
        assert_relative_eq!(ghosts[0].position, Cartesian::from([-10.5, 0.0, 0.0]));

        // Face: y-direction
        let ghosts = periodic.generate_ghosts(&Point::new(Cartesian::from([0.0, 4.5, 0.0])));
        assert_eq!(ghosts.len(), 1);
        assert_relative_eq!(ghosts[0].position, Cartesian::from([0.0, -5.5, 0.0]));

        // Face: z-direction
        let ghosts = periodic.generate_ghosts(&Point::new(Cartesian::from([0.0, 0.0, 19.5])));
        assert_eq!(ghosts.len(), 1);
        assert_relative_eq!(ghosts[0].position, Cartesian::from([0.0, 0.0, -20.5]));

        // Edge: x-y
        let ghosts = periodic.generate_ghosts(&Point::new(Cartesian::from([9.5, 4.5, 0.0])));
        assert_eq!(ghosts.len(), 3);

        // Vertex: x-y-z
        let ghosts = periodic.generate_ghosts(&Point::new(Cartesian::from([9.5, 4.5, 19.5])));
        assert_eq!(ghosts.len(), 7);
    }

    #[rstest]
    fn ghosts_tilted_box(sheared_triclinic: Triclinic) {
        // Test ghost generation with a tilted box
        let periodic = Periodic::new(0.1, sheared_triclinic.clone())
            .expect("hard-coded range should be valid");

        // Point inside the box should not generate ghosts unless near a boundary
        let _edge_vectors = sheared_triclinic.edge_vectors();
        let interior_pos = Cartesian::<3>::default();

        let ghosts = periodic.generate_ghosts(&Point::new(interior_pos));
        // Interior point far from boundary
        assert!(ghosts.is_empty());

        // Point at boundary vertex
        let frac_pos = Cartesian::<3>::from([0.499, 0.499, 0.499]);
        let abs_point = Point::new(periodic.shape.absolute(&frac_pos));

        let ghosts = periodic.generate_ghosts(&abs_point);
        // Should generate 7 ghosts, all of which should wrap back to same point
        assert!(
            ghosts.len() == 7,
            "Point near boundary should generate ghosts"
        );
        for i in 0..ghosts.len() {
            assert_relative_eq!(
                periodic.wrap(ghosts[i]).unwrap().position(),
                abs_point.position(),
                epsilon = 1e-8
            );
        }
    }
    #[test]
    fn replicate_222_cubic() -> anyhow::Result<()> {
        let cuboid = Triclinic::cuboid(10.0.try_into()?, 20.0.try_into()?, 30.0.try_into()?);

        let periodic = Periodic::new(1.0, cuboid)?;
        let microstate = Microstate::builder()
            .boundary(periodic)
            .bodies([Body::point(Cartesian::from([0.0, 0.0, 0.0]))])
            .try_build()?;

        let replicated = microstate.replicate([2, 2, 2])?;

        assert_eq!(replicated.bodies().len(), 8);
        assert_eq!(replicated.boundary().shape.extents[0].get(), 20.0);
        assert_eq!(replicated.boundary().shape.extents[1].get(), 40.0);
        assert_eq!(replicated.boundary().shape.extents[2].get(), 60.0);
        assert_eq!(
            replicated.boundary().shape.tilt_factors,
            microstate.boundary().shape().tilt_factors
        );
        assert_eq!(
            replicated.boundary().maximum_interaction_range(),
            microstate.boundary().maximum_interaction_range()
        );
        assert_eq!(
            replicated.bodies()[0].item.properties.position,
            [-5.0, -10.0, -15.0].into()
        );
        assert_eq!(
            replicated.bodies()[1].item.properties.position,
            [-5.0, -10.0, 15.0].into()
        );
        assert_eq!(
            replicated.bodies()[2].item.properties.position,
            [-5.0, 10.0, -15.0].into()
        );
        assert_eq!(
            replicated.bodies()[3].item.properties.position,
            [-5.0, 10.0, 15.0].into()
        );
        assert_eq!(
            replicated.bodies()[4].item.properties.position,
            [5.0, -10.0, -15.0].into()
        );
        assert_eq!(
            replicated.bodies()[5].item.properties.position,
            [5.0, -10.0, 15.0].into()
        );
        assert_eq!(
            replicated.bodies()[6].item.properties.position,
            [5.0, 10.0, -15.0].into()
        );
        assert_eq!(
            replicated.bodies()[7].item.properties.position,
            [5.0, 10.0, 15.0].into()
        );

        Ok(())
    }

    #[test]
    fn replicate_222_sheared() -> anyhow::Result<()> {
        let triclinic = Triclinic {
            extents: [10.0.try_into()?, 20.0.try_into()?, 30.0.try_into()?],
            tilt_factors: [1.0, 0.5, 0.25],
        };

        let periodic = Periodic::new(1.0, triclinic)?;
        let microstate = Microstate::builder()
            .boundary(periodic)
            .bodies([Body::point(Cartesian::from([0.0, 0.0, 0.0]))])
            .try_build()?;

        let replicated = microstate.replicate([2, 2, 2])?;

        assert_eq!(replicated.bodies().len(), 8);
        assert_eq!(replicated.boundary().shape.extents[0].get(), 20.0);
        assert_eq!(replicated.boundary().shape.extents[1].get(), 40.0);
        assert_eq!(replicated.boundary().shape.extents[2].get(), 60.0);
        assert_eq!(
            replicated.boundary().shape.tilt_factors,
            microstate.boundary().shape().tilt_factors
        );
        assert_eq!(
            replicated.boundary().maximum_interaction_range(),
            microstate.boundary().maximum_interaction_range()
        );
        assert_eq!(
            replicated.bodies()[0].item.properties.position,
            [-22.5, -13.75, -15.0].into()
        );
        assert_eq!(
            replicated.bodies()[1].item.properties.position,
            [-7.5, -6.25, 15.0].into()
        );
        assert_eq!(
            replicated.bodies()[2].item.properties.position,
            [-2.5, 6.25, -15.0].into()
        );
        assert_eq!(
            replicated.bodies()[3].item.properties.position,
            [12.5, 13.75, 15.0].into()
        );
        assert_eq!(
            replicated.bodies()[4].item.properties.position,
            [-12.5, -13.75, -15.0].into()
        );
        assert_eq!(
            replicated.bodies()[5].item.properties.position,
            [2.5, -6.25, 15.0].into()
        );
        assert_eq!(
            replicated.bodies()[6].item.properties.position,
            [7.5, 6.25, -15.0].into()
        );
        assert_eq!(
            replicated.bodies()[7].item.properties.position,
            [22.5, 13.75, 15.0].into()
        );

        Ok(())
    }

    #[test]
    fn replicate_with_maximum_interaction_range() -> anyhow::Result<()> {
        let triclinic = Triclinic {
            extents: [10.0.try_into()?, 20.0.try_into()?, 30.0.try_into()?],
            tilt_factors: [1.0, 0.5, 0.25],
        };

        let periodic = Periodic::new(0.0, triclinic)?;
        let microstate = Microstate::builder()
            .boundary(periodic)
            .bodies([Body::point(Cartesian::from([0.0, 0.0, 0.0]))])
            .try_build()?;

        let replicated = microstate.replicate_with_maximum_interaction_range([2, 2, 2], 3.0)?;

        assert_eq!(replicated.bodies().len(), 8);
        assert_eq!(replicated.boundary().shape.extents[0].get(), 20.0);
        assert_eq!(replicated.boundary().shape.extents[1].get(), 40.0);
        assert_eq!(replicated.boundary().shape.extents[2].get(), 60.0);
        assert_eq!(
            replicated.boundary().shape.tilt_factors,
            microstate.boundary().shape().tilt_factors
        );
        assert_eq!(replicated.boundary().maximum_interaction_range(), 3.0);

        Ok(())
    }
}

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement periodic boundary conditions for rhomboid boxes in cartesian space.

use arrayvec::ArrayVec;
use std::cmp::min;

use crate::{
    boundary::{
        Error, GenerateGhosts, MAX_GHOSTS, MaximumAllowableInteractionRange, Periodic, Wrap,
    },
    property::Position,
};
use hoomd_geometry::{IsPointInside, shape::Rhomboid};

use hoomd_vector::Cartesian;

impl MaximumAllowableInteractionRange for Rhomboid {
    /// The largest value that the maximum interaction range can take.
    ///
    /// For a rhomboid (2D), the maximum allowable interaction range is
    /// half the minimum distance to the nearest parallel planes:
    /// ```math
    /// r_\mathrm{max} = \frac{1}{2} \min(d_x, d_y)
    /// ```
    /// where $`d_i`$ are the distances to the nearest parallel planes
    /// in each direction, accounting for tilt factors.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rhomboid;
    /// use hoomd_microstate::boundary::MaximumAllowableInteractionRange;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rhomboid = Rhomboid::from((10.0.try_into()?, 10.0.try_into()?, 0.0));
    ///
    /// // For orthogonal rhomboid, max range is min(extent)/2 = 5.0
    /// assert_eq!(rhomboid.maximum_allowable_interaction_range(), 5.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn maximum_allowable_interaction_range(&self) -> f64 {
        let plane_distances = self.get_nearest_plane_distance();
        f64::min(plane_distances[0].get(), plane_distances[1].get())
    }
}

impl Periodic<Rhomboid> {
    pub fn to_fractional(&self, pos: &Cartesian<2>) -> Cartesian<2> {
        let lx = self.shape.lx().get();
        let ly = self.shape.ly().get();
        let xy = self.shape.xy();

        let s1 = (pos[0] - xy * pos[1]) / lx;
        let s2 = pos[1] / ly;

        Cartesian::from([s1, s2])
    }

    /// Convert fractional coordinates to absolute position.
    ///
    /// This is the inverse operation of `to_fractional`:
    /// ```math
    /// \begin{align*}
    ///     r_1 &= L_x s_1 + xy \cdot L_y s_2\\
    ///     r_2 &= L_y s_2
    /// \end{align*}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rhomboid;
    /// use hoomd_microstate::boundary::Periodic;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rhomboid = Rhomboid::from((2.0.try_into()?, 2.0.try_into()?, 0.0));
    /// let periodic = Periodic::new(1.0, rhomboid)?;
    ///
    /// let frac = Cartesian::from([0.5, 0.0]);
    /// let pos = periodic.to_absolute(&frac);
    /// assert_eq!(pos, Cartesian::from([1.0, 0.0]));
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_absolute(&self, frac: &Cartesian<2>) -> Cartesian<2> {
        let lx = self.shape.lx().get();
        let ly = self.shape.ly().get();
        let xy = self.shape.xy();

        let r1 = lx * frac[0] + xy * ly * frac[1];
        let r2 = ly * frac[1];

        Cartesian::from([r1, r2])
    }
}

impl<P> Wrap<P> for Periodic<Rhomboid>
where
    P: Position<Position = Cartesian<2>>,
{
    /// Wrap any cartesian vector to the inside of the rhomboid box.
    ///
    /// Points are wrapped using fractional coordinates that account for
    /// the rhomboid tilt factors. The wrapping ensures particles remain
    /// within the periodic boundaries.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rhomboid;
    /// use hoomd_microstate::{
    ///     boundary::{Periodic, Wrap},
    ///     property::Point,
    /// };
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rhomboid = Rhomboid::from((10.0.try_into()?, 10.0.try_into()?, 0.0));
    /// let periodic = Periodic::new(5.0, rhomboid)?;
    ///
    /// let point = Point::new(Cartesian::from([6.0, 0.0]));
    /// let wrapped = periodic.wrap(point)?;
    /// assert_eq!(wrapped.position, Cartesian::from([-4.0, 0.0]));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn wrap(&self, mut properties: P) -> Result<P, Error> {
        let r = properties.position_mut();
        let mut fractional = self.to_fractional(r);
        for i in 0..2 {
            fractional[i] -= fractional[i].round();
            fractional[i] = if fractional[i] == 0.5 {
                -0.5
            } else {
                fractional[i]
            };
        }
        *r = self.to_absolute(&fractional);
        Ok(properties)
    }
}

impl<S> GenerateGhosts<S> for Periodic<Rhomboid>
where
    S: Position<Position = Cartesian<2>> + Copy + Default,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }

    /// Generate periodic images of sites near the edge of the periodic boundary.
    ///
    /// For rhomboids (2D), `generate_ghosts` places ghosts near the 4 edges and
    /// 4 vertices of the parallelogram.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rhomboid;
    /// use hoomd_microstate::{
    ///     boundary::{GenerateGhosts, Periodic},
    ///     property::Point,
    /// };
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rhomboid = Rhomboid::from((4.0.try_into()?, 4.0.try_into()?, 0.0));
    /// let periodic = Periodic::new(1.0, rhomboid)?;
    ///
    /// // Point near the x-face
    /// let point = Point::new(Cartesian::from([1.9, 0.0]));
    /// let ghosts = periodic.generate_ghosts(&point);
    ///
    /// // Should generate ghost on the opposite side
    /// assert!(!ghosts.is_empty());
    /// assert_eq!(ghosts[0].position[0], -2.1); // wrapped around
    /// #
    /// # Ok(())
    /// # }
    /// ```
    fn generate_ghosts(&self, site_properties: &S) -> ArrayVec<S, MAX_GHOSTS> {
        let mut result = ArrayVec::new();

        let r = site_properties.position();

        if !self.shape.is_point_inside(r) {
            return result;
        }

        let edge_vectors = self.shape.get_edge_vectors();

        let new_site = |x, y| {
            let mut new_site = *site_properties;
            *new_site.position_mut() += x * edge_vectors[0];
            *new_site.position_mut() += y * edge_vectors[1];
            new_site
        };

        let plane_distances = self.shape.get_nearest_plane_distance();
        let frac = self.to_fractional(r);

        let near_right = frac[0] > 0.5 - self.maximum_interaction_range / plane_distances[0].get();
        let near_left = frac[0] < -0.5 + self.maximum_interaction_range / plane_distances[0].get();
        let near_top = frac[1] > 0.5 - self.maximum_interaction_range / plane_distances[1].get();
        let near_bottom =
            frac[1] < -0.5 + self.maximum_interaction_range / plane_distances[1].get();

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

#[cfg(test)]
mod tests {

    use super::*;
    use crate::property::Point;

    use approxim::assert_relative_eq;
    use rand::{Rng, SeedableRng, distr::Distribution, rngs::StdRng};
    use rstest::{fixture, rstest};

    const N_SAMPLES: usize = 1024;

    #[fixture]
    fn get_sheared_rhomboid() -> Rhomboid {
        Rhomboid::from((2.0.try_into()?, 2.0.try_into()?, f64::sqrt(2.)))
    }

    #[rstest]
    fn coordinate_conversion_roundtrip(get_sheared_rhomboid: Rhomboid) {
        // Test that converting to fractional and back gives the original position
        let periodic =
            Periodic::new(0.0, get_sheared_rhomboid).expect("hard-coded range should be valid");

        let test_frac_positions = vec![
            [0.0, 0.0],
            [0.5, 0.5],
            [-0.5, -0.5],
            [0.9, 0.8],
            [-0.9, -0.8],
        ];

        for frac_array in test_frac_positions {
            let frac = Cartesian::<2>::from(frac_array);
            let pos = periodic.to_absolute(&frac);
            let frac_back = periodic.to_fractional(&pos);
            assert_relative_eq!(frac, frac_back, epsilon = 1e-8);
        }
    }

    #[test]
    fn maximum_allowable_orthogonal() {
        // Test with orthogonal rhomboid (no tilt)
        let rhomboid = Rhomboid::from((20.0.try_into()?, 10.0.try_into()?, 0.0));
        assert_eq!(rhomboid.maximum_allowable_interaction_range(), 5.0);
    }

    #[rstest]
    fn maximum_allowable_tilted(get_sheared_rhomboid: Rhomboid) {
        // Test with tilted rhomboid
        let max_range = get_sheared_rhomboid.maximum_allowable_interaction_range();
        // For lx=ly=2, xy=sqrt(2), the distance to parallel planes in x is:
        // dx = 2 / sqrt(1 + 2) = 2 / sqrt(3)
        // dy = 2
        // max_range = min(dx, dy) / 2 = (2/sqrt(3)) / 2 = 1/sqrt(3)
        assert!(max_range == 1.0 / f64::sqrt(3.0));
    }

    #[test]
    fn wrap_orthogonal() {
        // Test wrapping with orthogonal rhomboid (no tilt)
        let rhomboid = Rhomboid::from((20.0.try_into()?, 20.0.try_into()?, 0.0));
        let periodic = Periodic::new(0.0, rhomboid).expect("hard-coded range should be valid");

        let point = Point::new([5.0, 3.0].into());
        assert_eq!(periodic.wrap(point), Ok(point));

        let point = Point::new([-10.0, -10.0].into());
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

    #[rstest]
    fn wrap_tilted(get_sheared_rhomboid: Rhomboid) {
        // Test wrapping with tilted rhomboid
        let periodic =
            Periodic::new(0.0, get_sheared_rhomboid).expect("hard-coded range should be valid");

        // Point inside rhomboid should not change
        let point = Point::new([0.5, 0.5].into());
        assert_relative_eq!(
            periodic.wrap(point).unwrap().position,
            point.position,
            epsilon = 1e-8
        );

        // Point at center should wrap
        let frac_point = [1.0, 1.0].into();
        let abs_point = Point::new(periodic.to_absolute(&frac_point));
        let wrapped = periodic.wrap(abs_point).expect("wrap should succeed");
        // Verify it's back inside the rhomboid
        assert_relative_eq!(wrapped.position, [0.0, 0.0].into(), epsilon = 1e-8);
    }

    #[rstest]
    fn no_ghosts_interior(get_sheared_rhomboid: Rhomboid) {
        // Test that interior points generate no ghosts and boundary points do
        let periodic = Periodic::new(0.01, get_sheared_rhomboid.clone())
            .expect("hard-coded range should be valid");

        // Test interior point (not at origin)
        let mut interior_pos = Cartesian::from([0.2, 0.2]);
        interior_pos = periodic.to_absolute(&interior_pos);

        let ghosts = periodic.generate_ghosts(&Point::new(interior_pos));
        assert!(
            ghosts.is_empty(),
            "Interior point should not generate ghosts: {:?}",
            interior_pos
        );
    }

    #[rstest]
    fn ghosts_face_centers(get_sheared_rhomboid: Rhomboid) {
        // Test that points near face centers generate appropriate ghosts
        let periodic = Periodic::new(0.3, get_sheared_rhomboid.clone())
            .expect("hard-coded range should be valid");

        // Point near the x-face (at maximum x)
        let frac_pos = Cartesian::<2>::from([0.49, 0.0]);
        let abs_point = Point::new(periodic.to_absolute(&frac_pos));

        let ghosts = periodic.generate_ghosts(&abs_point);
        // Should generate at least 1 ghost (one for the face)
        assert!(ghosts.len() >= 1, "Should generate ghosts near face");
    }

    #[test]
    fn ghosts_orthogonal_faces() {
        // Comprehensive test with orthogonal rhomboid to validate ghost generation
        let rhomboid = Rhomboid::from((20.0.try_into()?, 10.0.try_into()?, 0.0));
        let periodic = Periodic::new(1.0, rhomboid).expect("hard-coded range should be valid");

        // no ghosts for points outside the boundary
        let ghosts = periodic.generate_ghosts(&Point::new(Cartesian::from([10.5, 0.0])));
        assert!(ghosts.is_empty());

        let ghosts = periodic.generate_ghosts(&Point::new(Cartesian::from([0.0, 5.5])));
        assert!(ghosts.is_empty());

        // Face: x-direction
        let ghosts = periodic.generate_ghosts(&Point::new(Cartesian::from([9.5, 0.0])));
        assert_eq!(ghosts.len(), 1);
        assert_relative_eq!(ghosts[0].position, Cartesian::from([-10.5, 0.0]));

        // Face: y-direction
        let ghosts = periodic.generate_ghosts(&Point::new(Cartesian::from([0.0, 4.5])));
        assert_eq!(ghosts.len(), 1);
        assert_relative_eq!(ghosts[0].position, Cartesian::from([0.0, -5.5]));

        // Edge: x-y
        let ghosts = periodic.generate_ghosts(&Point::new(Cartesian::from([9.5, 4.5])));
        assert_eq!(ghosts.len(), 3);

        // Vertex: x-y
        let ghosts = periodic.generate_ghosts(&Point::new(Cartesian::from([9.5, 4.5])));
        assert_eq!(ghosts.len(), 3);
    }

    #[rstest]
    fn ghosts_tilted_box(get_sheared_rhomboid: Rhomboid) {
        // Test ghost generation with a tilted rhomboid
        let periodic = Periodic::new(0.1, get_sheared_rhomboid.clone())
            .expect("hard-coded range should be valid");

        // Point inside the rhomboid should not generate ghosts unless near a boundary
        let interior_pos = Cartesian::<2>::default();

        let ghosts = periodic.generate_ghosts(&Point::new(interior_pos));
        // Interior point far from boundary
        assert!(ghosts.is_empty());

        // Point at boundary vertex
        let frac_pos = Cartesian::<2>::from([0.499, 0.499]);
        let abs_point = Point::new(periodic.to_absolute(&frac_pos));

        let ghosts = periodic.generate_ghosts(&abs_point);
        // Should generate 3 ghosts (at vertex: 2 faces + 1 edge), all of which should wrap back to same point
        assert!(
            ghosts.len() == 3,
            "Point near boundary should generate ghosts, got {}",
            ghosts.len()
        );
        for i in 0..ghosts.len() {
            assert_relative_eq!(
                periodic.wrap(ghosts[i]).unwrap().position(),
                abs_point.position(),
                epsilon = 1e-8
            );
        }
    }
}

// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement periodic boundary conditions for triclinic boxes in cartesian space.

use arrayvec::ArrayVec;

use crate::{
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
    /// let triclinic =
    ///     Triclinic::with_box_dimensions([10.0, 10.0, 10.0, 0.0, 0.0, 0.0]);
    ///
    /// // For orthogonal box, max range is min(extent)/2 = 5.0
    /// assert_eq!(triclinic.maximum_allowable_interaction_range(), 5.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn maximum_allowable_interaction_range(&self) -> f64 {
        let plane_distances = self.get_nearest_plane_distance();
        plane_distances
            .iter()
            .map(|x| x.get() * 0.5)
            .fold(f64::INFINITY, f64::min)
    }
}

impl Periodic<Triclinic> {
    /// Convert a real space (absolute) position to fractional coordinates. We can express a
    /// general point in space $`\vec{r}`$ in terms of the basis vectors $`a_i`$,
    /// ```math
    ///     \vec{r} = s_1 \vec{a}_1 + s_2 \vec{a}_2 + s_3 \vec{a}_3.
    /// ```
    /// The vector, $`\vec{s}=(s_1, s_2, s_3)`$ then gives the fractional coordinates
    /// of this point. This can be expressed as $`\vec{r} = \mathbf{A} \vec{s}`$, where
    /// $`\mathbf{A}`$ is the matrix with columns equal to the box vectors. That is,
    /// $`\vec{s} = \mathbf{A}^{-1} \vec{r}`$, where $`\mathbf{A}^{-1}`$ can be viewed
    /// as linearly shearing the box into a unit cube centered at the origin.
    /// For a triclinic box, the transformation can be written as
    /// ```math
    /// \begin{align*}
    ///     s_1 &= \frac{r_1-(xy)r_2-(xz-yz\cdot xy) r_3}{L_x}\\
    ///     s_2 &= \frac{r_2-yz r_3}{L_y}\\
    ///     s_3 &= \frac{r_3}{L_z}.\\
    /// \end{align*}
    /// ```
    ///
    /// Each fractional coordinate is in the range $`[-0.5, 0.5)`$.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Triclinic;
    /// use hoomd_microstate::boundary::Periodic;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic =
    ///     Triclinic::with_box_dimensions([2.0, 2.0, 2.0, 0.0, 0.0, 0.0]);
    /// let periodic = Periodic::new(1.0, triclinic)?;
    ///
    /// let pos = Cartesian::from([1.0, 0.0, 0.0]);
    /// let frac = periodic.to_fractional(&pos);
    /// assert_eq!(frac, Cartesian::from([0.5, 0.0, 0.0]));
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_fractional(&self, pos: &Cartesian<3>) -> Cartesian<3> {
        let l: Cartesian<3> = self.shape.extents.map(|x| x.get()).into();
        let mut frac = *pos;
        frac[0] -= (self.shape.xz() - self.shape.yz() * self.shape.xy()) * pos[2]
            + self.shape.xy() * pos[1];
        frac[1] -= self.shape.yz() * pos[2];
        for i in 0..3 {
            frac[i] /= l[i];
        }
        frac
    }

    /// Convert fractional coordinates to absolute position.
    ///
    /// This is the inverse operation of `to_fractional`, $`\vec{r} = \mathbf{A} s`$.
    /// Namely,
    /// ```math
    /// \begin{align*}
    ///     r_1 &= L_x s_1 + xyL_y s_2 + xzL_z s_3\\
    ///     r_2 &= L_y s_2 + yz L_z s_3\\
    ///     r_3 &= L_z s_3.\\
    /// \end{align*}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Triclinic;
    /// use hoomd_microstate::boundary::Periodic;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic =
    ///     Triclinic::with_box_dimensions([2.0, 2.0, 2.0, 0.0, 0.0, 0.0]);
    /// let periodic = Periodic::new(1.0, triclinic)?;
    ///
    /// let frac = Cartesian::from([0.5, 0.0, 0.0]);
    /// let pos = periodic.to_absolute(&frac);
    /// assert_eq!(pos, Cartesian::from([1.0, 0.0, 0.0]));
    /// # Ok(())
    /// # }
    /// ```
    pub fn to_absolute(&self, frac: &Cartesian<3>) -> Cartesian<3> {
        let mut pos: Cartesian<3> = Cartesian::from([1.0, 1.0, 1.0]);
        for i in 0..3 {
            pos[i] = self.shape.extents[i].get() * frac[i];
        }
        pos[0] += self.shape.xy() * pos[1] + self.shape.xz() * pos[2];
        pos[1] += self.shape.yz() * pos[2];
        pos
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
    /// let triclinic =
    ///     Triclinic::with_box_dimensions([10.0, 10.0, 10.0, 0.0, 0.0, 0.0]);
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
        let mut fractional = self.to_fractional(r);
        for i in 0..3 {
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

    #[inline]
    /// Generate periodic images of sites near the edge of the periodic boundary. Ghosts
    /// are generated based on a `maximum_interaction_range`, the particle position, and
    /// box geometry.
    ///
    /// A ghost needs to be generated if it is closer than the maximum interaction range
    /// to one of the walls of the simulation box. The boundary of this zone (the set of
    /// all points exactly `maximum_interaction_range` away) is a plane parallel to the
    /// wall. In fractional coordinates, this boundary plane will be of the
    /// form $`s_i =\pm 0.5 \mp \frac{r_{\text{max}}}{r_{\perp}}`$, where
    /// $`r_{\text{max}}`$ is the maximum_interaction_range and $`r_{\perp}`$ is the
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
    /// let triclinic =
    ///     Triclinic::with_box_dimensions([4.0, 4.0, 4.0, 0.0, 0.0, 0.0]);
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
    fn generate_ghosts(&self, site_properties: &S) -> ArrayVec<S, MAX_GHOSTS> {
        let mut result = ArrayVec::new();

        let r = site_properties.position();

        if !self.shape.is_point_inside(r) {
            return result;
        }

        let edge_vectors = self.shape.get_edge_vectors();

        let new_site = |x, y, z| {
            let mut new_site = *site_properties;
            *new_site.position_mut() += x * edge_vectors[0];
            *new_site.position_mut() += y * edge_vectors[1];
            *new_site.position_mut() += z * edge_vectors[2];
            new_site
        };

        let plane_distances = self.shape.get_nearest_plane_distance();
        let frac = self.to_fractional(r);

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

#[cfg(test)]
mod tests {

    use super::*;
    use crate::property::Point;

    use approxim::assert_relative_eq;
    use rand::{Rng, SeedableRng, distr::Distribution, rngs::StdRng};
    use rstest::{fixture, rstest};

    const N_SAMPLES: usize = 1024;

    #[fixture]
    fn get_sheared_triclinic() -> Triclinic {
        Triclinic::with_box_dimensions([2., 2., 2., f64::sqrt(2.), f64::sqrt(2.), f64::sqrt(2.)])
    }

    #[rstest]
    fn coordinate_conversion_roundtrip(get_sheared_triclinic: Triclinic) {
        // Test that converting to fractional and back gives the original position
        let periodic =
            Periodic::new(0.0, get_sheared_triclinic).expect("hard-coded range should be valid");

        let test_frac_positions = vec![
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5],
            [0.9, 0.8, 0.9],
            [-0.9, -0.8, -0.9],
        ];

        for frac_array in test_frac_positions {
            let frac = Cartesian::<3>::from(frac_array);
            let pos = periodic.to_absolute(&frac);
            let frac_back = periodic.to_fractional(&pos);
            assert_relative_eq!(frac, frac_back, epsilon = 1e-8);
        }
    }

    #[test]
    fn maximum_allowable_orthogonal() {
        // Test with orthogonal box (no tilt)
        let triclinic = Triclinic::with_box_dimensions([20.0, 10.0, 40.0, 0.0, 0.0, 0.0]);
        assert_eq!(triclinic.maximum_allowable_interaction_range(), 5.0);
    }

    #[rstest]
    fn maximum_allowable_tilted(get_sheared_triclinic: Triclinic) {
        // Test with tilted box
        let max_range = get_sheared_triclinic.maximum_allowable_interaction_range();
        assert!(max_range == 1.0 / (9.0_f64 - 4.0 * 2.0_f64.sqrt()).sqrt());
    }

    #[test]
    fn wrap_orthogonal() {
        // Test wrapping with orthogonal box (no tilt)
        let triclinic = Triclinic::with_box_dimensions([20.0, 20.0, 20.0, 0.0, 0.0, 0.0]);
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
    fn wrap_tilted(get_sheared_triclinic: Triclinic) {
        // Test wrapping with tilted box
        let periodic =
            Periodic::new(0.0, get_sheared_triclinic).expect("hard-coded range should be valid");

        // Point inside box should not change
        let point = Point::new([0.5, 0.5, 0.5].into());
        assert_relative_eq!(
            periodic.wrap(point).unwrap().position,
            point.position,
            epsilon = 1e-8
        );

        // Point at center should wrap
        let frac_point = [1.0, 1.0, 1.0].into();
        let abs_point = Point::new(periodic.to_absolute(&frac_point));
        let wrapped = periodic.wrap(abs_point).expect("wrap should succeed");
        // Verify it's back inside the box
        assert_relative_eq!(wrapped.position, [0.0, 0.0, 0.0].into(), epsilon = 1e-8);
    }

    #[rstest]
    fn no_ghosts_interior(get_sheared_triclinic: Triclinic) {
        // Test that interior points generate no ghosts and boundary points do
        let periodic = Periodic::new(0.01, get_sheared_triclinic.clone())
            .expect("hard-coded range should be valid");

        let edge_vectors = get_sheared_triclinic.get_edge_vectors();

        // Test interior point (not at origin)
        let mut interior_pos = Cartesian::from([0.2, 0.2, 0.2]);
        interior_pos = periodic.to_absolute(&interior_pos);

        let ghosts = periodic.generate_ghosts(&Point::new(interior_pos));
        assert!(
            ghosts.is_empty(),
            "Interior point should not generate ghosts: {:?}",
            interior_pos
        );
    }

    #[rstest]
    fn ghosts_face_centers(get_sheared_triclinic: Triclinic) {
        // Test that points near face centers generate appropriate ghosts
        let periodic = Periodic::new(0.3, get_sheared_triclinic.clone())
            .expect("hard-coded range should be valid");

        let edge_vectors = get_sheared_triclinic.get_edge_vectors();

        // Point near the x-face (at maximum x)
        let frac_pos = Cartesian::<3>::from([0.49, 0.0, 0.0]);
        let abs_point = Point::new(periodic.to_absolute(&frac_pos));

        let ghosts = periodic.generate_ghosts(&abs_point);
        // Should generate at least 1 ghost (one for the face)
        assert!(ghosts.len() >= 1, "Should generate ghosts near face");
    }

    #[test]
    fn ghosts_orthogonal_faces() {
        // Comprehensive test with orthogonal box to validate ghost generation
        let triclinic = Triclinic::with_box_dimensions([20.0, 10.0, 40.0, 0.0, 0.0, 0.0]);
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
    fn ghosts_tilted_box(get_sheared_triclinic: Triclinic) {
        // Test ghost generation with a tilted box
        let periodic = Periodic::new(0.1, get_sheared_triclinic.clone())
            .expect("hard-coded range should be valid");

        // Point inside the box should not generate ghosts unless near a boundary
        let edge_vectors = get_sheared_triclinic.get_edge_vectors();
        let interior_pos = Cartesian::<3>::default();

        let ghosts = periodic.generate_ghosts(&Point::new(interior_pos));
        // Interior point far from boundary
        assert!(ghosts.is_empty());

        // Point at boundary vertex
        let frac_pos = Cartesian::<3>::from([0.499, 0.499, 0.499]);
        let abs_point = Point::new(periodic.to_absolute(&frac_pos));

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
}

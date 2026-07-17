// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement periodic boundary conditions for hyperparallelepipeds in cartesian space.

use std::array;

use crate::{
    boundary::{
        Error, GenerateGhosts, MAX_GHOSTS, MaximumAllowableInteractionRange, Periodic, Wrap,
    },
    property::Position,
};
use arrayvec::ArrayVec;
use hoomd_geometry::{IsPointInside, shape::Hyperparallelepiped};
use hoomd_linear_algebra::{
    MatMul,
    matrix::{Matrix, qr},
};
use hoomd_vector::Cartesian;
use itertools::Itertools;

impl<const N: usize> MaximumAllowableInteractionRange for Hyperparallelepiped<N> {
    /// The largest value that the maximum interaction range can take. While theoretically to avoid self-interaction the interaction distance may be as large as 1/2 the smallest box vector, we choose to take the maximum interaction range to be 1/2 the smallest perpendicular distance between pairs of parallel faces in order to avoid having to generating more than one ghost per particle.
    ///
    /// # Example
    #[inline]
    fn maximum_allowable_interaction_range(&self) -> f64 {
        let plane_distances = self.nearest_plane_distances();
        plane_distances
            .iter()
            .map(|x| x.get() * 0.5)
            .fold(f64::INFINITY, f64::min)
    }
}

impl<P, const N: usize> Wrap<P> for Periodic<Hyperparallelepiped<N>>
where
    P: Position<Position = Cartesian<N>>,
{
    /// Wrap any cartesian vector to the inside of the given hyperparallepiped.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Hyperparallelepiped;
    /// use hoomd_microstate::{
    ///     boundary::{Periodic, Wrap},
    ///     property::Point,
    /// };
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut box_ = Hyperparallelepiped::new([
    ///         [1.0, 0.0, 0.0].into(),
    ///         [0.5, f64::sqrt(3.0) / 2.0, 0.0].into(),
    ///         [0.0, 0.0, 1.0].into(),
    ///     ]);
    /// let periodic = Periodic::new(0.25, box_)?;
    /// let point = Point::new(Cartesian::from([1.0, f64::sqrt(3.0), 2.5]));
    ///
    /// let wrapped_point = periodic.wrap(point)?;
    /// assert_eq!(wrapped_point.position, [0.0, 0.0, -0.5].into());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn wrap(&self, properties: P) -> Result<P, Error> {
        let mut properties = properties;
        let r = properties.position_mut();

        let a = Matrix::<N, N> {
            rows: self.shape.edge_vectors().map(|v| v.coordinates),
        }
        .transpose();

        // TODO: Can this use the pre-computed qr?
        let fractional = qr::qr_solve(&a, &r.to_column_matrix());

        let position_offset = a.matmul(&fractional.map_elements(f64::round));
        *r -= Cartesian::from_column_matrix(&position_offset);

        Ok(properties)
    }
}

impl<S, const N: usize> GenerateGhosts<S> for Periodic<Hyperparallelepiped<N>>
where
    S: Position<Position = Cartesian<N>> + Copy + Default,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }

    // /// Place periodic images of sites near the edge of the periodic boundary.
    #[inline]
    fn generate_ghosts(&self, site_properties: &S) -> ArrayVec<S, MAX_GHOSTS> {
        let mut result = ArrayVec::new();

        let r = site_properties.position();

        if !self.shape.is_point_inside(r) {
            return result;
        }

        // Determine fractional coordinates of "twighlight zones," where ghosts must be generated
        let plane_distances = self.shape.nearest_plane_distances();
        let fractional_cutoffs: [f64; N] =
            array::from_fn(|i| self.maximum_interaction_range() / plane_distances[i].get());
        let fractional_coordinate = self.shape.fractional(*r);

        // For each axis, determine if the particle is near the negative or positive face.
        // Use -(i+1) for negative face and +(i+1) for positive face to avoid -0 encoding issues.
        let mut ghost_directions: ArrayVec<i32, N> = ArrayVec::new();
        for (i, fractional_cutoff) in fractional_cutoffs.iter().enumerate() {
            if fractional_coordinate[i] <= -0.5 + fractional_cutoff {
                ghost_directions
                    .push(-i32::try_from(i + 1).expect("Could not convert face dim to i32"));
            } else if fractional_coordinate[i] >= 0.5 - fractional_cutoff {
                ghost_directions
                    .push(i32::try_from(i + 1).expect("Could not convert face dim to i32"));
            }
        }

        // Generate ghosts for every non-empty subset of the relevant directions.
        // Note: a bit mask may be a faster alternative here since powerset allocates a vec under the hood.
        for subset in ghost_directions.iter().powerset().filter(|s| !s.is_empty()) {
            let mut offset = Cartesian::<N>::default();
            for &&direction in &subset {
                let axis = (direction.unsigned_abs() - 1) as usize;
                let sign = if direction.is_negative() { 1.0 } else { -1.0 };
                offset += self.shape.edge_vectors()[axis] * sign;
            }
            let mut ghost_site = *site_properties;
            *ghost_site.position_mut() += offset;
            result.push(ghost_site);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::Point;
    use approxim::assert_relative_eq;
    use hoomd_geometry::shape::{Hyperparallelepiped, Triclinic};
    use hoomd_vector::Cartesian;
    use rstest::{fixture, rstest};

    fn hyper_from_triclinic(tric: &Triclinic) -> Hyperparallelepiped<3> {
        Hyperparallelepiped::new(tric.edge_vectors())
    }

    #[fixture]
    fn sheared_triclinic() -> Triclinic {
        Triclinic::from_box_vector([
            2.0,
            2.0,
            2.0,
            f64::sqrt(2.0),
            f64::sqrt(2.0),
            f64::sqrt(2.0),
        ])
    }

    #[fixture]
    fn sheared_hyperparallelepiped(sheared_triclinic: Triclinic) -> Hyperparallelepiped<3> {
        hyper_from_triclinic(&sheared_triclinic)
    }

    #[rstest]
    fn coordinate_conversion_roundtrip(sheared_hyperparallelepiped: Hyperparallelepiped<3>) {
        let periodic = Periodic::new(0.0, sheared_hyperparallelepiped)
            .expect("valid periodic hyperparallelepiped");

        let test_frac_positions = vec![
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5],
            [0.9, 0.8, 0.9],
            [-0.9, -0.8, -0.9],
        ];

        for frac_array in test_frac_positions {
            let frac = Cartesian::<3>::from(frac_array);
            let pos = periodic.shape.absolute(frac);
            let frac_back = periodic.shape.fractional(pos);
            assert_relative_eq!(frac, frac_back, epsilon = 1e-8);
        }
    }

    #[rstest]
    fn no_ghosts_interior(sheared_hyperparallelepiped: Hyperparallelepiped<3>) {
        let periodic = Periodic::new(0.01, sheared_hyperparallelepiped)
            .expect("valid periodic hyperparallelepiped");

        let frac_pos = Cartesian::<3>::from([0.2, 0.2, 0.2]);
        let abs_pos = periodic.shape.absolute(frac_pos);

        let ghosts = periodic.generate_ghosts(&Point::new(abs_pos));
        assert!(
            ghosts.is_empty(),
            "Interior point should not generate ghosts"
        );
    }

    #[rstest]
    fn ghosts_face_centers(sheared_hyperparallelepiped: Hyperparallelepiped<3>) {
        let periodic = Periodic::new(0.3, sheared_hyperparallelepiped)
            .expect("valid periodic hyperparallelepiped");

        let frac_pos = Cartesian::<3>::from([0.49, 0.0, 0.0]);
        let abs_point = Point::new(periodic.shape.absolute(frac_pos));

        let ghosts = periodic.generate_ghosts(&abs_point);
        assert!(!ghosts.is_empty(), "Should generate ghosts near face");
    }

    #[test]
    fn same_behavior_as_triclinic() {
        let tric = Triclinic::from_box_vector([20.0, 10.0, 40.0, 0.2, -0.3, 0.4]);
        let hyper = hyper_from_triclinic(&tric);

        let periodic_tric = Periodic::new(1.0, tric.clone()).expect("valid periodic triclinic");
        let periodic_hyper = Periodic::new(1.0, hyper).expect("valid periodic hyperparallelepiped");

        let test_points = vec![
            Cartesian::<3>::from([0.0, 0.0, 0.0]),
            Cartesian::<3>::from([0.49, 0.49, 0.49]),
            Cartesian::<3>::from([-0.49, -0.49, -0.49]),
            Cartesian::<3>::from([1.1, -0.9, 0.2]),
        ];

        for frac_pos in test_points {
            let abs_point = Point::new(periodic_tric.shape.absolute(&frac_pos));

            let wrapped_tric = periodic_tric.wrap(abs_point).unwrap();
            let wrapped_hyper = periodic_hyper.wrap(abs_point).unwrap();
            assert_relative_eq!(
                wrapped_tric.position,
                wrapped_hyper.position,
                epsilon = 1e-8
            );

            let ghosts_tric = periodic_tric.generate_ghosts(&abs_point);
            let ghosts_hyper = periodic_hyper.generate_ghosts(&abs_point);
            assert_eq!(ghosts_tric.len(), ghosts_hyper.len());

            for ghost in &ghosts_tric {
                let found = ghosts_hyper.iter().any(|other| {
                    ghost
                        .position
                        .coordinates
                        .iter()
                        .zip(other.position.coordinates.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-8)
                });
                assert!(
                    found,
                    "Expected ghost position to exist in hyperparallelepiped result"
                );
            }
        }
    }
}

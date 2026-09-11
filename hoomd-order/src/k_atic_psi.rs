// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `k_atic_psi`

use num_complex::Complex;

use hoomd_vector::Cartesian;

/// Compute the 2D *k-atic* order parameter $` \psi_k `$.
///
/// Let the `neighbors` iterator produce `n` points $` \vec{r}_j `$ and
/// $` \theta_j `$ be the polar angle of the vector $`\vec{r}_j - \vec{r}`$.
/// The *k-atic* order parameter is then:
/// ```math
/// \psi_k = \frac{1}{n} \sum_j e^{i k \theta_j}
/// ```
///
/// # Example
///
/// Compute $` \psi_4 `$ for a single arrangement of points:
/// ```
/// use approxim::assert_relative_eq;
/// use num_complex::Complex;
///
/// let r = [0.0, 0.0].into();
/// let neighbors = [
///     [1.0, 0.0].into(),
///     [0.0, 1.0].into(),
///     [-1.0, 0.0].into(),
///     [0.0, -1.0].into(),
/// ];
/// let psi = hoomd_order::k_atic_psi(4.0, &r, neighbors);
///
/// assert_relative_eq!(psi, Complex::new(1.0, 0.0), epsilon = 1e-12);
/// ```
///
/// Compute $` \psi_4 `$ for all sites in a microstate:
/// ```
/// use std::collections::HashMap;
///
/// use hoomd_geometry::shape::Rectangle;
/// use hoomd_microstate::{Body, Microstate, Replicate, boundary::Periodic};
/// use hoomd_order::SitesInBall;
/// use hoomd_spatial::VecCell;
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model_maximum_interaction_range: f64 = 1.0;
/// let order_maximum_neighbor_distance: f64 = 1.5;
/// let maximum_interaction_range =
///     model_maximum_interaction_range.max(order_maximum_neighbor_distance);
///
/// let unit_cell_square = Rectangle::with_equal_edges(1.0.try_into()?);
/// let periodic_unit_cell = Periodic::new(0.0, unit_cell_square)?;
/// let vec_cell = VecCell::builder()
///     .nominal_search_radius(model_maximum_interaction_range.try_into()?)
///     .maximum_search_radius(maximum_interaction_range)
///     .build();
/// let microstate = Microstate::builder()
///     .boundary(periodic_unit_cell)
///     .spatial_data(vec_cell)
///     .bodies([Body::point(Cartesian::default())])
///     .try_build()?
///     .replicate_with_maximum_interaction_range(
///         [16; 2],
///         maximum_interaction_range,
///     )?;
///
/// let mut psi_4 = HashMap::new();
/// for (site_index, site) in microstate.sites().iter().enumerate() {
///     let neighbors = SitesInBall::near_site(
///         &microstate,
///         site_index,
///         order_maximum_neighbor_distance,
///     );
///     psi_4.insert(
///         site.site_tag,
///         hoomd_order::k_atic_psi(
///             4.0,
///             &site.properties.position,
///             neighbors.iter_site_positions().copied(),
///         ),
///     );
/// }
/// # Ok(())
/// # }
/// ```
#[inline]
pub fn k_atic_psi<I: IntoIterator<Item = Cartesian<2>>>(
    k: f64,
    r: &Cartesian<2>,
    neighbors: I,
) -> Complex<f64> {
    let mut total: Complex<f64> = Complex::default();
    let mut count: usize = 0;

    for r_j in neighbors {
        let delta_r = r_j - *r;
        let theta = delta_r[1].atan2(delta_r[0]);
        let (sin, cos) = (k * theta).sin_cos();
        total += Complex::new(cos, sin);
        count += 1;
    }

    total / Complex::new(count as f64, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;

    #[test]
    fn single() {
        let r = [0.0, 0.0].into();
        let neighbors = [[1.0, 0.0].into()];

        assert_relative_eq!(
            k_atic_psi(1.0, &r, neighbors),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(2.0, &r, neighbors),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(3.0, &r, neighbors),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(4.0, &r, neighbors),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(5.0, &r, neighbors),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(6.0, &r, neighbors),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn shifted() {
        let r = [-5.0, 4.0].into();
        let neighbors = [[-4.0, 4.0].into()];

        assert_relative_eq!(
            k_atic_psi(1.0, &r, neighbors),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(2.0, &r, neighbors),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(3.0, &r, neighbors),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(4.0, &r, neighbors),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(5.0, &r, neighbors),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(6.0, &r, neighbors),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn rotated() {
        let r = [0.0, 0.0].into();

        assert_relative_eq!(
            k_atic_psi(2.0, &r, [[1.0, 0.0].into()]),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(2.0, &r, [[-1.0, 0.0].into()]),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(2.0, &r, [[0.0, 1.0].into()]),
            Complex::new(-1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(2.0, &r, [[0.0, -1.0].into()]),
            Complex::new(-1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(2.0, &r, [[1.0, 1.0].into()]),
            Complex::new(0.0, 1.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(2.0, &r, [[1.0, -1.0].into()]),
            Complex::new(0.0, -1.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(2.0, &r, [[-1.0, 1.0].into()]),
            Complex::new(0.0, -1.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(2.0, &r, [[-1.0, -1.0].into()]),
            Complex::new(0.0, 1.0),
            epsilon = 1e-12
        );

        assert_relative_eq!(
            k_atic_psi(4.0, &r, [[1.0, 0.0].into()]),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(4.0, &r, [[0.0, 1.0].into()]),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(4.0, &r, [[-1.0, 0.0].into()]),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(4.0, &r, [[0.0, -1.0].into()]),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(4.0, &r, [[1.0, 1.0].into()]),
            Complex::new(-1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(4.0, &r, [[1.0, -1.0].into()]),
            Complex::new(-1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(4.0, &r, [[-1.0, 1.0].into()]),
            Complex::new(-1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(4.0, &r, [[-1.0, -1.0].into()]),
            Complex::new(-1.0, 0.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn average() {
        let r = [0.0, 0.0].into();

        assert_relative_eq!(
            k_atic_psi(2.0, &r, [[1.0, 0.0].into(), [-1.0, 0.0].into()]),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(2.0, &r, [[0.0, 1.0].into(), [0.0, -1.0].into()]),
            Complex::new(-1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(2.0, &r, [[-1.0, 1.0].into(), [-1.0, -1.0].into()]),
            Complex::new(0.0, 0.0),
            epsilon = 1e-12
        );

        assert_relative_eq!(
            k_atic_psi(
                4.0,
                &r,
                [
                    [1.0, 0.0].into(),
                    [0.0, 1.0].into(),
                    [-1.0, 0.0].into(),
                    [0.0, -1.0].into()
                ]
            ),
            Complex::new(1.0, 0.0),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            k_atic_psi(
                4.0,
                &r,
                [
                    [1.0, 1.0].into(),
                    [1.0, -1.0].into(),
                    [-1.0, 1.0].into(),
                    [-1.0, -1.0].into()
                ]
            ),
            Complex::new(-1.0, 0.0),
            epsilon = 1e-12
        );
    }
}

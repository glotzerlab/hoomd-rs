// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the necessary conversions are necessary and have been checked"
)]
#![expect(
    clippy::cast_sign_loss,
    reason = "the necessary conversions are necessary and have been checked"
)]

//! Implement Checkerboard for Hypercuboids

use itertools::izip;
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::array;

use hoomd_geometry::shape::Hypercuboid;
use hoomd_microstate::boundary::{Closed, Periodic};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Cartesian;

use crate::{Checkerboard, Cover};

/// `2^N` color checkerboard with axis-aligned hypercuboidal cells.
///
/// A `HypercuboidCheckerboard` is comprised of n x m x ... axis aligned
/// spaces. Each space has the same shape, but each axis might have a different
/// edge length. Each axis may be periodic or not.
///
/// Along the non-periodic axes, the checkerboard overhangs the boundary so that
/// the entire domain is always covered (for any origin shift up to 1 cell length).
/// Along periodic axes, the checkerboard has exactly the same width as the domain.
/// `HypercuboidCheckerboard` wraps points "outside" the checkerboard into the correct
/// periodic space.
///
/// Obviously, `HypercuboidCheckerboard` is a suitable checkerboard for
/// [`Hypercuboid`] boundary geometries. It can also be a good choice for
/// other boundaries. For example: cylindrical boundaries (periodic in one
/// direction) and closed boundaries of any shape. There may be many overhanging
/// spaces (some completely outside the boundary) in these cases. However, the
/// checkerboard is still valid and rayon's dynamic load balancing scheme should
/// be able to handle the empty cells efficiently.
#[serde_as]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HypercuboidCheckerboard<const N: usize> {
    /// Position of the 0,0,0 cell's lower left corner.
    origin: Cartesian<N>,

    /// Length of each axis aligned space edge.
    #[serde_as(as = "[_; N]")]
    space_width: [f64; N],

    /// Number of spaces along each axis.
    #[serde_as(as = "[_; N]")]
    shape: [usize; N],

    /// True when an axis is periodic.
    #[serde_as(as = "[_; N]")]
    periodic: [bool; N],

    /// The set of all space indices, grouped by color.
    space_indices_by_color: Vec<Vec<usize>>,
}

impl<const N: usize> Default for HypercuboidCheckerboard<N> {
    /// Construct a default `HypercuboidCheckerboard`.
    ///
    /// The default is a 2x2x... non-periodic checkerboard with 1.0 x 1.0 x ... spaces.
    #[inline]
    fn default() -> Self {
        Self {
            origin: Cartesian::default(),
            space_width: [1.0; N],
            shape: [2; N],
            periodic: [false; N],
            space_indices_by_color: Self::construct_space_indices_by_color([2; N]),
        }
    }
}

impl<const N: usize> Checkerboard<Cartesian<N>> for HypercuboidCheckerboard<N> {
    #[inline]
    fn point_to_space_index(&self, point: &Cartesian<N>) -> Option<usize> {
        let p = *point - self.origin;
        let mut space_multi_index: [i64; N] =
            array::from_fn(|i| (p.coordinates[i] / self.space_width[i]).floor() as i64);

        for (index, shape, periodic) in izip!(&mut space_multi_index, self.shape, self.periodic) {
            // The origin is in the lower left corner of the box and may be up
            // to one space width to the left of simulation boundary. Therefore,
            // negative indices are out of bounds (and should never been seen
            // for wrapped inputs).
            if *index < 0 {
                return None;
            }

            if periodic {
                // In periodic boundaries, the checkerboard spaces end before
                // the right side. The space at the rightmost edge is identical
                // to space 0 to make the checkerboard coloring commensurate
                // with the periodic boundary conditions.
                if *index as usize == shape {
                    *index = 0;
                }
                if *index as usize > shape {
                    return None;
                }
            } else if *index as usize >= shape {
                // In non-periodic boundaries, the checkerboard extends one full
                // space to the right of the simulation boundary so that when
                // it is shifted to the left it will still cover the boundary.
                // Therefore, any points outside the checkerboard shape are
                // invalid.
                return None;
            }
        }

        Some(Self::multi_index_to_index(
            array::from_fn(|i| space_multi_index[i] as usize),
            self.shape,
        ))
    }

    #[inline]
    fn space_indices_by_color(&self) -> &[Vec<usize>] {
        &self.space_indices_by_color
    }

    #[inline]
    fn num_spaces(&self) -> usize {
        self.shape.iter().product()
    }
}

impl<const N: usize> HypercuboidCheckerboard<N> {
    /// Collapse a multi-dimensional index to a single value in `[0, num_spaces]`.
    #[inline]
    fn multi_index_to_index(multi_index: [usize; N], shape: [usize; N]) -> usize {
        let mut index: usize = 0;
        let mut width = 1;

        for i in (0..N).rev() {
            index += multi_index[i] * width;
            width *= shape[i];
        }

        index
    }

    /// Compute the space width and checkerboard shape.
    #[inline]
    fn compute_dimensions(
        edge_lengths: [PositiveReal; N],
        interaction_range: PositiveReal,
        periodic: [bool; N],
    ) -> ([f64; N], [usize; N]) {
        let mut shape_inside: [usize; N] =
            array::from_fn(|i| (edge_lengths[i].get() / interaction_range.get()).floor() as usize);

        assert!(
            shape_inside.iter().all(|n| *n >= 2),
            "body interaction range {interaction_range} is too large for the boundary dimensions {edge_lengths:?}"
        );

        for (width, periodic) in shape_inside.iter_mut().zip(periodic) {
            // In periodic boundaries, the checkerboard must have an even number
            // of spaces on a side and fit entirely within the given boundary.
            // Spaces must be made larger to accommodate.
            if periodic && !width.is_multiple_of(2) {
                *width -= 1;
            }

            // In non-periodic boundaries, the checkerboard must have an even
            // number of spaces on a side with exactly 1 full space outside
            // the boundary. Therefore, there must be an odd number of spaces
            // inside.
            if !periodic && width.is_multiple_of(2) {
                *width -= 1;
            }
        }

        let space_width = array::from_fn(|i| edge_lengths[i].get() / shape_inside[i] as f64);
        let shape = array::from_fn(|i| {
            if periodic[i] {
                shape_inside[i]
            } else {
                shape_inside[i] + 1
            }
        });

        (space_width, shape)
    }

    /// Partition the space indices by color.
    #[expect(
        clippy::todo,
        reason = "there are no known use-cases for parallel 4D, 5D, ... simulations at this time"
    )]
    fn construct_space_indices_by_color(shape: [usize; N]) -> Vec<Vec<usize>> {
        for width in shape {
            assert!(width.is_multiple_of(2));
        }

        let mut result = Vec::new();

        if N == 2 {
            for offset_j in 0..=1 {
                for offset_i in 0..=1 {
                    let mut space_indices = Vec::new();
                    let mut multi_index = [0; N];

                    for j in 0..shape[0] / 2 {
                        multi_index[0] = 2 * j + offset_j;
                        for i in 0..shape[1] / 2 {
                            multi_index[1] = 2 * i + offset_i;
                            space_indices.push(Self::multi_index_to_index(multi_index, shape));
                        }
                    }

                    result.push(space_indices);
                }
            }

            return result;
        }

        if N == 3 {
            for offset_k in 0..=1 {
                for offset_j in 0..=1 {
                    for offset_i in 0..=1 {
                        let mut space_indices = Vec::new();
                        let mut multi_index = [0; N];

                        for k in 0..shape[0] / 2 {
                            multi_index[0] = 2 * k + offset_k;
                            for j in 0..shape[1] / 2 {
                                multi_index[1] = 2 * j + offset_j;
                                for i in 0..shape[2] / 2 {
                                    multi_index[2] = 2 * i + offset_i;
                                    space_indices
                                        .push(Self::multi_index_to_index(multi_index, shape));
                                }
                            }
                        }

                        result.push(space_indices);
                    }
                }
            }

            return result;
        }

        todo!("Implement a general method");
    }

    /// Construct a checkerboard with a given origin (for testing).
    #[cfg(test)]
    fn with_fixed_origin(
        interaction_range: PositiveReal,
        edge_lengths: [PositiveReal; N],
        periodic: [bool; N],
    ) -> Self {
        let (space_width, shape) =
            Self::compute_dimensions(edge_lengths, interaction_range, periodic);

        Self {
            space_width,
            shape,
            origin: Cartesian::from(array::from_fn(|i| -edge_lengths[i].get() / 2.0)),
            space_indices_by_color: Self::construct_space_indices_by_color(shape),
            periodic,
        }
    }

    /// Construct a new `HypercuboidCheckerboard`.
    ///
    /// Set `interaction_range` to the largest distance between any two
    /// interacting bodies. `new` will construct a checkerboard that covers
    /// the range `[-edge_lengths[i]/2.0, edge_lengths[i]/2.0)` (respecting
    /// `periodic[i]`) for each dimension `i`.
    #[inline]
    pub fn new<R: Rng + ?Sized>(
        rng: &mut R,
        interaction_range: PositiveReal,
        edge_lengths: [PositiveReal; N],
        periodic: [bool; N],
    ) -> Self {
        let (space_width, shape) =
            Self::compute_dimensions(edge_lengths, interaction_range, periodic);

        let offset: [f64; N] = array::from_fn(|_| rng.random());

        let origin = Cartesian {
            coordinates: array::from_fn(|i| {
                -edge_lengths[i].get() / 2.0 - offset[i] * space_width[i]
            }),
        };

        Self {
            space_width,
            shape,
            origin,
            space_indices_by_color: Self::construct_space_indices_by_color(shape),
            periodic,
        }
    }

    /// Update an existing checkerboard.
    ///
    /// `update` performs the same steps as `new`, but modifies `self` in place.
    /// Prefer `update` when possible, as it can reuse the space index partitioning
    /// when the shape doesn't change.
    #[inline]
    pub fn update<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        interaction_range: PositiveReal,
        edge_lengths: [PositiveReal; N],
        periodic: [bool; N],
    ) {
        let (space_width, shape) =
            Self::compute_dimensions(edge_lengths, interaction_range, periodic);

        let offset: [f64; N] = array::from_fn(|_| rng.random());

        let origin = Cartesian {
            coordinates: array::from_fn(|i| {
                -edge_lengths[i].get() / 2.0 - offset[i] * space_width[i]
            }),
        };

        if shape != self.shape || self.space_indices_by_color.is_empty() {
            self.space_indices_by_color = Self::construct_space_indices_by_color(shape);
            self.shape = shape;
        }

        self.space_width = space_width;
        self.origin = origin;
        self.periodic = periodic;
    }
}

impl<const N: usize> Cover<Cartesian<N>> for Closed<Hypercuboid<N>> {
    type Checkerboard = HypercuboidCheckerboard<N>;

    #[inline]
    fn cover<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        interaction_range: PositiveReal,
    ) -> Self::Checkerboard {
        HypercuboidCheckerboard::new(rng, interaction_range, self.0.edge_lengths, [false; N])
    }

    #[inline]
    fn cover_into<R: Rng + ?Sized>(
        &self,
        checkerboard: &mut Self::Checkerboard,
        rng: &mut R,
        interaction_range: PositiveReal,
    ) {
        checkerboard.update(rng, interaction_range, self.0.edge_lengths, [false; N]);
    }
}

impl<const N: usize> Cover<Cartesian<N>> for Periodic<Hypercuboid<N>> {
    type Checkerboard = HypercuboidCheckerboard<N>;

    #[inline]
    fn cover<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        interaction_range: PositiveReal,
    ) -> Self::Checkerboard {
        HypercuboidCheckerboard::new(rng, interaction_range, self.shape().edge_lengths, [true; N])
    }

    #[inline]
    fn cover_into<R: Rng + ?Sized>(
        &self,
        checkerboard: &mut Self::Checkerboard,
        rng: &mut R,
        interaction_range: PositiveReal,
    ) {
        checkerboard.update(rng, interaction_range, self.shape().edge_lengths, [true; N]);
    }
}

#[cfg(test)]
mod tests {
    use assert2::{assert, check};
    use rand::{SeedableRng, rngs::StdRng};
    use rstest::*;

    use super::*;

    #[test]
    fn test_multi_index_to_index_2d() {
        let shape = [12, 4];
        check!(HypercuboidCheckerboard::<2>::multi_index_to_index([0, 0], shape) == 0);
        check!(HypercuboidCheckerboard::<2>::multi_index_to_index([0, 1], shape) == 1);
        check!(HypercuboidCheckerboard::<2>::multi_index_to_index([0, 2], shape) == 2);
        check!(HypercuboidCheckerboard::<2>::multi_index_to_index([0, 3], shape) == 3);
        check!(HypercuboidCheckerboard::<2>::multi_index_to_index([1, 0], shape) == 4);
        check!(HypercuboidCheckerboard::<2>::multi_index_to_index([1, 1], shape) == 5);
        check!(HypercuboidCheckerboard::<2>::multi_index_to_index([1, 2], shape) == 6);
        check!(HypercuboidCheckerboard::<2>::multi_index_to_index([1, 3], shape) == 7);
        check!(HypercuboidCheckerboard::<2>::multi_index_to_index([11, 3], shape) == 47);
    }

    #[test]
    fn test_multi_index_to_index_3d() {
        let shape = [12, 4, 6];
        check!(HypercuboidCheckerboard::<3>::multi_index_to_index([0, 0, 0], shape) == 0);
        check!(HypercuboidCheckerboard::<3>::multi_index_to_index([0, 0, 1], shape) == 1);
        check!(HypercuboidCheckerboard::<3>::multi_index_to_index([0, 0, 2], shape) == 2);
        check!(HypercuboidCheckerboard::<3>::multi_index_to_index([0, 0, 3], shape) == 3);
        check!(HypercuboidCheckerboard::<3>::multi_index_to_index([0, 0, 4], shape) == 4);
        check!(HypercuboidCheckerboard::<3>::multi_index_to_index([0, 0, 5], shape) == 5);

        check!(HypercuboidCheckerboard::<3>::multi_index_to_index([0, 1, 0], shape) == 6);
        check!(HypercuboidCheckerboard::<3>::multi_index_to_index([0, 2, 0], shape) == 12);
        check!(HypercuboidCheckerboard::<3>::multi_index_to_index([0, 3, 0], shape) == 18);

        check!(HypercuboidCheckerboard::<3>::multi_index_to_index([1, 0, 0], shape) == 24);
        check!(HypercuboidCheckerboard::<3>::multi_index_to_index([2, 0, 0], shape) == 48);
        check!(HypercuboidCheckerboard::<3>::multi_index_to_index([3, 3, 2], shape) == 92);
    }

    #[test]
    fn test_compute_dimensions_exact() -> anyhow::Result<()> {
        let (space_width, shape) = HypercuboidCheckerboard::<2>::compute_dimensions(
            [16.0.try_into()?, 24.0.try_into()?],
            2.0.try_into()?,
            [true; 2],
        );
        check!(space_width == [2.0, 2.0]);
        check!(shape == [8, 12]);

        let (space_width, shape) = HypercuboidCheckerboard::<2>::compute_dimensions(
            [14.0.try_into()?, 22.0.try_into()?],
            2.0.try_into()?,
            [false; 2],
        );
        check!(space_width == [2.0, 2.0]);
        check!(shape == [8, 12]);

        let (space_width, shape) = HypercuboidCheckerboard::<2>::compute_dimensions(
            [16.0.try_into()?, 22.0.try_into()?],
            2.0.try_into()?,
            [true, false],
        );
        check!(space_width == [2.0, 2.0]);
        check!(shape == [8, 12]);

        Ok(())
    }

    #[test]
    fn test_compute_dimensions_expand() -> anyhow::Result<()> {
        let (space_width, shape) = HypercuboidCheckerboard::<2>::compute_dimensions(
            [15.0.try_into()?, 23.0.try_into()?],
            1.0.try_into()?,
            [true; 2],
        );
        check!(space_width == [15.0 / 14.0, 23.0 / 22.0]);
        check!(shape == [14, 22]);

        let (space_width, shape) = HypercuboidCheckerboard::<2>::compute_dimensions(
            [14.0.try_into()?, 22.0.try_into()?],
            1.0.try_into()?,
            [false; 2],
        );
        check!(space_width == [14.0 / 13.0, 22.0 / 21.0]);
        check!(shape == [14, 22]);

        Ok(())
    }

    #[test]
    fn test_construct_space_indices_by_color_2d() {
        let space_indices_by_color =
            HypercuboidCheckerboard::<2>::construct_space_indices_by_color([6, 4]);

        assert!(space_indices_by_color.len() == 4);
        assert!(space_indices_by_color[0].len() == 6);
        check!(space_indices_by_color[0][0] == 0);
        check!(space_indices_by_color[0][1] == 2);
        check!(space_indices_by_color[0][2] == 8);
        check!(space_indices_by_color[0][3] == 10);
        check!(space_indices_by_color[0][4] == 16);
        check!(space_indices_by_color[0][5] == 18);

        assert!(space_indices_by_color[1].len() == 6);
        check!(space_indices_by_color[1][0] == 1);
        check!(space_indices_by_color[1][1] == 3);
        check!(space_indices_by_color[1][2] == 9);
        check!(space_indices_by_color[1][3] == 11);
        check!(space_indices_by_color[1][4] == 17);
        check!(space_indices_by_color[1][5] == 19);

        assert!(space_indices_by_color[2].len() == 6);
        check!(space_indices_by_color[2][0] == 4);
        check!(space_indices_by_color[2][1] == 6);
        check!(space_indices_by_color[2][2] == 12);
        check!(space_indices_by_color[2][3] == 14);
        check!(space_indices_by_color[2][4] == 20);
        check!(space_indices_by_color[2][5] == 22);

        assert!(space_indices_by_color[3].len() == 6);
        check!(space_indices_by_color[3][0] == 5);
        check!(space_indices_by_color[3][1] == 7);
        check!(space_indices_by_color[3][2] == 13);
        check!(space_indices_by_color[3][3] == 15);
        check!(space_indices_by_color[3][4] == 21);
        check!(space_indices_by_color[3][5] == 23);
    }

    #[test]
    fn test_construct_space_indices_by_color_3d_222() {
        let space_indices_by_color =
            HypercuboidCheckerboard::<3>::construct_space_indices_by_color([2, 2, 2]);

        assert!(space_indices_by_color.len() == 8);
        assert!(space_indices_by_color[0].len() == 1);
        check!(space_indices_by_color[0][0] == 0);

        assert!(space_indices_by_color[1].len() == 1);
        check!(space_indices_by_color[1][0] == 1);

        assert!(space_indices_by_color[2].len() == 1);
        check!(space_indices_by_color[2][0] == 2);

        assert!(space_indices_by_color[3].len() == 1);
        check!(space_indices_by_color[3][0] == 3);

        assert!(space_indices_by_color[4].len() == 1);
        check!(space_indices_by_color[4][0] == 4);

        assert!(space_indices_by_color[5].len() == 1);
        check!(space_indices_by_color[5][0] == 5);

        assert!(space_indices_by_color[6].len() == 1);
        check!(space_indices_by_color[6][0] == 6);

        assert!(space_indices_by_color[7].len() == 1);
        check!(space_indices_by_color[7][0] == 7);
    }
    #[test]
    fn test_construct_space_indices_by_color_3d_general() {
        let space_indices_by_color =
            HypercuboidCheckerboard::<3>::construct_space_indices_by_color([2, 6, 4]);

        assert!(space_indices_by_color.len() == 8);
        assert!(space_indices_by_color[0].len() == 6);
        check!(space_indices_by_color[0][0] == 0);
        check!(space_indices_by_color[0][1] == 2);
        check!(space_indices_by_color[0][2] == 8);
        check!(space_indices_by_color[0][3] == 10);
        check!(space_indices_by_color[0][4] == 16);
        check!(space_indices_by_color[0][5] == 18);

        assert!(space_indices_by_color[1].len() == 6);
        check!(space_indices_by_color[1][0] == 1);
        check!(space_indices_by_color[1][1] == 3);
        check!(space_indices_by_color[1][2] == 9);
        check!(space_indices_by_color[1][3] == 11);
        check!(space_indices_by_color[1][4] == 17);
        check!(space_indices_by_color[1][5] == 19);

        assert!(space_indices_by_color[2].len() == 6);
        check!(space_indices_by_color[2][0] == 4);
        check!(space_indices_by_color[2][1] == 6);
        check!(space_indices_by_color[2][2] == 12);
        check!(space_indices_by_color[2][3] == 14);
        check!(space_indices_by_color[2][4] == 20);
        check!(space_indices_by_color[2][5] == 22);

        assert!(space_indices_by_color[3].len() == 6);
        check!(space_indices_by_color[3][0] == 5);
        check!(space_indices_by_color[3][1] == 7);
        check!(space_indices_by_color[3][2] == 13);
        check!(space_indices_by_color[3][3] == 15);
        check!(space_indices_by_color[3][4] == 21);
        check!(space_indices_by_color[3][5] == 23);

        assert!(space_indices_by_color[4].len() == 6);
        check!(space_indices_by_color[4][0] == 24);
        check!(space_indices_by_color[4][1] == 24 + 2);
        check!(space_indices_by_color[4][2] == 24 + 8);
        check!(space_indices_by_color[4][3] == 24 + 10);
        check!(space_indices_by_color[4][4] == 24 + 16);
        check!(space_indices_by_color[4][5] == 24 + 18);

        assert!(space_indices_by_color[5].len() == 6);
        check!(space_indices_by_color[5][0] == 24 + 1);
        check!(space_indices_by_color[5][1] == 24 + 3);
        check!(space_indices_by_color[5][2] == 24 + 9);
        check!(space_indices_by_color[5][3] == 24 + 11);
        check!(space_indices_by_color[5][4] == 24 + 17);
        check!(space_indices_by_color[5][5] == 24 + 19);

        assert!(space_indices_by_color[6].len() == 6);
        check!(space_indices_by_color[6][0] == 24 + 4);
        check!(space_indices_by_color[6][1] == 24 + 6);
        check!(space_indices_by_color[6][2] == 24 + 12);
        check!(space_indices_by_color[6][3] == 24 + 14);
        check!(space_indices_by_color[6][4] == 24 + 20);
        check!(space_indices_by_color[6][5] == 24 + 22);

        assert!(space_indices_by_color[7].len() == 6);
        check!(space_indices_by_color[7][0] == 24 + 5);
        check!(space_indices_by_color[7][1] == 24 + 7);
        check!(space_indices_by_color[7][2] == 24 + 13);
        check!(space_indices_by_color[7][3] == 24 + 15);
        check!(space_indices_by_color[7][4] == 24 + 21);
        check!(space_indices_by_color[7][5] == 24 + 23);
    }

    #[test]
    fn test_point_to_space_index_periodic() -> anyhow::Result<()> {
        let checkerboard = HypercuboidCheckerboard::with_fixed_origin(
            2.0.try_into()?,
            [16.0.try_into()?, 24.0.try_into()?],
            [true; 2],
        );
        check!(checkerboard.space_width == [2.0, 2.0]);
        check!(checkerboard.shape == [8, 12]);

        check!(checkerboard.point_to_space_index(&Cartesian::from([-8.0, -12.1])) == None);
        check!(checkerboard.point_to_space_index(&Cartesian::from([-8.0, -12.0])) == Some(0));
        check!(checkerboard.point_to_space_index(&Cartesian::from([-8.0, -10.0])) == Some(1));
        check!(checkerboard.point_to_space_index(&Cartesian::from([-8.0, -8.0])) == Some(2));
        check!(checkerboard.point_to_space_index(&Cartesian::from([-8.0, -6.0])) == Some(3));
        check!(checkerboard.point_to_space_index(&Cartesian::from([-8.0, 11.9])) == Some(11));
        check!(checkerboard.point_to_space_index(&Cartesian::from([-8.0, 12.0])) == Some(0));

        check!(checkerboard.point_to_space_index(&Cartesian::from([-8.1, 0.0])) == None);
        check!(checkerboard.point_to_space_index(&Cartesian::from([-8.0, 0.0])) == Some(6));
        check!(checkerboard.point_to_space_index(&Cartesian::from([-6.0, 0.0])) == Some(18));
        check!(checkerboard.point_to_space_index(&Cartesian::from([-4.0, 0.0])) == Some(30));
        check!(checkerboard.point_to_space_index(&Cartesian::from([-2.0, 0.0])) == Some(42));
        check!(checkerboard.point_to_space_index(&Cartesian::from([7.9, 0.0])) == Some(90));
        check!(checkerboard.point_to_space_index(&Cartesian::from([8.0, 0.0])) == Some(6));

        check!(checkerboard.point_to_space_index(&Cartesian::from([7.9, 11.9])) == Some(95));

        Ok(())
    }

    #[test]
    fn test_point_to_space_index_nonperiodic() -> anyhow::Result<()> {
        let checkerboard = HypercuboidCheckerboard::with_fixed_origin(
            2.0.try_into()?,
            [14.0.try_into()?, 22.0.try_into()?],
            [false; 2],
        );
        check!(checkerboard.space_width == [2.0, 2.0]);
        check!(checkerboard.shape == [8, 12]);

        check!(checkerboard.point_to_space_index(&Cartesian::from([-7.0, -11.1])) == None);
        check!(checkerboard.point_to_space_index(&Cartesian::from([-7.0, -11.0])) == Some(0));
        check!(checkerboard.point_to_space_index(&Cartesian::from([-7.0, -9.0])) == Some(1));
        check!(checkerboard.point_to_space_index(&Cartesian::from([-7.0, -7.0])) == Some(2));
        check!(checkerboard.point_to_space_index(&Cartesian::from([-7.0, -5.0])) == Some(3));
        check!(checkerboard.point_to_space_index(&Cartesian::from([-7.0, 10.0])) == Some(10));
        check!(checkerboard.point_to_space_index(&Cartesian::from([-7.0, 11.0])) == Some(11));
        check!(checkerboard.point_to_space_index(&Cartesian::from([-7.0, 12.9])) == Some(11));
        check!(checkerboard.point_to_space_index(&Cartesian::from([-7.0, 13.0])) == None);

        check!(checkerboard.point_to_space_index(&Cartesian::from([-7.1, 1.0])) == None);
        check!(checkerboard.point_to_space_index(&Cartesian::from([-7.0, 1.0])) == Some(6));
        check!(checkerboard.point_to_space_index(&Cartesian::from([-5.0, 1.0])) == Some(18));
        check!(checkerboard.point_to_space_index(&Cartesian::from([-3.0, 1.0])) == Some(30));
        check!(checkerboard.point_to_space_index(&Cartesian::from([-1.0, 1.0])) == Some(42));
        check!(checkerboard.point_to_space_index(&Cartesian::from([7.0, 1.0])) == Some(90));
        check!(checkerboard.point_to_space_index(&Cartesian::from([9.0, 1.0])) == None);

        check!(checkerboard.point_to_space_index(&Cartesian::from([8.9, 12.9])) == Some(95));

        Ok(())
    }

    #[rstest]
    fn test_all_points_inside(
        #[values(true, false)] periodic: bool,
        #[values(1, 2, 3, 4, 5, 6, 7)] seed: u64,
    ) -> anyhow::Result<()> {
        const N_SAMPLES: usize = 512;

        let interaction_range = 1.5.try_into()?;
        let edge_lengths: [PositiveReal; 2] = [10.0.try_into()?, 7.0.try_into()?];
        let periodic = [periodic; 2];

        let lower_left =
            Cartesian::from([-edge_lengths[0].get() / 2.0, -edge_lengths[1].get() / 2.0]);
        let upper_right = Cartesian::from([
            edge_lengths[0].get() / 2.0 - 0.1,
            edge_lengths[1].get() / 2.0 - 0.1,
        ]);

        let mut rng = StdRng::seed_from_u64(seed);

        for _ in 0..N_SAMPLES {
            let checkerboard =
                HypercuboidCheckerboard::new(&mut rng, interaction_range, edge_lengths, periodic);

            let boundary = Hypercuboid { edge_lengths };

            for _ in 0..N_SAMPLES {
                let v: Cartesian<2> = rng.sample(&boundary);

                check!(checkerboard.point_to_space_index(&v).is_some());
                check!(checkerboard.point_to_space_index(&lower_left) == Some(0));
                check!(checkerboard.point_to_space_index(&upper_right).is_some());
            }
        }

        Ok(())
    }
}

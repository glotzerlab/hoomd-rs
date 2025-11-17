// Copyright (c) 2024-2025 The Regents of the University of Michigan.
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

use std::array;

use itertools::izip;
use rand::Rng;

use hoomd_utility::valid::PositiveReal;
use hoomd_geometry::shape::Hypercuboid;
use hoomd_microstate::boundary::{Closed, Periodic};
use hoomd_vector::Cartesian;

use super::{Cover, Checkerboard};

// * MakeCheckerboard trait implemented by a boundary
// * HypercuboidCheckerboard type implements these
//   * Give it a `periodic` flag.
//   * When periodic, fit an even number of spaces across each dimension. Adjust the origin up to
//     one space width to the left. Points 1 space to the right of the end are still in the box,
//     but are logically in the 0 space because of periodic boundary conditions.
//   * When not periodic, fit an odd number of spaces in the box and add 1 more to make it even
//     When shifting the origin to the left by up to one space, all possible
//     positions are still covered by the existing spaces.

pub struct HypercuboidCheckerboard<const N: usize> {
    origin: Cartesian<N>,
    space_width: [f64; N],
    shape: [usize; N],
    periodic: [bool; N],
    space_indices_by_color: Vec<Vec<usize>>,
}

impl<const N: usize> Checkerboard<Cartesian<N>> for HypercuboidCheckerboard<N> {

    #[inline]
    fn point_to_space_index(&self, point: &Cartesian<N>) -> Option<usize> {
        let p = *point - self.origin;
        let mut space_multi_index: [i64; N] = array::from_fn(|i| (p.coordinates[i] / self.space_width[i]).floor() as i64);

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


        Some(Self::multi_index_to_index(array::from_fn(|i| space_multi_index[i] as usize), self.shape))
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

    #[inline]
    fn compute_dimensions(edge_lengths: [PositiveReal; N],
        interaction_range: PositiveReal,
        periodic: [bool; N]) -> ([f64; N], [usize; N]) {

        let mut shape_inside: [usize; N] = array::from_fn(|i| (edge_lengths[i].get() / interaction_range.get()).floor() as usize);

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
            }});

        (space_width, shape)
        }

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
                        multi_index[0] = 2*j + offset_j;
                        for i in 0..shape[1] / 2 {
                            multi_index[1] = 2*i + offset_i;
                            space_indices.push(Self::multi_index_to_index(multi_index, shape));
                        }
                    }

                result.push(space_indices);
                }
            }

        return result;
        }

        todo!("Implement a general method");
    }

    #[cfg(test)]
    fn with_fixed_origin(interaction_range: PositiveReal, edge_lengths: [PositiveReal; N], periodic: [bool; N]) -> Self {
        let (space_width, shape) = Self::compute_dimensions(edge_lengths, interaction_range, periodic);

        Self {
            space_width,
            shape,
            origin: Cartesian::from(array::from_fn(|i| -edge_lengths[i].get()/2.0)),
            space_indices_by_color: Self::construct_space_indices_by_color(shape),
            periodic,
        }
    }

    pub fn new<R: Rng + ?Sized>(rng: &mut R,
        interaction_range: PositiveReal,
        edge_lengths: [PositiveReal; N],
        periodic: [bool; N]) -> Self {
        let (space_width, shape) = Self::compute_dimensions(edge_lengths, interaction_range, periodic);

        let mut offset = [0.0; N];
        rng.fill(&mut offset);

        let origin = Cartesian {
            coordinates: array::from_fn(|i| -edge_lengths[i].get() / 2.0 - offset[i] * space_width[i]),
        };
        
        Self {
            space_width,
            shape,
            origin,
            space_indices_by_color: Self::construct_space_indices_by_color(shape),
            periodic,
        }        
    }
}

impl<const N: usize> Cover<Cartesian<N>> for Closed<Hypercuboid<N>> {
    type Checkerboard = HypercuboidCheckerboard<N>;

    #[inline]
    fn cover<R: Rng + ?Sized>(&self, rng: &mut R, interaction_range: PositiveReal) -> Self::Checkerboard {
        HypercuboidCheckerboard::new(rng, interaction_range, self.0.edge_lengths, [false; N])
    }
}

impl<const N: usize> Cover<Cartesian<N>> for Periodic<Hypercuboid<N>> {
    type Checkerboard = HypercuboidCheckerboard<N>;

    #[inline]
    fn cover<R: Rng + ?Sized>(&self, rng: &mut R, interaction_range: PositiveReal) -> Self::Checkerboard {
        HypercuboidCheckerboard::new(rng, interaction_range, self.shape().edge_lengths, [true; N])
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
            [true; 2]);
        check!(space_width == [2.0, 2.0]);
        check!(shape == [8, 12]);

        let (space_width, shape) = HypercuboidCheckerboard::<2>::compute_dimensions(
            [14.0.try_into()?, 22.0.try_into()?],
            2.0.try_into()?,
            [false; 2]);
        check!(space_width == [2.0, 2.0]);
        check!(shape == [8, 12]);

        let (space_width, shape) = HypercuboidCheckerboard::<2>::compute_dimensions(
            [16.0.try_into()?, 22.0.try_into()?],
            2.0.try_into()?,
            [true, false]);
        check!(space_width == [2.0, 2.0]);
        check!(shape == [8, 12]);

        Ok(())
    }

    #[test]
    fn test_compute_dimensions_expand() -> anyhow::Result<()> {
        let (space_width, shape) = HypercuboidCheckerboard::<2>::compute_dimensions(
            [15.0.try_into()?, 23.0.try_into()?],
            1.0.try_into()?,
            [true; 2]);
        check!(space_width == [15.0/14.0, 23.0/22.0]);
        check!(shape == [14, 22]);

        let (space_width, shape) = HypercuboidCheckerboard::<2>::compute_dimensions(
            [14.0.try_into()?, 22.0.try_into()?],
            1.0.try_into()?,
            [false; 2]);
        check!(space_width == [14.0/13.0, 22.0/21.0]);
        check!(shape == [14, 22]);

        Ok(())
    }

    #[test]
    fn test_construct_space_indices_by_color_2d() {
        let space_indices_by_color = HypercuboidCheckerboard::<2>::construct_space_indices_by_color([6, 4]);

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
    fn test_point_to_space_index_periodic() -> anyhow::Result<()> {
        let checkerboard = HypercuboidCheckerboard::with_fixed_origin(
            2.0.try_into()?,
            [16.0.try_into()?, 24.0.try_into()?],
            [true; 2]);
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
            [false; 2]);
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
        #[values(1, 2, 3, 4, 5, 6, 7)] seed: u64) -> anyhow::Result<()> {

        const N_SAMPLES: usize = 512;

        let interaction_range = 1.5.try_into()?;
        let edge_lengths: [PositiveReal; _] = [10.0.try_into()?,
                            7.0.try_into()?];
        let periodic = [periodic; 2];

        let lower_left = Cartesian::from([-edge_lengths[0].get() / 2.0, -edge_lengths[1].get() / 2.0]);
        let upper_right = Cartesian::from([edge_lengths[0].get() / 2.0 - 0.1, edge_lengths[1].get() / 2.0 - 0.1]);

        let mut rng = StdRng::seed_from_u64(seed);

        for _ in 0..N_SAMPLES {
            let checkerboard = HypercuboidCheckerboard::new(
                &mut rng,
                interaction_range,
                edge_lengths,
                periodic);

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

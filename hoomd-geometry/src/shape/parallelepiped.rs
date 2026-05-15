// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use hoomd_linear_algebra::{
    GeneralMatrix, SquareMatrix,
    matrix::{Matrix, qr},
};
// use crate::{BoundingSphereRadius, SupportMapping, Volume};
use hoomd_vector::{Cartesian, InnerProduct};

use crate::{IsPointInside, SupportMapping};

#[derive(Clone, Debug, PartialEq)]
pub struct Hyperparallelepiped<const N: usize> {
    /// The lengths of each edge of the cuboid.
    pub edge_vectors: [Cartesian<N>; N],
    pub _qr: Option<Matrix<N, N>>,
}

pub type Parallelogram = Hyperparallelepiped<2>;
pub type Parallelepiped = Hyperparallelepiped<3>;

impl<const N: usize> Default for Hyperparallelepiped<N> {
    #[inline]
    fn default() -> Self {
        Self {
            edge_vectors: std::array::from_fn(|i| {
                std::array::from_fn(|j| if i == j { 1. } else { 0. }).into()
            }),
            _qr: None,
        }
    }
}

impl<const N: usize> Hyperparallelepiped<N> {
    /// Construct a new hyperparallelepiped (parallelotope) from edge vectors.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Hyperparallelepiped;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let parallelepiped = Hyperparallelepiped::new([
    ///     Cartesian::from([10.0, 0.0, 0.0]),
    ///     Cartesian::from([0.0, 12.0, 0.0]),
    ///     Cartesian::from([0.0, 0.0, 14.0]),
    /// ]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(edge_vectors: [Cartesian<N>; N]) -> Self {
        Self {
            edge_vectors,
            _qr: None,
        }
    }

    pub fn calc_qr(mut self) {
        // bundle up vtors
        self._qr = Some(
            Matrix::<N, N> {
                rows: self.edge_vectors.map(|v| v.coordinates),
            }
            .transpose(),
        );
    }

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
        (0.5 * self
            .edge_vectors
            .iter()
            .fold(Cartesian::<N>::default(), |acc, v| v.map(f64::abs) + acc))
        .into()
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
        self.maximal_extents().map(|x| -x)
    }
}

impl<const N: usize> SupportMapping<Cartesian<N>> for Hyperparallelepiped<N> {
    #[inline]
    fn support_mapping(&self, direction: &Cartesian<N>) -> Cartesian<N> {
        0.5 * self
            .edge_vectors
            .iter()
            .fold(Cartesian::<N>::default(), |acc, v| {
                v.dot(direction).signum() * *v + acc
            })
    }
}

impl<const N: usize> IsPointInside<Cartesian<N>> for Hyperparallelepiped<N> {
    /// Check if a cartesian vector is inside a hyperparallelepiped.
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
    /// use hoomd_geometry::{IsPointInside, shape::Hyperparallelepiped};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let cuboid = Hyperparallelepiped {
    ///     // TODO
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
        let fractional = qr::qr_solve(
            &self._qr.as_ref().expect("_qr attribute is not computed"),
            point.to_column_matrix(),
        );

        fractional
            .rows
            .into_iter()
            .all(|x| -1.0 / 2.0 <= x[0] && x[0] < 1.0 / 2.0)
    }
}

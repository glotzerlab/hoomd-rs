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

//! Implement `SphericalVecCell`

use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::{array, cmp::Eq, f64::consts::PI, fmt, hash::Hash};

use hoomd_manifold::Spherical;
use hoomd_utility::valid::PositiveReal;

use super::{PointUpdate, PointsNearBall, WithSearchRadius};

use crate::{
    IndexFromPosition,
    vec_cell::{PointsIterator, VecCell, VecCellBuilder},
};

/// Implement [`VecCell`] for [`Spherical`] bodies.
///
/// `Spherical<N>` bodies exist on the surface of an $`(N-1)`$-sphere embedded
/// in $`N`$-dimensional Cartesian space. As such, methods for `VecCell<K, N>`
/// can be adapted to work for `Spherical<N>`.
///
/// `SphericalVecCell` only differs from `VecCell` in that the user may choose
/// to use either the `Spherical` geodesic distance or the Euclidean
/// "line-of-site" distance when specifying nominal and maximum search radii.
///
/// [`VecCell`]: hoomd_spatial::VecCell;
/// [`Spherical`]: hoomd_manifold::Spherical;
#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SphericalVecCell<K: Eq + Hash, const D: usize>(pub VecCell<K, D>);

/// Construct a [`SphericalVecCell`] with given parameters.
///
/// # Example
///
/// ```
/// use hoomd_manifold::Spherical;
/// use hoomd_spatial::SphericalVecCell;
/// use std::f64::consts::PI;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let two_sphere_vec_cell = SphericalVecCell::<usize, 3>::builder()
///     .spherical_nominal_search_radius((PI / 12.0).try_into()?)
///     .spherical_maximum_search_radius(PI / 4.0)
///     .build();
/// # Ok(())
/// # }
/// ```
pub struct SphericalVecCellBuilder<K, const D: usize>(pub VecCellBuilder<K, D>);

impl<K, const D: usize> SphericalVecCellBuilder<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Choose the search radius from a given `Spherical` search radius.
    ///
    /// The nominal search radius is the edge length of each hypercube in
    /// `SphericalVecCell`. Note that `Spherical` geodesic distances are always larger
    /// than the euclidean "line-of-sight" distance.
    ///
    /// # Panics
    /// Method will panic if `spherical_nominal_search_radius` is larger than $`\pi/2`$.
    #[inline]
    #[must_use]
    pub fn spherical_nominal_search_radius(
        mut self,
        spherical_nominal_search_radius: PositiveReal,
    ) -> Self {
        self.0.nominal_search_radius = (spherical_nominal_search_radius.get().clamp(0.0, PI / 2.0))
            .sin()
            .try_into()
            .expect("clamp ensures number is positive");
        self
    }

    /// Choose the largest search radius from a given `Spherical` search radius.
    ///
    /// As in `VecCell`, the maximum radius is rounded up to the nearest
    /// integer multiple of the nominal search radius. [`SphericalVecCell`]
    /// will panic when asked to search for points within a radius larger
    /// than the maximum.
    #[inline]
    #[must_use]
    pub fn spherical_maximum_search_radius(mut self, spherical_maximum_search_radius: f64) -> Self {
        self.0.maximum_search_radius = (spherical_maximum_search_radius.clamp(0.0, PI / 2.0)).sin();
        self
    }

    /// Choose the search radius from a given Euclidean search radius.
    #[inline]
    #[must_use]
    pub fn euclidean_nominal_search_radius(
        mut self,
        euclidean_nominal_search_radius: PositiveReal,
    ) -> Self {
        self.0.nominal_search_radius = euclidean_nominal_search_radius;
        self
    }

    /// Choose the largest search radius from a given Euclidean search radius.
    ///
    /// As in `VecCell`, the maximum radius is rounded up to the nearest
    /// integer multiple of the nominal search radius. [`SphericalVecCell`]
    /// will panic when asked to search for points within a radius larger
    /// than the maximum.
    #[inline]
    #[must_use]
    pub fn euclidean_maximum_search_radius(mut self, euclidean_maximum_search_radius: f64) -> Self {
        self.0.maximum_search_radius = euclidean_maximum_search_radius;
        self
    }

    /// Construct the [`SphericalVecCell`] with the chosen parameters.
    #[inline]
    #[must_use]
    pub fn build(self) -> SphericalVecCell<K, D> {
        SphericalVecCell(VecCellBuilder::<K, D>::build(self.0))
    }
}

impl<K, const D: usize> Default for SphericalVecCell<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Construct a default [`SphericalVecCell`].
    ///
    /// The default sets both the nominal and maximum search to 1.0.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_spatial::SphericalVecCell;
    ///
    /// let two_sphere_vec_cell = SphericalVecCell::<usize, 3>::default();
    /// ```
    #[inline]
    fn default() -> Self {
        Self::builder().build()
    }
}

impl<K, const D: usize> WithSearchRadius for SphericalVecCell<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Construct a [`SphericalVecCell`] with a given search radius in the
    /// `Spherical` metric.
    ///
    /// Both the nominal and maximum search radii are set to `spherical_radius`
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_manifold::Spherical;
    /// use hoomd_spatial::{SphericalVecCell, WithSearchRadius};
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let two_sphere_vec_cell = SphericalVecCell::<usize, 3>::with_search_radius(
    ///     (PI / 2.0).try_into()?,
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn with_search_radius(radius: PositiveReal) -> Self {
        let euclidean_radius = (radius.get().clamp(0.0, PI / 2.0)).sin();
        Self::builder()
            .euclidean_nominal_search_radius(
                PositiveReal::try_from(euclidean_radius)
                    .expect("positive number given previous check"),
            )
            .euclidean_maximum_search_radius(euclidean_radius)
            .build()
    }
}

impl<K, const D: usize> SphericalVecCell<K, D>
where
    K: Eq + Hash,
{
    /// Get the keys in a given cell index
    #[cfg(test)]
    #[inline]
    fn get_keys(&self, cell_index: &[i64; D]) -> &[K] {
        VecCell::<K, D>::get_keys(&self.0, cell_index)
    }
}

impl<K, const D: usize> SphericalVecCell<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Construct a `SphericalVecCell` builder.
    ///
    /// Use the builder to set any or all parameters and construct a [`SphericalVecCell`].
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_manifold::Spherical;
    /// use hoomd_spatial::SphericalVecCell;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let two_sphere_vec_cell = SphericalVecCell::<usize, 3>::builder()
    ///     .euclidean_nominal_search_radius(0.2.try_into()?)
    ///     .euclidean_maximum_search_radius(0.4)
    ///     .build();
    /// # Ok(())
    /// # }
    /// ```
    #[expect(
        clippy::missing_panics_doc,
        reason = "hard-coded constant will never panic"
    )]
    #[inline]
    #[must_use]
    pub fn builder() -> SphericalVecCellBuilder<K, D> {
        SphericalVecCellBuilder(VecCell::<K, D>::builder())
    }

    /// Remove excess capacity from dynamically allocated arrays.
    ///
    /// At this time, `shrink_to_fit` only reduces the memory utilized by the
    /// cell contents. It does not shrink the `D`-dimensional storage to match
    /// the range spanned by points currently in the data structure.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        VecCell::<K, D>::shrink_to_fit(&mut self.0);
    }
}

impl<K, const D: usize> PointUpdate<Spherical<D>, K> for SphericalVecCell<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Insert or update a point identified by a key.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Spherical;
    /// use hoomd_spatial::{PointUpdate, SphericalVecCell};
    /// use std::f64::consts::PI;
    ///
    /// let mut spherical_vec_cell = SphericalVecCell::<usize, 4>::default();
    ///
    /// spherical_vec_cell.insert(
    ///     0,
    ///     Spherical::<4>::from_polar_coordinates(
    ///         PI / 4.0,
    ///         PI / 2.0,
    ///         3.0 * PI / 2.0,
    ///     ),
    /// );
    /// ```
    #[inline]
    fn insert(&mut self, key: K, position: Spherical<D>) {
        VecCell::<K, D>::insert(&mut self.0, key, *position.point())
    }

    /// Remove the point with the given key.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Spherical;
    /// use hoomd_spatial::{PointUpdate, SphericalVecCell};
    /// use std::f64::consts::PI;
    ///
    /// let mut spherical_vec_cell = SphericalVecCell::<usize, 4>::default();
    /// spherical_vec_cell.insert(
    ///     0,
    ///     Spherical::<4>::from_polar_coordinates(PI / 2.0, PI / 4.0, PI),
    /// );
    ///
    /// spherical_vec_cell.remove(&0)
    /// ```
    #[inline]
    fn remove(&mut self, key: &K) {
        VecCell::<K, D>::remove(&mut self.0, key);
    }

    /// Get the number of points in the spatial data structure.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Spherical;
    /// use hoomd_spatial::{PointUpdate, SphericalVecCell};
    /// use std::f64::consts::PI;
    ///
    /// let mut spherical_vec_cell = SphericalVecCell::<usize, 3>::default();
    /// spherical_vec_cell
    ///     .insert(0, Spherical::<3>::from_polar_coordinates(0.0, 0.0));
    /// spherical_vec_cell
    ///     .insert(1, Spherical::<3>::from_polar_coordinates(PI / 4.0, 0.0));
    ///
    /// assert_eq!(spherical_vec_cell.len(), 2)
    /// ```
    #[inline]
    fn len(&self) -> usize {
        self.0.cell_index.len()
    }

    /// Test if the spatial data structure is empty.
    ///
    /// # Example
    /// ```
    /// use hoomd_spatial::{PointUpdate, SphericalVecCell};
    ///
    /// let mut spherical_vec_cell = SphericalVecCell::<usize, 4>::default();
    ///
    /// assert!(spherical_vec_cell.is_empty());
    /// ```
    #[inline]
    fn is_empty(&self) -> bool {
        self.0.cell_index.is_empty()
    }

    /// Test if the spatial data structure contains a key.
    /// ```
    /// use hoomd_manifold::Spherical;
    /// use hoomd_spatial::{PointUpdate, SphericalVecCell};
    /// use std::f64::consts::PI;
    ///
    /// let mut spherical_vec_cell = SphericalVecCell::<usize, 3>::default();
    /// spherical_vec_cell.insert(
    ///     0,
    ///     Spherical::<3>::from_polar_coordinates(3.0 * PI / 5.0, 0.0),
    /// );
    ///
    /// assert!(spherical_vec_cell.contains_key(&0));
    /// ```
    #[inline]
    fn contains_key(&self, key: &K) -> bool {
        self.0.cell_index.contains_key(key)
    }

    /// Remove all points.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Spherical;
    /// use hoomd_spatial::{PointUpdate, SphericalVecCell};
    /// use std::f64::consts::PI;
    ///
    /// let mut spherical_vec_cell = SphericalVecCell::<usize, 4>::default();
    /// spherical_vec_cell.insert(
    ///     0,
    ///     Spherical::<4>::from_polar_coordinates(PI / 4.0, PI / 2.0, 0.0),
    /// );
    ///
    /// spherical_vec_cell.clear();
    /// ```
    #[inline]
    fn clear(&mut self) {
        self.0.clear()
    }
}

impl<K, const D: usize> Iterator for PointsIterator<'_, K, D, SphericalVecCell<K, D>>
where
    K: Copy + Eq + Hash,
{
    type Item = K;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(keys) = self.keys
                && self.index_in_current_cell < keys.len()
            {
                let last_index = self.index_in_current_cell;
                self.index_in_current_cell += 1;
                return Some(keys[last_index]);
            }

            self.index_in_current_cell = 0;
            self.current_stencil += 1;

            if self.current_stencil >= self.stencil.len() {
                return None;
            }

            let cell_index =
                array::from_fn(|i| self.center[i] + self.stencil[self.current_stencil][i]);
            let map_index =
                VecCell::<K, D>::map_index_from_cell(self.cell_list.0.half_extent, &cell_index);
            self.keys = map_index.map(|index| &self.cell_list.0.keys_map[index]);
        }
    }
}

impl<const D: usize, K> PointsNearBall<Spherical<D>, K> for SphericalVecCell<K, D>
where
    K: Copy + Eq + Hash,
{
    /// Find all the points that *might* be in the given ball with a specified
    /// radius in the `Spherical` distance metric.
    ///
    /// `points_near_ball` will iterate over all points in the given ball *and
    /// possibly others as well*. [`SphericalVecCell`] may iterate over the points in
    /// any order.
    ///
    /// # Example
    /// ```
    /// use hoomd_manifold::Spherical;
    /// use hoomd_spatial::{PointUpdate, PointsNearBall, SphericalVecCell};
    /// use std::f64::consts::PI;
    ///
    /// let mut spherical_vec_cell = SphericalVecCell::<usize, 3>::default();
    /// spherical_vec_cell
    ///     .insert(0, Spherical::<3>::from_polar_coordinates(PI / 12.0, 0.0));
    /// spherical_vec_cell
    ///     .insert(1, Spherical::<3>::from_polar_coordinates(PI / 12.0, PI));
    /// spherical_vec_cell.insert(
    ///     2,
    ///     Spherical::<3>::from_polar_coordinates(2.0 * PI / 3.0, 0.0),
    /// );
    ///
    /// for key in spherical_vec_cell.points_near_ball(
    ///     &Spherical::<3>::from_cartesian_coordinates([0.0, 0.0, 1.0].into()),
    ///     PI / 4.0,
    /// ) {
    ///     println!("{key}");
    /// }
    /// ```
    /// Prints (in any order):
    /// ```text
    /// 0
    /// 1
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when `radius` is larger than the *maximum search radius*
    /// provided at construction, rounded up to the nearest integer multiple
    /// of the *nominal search radius*.
    #[inline]
    fn points_near_ball(&self, position: &Spherical<D>, radius: f64) -> impl Iterator<Item = K> {
        // convert spherical distance to Euclidean distance
        let euclidean_radius = radius.asin();
        VecCell::<K, D>::points_near_ball(&self.0, position.point(), euclidean_radius)
    }
}

impl<K, const D: usize> fmt::Display for SphericalVecCell<K, D>
where
    K: Eq + Hash,
{
    /// Summarize the contents of the cell list.
    ///
    /// This is a slow operation. It is meant to be printed to logs only
    /// occasionally, such as at the end of a benchmark or simulation.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_spatial::SphericalVecCell;
    /// use log::info;
    ///
    /// let spherical_vec_cell = SphericalVecCell::<usize, 3>::default();
    ///
    /// info!("{spherical_vec_cell}");
    /// ```
    #[allow(
        clippy::missing_inline_in_public_items,
        reason = "no need to inline display"
    )]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        VecCell::<K, D>::fmt(&self.0, f)
    }
}

impl<K, const D: usize> IndexFromPosition<Spherical<D>> for SphericalVecCell<K, D>
where
    K: Eq + Hash,
{
    type Location = [i64; D];

    #[inline]
    fn location_from_position(&self, position: &Spherical<D>) -> Self::Location {
        self.0.cell_index_from_position(position.point())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash_cell::CellIndex;
    use hoomd_manifold::{Spherical, SphericalDisk};
    use rand::{
        RngExt, SeedableRng,
        distr::{Distribution, Uniform},
        rngs::StdRng,
    };
    use rstest::*;

    use approxim::assert_relative_eq;
    use assert2::{assert, check};
    use rustc_hash::FxHashMap;
    use std::f64::consts::PI;

    #[test]
    fn two_sphere_cell_index() {
        let spherical_cell_list = SphericalVecCell::<usize, 3>::builder()
            .euclidean_nominal_search_radius(0.5.try_into().expect("hard-coded positive number"))
            .build();
        check!(
            spherical_cell_list.0.cell_index_from_position(
                &Spherical::<3>::from_polar_coordinates(PI / 2.0, 0.0).point()
            ) == [2, 0, 0]
        );
        check!(
            spherical_cell_list.0.cell_index_from_position(
                &Spherical::<3>::from_polar_coordinates(PI / 2.0, PI / 2.0).point()
            ) == [0, 2, 0]
        );
        check!(
            spherical_cell_list.0.cell_index_from_position(
                &Spherical::<3>::from_polar_coordinates(0.0, 0.0).point()
            ) == [0, 0, 2]
        );
    }

    #[test]
    fn three_sphere_cell_index() {
        let spherical_cell_list = SphericalVecCell::<usize, 4>::builder()
            .euclidean_nominal_search_radius(0.5.try_into().expect("hard-coded positive number"))
            .build();
        check!(
            spherical_cell_list.0.cell_index_from_position(
                &Spherical::<4>::from_polar_coordinates(PI / 2.0, 0.0, 0.0).point()
            ) == [2, 0, 0, 0]
        );
        check!(
            spherical_cell_list.0.cell_index_from_position(
                &Spherical::<4>::from_polar_coordinates(PI / 2.0, PI / 2.0, 0.0).point()
            ) == [0, 2, 0, 0]
        );
        check!(
            spherical_cell_list.0.cell_index_from_position(
                &Spherical::<4>::from_polar_coordinates(PI / 2.0, PI / 2.0, PI / 2.0).point()
            ) == [0, 0, 2, 0]
        );
        check!(
            spherical_cell_list.0.cell_index_from_position(
                &Spherical::<4>::from_polar_coordinates(0.0, 0.0, 0.0).point()
            ) == [0, 0, 0, 2]
        );
    }

    #[test]
    fn two_sphere_geodesic_search_radius() {
        let spherical_cell_list = SphericalVecCell::<usize, 3>::with_search_radius(
            (PI / 2.0).try_into().expect("hard-coded positive number"),
        );
        assert!(spherical_cell_list.0.cell_width.get() == 1.0);

        let spherical_cell_list = SphericalVecCell::<usize, 3>::with_search_radius(
            (PI / 4.0).try_into().expect("hard-coded positive number"),
        );
        assert_relative_eq!(
            spherical_cell_list.0.cell_width.get(),
            (0.5_f64).sqrt(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn three_sphere_geodesic_search_radius() {
        let spherical_cell_list = SphericalVecCell::<usize, 4>::with_search_radius(
            (PI / 2.0).try_into().expect("hard-coded positive number"),
        );
        assert!(spherical_cell_list.0.cell_width.get() == 1.0);

        let spherical_cell_list = SphericalVecCell::<usize, 4>::with_search_radius(
            (PI / 4.0).try_into().expect("hard-coded positive number"),
        );
        assert_relative_eq!(
            spherical_cell_list.0.cell_width.get(),
            (0.5_f64).sqrt(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn two_sphere_insert() {
        let mut sphere_cell_list = SphericalVecCell::<usize, 3>::default();

        sphere_cell_list.insert(0, Spherical::<3>::from_polar_coordinates(PI / 2.0, 0.0));
        sphere_cell_list.insert(
            1,
            Spherical::<3>::from_polar_coordinates(PI / 2.0 - 0.1, PI),
        );
        sphere_cell_list.insert(2, Spherical::<3>::from_polar_coordinates(PI / 4.0, 0.0));
        sphere_cell_list.insert(
            3,
            Spherical::<3>::from_polar_coordinates(PI / 2.0, 3.0 * PI / 2.0 + 0.001),
        );
        sphere_cell_list.insert(
            4,
            Spherical::<3>::from_polar_coordinates(PI / 4.0, 3.0 * PI / 2.0 + 0.001),
        );

        assert!(sphere_cell_list.0.cell_index.get(&0) == Some(&CellIndex([1, 0, 0])));
        assert!(sphere_cell_list.0.cell_index.get(&1) == Some(&CellIndex([-1, 0, 0])));
        assert!(sphere_cell_list.0.cell_index.get(&2) == Some(&CellIndex([0, 0, 0])));
        assert!(sphere_cell_list.0.cell_index.get(&3) == Some(&CellIndex([0, -1, 0])));
        assert!(sphere_cell_list.0.cell_index.get(&4) == Some(&CellIndex([0, -1, 0])));

        let keys = sphere_cell_list.get_keys(&[0, 0, 0]);
        assert!(keys.len() == 1);
        check!(keys.contains(&2));

        let keys = sphere_cell_list.get_keys(&[1, 0, 0]);
        assert!(keys.len() == 1);
        check!(keys.contains(&0));

        let keys = sphere_cell_list.get_keys(&[-1, 0, 0]);
        assert!(keys.len() == 1);
        check!(keys.contains(&1));

        let keys = sphere_cell_list.get_keys(&[0, -1, 0]);
        assert!(keys.len() == 2);
        check!(keys.contains(&3));
        check!(keys.contains(&4));
    }

    #[test]
    fn three_sphere_insert() {
        let mut sphere_cell_list = SphericalVecCell::<usize, 4>::default();

        sphere_cell_list.insert(
            0,
            Spherical::<4>::from_polar_coordinates(PI / 2.0 - 0.1, 0.0, 0.0),
        );
        sphere_cell_list.insert(
            1,
            Spherical::<4>::from_polar_coordinates(
                PI / 2.0 - 0.1,
                PI / 2.0,
                3.0 * PI / 2.0 + 0.001,
            ),
        );
        sphere_cell_list.insert(
            2,
            Spherical::<4>::from_polar_coordinates(PI / 4.0, PI / 4.0, 3.0 * PI / 2.0),
        );
        sphere_cell_list.insert(
            3,
            Spherical::<4>::from_polar_coordinates(PI / 2.0, 3.0 * PI / 2.0 + 0.001, 0.0),
        );
        sphere_cell_list.insert(
            4,
            Spherical::<4>::from_polar_coordinates(PI / 4.0, 3.0 * PI / 2.0 + 0.001, 0.0),
        );

        assert!(sphere_cell_list.0.cell_index.get(&0) == Some(&CellIndex([0, 0, 0, 0])));
        assert!(sphere_cell_list.0.cell_index.get(&1) == Some(&CellIndex([0, 0, -1, 0])));
        assert!(sphere_cell_list.0.cell_index.get(&2) == Some(&CellIndex([0, -1, -1, 0])));
        assert!(sphere_cell_list.0.cell_index.get(&3) == Some(&CellIndex([-1, 0, 0, 0])));
        assert!(sphere_cell_list.0.cell_index.get(&4) == Some(&CellIndex([-1, 0, 0, 0])));

        let keys = sphere_cell_list.get_keys(&[0, 0, 0, 0]);
        assert!(keys.len() == 1);
        check!(keys.contains(&0));

        let keys = sphere_cell_list.get_keys(&[-1, 0, 0, 0]);
        assert!(keys.len() == 2);
        check!(keys.contains(&3));
        check!(keys.contains(&4));

        let keys = sphere_cell_list.get_keys(&[0, 0, -1, 0]);
        assert!(keys.len() == 1);
        check!(keys.contains(&1));

        let keys = sphere_cell_list.get_keys(&[0, -1, -1, 0]);
        assert!(keys.len() == 1);
        check!(keys.contains(&2));
    }

    #[rstest]
    fn two_sphere_consistency() {
        const N_STEPS: usize = 10_000;
        let mut rng = StdRng::seed_from_u64(0);
        let mut reference = FxHashMap::default();

        let cell_width = 0.2;
        let mut cell_list = SphericalVecCell::<usize, 3>::builder()
            .euclidean_nominal_search_radius(
                cell_width
                    .try_into()
                    .expect("hard-coded cell with should be positive"),
            )
            .build();
        let position_distribution = SphericalDisk {
            disk_radius: (PI / 4.0).try_into().expect("hard-coded positive numbers"),
            point: Spherical::<3>::from_polar_coordinates(0.0, 0.0),
        };
        let key_distribution =
            Uniform::new(0, N_STEPS / 4).expect("hardcoded distribution should be valid");

        for _ in 0..N_STEPS {
            // Add more keys than removing
            if rng.random_bool(0.7) {
                let position: Spherical<3> = position_distribution.sample(&mut rng);
                let key = key_distribution.sample(&mut rng);

                cell_list.insert(key, position);
                reference.insert(key, cell_list.0.cell_index_from_position(&position.point()));
            } else {
                let key = key_distribution.sample(&mut rng);
                cell_list.remove(&key);
                reference.remove(&key);
            }
        }

        // Validate that cell_index contains the expected keys and that
        // keys_map is consistent.
        assert!(cell_list.0.cell_index.len() == reference.len());
        for (reference_key, reference_value) in reference.drain() {
            let value = cell_list.0.cell_index.get(&reference_key);
            check!(value == Some(&CellIndex(reference_value)));

            let keys = cell_list.get_keys(&reference_value);
            check!(keys.contains(&reference_key));
        }

        // Ensure that there are no extra values in keys_map.
        let total = cell_list.0.keys_map.iter().map(Vec::len).sum();
        check!(cell_list.0.cell_index.len() == total);
        check!(total > 300);
    }

    #[rstest]
    fn three_sphere_consistency() {
        const N_STEPS: usize = 10_000;
        let mut rng = StdRng::seed_from_u64(0);
        let mut reference = FxHashMap::default();

        let cell_width = 0.2;
        let mut cell_list = SphericalVecCell::<usize, 4>::builder()
            .euclidean_nominal_search_radius(
                cell_width
                    .try_into()
                    .expect("hard-coded cell with should be positive"),
            )
            .build();
        let position_distribution = SphericalDisk {
            disk_radius: (PI / 4.0).try_into().expect("hard-coded positive numbers"),
            point: Spherical::<4>::from_polar_coordinates(0.0, 0.0, 0.0),
        };
        let key_distribution =
            Uniform::new(0, N_STEPS / 4).expect("hardcoded distribution should be valid");

        for _ in 0..N_STEPS {
            // Add more keys than removing
            if rng.random_bool(0.7) {
                let position: Spherical<4> = position_distribution.sample(&mut rng);
                let key = key_distribution.sample(&mut rng);

                cell_list.insert(key, position);
                reference.insert(key, cell_list.0.cell_index_from_position(&position.point()));
            } else {
                let key = key_distribution.sample(&mut rng);
                cell_list.remove(&key);
                reference.remove(&key);
            }
        }

        // Validate that cell_index contains the expected keys and that
        // keys_map is consistent.
        assert!(cell_list.0.cell_index.len() == reference.len());
        for (reference_key, reference_value) in reference.drain() {
            let value = cell_list.0.cell_index.get(&reference_key);
            check!(value == Some(&CellIndex(reference_value)));

            let keys = cell_list.get_keys(&reference_value);
            check!(keys.contains(&reference_key));
        }

        // Ensure that there are no extra values in keys_map.
        let total = cell_list.0.keys_map.iter().map(Vec::len).sum();
        check!(cell_list.0.cell_index.len() == total);
        check!(total > 300);
    }
}

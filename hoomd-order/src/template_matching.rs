// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! TODO

#![allow(dead_code, reason = "wip")]

use crate::CrossCovariance;
use hoomd_linear_algebra::{GeneralMatrix, MatMul, matrix::Matrix};
use hoomd_vector::{Cartesian, Rotate, RotationMatrix};

/// Store a [`Template`] with origin-centered coordinates and a known center of mass.
///
/// This struct implements ``template_match``, enabling fast determination of similarity
/// between a [`Template`] reference and an slice containing points to match against.
///
/// # Examples
/// ```
/// use hoomd_order::template_matching::Template;
/// use hoomd_vector::{Angle, RotationMatrix, Cartesian};
/// use approx::assert_relative_eq;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Align and measure similarity between two triangles
/// let equilateral = Template::from(vec![[0.0, 0.0], [1.0,0.0], [0.5, f64::sqrt(3.0)/2.0]]);
/// // let rotation = Angle::from(f64::consts::PI / 2.0);
/// let rotated_points = vec![[0.0,0.0],[0.0,1.0], [f64::sqrt(3.0)/2.0, 0.5]];
///
/// // Align the point sets
/// let (rotation_matrix, t, rmsd) = equilateral.template_match(&rotated_points)?;
///
/// // The rotation angle should be π / 2.0
/// assert_relative_eq!(rotation_matrix.to_angle().theta, std::f64::consts::PI / 2., epsilon = 1e-14);
/// // No translation is required to align the points
/// // Therefore, the output should be the same as the center of mass of the input equilateral
/// assert_relative_eq!(t[0], equilateral.center()[0], epsilon=1e-14);
/// assert_relative_eq!(t[1], equilateral.center()[1], epsilon=1e-14);
/// // The shapes are identical save for the rotation
/// assert_relative_eq!(rmsd, 0.0, epsilon=1e-14);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Template<P> {
    /// The coordinates defining the geometry of the template, centered at the origin.
    pub(crate) coordinates: Vec<P>,

    /// The center of mass of the `coordinates`.
    pub(crate) center: P,
}

impl<const N: usize, P> From<Vec<P>> for Template<Cartesian<N>>
where
    P: Into<Cartesian<N>> + Copy,
{
    #[inline]
    fn from(value: Vec<P>) -> Self {
        let centroid = value
            .iter()
            .fold(Cartesian::default(), |acc, &v| acc + v.into())
            / value.len() as f64;
        Self {
            coordinates: value
                .iter()
                .map(|&v| v.into() - centroid)
                .collect::<Vec<_>>(),
            center: centroid,
        }
    }
}

impl<const N: usize, I> CrossCovariance<I, Matrix<N, N>> for Template<Cartesian<N>>
where
    I: ExactSizeIterator<Item = Cartesian<N>>,
{
    /// Compute the cross-covariance between two sets of vectors.
    ///
    /// The result will be `None` if the two sets of points have differing numbers of
    /// points.
    #[inline]
    fn cross_covariance(self, other: I) -> Option<Matrix<N, N>> {
        // TODO: better error?
        if self.coordinates.len() != other.len() {
            return None;
        }
        Some(
            self.coordinates
                .iter()
                .zip(other)
                .fold(Matrix::<N, N>::zeros(), |mut acc, (l, r)| {
                    for i in 0..N {
                        for j in 0..N {
                            acc[(i, j)] += l[i] * r[j];
                        }
                    }
                    acc
                }),
        )
    }
}

/// Compute the root-mean squared deviation between two sets of points.
fn compute_rmsd<const N: usize, I>(test_set: I, reference_set: &[Cartesian<N>]) -> f64
where
    I: IntoIterator<Item = Cartesian<N>>,
{
    test_set
        .into_iter()
        .zip(reference_set.iter())
        .fold(0.0, |acc, (x, &y)| {
            acc + (x - y)
                .coordinates
                .iter()
                .fold(0.0, |sum, p| sum + p.powi(2))
        })
}

impl<const N: usize> Template<Cartesian<N>> {
    /// Get the `center` from a [`Template`]
    /// # Examples
    /// ```
    /// use hoomd_order::template_matching::Template;
    /// let equilateral = Template::from(vec![
    ///     [0.0, 0.0],
    ///     [1.0, 0.0],
    ///     [0.5, f64::sqrt(3.0) / 2.0],
    /// ]);
    /// assert_eq!(equilateral.center(), [0.5, f64::sqrt(3.0) / 6.0].into());
    /// ```
    #[must_use]
    #[inline]
    pub fn center(&self) -> Cartesian<N> {
        self.center
    }
}

impl Template<Cartesian<3>> {
    /// Compute the rotation and translation that optimally align points in `test_set` to a [`Template`].
    ///
    /// # Errors
    ///
    /// Returns [`Err(super::Error::MismatchedPointSetSize)`](super::Error::MismatchedPointSetSize) when `test_set` and `self` have different numbers of points.
    #[inline]
    pub fn template_match<V>(
        &self,
        test_set: &[V],
    ) -> Result<(RotationMatrix<3>, Cartesian<3>, f64), super::Error>
    where
        V: Into<Cartesian<3>> + Copy,
    {
        let test_set_centroid = test_set
            .iter()
            .fold(Cartesian::default(), |acc, &v| acc + v.into())
            / self.coordinates.len() as f64;
        let test_set_centered = test_set.iter().map(|&v| v.into() - test_set_centroid);

        let m = self
            .clone()
            .cross_covariance(test_set_centered)
            .ok_or(super::Error::MismatchedPointSetSize)?;

        let (u, _, vt) = m.svd();

        let r: RotationMatrix<3> = u
            .matmul(&vt)
            .try_into()
            .map_err(|_| super::Error::NonUnitaryMatrix)?;

        let t = r.rotate(&test_set_centroid);

        Ok((
            r,
            t,
            compute_rmsd(
                test_set.iter().map(|&v| r.rotate(&v.into())),
                &self.coordinates,
            ),
        ))
    }
}
impl Template<Cartesian<2>> {
    /// Compute the rotation and translation that optimally align points in `test_set` to a [`Template`].
    ///
    /// # Errors
    ///
    /// Returns [`Err(super::Error::MismatchedPointSetSize)`](super::Error::MismatchedPointSetSize) when `test_set` and `self` have different numbers of points.
    #[inline]
    pub fn template_match<V>(
        &self,
        test_set: &[V],
    ) -> Result<(RotationMatrix<2>, Cartesian<2>, f64), super::Error>
    where
        V: Into<Cartesian<2>> + Copy,
    {
        let test_set_centroid = test_set
            .iter()
            .fold(Cartesian::default(), |acc, &v| acc + v.into())
            / self.coordinates.len() as f64;
        let test_set_centered = test_set.iter().map(|&v| v.into() - test_set_centroid);

        let m = self
            .clone()
            .cross_covariance(test_set_centered)
            .ok_or(super::Error::MismatchedPointSetSize)?;

        let (u, _, vt) = m.svd();
        let r: RotationMatrix<2> = u
            .matmul(&vt)
            .try_into()
            .map_err(|_| super::Error::NonUnitaryMatrix)?;
        let r_transpose = r.inverted();

        let t = r.rotate(&test_set_centroid);

        Ok((
            r,
            t,
            compute_rmsd(
                test_set.iter().map(|&v| r_transpose.rotate(&v.into()) - t),
                &self.coordinates,
            ),
        ))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    use hoomd_vector::InnerProduct;
    use rstest::rstest;

    #[rstest(
        test_set,
        case(
            vec![[-99.0, -1.0, 1.0].into(), [9.3, 4.5, 8.1].into()],
        ),
        case(
            vec![[0.0, 0.0].into(), [99.3, 0.0].into(), [0.0, 99.3].into(), [99.3, 99.3].into()],
        )
    )]
    fn test_rmsd_matching<const N: usize>(test_set: Vec<Cartesian<N>>) {
        assert_eq!(compute_rmsd(test_set.clone(), &test_set), 0.0);
    }

    #[rstest(
        test_set,
        case(
            vec![[-99.0, -1.0, 1.0].into(), [9.3, 4.5, 8.1].into()],
        ),
        case(
            vec![[0.0, 0.0].into(), [99.3, 0.0].into(), [0.0, 99.3].into(), [99.3, 99.3].into()],
        )
    )]
    fn test_rmsd_scaled<const N: usize>(
        test_set: Vec<Cartesian<N>>,
        #[values(0.0, 0.003, 1.0, 3.5, 98.9)] scale: f64,
    ) {
        // Closed form for points varying solely by a scale factor
        let rmsd = (1.0 - scale).powi(2)
            * test_set
                .clone()
                .into_iter()
                .fold(0.0, |acc, v| acc + v.norm_squared());
        assert_relative_eq!(
            compute_rmsd(test_set.iter().map(|&v| v * scale), &test_set),
            rmsd,
            epsilon = 1e-14
        );
    }

    #[rstest(
        test_set,
        translation,
        case(
            vec![[-99.0, -1.0, 1.0].into(), [9.3, 4.5, 8.1].into()],
            [1.0, -2.0, 3.0].into()
        ),
        case(
            vec![[0.0, 0.0].into(), [99.3, 0.0].into(), [0.0, 99.3].into(), [99.3, 99.3].into()],
            [10.0, -20.0].into()
        ),
        case(
            vec![[-99.0, -1.0, 1.0].into(), [9.3, 4.5, 8.1].into()],
            [0.0, 3.0, -9.1].into()
        )
    )]
    fn test_rmsd_translated<const N: usize>(
        test_set: Vec<Cartesian<N>>,
        translation: Cartesian<N>,
    ) {
        // Closed form for points varying solely by a translation
        let rmsd = test_set.len() as f64 * translation.norm_squared();
        let translated_set: Vec<Cartesian<N>> = test_set.iter().map(|&v| v + translation).collect();
        assert_eq!(compute_rmsd(translated_set, &test_set), rmsd);
    }
}

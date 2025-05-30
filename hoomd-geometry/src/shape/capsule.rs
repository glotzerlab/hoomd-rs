// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`Capsule`] */

use crate::{BoundingSphereRadius, SupportMapping, Volume};

use hoomd_vector::{Cartesian, Vector};

use super::sphere::sphere_volume_prefactor;

/** All points less than or equal to a distance `r` along a line of length `h`.
This line is oriented along the `[0 0 ... 1]` direction, and has extents `+h/2`, `-h/2`
along that axis.
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Capsule<const N: usize> {
    /// Radius of of points that are considered enclosed in the shape.
    pub radius: f64,
    /// Length of the line segment.
    pub height: f64,
}

impl<const N: usize> SupportMapping<Cartesian<N>> for Capsule<N> {
    #[inline]
    fn support_mapping(&self, n: &Cartesian<N>) -> Cartesian<N> {
        // Same support function as a ConvexPolyhedron with 2 vertices, plus the radius.
        let mut v_tip = [0.0; N];
        v_tip[N - 1] = self.height / 2.0;
        let v_tip = v_tip.into();

        let mut v_base = [0.0; N];
        v_base[N - 1] = -self.height / 2.0;
        let v_base = v_base.into();

        let (v_tip_dot_n, v_base_dot_n) = (n.dot(&v_tip), n.dot(&v_base));

        let rshift = *n * self.radius * n.norm();
        if v_tip_dot_n > v_base_dot_n {
            v_tip / n.norm() + rshift
        } else {
            v_base / n.norm() + rshift
        }
    }
}

impl<const N: usize> BoundingSphereRadius for Capsule<N> {
    #[inline]
    fn bounding_sphere_radius(&self) -> f64 {
        self.height / 2.0 + self.radius
    }
}

impl<const N: usize> Volume for Capsule<N> {
    #[inline]
    fn volume(&self) -> f64 {
        if N == 0 {
            return 0.0;
        }
        let r_n_minus_one = self.radius.powi(
            (N - 1)
                .try_into()
                .expect("Dimension {N}-1 would overflow i32!"),
        );
        let cylinder_volume = sphere_volume_prefactor(N - 1) * r_n_minus_one * self.height;
        cylinder_volume + sphere_volume_prefactor(N) * (r_n_minus_one * self.radius)
    }
}

#[cfg(test)]
mod tests {

    use crate::shape::{Cylinder, Hypersphere};

    use super::*;
    use rstest::*;
    use std::marker::PhantomData;

    #[rstest(
        _n => [
            PhantomData::<Capsule<1>>,
            PhantomData::<Capsule<2>>,
            PhantomData::<Capsule<3>>,
            PhantomData::<Capsule<4>>,
            PhantomData::<Capsule<5>>
        ],
        radius => [0.0, 1e-6, 1.0, 34.56],
    )]
    fn test_capsule_volume<const N: usize>(_n: PhantomData<Capsule<N>>, radius: f64) {
        assert_eq!(
            Capsule::<N> {
                radius,
                height: 0.0
            }
            .volume(),
            Hypersphere::<N> { radius }.volume()
        );
    }
    #[rstest(
        radius => [0.0, 1e-6, 1.0, 34.56],
        height => [0.0, 1e-6, 1.0, 34.56],
    )]
    fn test_elongated_capsule_volume(radius: f64, height: f64) {
        let cap = Capsule::<3> { radius, height };
        assert_eq!(
            cap.volume(),
            Hypersphere::<3> { radius }.volume()
                + Cylinder {
                    radius,
                    height: cap.height
                }
                .volume()
        );
    }
}

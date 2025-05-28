// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement approximate comparisons for vector and rotation types
*/

use crate::{Cartesian, Quaternion, Versor};

use approx::{AbsDiffEq, RelativeEq};
use std::iter::zip;

impl<const N: usize> AbsDiffEq for super::Cartesian<N> {
    type Epsilon = <f64 as AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        zip(self.coordinates.iter(), other.coordinates.iter())
            .all(|x| f64::abs_diff_eq(x.0, x.1, epsilon))
    }
}

impl<const N: usize> RelativeEq for super::Cartesian<N> {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        zip(self.coordinates.iter(), other.coordinates.iter())
            .all(|x| f64::relative_eq(x.0, x.1, epsilon, max_relative))
    }
}

impl AbsDiffEq for Quaternion {
    type Epsilon = <f64 as AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        f64::abs_diff_eq(&self.scalar, &other.scalar, epsilon)
            && Cartesian::abs_diff_eq(&self.vector, &other.vector, epsilon)
    }
}

impl AbsDiffEq for Versor {
    type Epsilon = <f64 as AbsDiffEq>::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        super::Quaternion::abs_diff_eq(&self.0, &other.0, epsilon)
    }
}

impl RelativeEq for Quaternion {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        f64::relative_eq(&self.scalar, &other.scalar, epsilon, max_relative)
            && Cartesian::relative_eq(&self.vector, &other.vector, epsilon, max_relative)
    }
}

impl RelativeEq for Versor {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        Quaternion::relative_eq(&self.0, &other.0, epsilon, max_relative)
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Helpers that enable consistent use of `ChIMES` potential
transformation style.
 */
mod cheby;
pub use cheby::Chebyshev;

use arrayvec::ArrayVec;
/** Implement the `ChIMES` transformation styles.

Implement [`Transformation`] on a custom type or use one of the provided
transformations in [`transformation`](crate::transformation) for
`ChIMES` potential.

*/
pub trait Basis<const N: usize> {
    /**
    Implement the basis function `f(s)`
     */
    #[must_use]
    fn evaluate(&self, s: &f64) -> ArrayVec<f64, N>;

    /**
    Implement the derivative of the basis fucntion
     $`\frac{df}{ds}`$.
    */
    #[must_use]
    fn evaluate_derivative(&self, s: &f64) -> ArrayVec<f64, N>;
}

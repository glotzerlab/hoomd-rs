// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*!
Helpers that enable consistent use of
`ChIMES` potential transformation style.
*/

mod morse_transformation;
pub use morse_transformation::MorseTransformation;

/**
Implement the transformation function applied
on particle pairwise distances of `ChIMES`
potential.
 */
pub trait Transformation {
    /** Construct the Transformation trait.

    Implement the transformation function `f(r)`
    that transform `r` into `s`, `s = f(r)`

    `s` is a variable fall in the interval [-1, 1].

     */
    #[must_use]
    fn s(&self, r: &f64) -> f64;

    /**
    Implement the derivative of transformation
    function `\frac{ds}{dr} = \frac{f(r)}{r}`

    `s` is a variable fall in the interval [-1, 1].

     */
    #[must_use]
    fn ds_dr(&self, r: &f64) -> f64;
}

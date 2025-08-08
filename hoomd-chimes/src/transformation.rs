// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Helpers that enable consistent use of `ChIMES` potential
transformation style.
 */
mod morse_transformation;
pub use morse_transformation::MorseTransformation;

/** Implement the `ChIMES` transformation styles.

Implement [`Transformation`] on a custom type or use one of the provided
transformations in [`transformation`](crate::transformation) for
`ChIMES` potential.

*/
pub trait Transformation {
    /**
    Implement the transformation function `f(r)`
    that transform `r` into `s`, `s = f(r)`

    `s` is a variable fall in the interval [-1, 1].

    # Note:
    To be consistent with the origianl `ChIMES`
    potential implementation, when `s` is equal
    or larger than 1 (inner distance cut-off),
    this function always return 1, and do not
    take care of the case when `s` is equal or
    smaller than -1.

     */
    #[must_use]
    fn s(&self, r: &f64) -> f64;

    /**
    Implement the derivative of transformation
    function $`\frac{ds}{dr} = \frac{df(r)}{dr}`$.

    Follows the same behaviour as the fucntion
    `Transformation.s`.
     */
    #[must_use]
    fn ds_dr(&self, r: &f64) -> f64;
}

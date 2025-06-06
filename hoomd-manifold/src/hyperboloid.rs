// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement vector types on a single-sheeted hyperboloid.
 */

use std::array;
use std::fmt;
use std::iter::{Sum, zip};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

use rand::Rng;
use rand::distr::{Distribution, StandardUniform, Uniform};
use crate::{Geodesic,Error};

/** Description of sphere, examples of usage
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hyperboloid<const N: usize> {
    // The vector's coordinates
    pub coordinates: [f64; N],
}
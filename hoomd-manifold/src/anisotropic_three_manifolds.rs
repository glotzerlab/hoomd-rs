// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement vector types in Minkowski space.

use approxim::{approx_derive::RelativeEq, assert_relative_eq};
use rand::{
    Rng,
    distr::{Distribution, StandardUniform, Uniform},
};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::{
    array,
    f64::consts::PI,
    fmt,
    iter::zip,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{Error, Hyperbolic, HyperbolicDisk};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Metric, Vector};

/// A point in $`\mathbb{H}^2\times\mathbb{R}`$.
/// TODO
pub struct H2CrossR {
    /// The two-dimensional hyperbolic component.
    pub hyperbolic: Hyperbolic<3>,
    /// The one-dimensional real component.
    pub real: f64
}

impl Default for H2CrossR {
    /// The origin of $`\mathbb{H}^2\times\mathbb{R}`$. Using the Poincaré disk 
    /// representation for the hyperbolic component, the `default` for 
    /// `H2CrossR` is the point $`(0,0)_{\mathbb{H}}\times \{0\}`$
    /// 
    /// # Example 
    /// ```
    /// use hoomd_manifold::H2CrossR;
    /// 
    /// let origin = H2CrossR::default();
    /// assert!([0.0,0.0,1.0] == *origin.hyperbolic.coordinates());
    /// assert!(0.0 == origin.real);
    /// ```
    #[inline]
    fn default() -> Self {
        H2CrossR {
            hyperbolic: Hyperbolic::<3>::default(),
            real: 0.0
        }
    }
}

impl Metric for H2CrossR {
    /// The squared distance between two `H2CrossR` points.
    /// 
    /// Explicitly, for two points $`(p,q)`$ and $`(u,v)`$ in 
    /// $`\mathbb{H}^2\times\mathbb{R}`$, the squared distance between them 
    /// is given by 
    /// ```math
    /// d^2_{H^2}(p,u) + (q-v)^2
    /// ```
    /// where $`d^2_{H^2}`$ is the squared two-dimensional hyperbolic metric. 
    /// For example, expressing the hyperbolic components in Minkowski
    /// coordinates $`(p_x,p_y,p_z)_{\mathbb{H}}`$ and $`(u_x,u_y,u_z)`$, the 
    /// metric for $`\mathbb{H}^2\times\mathbb{R}`$ is
    /// ```math
    /// \operatorname{arccosh}^2\left[p_zu_z - p_xu_x - p_yu_y\right] + (q-v)^2
    /// ```
    #[inline]
    fn distance_squared(&self, other: &Self) -> f64 {
        let hyperbolic_part = self.hyperbolic.distance_squared(&other.hyperbolic);
        let real_part = (self.real - other.real).powi(2);
        hyperbolic_part + real_part
    }
    #[inline]
    fn distance(&self, other: &Self) -> f64 {
        (self.distance_squared(&other)).sqrt()
    }
    #[inline]
    fn n_dimensions() -> usize {
        3
    }
}

impl H2CrossR {
    /// Project the hyperbolic component onto the Poincaré disk.
    #[inline]
    #[must_use]
    pub fn to_poincare_cross_R(&self) -> ([f64;2], f64) {
        let poincare = self.hyperbolic.to_poincare();
        ([poincare[0], poincare[1]], self.real)
    }
}

/// Randomly distribute points locally on `H2CrossR`.
/// 
/// `H2CrossRBall` is a distribution of points within a specified distance 
/// `radius` from a given point `center` on $`\mathbb{H}^2\times\mathbb{R}`$.
pub struct H2CrossRBall {
    /// Maximum distance away from the ball center.
    pub radius: PositiveReal,
    /// Center of the Ball
    pub center: H2CrossR,
}

impl Distribution<H2CrossR> for H2CrossRBall {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> H2CrossR {
        let hyperbolic_part = HyperbolicDisk {
            disk_radius: self.radius,
            point: self.center.hyperbolic,
        };
        let random_hyperbolic: Hyperbolic<3> = hyperbolic_part.sample(rng);
        let remaining = self.radius.get() - random_hyperbolic.distance(&self.center.hyperbolic);
        let random_real = Uniform::new(0.0, remaining).expect("remaining is positive and real by construction").sample(rng);
        H2CrossR { hyperbolic: random_hyperbolic, real: random_real }
    }
}
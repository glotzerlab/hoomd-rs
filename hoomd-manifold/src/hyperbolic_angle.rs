// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement SO(N-1,1) elements
 */

use num::complex::Complex;
use std::fmt;
use std::f64::consts::PI;

use crate::{Error,Minkowski,HyperbolicRotationMatrix, HyperbolicRotate};

/** Description of hyperbolic rotations

## Operations using [`HyperbolicAngle`] on Minkowski space
*/
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HyperbolicAngle {
    /// Rotation angle (radians).
    pub angles: (f64, f64, f64),
}

impl HyperbolicAngle {
    /** Reduce the rotation part of the tuple.
    */
    #[inline]
    #[must_use]
    pub fn to_reduced(self) -> Self {
        Self {
            angles: (self.angles.0,self.angles.1, self.angles.2.rem_euclid(2.0 * PI)),
        }
    }
}

impl From<HyperbolicAngle> for HyperbolicRotationMatrix<3> {
    /** Description and example
    */
    #[inline]
    fn from(angle_list: HyperbolicAngle) -> HyperbolicRotationMatrix<3> {
        let (a,b,c) = angle_list.angles;
        if (a,b,c)  == (0.0_f64, 0.0_f64, 0.0_f64) {
            HyperbolicRotationMatrix {
                rows : [[1.0,0.0,0.0].into(),
                        [0.0,1.0,0.0].into(),
                        [0.0,0.0,1.0].into(),],
            }
        } else {
            let arg_sq = -a.powi(2)+b.powi(2)+c.powi(2);
            let arg = Complex::new(arg_sq,0.0);
            let arg_sqrt = arg.sqrt();
            let ch = arg_sqrt.cosh();
            let sh = arg_sqrt.sinh();
            let sh_c = Complex::new(arg_sqrt.re*sh.re - arg_sqrt.im*sh.im, arg_sqrt.re*sh.im + arg_sqrt.im*sh.re);
            HyperbolicRotationMatrix { 
                rows: [
                    [((Complex::new(c.powi(2),0.0) + ch.scale(b.powi(2) - a.powi(2))).scale(1.0/arg_sq)).re,
                    ((Complex::new(b*c,0.0) - ch.scale(b*c) + sh_c.scale(a)).scale(-1.0/arg_sq)).re, 
                    ((Complex::new(-1.0*a*c,0.0) + ch.scale(a*c)-sh_c.scale(b)).scale(-1.0/arg_sq)).re]
                    .into(),
                    [((Complex::new(b*c,0.0) - ch.scale(b*c) - sh_c.scale(a)).scale(-1.0/arg_sq)).re,
                    ((Complex::new(b.powi(2),0.0) + ch.scale(c.powi(2)-a.powi(2))).scale(1.0/arg_sq)).re, 
                    ((Complex::new(a*b,0.0) - ch.scale(a*b) - sh_c.scale(c)).scale(-1.0/arg_sq)).re]
                    .into(),
                    [((Complex::new(a*c,0.0) - ch.scale(a*c) - sh_c.scale(b)).scale(-1.0/arg_sq)).re, 
                    ((Complex::new(-1.0*a*b,0.0) + ch.scale(a*b) - sh_c.scale(c)).scale(-1.0/arg_sq)).re, 
                    ((Complex::new(a.powi(2),0.0) - ch.scale(b.powi(2)+c.powi(2))).scale(-1.0/arg_sq)).re]
                    .into(),
                ], 
            }
        }
    }
}

impl From<(f64,f64,f64)> for HyperbolicAngle {
    /** Documentation and examples 
    */
    #[inline]
    fn from(value: (f64,f64,f64)) -> Self {
        Self {angles: (value.0,value.1,value.2)}
    }
}

impl fmt::Display for HyperbolicAngle {
    /// Format a [`HyperbolicAngle`] as `[{v[0]}, {v[1]}, {v[2]}]`.
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {}, {}]", self.angles.0, self.angles.1,self.angles.2)
    }
}
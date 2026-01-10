// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Boxcar`]

use serde::{Serialize, Deserialize};

use super::UnivariateEnergy;

/// Constant valued potential in a given range of `r` (_not differentiable_).
///
/// ```math
/// U(r) = \begin{cases}
/// 0 & r \lt a \\
/// \varepsilon & a \le r \lt b \\
/// 0 & r \ge b
/// \end{cases}
/// ```
///
/// Compute boxcar potential function. Some uses of this in the literature call it
/// the "square well" potential.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use hoomd_interaction::univariate::{Boxcar, UnivariateEnergy};
///
/// let epsilon = 1.5;
/// let (left, right) = (1.0, 2.5);
///
/// let boxcar = Boxcar {
///     epsilon,
///     left,
///     right,
/// };
/// assert_eq!(boxcar.energy(0.0), 0.0);
/// assert_eq!(boxcar.energy(1.0), 1.5);
/// assert_eq!(boxcar.energy(2.0), 1.5);
/// assert_eq!(boxcar.energy(2.5), 0.0);
/// assert_eq!(boxcar.energy(1000.0), 0.0);
/// ```
///
/// The parameters are public fields and may be accessed directly:
///
/// ```
/// use hoomd_interaction::univariate::{Boxcar, UnivariateEnergy};
///
/// let mut boxcar = Boxcar {
///     epsilon: 1.5,
///     left: 1.0,
///     right: 2.5,
/// };
/// boxcar.epsilon = -2.0;
/// boxcar.left = 0.0;
/// boxcar.right = 1.0;
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Boxcar {
    /// Energy scale *(\[energy\])*.
    pub epsilon: f64,
    /// Left side of the boxcar *(\[length\])*.
    pub left: f64,
    /// Right side of the boxcar *(\[length\])*.
    pub right: f64,
}

impl UnivariateEnergy for Boxcar {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        match r {
            x if x < self.left => 0.0,
            x if x < self.right => self.epsilon,
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    fn general_case(
        #[values(1.0, -2.0, 12.125, 0.25)] epsilon: f64,
        #[values(1.0, 2.0, 0.5)] left: f64,
        #[values(0.5, 0.125)] w: f64,
    ) {
        let right = left + w;
        let boxcar = Boxcar {
            epsilon,
            left,
            right,
        };

        assert_eq!(boxcar.epsilon, epsilon);
        assert_eq!(boxcar.left, left);
        assert_eq!(boxcar.right, right);

        // Left
        assert_eq!(boxcar.energy(0.0), 0.0);
        assert_eq!(boxcar.energy(left.next_down()), 0.0);

        // Center
        assert_eq!(boxcar.energy(left), epsilon);
        assert_eq!(boxcar.energy(left.next_up()), epsilon);
        assert_eq!(boxcar.energy(left + w / 2.0), epsilon);
        assert_eq!(boxcar.energy(right.next_down()), epsilon);

        // Right
        assert_eq!(boxcar.energy(right), 0.0);
        assert_eq!(boxcar.energy(right * 10.0), 0.0);
    }
}

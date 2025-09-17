// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! [`AngularMask`] and related data structures.
 */

use super::{AnisotropicEnergy, IsotropicEnergy};
use hoomd_vector::{InnerProduct, Rotate, Unit, Vector};

/** A single patch in the [`AngularMask`] potential.

The width of the patch is given as the cosine of its half-angle.

# Example

```
use hoomd_interaction::pairwise::angular_mask::Patch;
use std::f64::consts::PI;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let patch = Patch { director: [0.0, 1.0, 0.0].try_into()?, cos_delta: (PI / 4.0).cos() };
# Ok(())
# }
```
 */
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Patch<V> {
    /// Vector pointing from the center of the particle to the center of the mask `[unitless]`.
    pub director: Unit<V>,
    /// Cosine of the half-angle width of the mask `[unitless]`.
    pub cos_delta: f64,
}

/** Evaluate an isotropic pairwise energy masked by angular patches (_not differentiable_).

```math
U(\vec{r}_{ij}, \mathbf{o}_{ij}) = f(|\vec{r}_{ij}|) \cdot \max
    \left(1,
    \sum_{m=1}^{N_{\mathrm{masks},i}}
    \sum_{n=1}^{N_{\mathrm{masks},j}}
    s(\vec{d}_{m,i},
      \mathbf{o}_{ij} \vec{d}_{n,j} \mathbf{o}_{ij}^*,
      \delta_{m,i},
      \delta_{n,j}) \right)
```
where
```math
s(\vec{a}, \vec{b}, \delta_a, \delta_b) =
 \begin{cases}
 1 & \hat{a} \cdot \hat{r}_{ij} \ge \cos \delta_{a} \land
 \hat{b} \cdot \hat{r}_{ji} \ge \cos \delta_{b} \\
 0 & \text{otherwise} \\
\end{cases}
```

Implement the [Kern-Frenkel] potential with the [`Boxcar`](super::Boxcar) isotropic potential
and single patch in both `masks_i` and `masks_j`.

[Kern-Frenkel]: http://dx.doi.org/10.1063/1.1569473

# Examples

Construction:

```
use hoomd_interaction::pairwise::{AngularMask, Boxcar, angular_mask::Patch};
use hoomd_vector::Angle;
use std::f64::consts::PI;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let boxcar = Boxcar { epsilon: -1.0, left: 1.0, right: 1.5 };
let masks = [Patch { director: [1.0, 0.0].try_into()?, cos_delta: (PI / 8.0).cos() }];
let angular_mask = AngularMask::new(boxcar, masks);
# Ok(())
# }
```

All fields are public and can be directly manipulated:
```
use hoomd_interaction::pairwise::{AngularMask, Boxcar, angular_mask::Patch};
use hoomd_vector::Angle;
use std::f64::consts::PI;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let boxcar = Boxcar { epsilon: -1.0, left: 1.0, right: 1.5 };
let masks = [Patch { director: [1.0, 0.0].try_into()?, cos_delta: (PI / 8.0).cos() }];
let mut angular_mask = AngularMask::new(boxcar, masks);

angular_mask.masks_i[0].cos_delta = (PI / 4.0).cos();
angular_mask.isotropic.epsilon = -2.0;
# Ok(())
# }
```

Evaluate energy between particles:

```
use hoomd_interaction::pairwise::{AngularMask, AnisotropicEnergy, Boxcar, angular_mask::Patch};
use hoomd_vector::Angle;
use std::f64::consts::PI;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let boxcar = Boxcar { epsilon: -1.0, left: 1.0, right: 1.5 };
let masks = [Patch { director: [1.0, 0.0].try_into()?, cos_delta: (PI / 8.0).cos() }];
let angular_mask = AngularMask::new(boxcar, masks);

// With the same relative orientation, the patches do not overlap and the
// energy is 0.
let energy = angular_mask.energy(&[1.0, 0.0].into(), &Angle::from(0.0));
assert_eq!(energy, 0.0);

// Rotate the j particle to point at the i particle so the patches overlap.
let energy = angular_mask.energy(&[1.0, 0.0].into(), &Angle::from(PI));
assert_eq!(energy, -1.0);
# Ok(())
# }
```

Apply different patches to the _i_ and _j_ particles:
```
use hoomd_interaction::pairwise::{AngularMask, AnisotropicEnergy, Boxcar, angular_mask::Patch};
use hoomd_vector::Angle;
use std::f64::consts::PI;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let boxcar = Boxcar { epsilon: -1.0, left: 1.0, right: 1.5 };
let masks_i = vec![Patch { director: [1.0, 0.0].try_into()?, cos_delta: (PI / 8.0).cos() },
    Patch { director: [-1.0, 0.0].try_into()?, cos_delta: (PI / 8.0).cos() }];
let masks_j = vec![Patch { director: [0.0, 1.0].try_into()?, cos_delta: (PI / 8.0).cos() }];
let angular_mask = AngularMask { isotropic: boxcar, masks_i, masks_j, };

// With the same relative orientation, the patches do not overlap and the
// energy is 0.
let energy = angular_mask.energy(&[-1.0, 0.0].into(), &Angle::from(0.0));
assert_eq!(energy, 0.0);

// Rotate the j particle to point at the i particle so the patches overlap.
let energy = angular_mask.energy(&[-1.0, 0.0].into(), &Angle::from(-PI / 2.0));
assert_eq!(energy, -1.0);
# Ok(())
# }
```

Evaluate the angular mask potential on 3D particles:
```
use hoomd_interaction::pairwise::{AngularMask, AnisotropicEnergy, Boxcar, angular_mask::Patch};
use hoomd_vector::{Cartesian, InnerProduct, Versor};
use std::f64::consts::PI;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let boxcar = Boxcar { epsilon: -1.0, left: 1.0, right: 1.5 };

let mask = [Patch {
    director: [0.0, 0.0, 1.0].try_into()?,
    cos_delta: (PI / 8.0).cos(),
}];
let (x_axis, _) = Cartesian::from([1.0, 0.0, 0.0]).to_unit_unchecked();

let angular_mask = AngularMask::new(boxcar, mask);

assert_eq!(
    angular_mask.energy(
        &Cartesian::from([0.0, 0.0, 1.0]),
        &Versor::from_axis_angle(x_axis, 0.0)
    ),
    0.0
);
assert_eq!(
    angular_mask.energy(
        &Cartesian::from([0.0, 0.0, 1.0]),
        &Versor::from_axis_angle(x_axis, PI)
    ),
    -1.0
);
# Ok(())
# }

```
*/
#[derive(Clone, Debug, PartialEq)]
pub struct AngularMask<E, V> {
    /// The original potential.
    pub isotropic: E,

    /// Masks on the i particle.
    pub masks_i: Vec<Patch<V>>,

    /// Masks on the j particle.
    pub masks_j: Vec<Patch<V>>,
}

impl<E, V> AngularMask<E, V>
where
    V: Vector,
{
    /** Construct a [`AngularMask`] with the given function and masks.

    To obtain the best performance, construct [`AngularMask`] once and
    call use it many times. `new` dynamically allocates `Vec` types
    and is therefore not suitable to be called per particle,
    unlike other potentials such as [`LennardJones`](super::LennardJones)
    or [`Boxcar`](super::Boxcar).

    `new` sets both `masks_i` and `masks_j` to `masks`. Use struct initialization
    syntax to set these separately.

    # Example

    ```
    use hoomd_interaction::pairwise::{AngularMask, Boxcar, angular_mask::Patch};
    use std::f64::consts::PI;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let boxcar = Boxcar { epsilon: -1.0, left: 1.0, right: 1.5 };
    let masks = [Patch { director: [1.0, 0.0].try_into()?, cos_delta: (PI / 8.0).cos() }];
    let angular_mask = AngularMask::new(boxcar, masks);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn new<I>(isotropic: E, masks: I) -> Self
    where
        I: IntoIterator<Item = Patch<V>>,
    {
        let masks = Vec::from_iter(masks);
        Self {
            isotropic,
            masks_i: masks.clone(),
            masks_j: masks,
        }
    }
}

impl<E, V, R> AnisotropicEnergy<V, R> for AngularMask<E, V>
where
    E: IsotropicEnergy,
    V: InnerProduct,
    R: Rotate<V> + Into<R::Matrix> + Copy,
{
    #[inline]
    fn energy(&self, r_ij: &V, o_ij: &R) -> f64 {
        let o_ij_matrix: R::Matrix = (*o_ij).into();
        let (unit_r_ij, r_ij_norm) = r_ij.to_unit_unchecked();
        let unit_r_ji = -(*unit_r_ij.get());

        for mask_j in &self.masks_j {
            let d_j = o_ij_matrix.rotate(mask_j.director.get());

            for mask_i in &self.masks_i {
                if mask_i.director.get().dot(unit_r_ij.get()) >= mask_i.cos_delta
                    && d_j.dot(&unit_r_ji) >= mask_j.cos_delta
                {
                    return self.isotropic.energy(r_ij_norm);
                }
            }
        }

        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::approx::assert_relative_eq;
    use rstest::*;
    use std::f64::consts::PI;

    use crate::pairwise::{Boxcar, LennardJones};
    use hoomd_vector::{Angle, Cartesian, InnerProduct, Versor};

    #[test]
    fn single_patch_2d() {
        // Evaluate that patch directors, widths, and relative orientations are
        // handled properly.
        let epsilon = 1.125;
        let boxcar = Boxcar {
            epsilon,
            left: 0.0,
            right: 1000.0,
        };

        // First case: identical directors in the +x direction
        let mask = [Patch {
            director: [1.0, 0.0]
                .try_into()
                .expect("hard-coded vector should have non-zero length"),
            cos_delta: (PI / 8.0).cos(),
        }];
        let angular_mask = AngularMask::new(boxcar, mask);

        // Check corner cases when the j particle is along the patch direction.
        assert_eq!(
            angular_mask.energy(&Cartesian::from([1.0, 0.0]), &Angle::from(0.0)),
            0.0
        );
        assert_eq!(
            angular_mask.energy(&Cartesian::from([1.0, 0.0]), &Angle::from(PI)),
            epsilon
        );
        assert_eq!(
            angular_mask.energy(
                &Cartesian::from([1.0, 0.0]),
                &Angle::from(PI + PI / 8.0 - 0.001)
            ),
            epsilon
        );
        assert_eq!(
            angular_mask.energy(
                &Cartesian::from([1.0, 0.0]),
                &Angle::from(PI + PI / 8.0 + 0.001)
            ),
            0.0
        );
        assert_eq!(
            angular_mask.energy(
                &Cartesian::from([1.0, 0.0]),
                &Angle::from(PI + PI / 8.0 + 0.001)
            ),
            0.0
        );

        // When the j particle is orthogonal to the patch direction, no orientation will interact.
        for theta in (0..100).map(|x| f64::from(x) * 2.0 * PI / 100.0) {
            assert_eq!(
                angular_mask.energy(&Cartesian::from([0.0, 1.0]), &Angle::from(theta)),
                0.0
            );
        }

        // Second case: identical directors in the 1,1 direction
        let mask = [Patch {
            director: [1.0, 1.0]
                .try_into()
                .expect("hard-coded vector should have non-zero length"),
            cos_delta: (PI / 3.0).cos(),
        }];
        let angular_mask = AngularMask::new(boxcar, mask);

        // Check corner cases when the j particle is along the patch direction
        assert_eq!(
            angular_mask.energy(&Cartesian::from([1.0, 1.0]), &Angle::from(0.0)),
            0.0
        );
        assert_eq!(
            angular_mask.energy(&Cartesian::from([1.0, 1.0]), &Angle::from(PI)),
            epsilon
        );
        assert_eq!(
            angular_mask.energy(
                &Cartesian::from([1.0, 1.0]),
                &Angle::from(PI + PI / 3.0 - 0.001)
            ),
            epsilon
        );
        assert_eq!(
            angular_mask.energy(
                &Cartesian::from([1.0, 1.0]),
                &Angle::from(PI + PI / 3.0 + 0.001)
            ),
            0.0
        );
        assert_eq!(
            angular_mask.energy(
                &Cartesian::from([1.0, 1.0]),
                &Angle::from(PI + PI / 3.0 + 0.001)
            ),
            0.0
        );
        assert_eq!(
            angular_mask.energy(
                &Cartesian::from([1.0, 1.0]),
                &Angle::from(PI + PI / 3.0 + 0.001)
            ),
            0.0
        );

        // With the large PI / 3.0 patch, a PI / 4 offset r_ij can interact.
        assert_eq!(
            angular_mask.energy(&Cartesian::from([0.0, 1.0]), &Angle::from(0.0)),
            0.0
        );
        assert_eq!(
            angular_mask.energy(&Cartesian::from([0.0, 1.0]), &Angle::from(-3.0 * PI / 4.0)),
            epsilon
        );
    }

    #[rstest]
    #[case([0.0, 1.0].into(), 0.0, 1.0)]
    #[case([0.0, 1.0].into(), PI / 2.0, 0.0)]
    #[case([0.0, 1.0].into(), PI, 1.0)]
    #[case([0.0, -1.0].into(), 0.0, 1.0)]
    #[case([0.0, -1.0].into(), PI / 2.0, 0.0)]
    #[case([0.0, -1.0].into(), PI, 1.0)]
    #[case([1.0, 0.0].into(), 0.0, 0.0)]
    #[case([1.0, 0.0].into(), PI / 2.0, 1.0)]
    #[case([1.0, 0.0].into(), PI, 0.0)]
    #[case([-1.0, 0.0].into(), 0.0, 0.0)]
    #[case([-1.0, 0.0].into(), PI / 2.0, 1.0)]
    #[case([-1.0, 0.0].into(), PI, 0.0)]
    fn multiple_patches_2d(#[case] r_ij: Cartesian<2>, #[case] theta: f64, #[case] expected: f64) {
        let epsilon = 1.0;
        let boxcar = Boxcar {
            epsilon,
            left: 0.0,
            right: 1000.0,
        };

        // Third case: multiple patches and different i,j masks.
        let masks_i = vec![
            Patch {
                director: [0.0, 1.0]
                    .try_into()
                    .expect("hard-coded vector should have non-zero length"),
                cos_delta: (PI / 8.0).cos(),
            },
            Patch {
                director: [0.0, -1.0]
                    .try_into()
                    .expect("hard-coded vector should have non-zero length"),
                cos_delta: (PI / 8.0).cos(),
            },
            Patch {
                director: [1.0, 0.0]
                    .try_into()
                    .expect("hard-coded vector should have non-zero length"),
                cos_delta: (PI / 8.0).cos(),
            },
            Patch {
                director: [-1.0, 0.0]
                    .try_into()
                    .expect("hard-coded vector should have non-zero length"),
                cos_delta: (PI / 8.0).cos(),
            },
        ];
        let masks_j = vec![
            Patch {
                director: [0.0, 1.0]
                    .try_into()
                    .expect("hard-coded vector should have non-zero length"),
                cos_delta: (PI / 8.0).cos(),
            },
            Patch {
                director: [0.0, -1.0]
                    .try_into()
                    .expect("hard-coded vector should have non-zero length"),
                cos_delta: (PI / 8.0).cos(),
            },
        ];
        let angular_mask = AngularMask {
            isotropic: boxcar,
            masks_i,
            masks_j,
        };

        assert_eq!(angular_mask.energy(&r_ij, &Angle::from(theta)), expected);
    }

    #[rstest]
    fn smooth_potential(#[values(0.9, 1.1, 1.2, 3.0)] r: f64) {
        let epsilon = 1.0;
        let sigma = 1.0;
        let lj: LennardJones = LennardJones { epsilon, sigma };

        let mask = [Patch {
            director: [1.0, 0.0]
                .try_into()
                .expect("hard-coded vector should have non-zero length"),
            cos_delta: (PI).cos(),
        }];
        let angular_mask = AngularMask::new(lj, mask);

        // The patch covers the full surface. angular_mask.energy() should evaluate to the same
        // as lj.energy() for all orientations.
        for theta in (0..100).map(|x| f64::from(x) * 2.0 * PI / 100.0) {
            let r_ij = Angle::from(theta).rotate(&Cartesian::from([0.0, r]));
            assert_relative_eq!(
                angular_mask.energy(&r_ij, &Angle::from(0.0)),
                lj.energy(r),
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn single_patch_3d() {
        // Evaluate that patch directors, widths, and relative orientations are
        // handled properly in 3D.
        let epsilon = 1.125;
        let boxcar = Boxcar {
            epsilon,
            left: 0.0,
            right: 1000.0,
        };

        // First case: identical directors in the +z direction
        let mask = [Patch {
            director: [0.0, 0.0, 1.0]
                .try_into()
                .expect("hard-coded vector should have non-zero length"),
            cos_delta: (PI / 8.0).cos(),
        }];
        let angular_mask = AngularMask::new(boxcar, mask);

        let (x_axis, _) = Cartesian::from([1.0, 0.0, 0.0]).to_unit_unchecked();
        let (y_axis, _) = Cartesian::from([1.0, 0.0, 0.0]).to_unit_unchecked();

        // Check corner cases when the j particle is along the patch direction.
        assert_eq!(
            angular_mask.energy(
                &Cartesian::from([0.0, 0.0, 1.0]),
                &Versor::from_axis_angle(x_axis, 0.0)
            ),
            0.0
        );
        assert_eq!(
            angular_mask.energy(
                &Cartesian::from([0.0, 0.0, 1.0]),
                &Versor::from_axis_angle(y_axis, PI)
            ),
            epsilon
        );
        assert_eq!(
            angular_mask.energy(
                &Cartesian::from([0.0, 0.0, 1.0]),
                &Versor::from_axis_angle(x_axis, PI + PI / 8.0 - 0.001)
            ),
            epsilon
        );
        assert_eq!(
            angular_mask.energy(
                &Cartesian::from([0.0, 0.0, 1.0]),
                &Versor::from_axis_angle(y_axis, PI + PI / 8.0 + 0.001)
            ),
            0.0
        );
        assert_eq!(
            angular_mask.energy(
                &Cartesian::from([0.0, 0.0, 1.0]),
                &Versor::from_axis_angle(x_axis, PI + PI / 8.0 + 0.001)
            ),
            0.0
        );

        // When the j particle is orthogonal to the patch direction, no orientation will interact.
        for theta in (0..100).map(|x| f64::from(x) * 2.0 * PI / 100.0) {
            assert_eq!(
                angular_mask.energy(
                    &Cartesian::from([0.0, 1.0, 0.0]),
                    &Versor::from_axis_angle(x_axis, theta)
                ),
                0.0
            );
        }
    }
}

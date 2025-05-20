// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`AngularMask`] and related data structures.
 */

use super::{AnisotropicEnergy, IsotropicEnergy};
use hoomd_vector::{Rotate, Unit, Vector};

/** A single patch in the [`AngularMask`] potential.

The width of the patch is given as the cosine of its half-angle.

# Example

```
use hoomd_interaction::pairwise::angular_mask::Patch;
use std::f64::consts::PI;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let patch = Patch { director: [0.0, 1.0, 0.0].try_into()?, cos_delta: (PI/4.0).cos() };
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

/** Evaluate an isotropic pairwise energy masked by angular patches.

<!--
U(\vec{r}_{ij}, \mathbf{o}_{ij}) = f(|\vec{r}_{ij}|) \cdot \max
    \left(1,
    \sum_{m=1}^{N_{\mathrm{masks},i}}
    \sum_{n=1}^{N_{\mathrm{masks},j}}
    s(\vec{d}_{m,i},
      \mathbf{o}_{ij} \vec{d}_{n,j} \mathbf{o}_{ij}^*,
      \delta_{m,i},
      \delta_{n,j}) \right)
-->
<math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo form="prefix" stretchy="false">(</mo><msub><mover><mi>r</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo separator="true">,</mo><msub><mi>𝐨</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo form="postfix" stretchy="false">)</mo><mo>=</mo><mi>f</mi><mo form="prefix" stretchy="false">(</mo><mi>|</mi><msub><mover><mi>r</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mrow><mi>i</mi><mi>j</mi></mrow></msub><mi>|</mi><mo form="postfix" stretchy="false">)</mo><mo>⋅</mo><mrow><mi>max</mi><mo>⁡</mo></mrow><mrow><mo fence="true" form="prefix">(</mo><mn>1</mn><mo separator="true">,</mo><mrow><munderover><mo stretchy="true">∑</mo><mrow><mi>m</mi><mo>=</mo><mn>1</mn></mrow><msub><mi>N</mi><mrow><mrow><mtext></mtext><mi>masks</mi></mrow><mo separator="true">,</mo><mi>i</mi></mrow></msub></munderover></mrow><mrow><munderover><mo stretchy="true">∑</mo><mrow><mi>n</mi><mo>=</mo><mn>1</mn></mrow><msub><mi>N</mi><mrow><mrow><mtext></mtext><mi>masks</mi></mrow><mo separator="true">,</mo><mi>j</mi></mrow></msub></munderover></mrow><mi>s</mi><mo form="prefix" stretchy="false">(</mo><msub><mover><mi>d</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mrow><mi>m</mi><mo separator="true">,</mo><mi>i</mi></mrow></msub><mo separator="true">,</mo><msub><mi>𝐨</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><msub><mover><mi>d</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mrow><mi>n</mi><mo separator="true">,</mo><mi>j</mi></mrow></msub><msubsup><mi>𝐨</mi><mrow><mi>i</mi><mi>j</mi></mrow><mo>*</mo></msubsup><mo separator="true">,</mo><msub><mi>δ</mi><mrow><mi>m</mi><mo separator="true">,</mo><mi>i</mi></mrow></msub><mo separator="true">,</mo><msub><mi>δ</mi><mrow><mi>n</mi><mo separator="true">,</mo><mi>j</mi></mrow></msub><mo form="postfix" stretchy="false">)</mo><mo fence="true" form="postfix">)</mo></mrow></mrow></math>
where
<!--
s(\vec{a}, \vec{b}, \delta_a, \delta_b) =
 \begin{cases}
 1 & \hat{a} \cdot \hat{r}_{ij} \ge \cos \delta_{a} \land
 \hat{b} \cdot \hat{r}_{ji} \ge \cos \delta_{b} \\
 0 & \text{otherwise} \\
\end{cases}
-->
<math display="block" class="tml-display" style="display:block math;"><mrow><mi>s</mi><mo form="prefix" stretchy="false">(</mo><mover><mi>a</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo separator="true">,</mo><mover><mi>b</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo separator="true">,</mo><msub><mi>δ</mi><mi>a</mi></msub><mo separator="true">,</mo><msub><mi>δ</mi><mi>b</mi></msub><mo form="postfix" stretchy="false">)</mo><mo>=</mo><mrow><mo fence="true" form="prefix">{</mo><mtable><mtr><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 0em;"><mn>1</mn></mtd><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 1em;"><mrow><mover><mi>a</mi><mo stretchy="false" class="tml-xshift" style="math-style:normal;math-depth:0;">^</mo></mover><mo>⋅</mo><msub><mover><mi>r</mi><mo stretchy="false" class="tml-xshift" style="math-style:normal;math-depth:0;">^</mo></mover><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo>≥</mo><mrow><mi>cos</mi><mo>⁡</mo><mspace width="0.1667em"></mspace></mrow><msub><mi>δ</mi><mi>a</mi></msub><mo>∧</mo><mover><mi>b</mi><mo stretchy="false" class="tml-capshift" style="math-style:normal;math-depth:0;">^</mo></mover><mo>⋅</mo><msub><mover><mi>r</mi><mo stretchy="false" class="tml-xshift" style="math-style:normal;math-depth:0;">^</mo></mover><mrow><mi>j</mi><mi>i</mi></mrow></msub><mo>≥</mo><mrow><mi>cos</mi><mo>⁡</mo><mspace width="0.1667em"></mspace></mrow><msub><mi>δ</mi><mi>b</mi></msub></mrow></mtd></mtr><mtr><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 0em;"><mn>0</mn></mtd><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 1em;"><mtext>otherwise</mtext></mtd></mtr></mtable><mo fence="true" form="postfix"></mo></mrow></mrow></math>

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
let boxcar = Boxcar { epsilon: -1.0, a: 1.0, b: 1.5 };
let masks = [Patch { director: [1.0, 0.0].try_into()?, cos_delta: (PI/8.0).cos() }];
let angular_mask = AngularMask::new(boxcar, masks, masks);
# Ok(())
# }
```

All fields are public and can be directly manupipated:
```
use hoomd_interaction::pairwise::{AngularMask, Boxcar, angular_mask::Patch};
use hoomd_vector::Angle;
use std::f64::consts::PI;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let boxcar = Boxcar { epsilon: -1.0, a: 1.0, b: 1.5 };
let masks = [Patch { director: [1.0, 0.0].try_into()?, cos_delta: (PI/8.0).cos() }];
let mut angular_mask = AngularMask::new(boxcar, masks, masks);

angular_mask.masks_i[0].cos_delta = (PI/4.0).cos();
angular_mask.f.epsilon = -2.0;
# Ok(())
# }
```

Evaluate energy between particles:

```
use hoomd_interaction::pairwise::{AngularMask, AnisotropicEnergy, Boxcar, angular_mask::Patch};
use hoomd_vector::Angle;
use std::f64::consts::PI;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let boxcar = Boxcar { epsilon: -1.0, a: 1.0, b: 1.5 };
let masks = [Patch { director: [1.0, 0.0].try_into()?, cos_delta: (PI/8.0).cos() }];
let angular_mask = AngularMask::new(boxcar, masks, masks);

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
let boxcar = Boxcar { epsilon: -1.0, a: 1.0, b: 1.5 };
let masks_i = [Patch { director: [1.0, 0.0].try_into()?, cos_delta: (PI/8.0).cos() },
    Patch { director: [-1.0, 0.0].try_into()?, cos_delta: (PI/8.0).cos() }];
let masks_j = [Patch { director: [0.0, 1.0].try_into()?, cos_delta: (PI/8.0).cos() }];
let angular_mask = AngularMask::new(boxcar, masks_i, masks_j);

// With the same relative orientation, the patches do not overlap and the
// energy is 0.
let energy = angular_mask.energy(&[-1.0, 0.0].into(), &Angle::from(0.0));
assert_eq!(energy, 0.0);

// Rotate the j particle to point at the i particle so the patches overlap.
let energy = angular_mask.energy(&[-1.0, 0.0].into(), &Angle::from(-PI/2.0));
assert_eq!(energy, -1.0);
# Ok(())
# }
```

Evaluate the angular mask potential on 3D particles:
```
use hoomd_interaction::pairwise::{AngularMask, AnisotropicEnergy, Boxcar, angular_mask::Patch};
use hoomd_vector::{Cartesian, Vector, Versor};
use std::f64::consts::PI;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let boxcar = Boxcar { epsilon: -1.0, a: 1.0, b: 1.5 };

let mask = [Patch {
    director: [0.0, 0.0, 1.0].try_into()?,
    cos_delta: (PI / 8.0).cos(),
}];
let (x_axis, _) = Cartesian::from([1.0, 0.0, 0.0]).to_unit_unchecked();

let angular_mask = AngularMask::new(boxcar, mask, mask);

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
pub struct AngularMask<F, V> {
    /// The original potential.
    pub f: F,

    /// Masks on the i particle.
    pub masks_i: Vec<Patch<V>>,

    /// Masks on the j particle.
    pub masks_j: Vec<Patch<V>>,
}

impl<F, V> AngularMask<F, V>
where
    V: Vector,
{
    /** Construct a [`AngularMask`] with the given function and masks.

    To obtain the best performance, construct [`AngularMask`] once and
    call use it many times. `new` dynamically allocates `Vec` types
    and is therefore not suitable to be called per particle,
    unlike other potentials such as [`LennardJones`](super::LennardJones)
    or [`Boxcar`](super::Boxcar).

    # Example

    ```
    use hoomd_interaction::pairwise::{AngularMask, Boxcar, angular_mask::Patch};
    use std::f64::consts::PI;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let boxcar = Boxcar { epsilon: -1.0, a: 1.0, b: 1.5 };
    let masks = [Patch { director: [1.0, 0.0].try_into()?, cos_delta: (PI/8.0).cos() }];
    let angular_mask = AngularMask::new(boxcar, masks, masks);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn new<I1, I2>(f: F, masks_i: I1, masks_j: I2) -> Self
    where
        I1: IntoIterator<Item = Patch<V>>,
        I2: IntoIterator<Item = Patch<V>>,
    {
        Self {
            f,
            masks_i: Vec::from_iter(masks_i),
            masks_j: Vec::from_iter(masks_j),
        }
    }
}

impl<F, V, R> AnisotropicEnergy<V, R> for AngularMask<F, V>
where
    F: IsotropicEnergy,
    V: Vector,
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
                    return self.f.energy(r_ij_norm);
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
    use hoomd_vector::{Angle, Cartesian, Versor};

    #[test]
    fn single_patch_2d() {
        // Evaluate that patch directors, widths, and relative orientations are
        // handled properly.
        let epsilon = 1.125;
        let boxcar = Boxcar {
            epsilon,
            a: 0.0,
            b: 1000.0,
        };

        // First case: identical directors in the +x direction
        let mask = [Patch {
            director: [1.0, 0.0]
                .try_into()
                .expect("hard-coded vector should have non-zero length"),
            cos_delta: (PI / 8.0).cos(),
        }];
        let angular_mask = AngularMask::new(boxcar, mask, mask);

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
        let angular_mask = AngularMask::new(boxcar, mask, mask);

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

        // With the large PI/3.0 patch, a PI/4 offset r_ij can interact.
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
    #[case([0.0, 1.0].into(), PI/2.0, 0.0)]
    #[case([0.0, 1.0].into(), PI, 1.0)]
    #[case([0.0, -1.0].into(), 0.0, 1.0)]
    #[case([0.0, -1.0].into(), PI/2.0, 0.0)]
    #[case([0.0, -1.0].into(), PI, 1.0)]
    #[case([1.0, 0.0].into(), 0.0, 0.0)]
    #[case([1.0, 0.0].into(), PI/2.0, 1.0)]
    #[case([1.0, 0.0].into(), PI, 0.0)]
    #[case([-1.0, 0.0].into(), 0.0, 0.0)]
    #[case([-1.0, 0.0].into(), PI/2.0, 1.0)]
    #[case([-1.0, 0.0].into(), PI, 0.0)]
    fn multiple_patches_2d(#[case] r_ij: Cartesian<2>, #[case] theta: f64, #[case] expected: f64) {
        let epsilon = 1.0;
        let boxcar = Boxcar {
            epsilon,
            a: 0.0,
            b: 1000.0,
        };

        // Third case: multiple patches and different i,j masks.
        let mask_i = [
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
        let mask_j = [
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
        let angular_mask = AngularMask::new(boxcar, mask_i, mask_j);

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
        let angular_mask = AngularMask::new(lj, mask, mask);

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
            a: 0.0,
            b: 1000.0,
        };

        // First case: identical directors in the +z direction
        let mask = [Patch {
            director: [0.0, 0.0, 1.0]
                .try_into()
                .expect("hard-coded vector should have non-zero length"),
            cos_delta: (PI / 8.0).cos(),
        }];
        let angular_mask = AngularMask::new(boxcar, mask, mask);

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

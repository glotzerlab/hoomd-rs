// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`AngularMask`]
*/

use super::{AnisotropicEnergy, IsotropicEnergy};
use hoomd_vector::{Rotate, Rotation, Vector, Unit};

/** A single patch in the [`AngularMask`] potential.
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Patch<V> {
    /// Vector pointing from the center of the particle to the center of the mask `[unitless]`.
    pub director: Unit<V>,
    /// Cosine of the half-angle width of the mask `[unitless]`.
    pub cos_delta: f64,
}

impl<V> Patch<V> {
    /** Construct a new patch with the given direction and width.

    The width of the patch is given as the cosine of its half-angle.

    TODO: Possibly this name is too general? Should we consider exposing this module
    as public to group it with [`AngularMask`]? Patches for the MD potential will have different
    parameters.

    # Example

    ```
    use hoomd_interaction::pairwise::Patch;
    use std::f64::consts::PI;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let patch = Patch::new([0.0, 1.0, 0.0].try_into()?, (PI/4.0).cos());
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn new(director: Unit<V>, cos_delta: f64) -> Self {
        Self { director, cos_delta }
    }
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

Basic usage:

```
use hoomd_interaction::pairwise::{AngularMask, AnisotropicEnergy, Boxcar, Patch};
use hoomd_vector::Angle;
use std::f64::consts::PI;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let boxcar = Boxcar::new(-1.0, 1.0, 1.5);
let masks = vec![Patch::new([1.0, 0.0].try_into()?, (PI/8.0).cos())];
let angular_mask = AngularMask::new(boxcar, &masks, &masks);

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
use hoomd_interaction::pairwise::{AngularMask, AnisotropicEnergy, Boxcar, Patch};
use hoomd_vector::Angle;
use std::f64::consts::PI;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let boxcar = Boxcar::new(-1.0, 1.0, 1.5);
let masks_i = vec![Patch::new([1.0, 0.0].try_into()?, (PI/8.0).cos()),
    Patch::new([-1.0, 0.0].try_into()?, (PI/8.0).cos())];
let masks_j = vec![Patch::new([0.0, 1.0].try_into()?, (PI/8.0).cos())];
let angular_mask = AngularMask::new(boxcar, &masks_i, &masks_j);

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
```
*/
#[derive(Clone, Debug, PartialEq)]
pub struct AngularMask<F, I1, I2>
{

    /// The original potential.
    pub f: F,

    /// Masks on the i particle.
    pub masks_i: I1, 

    /// Masks on the j particle.
    pub masks_j: I2, 
}

impl<'a, F, V, I1, I2> AngularMask<F, I1, I2>
where
    V: Vector + 'a,
    I1: IntoIterator<Item = &'a Patch<V>>,
    I2: IntoIterator<Item = &'a Patch<V>>,
 {
    /** Construct a [`AngularMask`] with the given function and masks.

    # Example

    ```
    use hoomd_interaction::pairwise::{AngularMask, Boxcar, Patch};
    use std::f64::consts::PI;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let boxcar = Boxcar::new(-1.0, 1.0, 1.5);
    let masks = vec![Patch::new([1.0, 0.0].try_into()?, (PI/8.0).cos())];
    let angular_mask = AngularMask::new(boxcar, &masks, &masks);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn new(f: F, masks_i: I1, masks_j: I2) -> Self {
        Self { f, masks_i, masks_j }
        }
}

/** [`AngularMask`] is intended to be constructed once and then allow `energy`
to be called multiple times. The trait bound
`IntoIterator<Item = &'a Patch<V>> + Copy` ensures that the masks provide a
cheaply copyable iterator over references to patches. This works with references
to arrays `&[Patch<V>; N]` and vectors `&Vec<Patch<V>>`.

As a consequence, [`AngularMask`] does not take ownership of the mask data
structures. It is meant to be constructed when needed to evaluate energies
and then dropped. You should maintain your own separate data structure
to store the masks and other potential parameters as needed.

TODO: determine if there is some way to store owned masks as well.
Some use-cases may find that more conveneient as they will not need to
implement a separate storage type. It is weird that AngularMask owns `f`
but not the masks... Maybe the best solution is to remove the general
I1, I2 and make AnulgarMask store owned Vec<Patch<V>> directly....
The IntoIterator could be accepted in new and users should then be
cautioned that new is possibly an expensive operation that should be
done once per simulation - not per pair of particles.
*/
impl<'a, F, V, R, I1, I2> AnisotropicEnergy<V, R> for AngularMask<F, I1, I2>
where
    F: IsotropicEnergy,
    V: Vector + 'a,
    R: Rotation+Rotate<V>,
    I1: IntoIterator<Item = &'a Patch<V>> + Copy,
    I2: IntoIterator<Item = &'a Patch<V>> + Copy,
{
    #[inline]
    fn energy(&self, r_ij: &V, o_ij: &R) -> f64 {
        // TODO: to_rotation_matrix in base rotate trate
        let o_ij_matrix = o_ij; // .to_rotation_matrix();
        let unit_r_ij = r_ij.to_unit_unchecked();
        let unit_r_ji: V = -(*unit_r_ij.get());
    
        for mask_j in self.masks_j {
            let d_j = o_ij_matrix.rotate(mask_j.director.get());

            for mask_i in self.masks_i {
                if mask_i.director.get().dot(unit_r_ij.get()) >= mask_i.cos_delta
                     && d_j.dot(&unit_r_ji) >= mask_j.cos_delta {
                    return self.f.energy(r_ij.norm());
                    }
            }
        }

    0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;
    use std::f64::consts::PI;
    use ::approx::assert_relative_eq;

    use crate::pairwise::{Boxcar, LennardJones};
    use hoomd_vector::{Angle, Cartesian};

    #[test]
    fn single_patch_2d() {
        // Evaluate that patch directors, widths, and relative orientations are
        // handled properly.
        let epsilon = 1.125;
        let boxcar = Boxcar::new(epsilon, 0.0, 1000.0);

        // First case: identical directors in the +x direction
        let mask = [Patch::new([1.0, 0.0].try_into().expect("valid unit vector"), (PI/8.0).cos())];
        let angular_mask = AngularMask::new(boxcar, &mask, &mask);

        // Check corner cases when the j particle is along the patch direction.
        assert_eq!(angular_mask.energy(&Cartesian::from([1.0, 0.0]), &Angle::from(0.0)), 0.0);
        assert_eq!(angular_mask.energy(&Cartesian::from([1.0, 0.0]), &Angle::from(PI)), epsilon);
        assert_eq!(angular_mask.energy(&Cartesian::from([1.0, 0.0]), &Angle::from(PI + PI/8.0 - 0.001)), epsilon);
        assert_eq!(angular_mask.energy(&Cartesian::from([1.0, 0.0]), &Angle::from(PI + PI/8.0 + 0.001)), 0.0);
        assert_eq!(angular_mask.energy(&Cartesian::from([1.0, 0.0]), &Angle::from(PI + PI/8.0 + 0.001)), 0.0);

        // When the j particle is orthogonal to the patch direction, no orientation will interact.
        for theta in (0..100).map(|x| f64::from(x) * 2.0 * PI / 100.0) {
            assert_eq!(angular_mask.energy(&Cartesian::from([0.0, 1.0]), &Angle::from(theta)), 0.0);
        }

        // Second case: identical directors in the 1,1 direction
        let mask = [Patch::new([1.0, 1.0].try_into().expect("valid unit vector"), (PI/3.0).cos())];
        let angular_mask = AngularMask::new(boxcar, &mask, &mask);

        // Check corner cases when the j particle is along the patch direction
        assert_eq!(angular_mask.energy(&Cartesian::from([1.0, 1.0]), &Angle::from(0.0)), 0.0);
        assert_eq!(angular_mask.energy(&Cartesian::from([1.0, 1.0]), &Angle::from(PI)), epsilon);
        assert_eq!(angular_mask.energy(&Cartesian::from([1.0, 1.0]), &Angle::from(PI + PI/3.0 - 0.001)), epsilon);
        assert_eq!(angular_mask.energy(&Cartesian::from([1.0, 1.0]), &Angle::from(PI + PI/3.0 + 0.001)), 0.0);
        assert_eq!(angular_mask.energy(&Cartesian::from([1.0, 1.0]), &Angle::from(PI + PI/3.0 + 0.001)), 0.0);
        assert_eq!(angular_mask.energy(&Cartesian::from([1.0, 1.0]), &Angle::from(PI + PI/3.0 + 0.001)), 0.0);

        // With the large PI/3.0 patch, a PI/4 offset r_ij can interact.
        assert_eq!(angular_mask.energy(&Cartesian::from([0.0, 1.0]), &Angle::from(0.0)), 0.0);
        assert_eq!(angular_mask.energy(&Cartesian::from([0.0, 1.0]), &Angle::from(-3.0 * PI/4.0)), epsilon);
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
    fn multiple_patches_2d(#[case] r_ij: Cartesian<2>,
    #[case] theta: f64,
    #[case] expected: f64) {
        let epsilon = 1.0;
        let boxcar = Boxcar::new(epsilon, 0.0, 1000.0);

        // Third case: multiple patches and different i,j masks.
        let mask_i = [
            Patch::new([0.0, 1.0].try_into().expect("valid unit vector"), (PI/8.0).cos()),
            Patch::new([0.0, -1.0].try_into().expect("valid unit vector"), (PI/8.0).cos()),
            Patch::new([1.0, 0.0].try_into().expect("valid unit vector"), (PI/8.0).cos()),
            Patch::new([-1.0, 0.0].try_into().expect("valid unit vector"), (PI/8.0).cos()),
        ];
        let mask_j = [
            Patch::new([0.0, 1.0].try_into().expect("valid unit vector"), (PI/8.0).cos()),
            Patch::new([0.0, -1.0].try_into().expect("valid unit vector"), (PI/8.0).cos()),
        ];
        let angular_mask = AngularMask::new(boxcar, &mask_i, &mask_j);

        assert_eq!(angular_mask.energy(&r_ij, &Angle::from(theta)), expected);
        }

    #[rstest]
    fn smooth_potential(#[values(0.9, 1.1, 1.2, 3.0)] r: f64) {
        let epsilon = 1.0;
        let sigma = 1.0;
        let lj: LennardJones = LennardJones::new(epsilon, sigma);

        let mask = [Patch::new([1.0, 0.0].try_into().expect("valid unit vector"), (PI).cos())];
        let angular_mask = AngularMask::new(lj, &mask, &mask);

        // The patch covers the full surface. angular_mask.energy() should evaluate to the same
        // as lj.energy() for all orientations.
        for theta in (0..100).map(|x| f64::from(x) * 2.0 * PI / 100.0) {
            let r_ij = Angle::from(theta).rotate(&Cartesian::from([0.0, r]));
            assert_relative_eq!(angular_mask.energy(&r_ij, &Angle::from(0.0)), lj.energy(r), epsilon=1e-12);
            }
        }

    // TODO: 3D implementation
    
    }


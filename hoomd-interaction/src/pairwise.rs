// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Pairwise interactions.
*/

use hoomd_vector::{Rotate, Rotation, Vector};

pub trait IsotropicEnergy {
    /** Compute the pairwise energy between two point particles.
    <!-- U(r) -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo form="prefix" stretchy="false">(</mo><mi>r</mi><mo form="postfix" stretchy="false">)</mo></mrow></math>
    */
    #[must_use]
    fn energy(&self, r: f64) -> f64;
}

pub trait IsotropicForce {
    /** Compute the radial component of the pairwise force between two point
    particles.

    The direction of the force is along the unit vector between the two
    particles.
    <!-- -\frac{\mathrm{d} U}{\mathrm{d} r} -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><mo>−</mo><mfrac><mrow><mrow><mi mathvariant="normal">d</mi></mrow><mi>U</mi></mrow><mrow><mrow><mi mathvariant="normal">d</mi></mrow><mi>r</mi></mrow></mfrac></mrow></math>
    */
    #[must_use]
    fn force(&self, r: f64) -> f64;
}

pub trait AnisotropicEnergy<V: Vector, R: Rotation+Rotate<V>> {
    /** Compute the pairwise energy between two oriented particles.
    <!-- U(\vec{r}_{ij}, \mathbf{o}_{ij}) -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo form="prefix" stretchy="false">(</mo><msub><mover><mi>r</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo separator="true">,</mo><msub><mi>𝐨</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo form="postfix" stretchy="false">)</mo></mrow></math>    */
    #[must_use]
    fn energy(&self, r_ij: &V, o_ij: &R) -> f64;
}

// TODO: determine how to express the torque return type in a general way. Possibly use
// an associated type of Rotation.
// pub trait AnisotropicForce<V: Vector, R: Rotation+Rotate<V>> {
//     /** Compute the pairwise force and torque between two oriented particles.
//     TODO: math expression.
//     */
//     #[must_use]
//     fn energy(&self, r_ij: &V, o_ij: &R) -> f64;
// }


/** Lennard-Jones pairwise potential
    
<!-- U(r) = 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{N} - \left( \frac{\sigma}{r} \right)^{M} \right] -->
<math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo form="prefix" stretchy="false">(</mo><mi>r</mi><mo form="postfix" stretchy="false">)</mo><mo>=</mo><mn>4</mn><mi>ε</mi><mrow><mo fence="true" form="prefix">[</mo><msup><mrow><mo fence="true" form="prefix">(</mo><mfrac><mi>σ</mi><mi>r</mi></mfrac><mo fence="true" form="postfix">)</mo></mrow><mi>N</mi></msup><mo>−</mo><msup><mrow><mo fence="true" form="prefix">(</mo><mfrac><mi>σ</mi><mi>r</mi></mfrac><mo fence="true" form="postfix">)</mo></mrow><mi>M</mi></msup><mo fence="true" form="postfix">]</mo></mrow></mrow></math>*/
pub struct LennardJones<const N: i32 = 12, const M: i32 = 6> {
    /// Energy scale `[energy]`.
    pub epsilon: f64,
    /// Interaction width `[length]`.
    pub sigma: f64
}

impl<const N: i32, const M: i32> LennardJones<N, M> {
    #[inline]
    #[must_use]
    pub fn new(epsilon: f64, sigma: f64) -> Self {
        Self { epsilon, sigma }
        }
}

impl<const N: i32, const M: i32> IsotropicEnergy for LennardJones<N, M> {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        let sigma_r = self.sigma / r;
        
        4.0 * self.epsilon * (sigma_r.powi(N) - sigma_r.powi(M))
    }
}

impl<const N: i32, const M: i32> IsotropicForce for LennardJones<N, M> {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        let r_inv = r.recip();
        let sigma_r = self.sigma * r_inv;
        
        self.epsilon * r_inv * (4.0 * f64::from(N) * sigma_r.powi(N) - 4.0 * f64::from(M) * sigma_r.powi(M))
    }
}

/** Weeks-Chandler-Anderson pairwise potential
    
<!--
U(r) = \begin{cases}
4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r} \right)^{6} \right] + \varepsilon & r \lt 2^{1/6} \sigma \\

0 & r \ge 2^{1/6} \sigma
\end{cases}
-->
<math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo form="prefix" stretchy="false">(</mo><mi>r</mi><mo form="postfix" stretchy="false">)</mo><mo>=</mo><mrow><mo fence="true" form="prefix">{</mo><mtable><mtr><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 0em;"><mrow><mn>4</mn><mi>ε</mi><mrow><mo fence="true" form="prefix">[</mo><msup><mrow><mo fence="true" form="prefix">(</mo><mfrac><mi>σ</mi><mi>r</mi></mfrac><mo fence="true" form="postfix">)</mo></mrow><mn>12</mn></msup><mo>−</mo><msup><mrow><mo fence="true" form="prefix">(</mo><mfrac><mi>σ</mi><mi>r</mi></mfrac><mo fence="true" form="postfix">)</mo></mrow><mn>6</mn></msup><mo fence="true" form="postfix">]</mo></mrow><mo>+</mo><mi>ε</mi></mrow></mtd><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 1em;"><mrow><mi>r</mi><mo>&lt;</mo><msup><mn>2</mn><mrow><mn>1</mn><mi>/</mi><mn>6</mn></mrow></msup><mi>σ</mi></mrow></mtd></mtr><mtr><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 0em;"><mn>0</mn></mtd><mtd class="tml-left" style="padding:0.5ex 0em 0.5ex 1em;"><mrow><mi>r</mi><mo>≥</mo><msup><mn>2</mn><mrow><mn>1</mn><mi>/</mi><mn>6</mn></mrow></msup><mi>σ</mi></mrow></mtd></mtr></mtable><mo fence="true" form="postfix"></mo></mrow></mrow></math>
*/
pub struct WeeksChandlerAnderson {
    /// Energy scale `[energy]`.
    pub epsilon: f64,
    /// Interaction width `[length]`.
    pub sigma: f64
}

impl WeeksChandlerAnderson {
    #[inline]
    #[must_use]
    pub fn new(epsilon: f64, sigma: f64) -> Self {
        Self { epsilon, sigma }
        }
}

impl IsotropicEnergy for WeeksChandlerAnderson {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        if r < 2.0_f64.powf(1.0/6.0) * self.sigma {
            let lj = LennardJones::<12,6>::new(self.epsilon, self.sigma);
            lj.energy(r) + self.epsilon
            }
        else
            {
            0.0
            }
        }
    }

impl IsotropicForce for WeeksChandlerAnderson {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        if r < 2.0_f64.powf(1.0/6.0) * self.sigma {
            let lj = LennardJones::<12,6>::new(self.epsilon, self.sigma);
            lj.force(r)
            }
        else
            {
            0.0
            }
        }
    }

/** Shifted pairwise potential

<!-- U(r) = f(r) - f(r_\mathrm{shift}) -->
<math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo form="prefix" stretchy="false">(</mo><mi>r</mi><mo form="postfix" stretchy="false">)</mo><mo>=</mo><mi>f</mi><mo form="prefix" stretchy="false">(</mo><mi>r</mi><mo form="postfix" stretchy="false">)</mo><mo>−</mo><mi>f</mi><mo form="prefix" stretchy="false">(</mo><msub><mi>r</mi><mrow><mtext></mtext><mi>shift</mi></mrow></msub><mo form="postfix" stretchy="false">)</mo></mrow></math>
*/
pub struct Shifted<F> {
    /// The original potential.
    pub f: F,
    /// `r` value `[length]` where the shifted potential will be 0.
    pub r_shift: f64,
}

impl<F> Shifted<F> {
    #[inline]
    #[must_use]
    pub fn new(f: F, r_shift: f64) -> Self {
        Self { f, r_shift }
    }
}

impl<F: IsotropicEnergy> IsotropicEnergy for Shifted<F> {
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        self.f.energy(r) - self.f.energy(self.r_shift)
    }
}

impl<F: IsotropicForce> IsotropicForce for Shifted<F> {
    #[inline]
    fn force(&self, r: f64) -> f64 {
        self.f.force(r)
    }
}

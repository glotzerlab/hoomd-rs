// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement [`HarmonicField`]
 */

use super::super::SiteEnergy;
use hoomd_microstate::property::Position;
use hoomd_vector::{Vector};

/** Harmonic external field based on position.

<!-- U = \frac{1}{2} k (||\vec{r} - \vec{p}|| - d)^2-->
<math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo>=</mo><mi>α</mi><mo>⋅</mo><mover><mi>n</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo>⋅</mo><mo form="prefix" stretchy="false">(</mo><mover><mi>r</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo>−</mo><mover><mi>p</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo form="postfix" stretchy="false">)</mo></mrow></math>

Computes a harmonic field at a point in space relative to the center
point of harmonic field `p`, spring constant `k`, and the equilibrium spring 
length `d`.

# Example

Basic usage:

```
use hoomd_interaction::external::HarmonicField;
use hoomd_vector::{Cartesian, Unit};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let HarmonicField = HarmonicField { k: 2.0,
    center_point: [0.0, 0.0].into(),
    eq_distance: 1.0,
};
# Ok(())
# }
```
*/
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarmonicField<V> {
    /// Spring constant (`[energy] [length]^(-2)`).
    pub k: f64,
    /// Center of harmonic field (`[length]`).
    pub center_point: V,
    /// Equilibrium spring length (`[length]`).
    pub eq_distance: f64,  // should we assume it's PositiveReal?
}

impl<V> HarmonicField<V>
where
    V: Vector,
{
    /** Compute the energy of a point in the harmonic field.

    # Example

    ```
    use hoomd_interaction::external::HarmonicField;
    use hoomd_vector::{Cartesian, Unit};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let harmonicfield = HarmonicField { k: 2.0,
        center_point: [0.0, 0.0].into(),
        eq_distance: 1.0,
    };

    let energy = harmonicfield.energy(&[2.0, 0.0].into());
    assert_eq!(energy, 1.0);
    # Ok(())
    # }
    ```
    */
    #[inline]
    #[must_use]
    pub fn energy(&self, r: &V) -> f64 {
        let r_to_origin_vec = *r - self.center_point;
        let r_to_origin = (r_to_origin_vec.dot(&r_to_origin_vec)).sqrt();
        let current_spring_length = r_to_origin - self.eq_distance;

        0.5 * self.k * current_spring_length * current_spring_length
    }
}

impl<S, V> SiteEnergy<S> for HarmonicField<V>
where
    S: Position<V>,
    V: Vector,
{
    #[inline]
    fn site_energy(&self, site_properties: &S) -> f64 where {
        self.energy(site_properties.position())
    }
}

#[cfg(test)]
mod tests {
    use hoomd_vector::Cartesian;

    use super::*;
    use ::approx::{assert_relative_eq};
    use rstest::*;

    #[rstest]
    fn energy_2d(
        #[values(1.0, 2.0, 5.0)] k: f64,
        #[values([0.0, 0.0], [-1.0, 0.0], [2.0, 0.0])] center_point: [f64; 2],
        #[values(0.0, 1.0, 2.0)] eq_distance: f64,
    ) {
        
        let r = Cartesian::from([1.0, 0.0]);
        let harmonicfield = HarmonicField {
            k: k,
            center_point: center_point.into(),
            eq_distance: eq_distance,
        };
        
        let r_to_origin_vec = r - center_point.into();
        let r_to_origin = (r_to_origin_vec.dot(&r_to_origin_vec)).sqrt();
        let current_spring_length = r_to_origin - eq_distance;

        let expected_energy = 0.5 * k * current_spring_length * current_spring_length;

        assert_relative_eq!(harmonicfield.energy(&r), expected_energy, epsilon = 1e-9);
    }
}
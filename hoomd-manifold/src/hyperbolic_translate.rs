// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement HyperbolicTranslate
*/

use hoomd_mc::LocalTrial;
use hoomd_microstate::property::Position;
use hoomd_utility::valid::PositiveReal;
use crate::{Minkowski, HyperbolicDisk};

use rand::Rng;
use rand::distr::Distribution;

/** Move the position of a body in hyperbolic space by a small distance 

TODO: documentation, examples
*/

pub struct HyperbolicTranslate {
    // The max distance a body can be translated in one trial move
    pub maximum_distance: PositiveReal,
    pub skirt: f64,
}

impl<B> LocalTrial<B> for HyperbolicTranslate 
where
    B: Position<Vector = Minkowski<3>>,
    HyperbolicDisk: Distribution<Minkowski<3>>
{
    /** TODO: documentation, examples

    # Example 
    ```
    use hoomd_mc::{LocalTrial, Translate};
    use hoomd_microstate::property::{Point, Position};
    use hoomd_manifold::{Minkowski, Hyperboloid, HyperbolicTranslate};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(131);
    let rho : f64 = 1.0;
    let body_properties = Point::new(Minkowski::from([0.0, 0.0, rho]));
    let d = 0.1;
    let hyperbolic_translate = HyperbolicTranslate {maximum_distance: d.try_into()? ,
                                                    skirt: rho};

    let new_body_properties = hyperbolic_translate.propose(&mut rng, body_properties);
    assert!(d > new_body_properties.position().distance_from_cusp(rho));
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn propose<R: Rng>(&self, rng: &mut R, body_properties: B) -> B {
        let mut trial = body_properties; 
        let disk = HyperbolicDisk {
            r: self.maximum_distance,
            point: *trial.position_mut(),
            skirt: self.skirt,
        };
        *trial.position_mut() = disk.sample(rng);
        trial
    }
}

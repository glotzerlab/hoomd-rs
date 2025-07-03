// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement Translation moves on curved surfaces
*/

use hoomd_mc::LocalTrial;
use hoomd_microstate::property::Position;
use hoomd_utility::valid::PositiveReal;
use crate::{Minkowski, HyperbolicDisk, SphericalDisk};
use hoomd_vector::{Cartesian, Vector, InnerProduct};

use rand::Rng;
use rand::distr::Distribution;

/** Move the position of a body on a hyperbolic surface by a small distance 

TODO: documentation, examples

HyperbolicTranslate used with Sweep:
# Example
```
use hoomd_mc::{LocalTrial, Translate, Sweep, Trial, Zero};
use hoomd_microstate::{property::Position, Body, Microstate};
use hoomd_manifold::{Minkowski, Hyperboloid, HyperbolicTranslate};
use hoomd_vector::Vector;
use rand::{rngs::StdRng, Rng, SeedableRng};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = Microstate::new();
microstate.add_body(Body::point(Minkowski::from([1.0, 1.0, 3.0_f64.sqrt()])));
let d = 0.1;
let translate = HyperbolicTranslate { maximum_distance: d.try_into()?, skirt: 1.0,};
let translate_sweep = Sweep { local: translate };

let hamiltonian = Zero;
let kt = 1.0;

for _ in 0..1_000 {
    translate_sweep.apply(&mut microstate, &hamiltonian, &kt);
    microstate.increment_step();
}
# Ok(())
# }
```
*/

pub struct HyperbolicTranslate {
    /// The max distance a body can be translated in one trial move
    pub maximum_distance: PositiveReal,
    /// The skirt width of the hyperboloid
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
    use hoomd_vector::Vector;
    use libm::sqrt;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(13);
    let rho : f64 = 0.8;
    let body_properties = Point::new(Minkowski::from([1.0, -1.0, sqrt(2.0 + rho.powi(2))]));
    let d = 0.1 * rho;
    let hyperbolic_translate = HyperbolicTranslate {maximum_distance: d.try_into()? ,
                                                    skirt: rho};

    let new_body_properties = hyperbolic_translate.propose(&mut rng, body_properties);
    // assert_eq!(new_body_properties.position().distance_squared(&Minkowski::from([0.0,0.0,0.0])), -1.0* rho.powi(2));
    assert!(d > new_body_properties.position().hyperbolic_distance(&Minkowski::from([1.0, -1.0, sqrt(2.0 + rho.powi(2))]), rho));
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
        let z = (trial.position_mut()[0].powi(2) + trial.position_mut()[1].powi(2) + self.skirt.powi(2)).sqrt();
        trial.position_mut()[2] = z;
        trial
    }
}

/** Move the position of a body on the surface of a sphere by a small distance

TODO: documentation, examples

SphericalTranslate used with Sweep:
# Example
```
use hoomd_mc::{LocalTrial, Translate, Sweep, Trial, Zero};
use hoomd_microstate::{property::Position, Body, Microstate};
use hoomd_manifold::{Sphere, SphericalDisk, SphericalTranslate};
use hoomd_vector::{Cartesian, Vector};
use rand::{rngs::StdRng, Rng, SeedableRng};
use approx::assert_relative_eq;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut microstate = Microstate::new();
microstate.add_body(Body::point(Cartesian::from([0.5_f64.sqrt(), 0.5_f64.sqrt(), 0.0])));
let d = 0.1;
let radius: f64 = 0.6;
let translate = SphericalTranslate { maximum_distance: d.try_into()?, radius: radius,};
let translate_sweep = Sweep { local: translate };

let hamiltonian = Zero;
let kt = 1.0;

for _ in 0..1_000 {
    translate_sweep.apply(&mut microstate, &hamiltonian, &kt);
    microstate.increment_step();
}
assert_relative_eq!(
    microstate.bodies()[0].item.properties.position().distance_squared(&Cartesian::from([0.0,0.0,0.0])), 
    radius.powi(2),
    epsilon=1e-12);
# Ok(())
# }
```
*/

pub struct SphericalTranslate {
    /// The max distance a body can be translated in one trial move
    pub maximum_distance: PositiveReal,
    /// The radius of the sphere
    pub radius: f64,
}

impl<B> LocalTrial<B> for SphericalTranslate 
where
    B: Position<Vector = Cartesian<3>>,
    SphericalDisk: Distribution<Cartesian<3>>
{
    /** TODO: documentation, examples

    # Example 
    ```
    use hoomd_mc::{LocalTrial, Translate};
    use hoomd_microstate::property::{Point, Position};
    use hoomd_manifold::{Sphere, SphericalTranslate, SphericalDisk};
    use hoomd_vector::{Vector, Cartesian};
    use libm::sqrt;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use approx::assert_relative_eq;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(13);
    let radius : f64 = 2.0;
    let body_properties = Point::new(Cartesian::from([0.5_f64.sqrt(), 0.5_f64.sqrt(), 0.0]));
    let d = 0.1;
    let spherical_translate = SphericalTranslate {maximum_distance: d.try_into()? ,
                                                    radius: radius};

    let new_body_properties = spherical_translate.propose(&mut rng, body_properties);
    assert_relative_eq!(new_body_properties.position().distance_squared(&Cartesian::from([0.0,0.0,0.0])), radius.powi(2), epsilon=1e-12);
    assert!(d > new_body_properties.position().sphere_distance(&Cartesian::from([0.5_f64.sqrt(), 0.5_f64.sqrt(), 0.0]), radius));
    # Ok(())
    # }
    ```
    */
    #[inline]
    fn propose<R: Rng>(&self, rng: &mut R, body_properties: B) -> B {
        let mut trial = body_properties; 
        let disk = SphericalDisk {
            r: self.maximum_distance,
            point: *trial.position_mut(),
            radius: self.radius,
        };
        *trial.position_mut() = disk.sample(rng);
        let rescale = self.radius/trial.position_mut().norm();
        *trial.position_mut() *= rescale;
        trial
    }
}

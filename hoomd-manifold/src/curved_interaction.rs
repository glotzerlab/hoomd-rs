// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implement CurvedIsotropic
*/

use hoomd_interaction::{pairwise::IsotropicEnergy, SitePairEnergy};
use hoomd_microstate::property::Position;
use hoomd_vector::Vector;
use crate::CurvedManifold;
use approx::assert_relative_eq;
use std::marker::PhantomData;

/** [`CurvedManifold`] for Cartesian computes the arc length bewtween two points on an N-sphere of radius R. 
    For two points $\vec{u}$ and $\vec{v}$ on an N-sphere
    embedded in cartesian space, the arclength between \vec{u} and \vec{v} is given by 
    ```math
    d_S(\vec{u},\vec{v}) = R\delta\psi = R\arccos\left(\frac{\vec{u}\cdot\vec{v}}{R^2}\right)
    ```

    # Example
    ```
    use libm::acos;
    use std::f64::consts::PI;
    use hoomd_vector::{Cartesian, InnerProduct};
    use hoomd_manifold::Sphere;

    # fn main() -> Result<(), Box<dyn std::error::Error>> {
    let radius : f64 = 5.0;
    let x = Sphere::from(Cartesian::from([radius, 0.0, 0.0]));
    let y = Sphere::from(Cartesian::from([0.0, radius, 0.0]));
    assert_eq!(radius* PI/2.0, x.sphere_distance(&y));
    # Ok(())
    # }
    ```
    */

/** Compute isotropic properties from a pair of sites on a curved manifold, embedded 
in some vector space. Function is similar to [`Isotropic`], 
but the metric passed to the energy function is from the embedded manifold. 
*/
pub struct CurvedIsotropic<E, M> { 
    pub isotropic: E,
    pub manifold: M,
}

impl<V, S, E, M> SitePairEnergy<S> for CurvedIsotropic<E,M>
where
    S: Position<Vector = V>,
    V: Vector,
    E: IsotropicEnergy,
    M: CurvedManifold,
{
    #[inline]
    fn site_pair_energy(&self, a: &S, b: &S) -> f64 {
        let site_a :M = CurvedManifold::to_manifold(a.position().to_vec());
        let site_b :M = CurvedManifold::to_manifold(b.position().to_vec());
        self.isotropic.energy(site_a.geodesic_distance(&site_b))
    }
}

impl<E,M> IsotropicEnergy for CurvedIsotropic<E,M>
where
    E: IsotropicEnergy,
    M: CurvedManifold
{
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        self.isotropic.energy(r)
    }
}

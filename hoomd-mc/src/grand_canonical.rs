// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Grand Canonical ensemble simulation
use serde::{Deserialize, Serialize};

use hoomd_interaction::{DeltaEnergyInsert, DeltaEnergyRemove};
use hoomd_microstate::{
    Body, Microstate, SiteKey, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::Position,
};
use hoomd_simulation::macrostate::{Fugacity, Temperature};
use rand::RngExt;

use super::Trial;
use crate::BodyDistribution;
use hoomd_geometry::Volume;
use hoomd_spatial::PointUpdate;

/// Setup trial moves in the microstate that insert or remove bodies
///
/// The first field in the tuple struct is the `BodyDistribution`.
/// [`GrandCanonical::apply`] applies the insert/remove trial move to the microstate.
///
/// # Example
///
/// ```
/// use hoomd_geometry::shape::Rectangle;
/// use hoomd_mc::{GrandCanonical, UniformIn};
/// use hoomd_microstate::property::Point;
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let rectangle = Rectangle::with_equal_edges(10.0.try_into()?);
///
/// let distribution = UniformIn {
///     boundary: rectangle,
///     template_sites: vec![Point::<Cartesian<2>>::default()],
/// };
///
/// let gcmc = GrandCanonical(distribution);
///
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrandCanonical<D>(pub D);

/// Accepted and rejected counts of insert and remove moves
pub struct InsertRemoveCount {
    /// The number of insert accepted moves.
    pub insert_accepted: u64,
    /// The number of insert rejected moves.
    pub insert_rejected: u64,
    /// The number of remove accepted moves.
    pub remove_accepted: u64,
    /// The number of remove rejected moves.
    pub remove_rejected: u64,
}

impl<D, P, B, S, X, C, H, MA> Trial<Microstate<B, S, X, C>, H, MA> for GrandCanonical<D>
where
    D: BodyDistribution<Body<B, S>>,
    P: Copy,
    B: Copy + Default + Transform<S> + Position<Position = P>,
    S: Copy + Default + Position<Position = P>,
    X: PointUpdate<P, SiteKey>,
    H: DeltaEnergyInsert<B, S, X, C> + DeltaEnergyRemove<B, S, X, C>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S> + Volume,
    MA: Temperature + Fugacity,
{
    type Count = InsertRemoveCount;

    /// Apply the insert/remove moves to a microstate at constant
    /// temperature and fugacity set using [`hoomd_simulation::macrostate::IsothermalIsofugacity`].
    ///
    /// Combine [`GrandCanonical::apply`] with local trial moves that translate and/or
    /// rotate bodies by small amounts to sample the grand cannonical ensemble.
    ///
    /// Insert move is accepted when:
    /// ```math
    /// r < \frac{V f}{n + 1} \exp\left(\frac{-\Delta H}{kT}\right)
    /// ```
    ///
    /// Remove moves are accepted when:
    /// ```math
    /// r < \frac{n}{V f} \exp\left(\frac{-\Delta H}{kT}\right)
    /// ```
    ///
    /// where `r` is a random value uniformly distributed in `[0,1)`, $`V`$ is the volume, $`f`$ is the fugacity, $`n`$ is the number of particles, $`\Delta H`$ is
    /// the change in energy computed by the given `hamiltonian` and $`kT`$ is the
    /// `temperature` given in `macrostate`.
    ///
    ///
    /// # Example
    ///
    /// Hard spheres
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_interaction::{
    ///     PairwiseCutoff,
    ///     pairwise::Isotropic,
    ///     univariate::{Expanded, OverlapPenalty},
    /// };
    /// use hoomd_mc::{GrandCanonical, Sweep, Translate, Trial, UniformIn};
    /// use hoomd_microstate::{
    ///     Body, Microstate, boundary::Periodic, property::Point,
    /// };
    /// use hoomd_simulation::macrostate::IsothermalIsofugacity;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rectangle = Rectangle::with_equal_edges(10.0.try_into()?);
    ///
    /// let distribution = UniformIn {
    ///     boundary: rectangle.clone(),
    ///     template_sites: vec![Point::default()],
    /// };
    /// let mut gcmc = GrandCanonical(distribution);
    ///
    /// let translate = Translate::with_maximum_distance(0.1.try_into()?);
    /// let mut translate_sweep = Sweep(translate);
    ///
    /// let pairwise_cutoff = PairwiseCutoff(Isotropic {
    ///     interaction: Expanded {
    ///         delta: 1.0,
    ///         f: OverlapPenalty::default(),
    ///     },
    ///     r_cut: 1.0,
    /// });
    ///
    /// let macrostate = IsothermalIsofugacity {
    ///     temperature: 1.0,
    ///     fugacity: 1.0,
    /// };
    /// let mut microstate = Microstate::builder()
    ///     .boundary(Periodic::new(1.0, rectangle)?)
    ///     .bodies([Body::point(Cartesian::from([0.0, 0.0]))])
    ///     .try_build()?;
    ///
    /// gcmc.apply(&mut microstate, &pairwise_cutoff, &macrostate);
    ///
    /// translate_sweep.apply(&mut microstate, &pairwise_cutoff, &macrostate);
    ///
    /// assert!(microstate.bodies().len() > 1);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn apply(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        hamiltonian: &H,
        macrostate: &MA,
    ) -> Self::Count {
        let kt = *macrostate.temperature();
        let fugacity = *macrostate.fugacity();
        let n = microstate.bodies().len();
        let vol = microstate.boundary().volume();
        let mut rng = microstate.counter().make_rng();
        let mut count = InsertRemoveCount {
            insert_accepted: 0,
            insert_rejected: 0,
            remove_accepted: 0,
            remove_rejected: 0,
        };

        let move_type_r: f64 = rng.random();

        // in case n is zero and f is zero, no moves can be attempted
        if n == 0 && fugacity <= 0.0 {
            microstate.increment_substep();
            return count;
        }

        // insert
        if move_type_r > 0.5 || microstate.bodies().is_empty() {
            let new_body = self.0.sample(0, &mut rng);

            let delta_energy = hamiltonian.delta_energy_insert(microstate, &new_body);
            if delta_energy.is_finite() {
                let p_insert = (vol * fugacity * (-delta_energy / kt).exp()) / (n as f64 + 1.0);

                if p_insert > rng.random() && microstate.add_body(new_body).is_ok() {
                    count.insert_accepted += 1;
                } else {
                    count.insert_rejected += 1;
                }
            } else {
                count.insert_rejected += 1;
            }
        }
        // remove
        else {
            let index = rng.random_range(..n);
            let delta_energy = hamiltonian.delta_energy_remove(microstate, index);

            let p_remove = if fugacity <= 0.0 {
                1.0
            } else {
                (n as f64 * (-delta_energy / kt).exp()) / (vol * fugacity)
            };

            if p_remove > rng.random() {
                microstate.remove_body(index);
                count.remove_accepted += 1;
            } else {
                count.remove_rejected += 1;
            }
        }
        microstate.increment_substep();
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{QuickInsert, Sweep, Translate, Trial, UniformIn};
    use assert2::check;
    use hoomd_geometry::shape::Rectangle;
    use hoomd_interaction::{PairwiseCutoff, TotalEnergy, pairwise::Isotropic, univariate::Boxcar};
    use hoomd_microstate::{Microstate, boundary::Closed, property::Point};
    use hoomd_simulation::macrostate::IsothermalIsofugacity;
    use hoomd_vector::Cartesian;

    #[test]
    fn hard_spheres_fugacity() {
        let sigma = 1.0;
        let epsilon = f64::INFINITY;
        let kt = 1.0;

        let hamiltonian = PairwiseCutoff(Isotropic {
            interaction: Boxcar {
                left: 0.0,
                right: sigma,
                epsilon,
            },
            r_cut: sigma,
        });

        let translate =
            Translate::with_maximum_distance(0.1.try_into().expect("hard-coded value is non-zero"));
        let mut translate_sweep = Sweep(translate);

        let rectangle = Closed(Rectangle::with_equal_edges(
            6.0.try_into().expect("hard-coded value is non-zero"),
        ));

        let mut microstate = Microstate::builder()
            .boundary(rectangle.clone())
            .bodies(vec![Body::point(Cartesian::from([0.0, 0.0]))])
            .try_build()
            .expect("hard-coded point is in the boundary");
        let macrostate = IsothermalIsofugacity {
            temperature: kt,
            fugacity: 0.0,
        };

        let distribution = UniformIn {
            boundary: rectangle.clone(),
            template_sites: vec![Point::new([0.0, 0.0].into())],
        };
        let mut quick_insert = QuickInsert::new(distribution.clone(), 10);

        for _ in 0..100 {
            quick_insert.apply(&mut microstate, &hamiltonian);
            if quick_insert.is_complete() {
                break;
            }
        }

        translate_sweep.apply(&mut microstate, &hamiltonian, &macrostate);

        assert!(quick_insert.is_complete());
        assert_eq!(microstate.bodies().len(), 11);
        assert_eq!(hamiltonian.total_energy(&microstate), 0.0);

        let mut gcmc = GrandCanonical(distribution);

        for _ in 0..100 {
            gcmc.apply(&mut microstate, &hamiltonian, &macrostate);
        }

        assert_eq!(microstate.bodies().len(), 0);

        let new_macrostate = IsothermalIsofugacity {
            temperature: kt,
            fugacity: 1.0,
        };
        for _ in 0..100 {
            gcmc.apply(&mut microstate, &hamiltonian, &new_macrostate);
        }

        check!(microstate.bodies().len() > 0);
    }
}

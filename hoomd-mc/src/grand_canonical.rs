// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement Grand Canonical ensemble simulation

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
/// use hoomd_mc::{UniformIn, GrandCanonical};
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
    /// use hoomd_mc::{Sweep, Translate, Trial, UniformIn, GrandCanonical};
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
    /// let gcmc = GrandCanonical(distribution);
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
    /// let macrostate = IsothermalIsofugacity { temperature: 1.0, fugacity: 1.0 };
    /// let mut microstate = Microstate::builder()
    ///     .boundary(Periodic::new(1.0, rectangle)?)
    ///     .bodies([Body::point(Cartesian::from([0.0, 0.0]))])
    ///     .try_build()?;
    ///
    /// gcmc.apply(&mut microstate, &pairwise_cutoff);
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
        let kt = macrostate.temperature();
        let fugacity = macrostate.fugacity();
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

        // insert
        if move_type_r > 0.5 || microstate.bodies().is_empty() {
            let new_body = self.0.sample(0, &mut rng);

            let delta_energy = hamiltonian.delta_energy_insert(microstate, &new_body);
            if delta_energy.is_finite() {
                let p_insert = (vol * fugacity * (-delta_energy / kt).exp()) / (n as f64 + 1.0);

                if p_insert > rng.random() && microstate.add_body(new_body).is_ok() {
                    count.insert_accepted += 1;
                }

                else {
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

            let p_remove = (n as f64 * (-delta_energy / kt).exp()) / (vol * fugacity);

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

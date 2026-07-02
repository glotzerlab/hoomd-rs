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

pub struct GrandCanonical<D> {
    /// Sample random bodies to insert.
    distribution: D,
}

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

impl<D> GrandCanonical<D> {
    #[inline]
    pub fn new(distribution: D) -> Self {
        Self { distribution }
    }

    #[inline]
    pub fn apply_move<P, B, S, X, C, H, MA>(
        &self,
        microstate: &mut Microstate<B, S, X, C>,
        hamiltonian: &H,
        macrostate: &MA,
    ) -> InsertRemoveCount
    where
        P: Copy,
        B: Copy + Default + Transform<S> + Position<Position = P>,
        S: Copy + Default + Position<Position = P>,
        X: PointUpdate<P, SiteKey>,
        D: BodyDistribution<Body<B, S>>,
        H: DeltaEnergyInsert<B, S, X, C> + DeltaEnergyRemove<B, S, X, C>,
        C: Wrap<B> + Wrap<S> + GenerateGhosts<S> + Volume,
        MA: Temperature + Fugacity,
    {
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
            let new_body = self.distribution.sample(0, &mut rng);
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

    #[inline]
    fn apply(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        hamiltonian: &H,
        macrostate: &MA,
    ) -> Self::Count {
        self.apply_move(microstate, hamiltonian, macrostate)
    }
}

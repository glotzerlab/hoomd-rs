// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `ParallelSweep`

use rand::{Rng, seq::IndexedRandom};
use rayon::prelude::*;
use std::fmt::Display;
use serde::{Serialize, Deserialize};

use super::{Adjust, Count, LocalTrial, Trial, Tune, tune_local::tune_local_trial};
use hoomd_interaction::DeltaEnergyOne;
use hoomd_microstate::{
    Body, Microstate, SiteKey, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::Position,
};
use hoomd_simulation::macrostate::Temperature;
use hoomd_spatial::PointUpdate;
use hoomd_utility::valid::{OpenUnitIntervalNumber, PositiveReal};

use crate::{Checkerboard, Cover};

/// Result of a trial move
#[derive(Clone, Debug, Default, PartialEq)]
enum TrialStatus {
    /// The trial move was rejected by the Metropolis criterion.
    #[default]
    Rejected,

    /// The trial move was accepted.
    Accepted,

    /// Either no trial move was performed (the space was empty) or the trial move left the space.
    Invalid,
}

/// A body trial move and its status
///
/// `ParallelSweep` generates and evaluates many body trial moves in parallel. `BodyTrial`
/// stores both the trial and its status.
#[derive(Clone, Debug, Default, PartialEq)]
struct BodyTrial<B, S> {
    /// The trial body configuration.
    body: Body<B, S>,

    /// Index of the body in the microstate.
    body_index: usize,

    /// Status of the trial.
    status: TrialStatus,
}

/// In parallel, apply local trial moves to bodies in the microstate.
///
/// The generic type names are:
/// * `L`: The [`LocalTrial`](crate::LocalTrial) type.
/// * `K`: The [`Checkerboard`](crate::Checkerboard) type.
/// * `B`: The [`Body::properties`](hoomd_microstate::Body) type.
/// * `S`: The [`Site::properties`](hoomd_microstate::Site) type.
///
/// # Example
///
/// ```
/// use hoomd_mc::{HypercuboidCheckerboard, ParallelSweep, Translate};
/// use hoomd_microstate::property::Point;
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let d = 0.1;
/// let translate =
///     Translate::<Cartesian<2>>::with_maximum_distance(d.try_into()?);
/// let translate_sweep = ParallelSweep::<
///     _,
///     HypercuboidCheckerboard<2>,
///     Point<Cartesian<2>>,
///     Point<Cartesian<2>>,
/// >::new(1.0.try_into()?, translate);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParallelSweep<L, K, B, S> {
    /// The largest distance between any two interacting bodies.
    body_interaction_range: PositiveReal,

    /// Apply this local trial to bodies in the microstate.
    local_trial: L,

    /// Cached checkerboard generated on the last call (can be updated incrementally).
    checkerboard: K,

    /// Cached storage of the body indices assigned to each space.
    spaces: Vec<Vec<usize>>,

    /// Cached storage of the body trial moves in each space.
    #[serde(skip)]
    body_trials: Vec<BodyTrial<B, S>>,
}

impl<L, K, B, S> ParallelSweep<L, K, B, S>
where
    K: Default,
{
    /// Construct a new `ParallelSweep`.
    ///
    /// To avoid applying conflicting moves at the same time, you need to
    /// provide the largest distance between any two interacting bodies:
    /// `body_interaction_range` (measured between the body positions).
    /// `ParallelSweep` cannot compute this value, nor can it validate
    /// whether the provided value is correct.
    ///
    /// The `local_trial` arguments determines what trial moves `ParallelSweep`
    /// attempts.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_mc::{HypercuboidCheckerboard, ParallelSweep, Translate};
    /// use hoomd_microstate::property::Point;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let d = 0.1;
    /// let translate =
    ///     Translate::<Cartesian<2>>::with_maximum_distance(d.try_into()?);
    /// let translate_sweep = ParallelSweep::<
    ///     _,
    ///     HypercuboidCheckerboard<2>,
    ///     Point<Cartesian<2>>,
    ///     Point<Cartesian<2>>,
    /// >::new(1.0.try_into()?, translate);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn new(body_interaction_range: PositiveReal, local_trial: L) -> Self {
        Self {
            body_interaction_range,
            local_trial,
            checkerboard: K::default(),
            spaces: Vec::new(),
            body_trials: Vec::new(),
        }
    }
}

impl<L, K, B, S> ParallelSweep<L, K, B, S> {
    /// Get the body interaction range.
    #[inline]
    pub fn body_interaction_range(&self) -> &PositiveReal {
        &self.body_interaction_range
    }

    /// Get the body interaction range (mutable).
    #[inline]
    pub fn body_interaction_range_mut(&mut self) -> &mut PositiveReal {
        &mut self.body_interaction_range
    }

    /// Get the local trial.
    #[inline]
    pub fn local_trial(&self) -> &L {
        &self.local_trial
    }

    /// Get the local trial (mutable).
    #[inline]
    pub fn local_trial_mut(&mut self) -> &mut L {
        &mut self.local_trial
    }

    /// Make the cached checkerboard consistent with the current microstate's boundary.
    #[inline(always)]
    fn update_checkerboard<P, X, C>(&mut self, microstate: &Microstate<B, S, X, C>)
    where
        B: Position<Position = P>,
        K: Checkerboard<P>,
        C: Cover<P, Checkerboard = K>,
    {
        let mut checkerboard_rng = microstate.counter().index(u64::MAX).make_rng();
        microstate.boundary().cover_into(
            &mut self.checkerboard,
            &mut checkerboard_rng,
            self.body_interaction_range,
        );

        self.spaces
            .resize_with(self.checkerboard.num_spaces(), Vec::default);
        for space in &mut self.spaces {
            space.truncate(0);
        }
        for (body_index, body) in microstate.bodies().iter().enumerate() {
            let space_index = self
                .checkerboard
                .point_to_space_index(body.item.properties.position());
            self.spaces[space_index.expect("body should be inside the checkerboard")]
                .push(body_index);
        }
    }

    /// Generate trial moves in parallel.
    #[expect(
        clippy::too_many_arguments,
        reason = "many arguments must be passed to avoid too many lines of code in other methods"
    )]
    #[inline(always)]
    fn generate_trial_moves<P, X, C, H>(
        local_trial: &L,
        body_trials: &mut Vec<BodyTrial<B, S>>,
        microstate: &Microstate<B, S, X, C>,
        hamiltonian: &H,
        kt: f64,
        checkerboard: &K,
        spaces: &[Vec<usize>],
        space_indices: &Vec<usize>,
    ) where
        P: Copy,
        B: Copy + Default + Transform<S> + Position<Position = P> + Send + Sync,
        S: Copy + Default + Position<Position = P> + Send + Sync,
        L: LocalTrial<B> + Sync,
        H: DeltaEnergyOne<B, S, X, C> + Sync,
        C: Wrap<B> + Wrap<S> + GenerateGhosts<S> + Sync,
        K: Checkerboard<P> + Sync,
        X: Sync,
    {
        body_trials.resize_with(space_indices.len(), Default::default);

        body_trials
            .par_iter_mut()
            .zip(space_indices)
            .for_each(|(body_trial, space_index)| {
                // body_trials.iter_mut().zip(space_indices).for_each(|(body_trial, space_index)| {
                body_trial.status = TrialStatus::Invalid;
                let space_index = *space_index;
                let mut rng = microstate.counter().index(space_index as u64).make_rng();

                if let Some(body_index) = spaces[space_index].choose(&mut rng) {
                    let body_index = *body_index;
                    body_trial
                        .body
                        .clone_from(&microstate.bodies()[body_index].item);
                    body_trial.body_index = body_index;

                    match microstate
                        .boundary()
                        .wrap(local_trial.propose(&mut rng, body_trial.body.properties))
                    {
                        Ok(new_properties)
                            if checkerboard.point_to_space_index(new_properties.position())
                                != Some(space_index) =>
                        {
                            body_trial.status = TrialStatus::Invalid;
                        }
                        Ok(new_properties) => {
                            body_trial.body.properties = new_properties;

                            let delta_h = hamiltonian.delta_energy_one(
                                microstate,
                                body_index,
                                &body_trial.body,
                            );
                            if delta_h != f64::INFINITY
                                && (delta_h <= 0.0 || rng.random::<f64>() < (-delta_h / kt).exp())
                            {
                                body_trial.status = TrialStatus::Accepted;
                            } else {
                                body_trial.status = TrialStatus::Rejected;
                            }
                        }
                        Err(_) => body_trial.status = TrialStatus::Rejected,
                    }
                }
            });
    }

    /// Update the properties of bodies with accepted moves.
    #[inline(always)]
    fn update_bodies<P, X, C>(
        microstate: &mut Microstate<B, S, X, C>,
        body_trials: &Vec<BodyTrial<B, S>>,
    ) -> Count
    where
        P: Copy,
        B: Copy + Default + Transform<S> + Position<Position = P>,
        S: Copy + Default + Position<Position = P>,
        X: PointUpdate<P, SiteKey>,
        C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    {
        let mut count = Count::default();

        for body_trial in body_trials {
            match body_trial.status {
                TrialStatus::Rejected => count.rejected += 1,
                TrialStatus::Invalid => {}
                TrialStatus::Accepted => {
                    if microstate
                        .update_body_properties(body_trial.body_index, body_trial.body.properties)
                        .is_ok()
                    {
                        count.accepted += 1;
                    } else {
                        count.rejected += 1;
                    }
                }
            }
        }
        count
    }
}

impl<P, B, S, X, C, L, H, MA, K> Trial<Microstate<B, S, X, C>, H, MA> for ParallelSweep<L, K, B, S>
where
    P: Copy,
    B: Copy + Default + Transform<S> + Position<Position = P> + Send + Sync,
    S: Copy + Default + Position<Position = P> + Send + Sync,
    X: PointUpdate<P, SiteKey> + Sync,
    L: LocalTrial<B> + Sync,
    H: DeltaEnergyOne<B, S, X, C> + Sync,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S> + Cover<P, Checkerboard = K> + Sync,
    MA: Temperature,
    K: Checkerboard<P> + Sync,
{
    type Count = Count;

    /// In parallel, apply local trial moves to bodies in the microstate.
    ///
    /// Each trial move is accepted when:
    /// ```math
    /// r < \exp\left(\frac{-\Delta H}{kT}\right)
    /// ```
    /// where `r` is a random value uniformly distributed in `[0,1)`, $`\Delta H`$ is
    /// the change in energy computed by the given `hamiltonian` and $`kT`$ is the
    /// `temperature` given in `macrostate`.
    ///
    /// `ParallelSweep` covers the simulation domain with a colored checkerboard
    /// (given by the `K` type). For each color, `apply` applies trial moves
    /// in parallel to bodies checkerboard spaces of that color (one trial move
    /// per space). A move is invalid when the body center leaves the initial
    /// checkerboard space. The return value counts only valid moves.
    ///
    /// One invocation of `apply` continues to apply trial moves until it
    /// has attempted at least as many trial moves as there are bodies in
    /// the microstate. Therefore, one call to `ParallelSweep::apply` roughly
    /// equivalent to one call to [`Sweep::apply`]. However, `ParallelSweep`
    /// will *always* attempt more trial moves because it rounds up. To make
    /// quantitative comparisons between the methods, normalize by the number of
    /// attempted moves.
    ///
    /// [`Sweep::apply`]: crate::Sweep::apply
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_interaction::Zero;
    /// use hoomd_mc::{ParallelSweep, Translate, Trial};
    /// use hoomd_microstate::{
    ///     Body, Microstate, boundary::Closed, property::Position,
    /// };
    /// use hoomd_simulation::macrostate::Isothermal;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let square = Rectangle::with_equal_edges(10.0.try_into()?);
    /// let mut microstate = Microstate::builder()
    ///     .boundary(Closed(square))
    ///     .bodies([Body::point(Cartesian::from([0.0, 0.0]))])
    ///     .try_build()?;
    /// let d = 0.1;
    /// let translate = Translate::with_maximum_distance(d.try_into()?);
    /// let mut translate_sweep = ParallelSweep::new(1.0.try_into()?, translate);
    ///
    /// let hamiltonian = Zero;
    /// let macrostate = Isothermal { temperature: 1.0 };
    ///
    /// translate_sweep.apply(&mut microstate, &hamiltonian, &macrostate);
    /// microstate.increment_step();
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

        self.update_checkerboard(microstate);

        let mut count = Self::Count::default();

        while count.total() < microstate.bodies().len() as u64 {
            for space_indices in self.checkerboard.space_indices_by_color() {
                Self::generate_trial_moves(
                    &self.local_trial,
                    &mut self.body_trials,
                    microstate,
                    hamiltonian,
                    *kt,
                    &self.checkerboard,
                    &self.spaces,
                    space_indices,
                );

                count += Self::update_bodies(microstate, &self.body_trials);
            }
            microstate.increment_substep();
        }

        count
    }
}

impl<P, B, S, X, C, L, H, MA, K> Tune<P, B, S, X, C, L, H, MA> for ParallelSweep<L, K, B, S>
where
    P: Copy,
    B: Copy + Default + Transform<S> + Position<Position = P>,
    S: Copy + Default + Position<Position = P>,
    X: PointUpdate<P, SiteKey>,
    L: LocalTrial<B> + Adjust + Display,
    H: DeltaEnergyOne<B, S, X, C>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    MA: Temperature,
{
    /// Tune the trial move maximum size to achieve a given acceptance ratio.
    ///
    /// Use [`tune_default`] unless you have a specific need to adjust the
    /// tuning parameters.
    ///
    /// [`tune_default`]: Self::tune_default
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_interaction::{
    ///     MaximumInteractionRange, PairwiseCutoff, pairwise::HardSphere,
    /// };
    /// use hoomd_mc::{ParallelSweep, Translate, Trial, Tune};
    /// use hoomd_microstate::{
    ///     Body, Microstate, boundary::Periodic, property::Position,
    /// };
    /// use hoomd_simulation::macrostate::Isothermal;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let square = Rectangle::with_equal_edges(10.0.try_into()?);
    /// let mut microstate = Microstate::builder()
    ///     .boundary(Periodic::new(1.0, square)?)
    ///     .try_build()?;
    /// microstate.add_body(Body::point(Cartesian::from([-0.6, -0.6])))?;
    /// microstate.add_body(Body::point(Cartesian::from([-0.6, 0.6])))?;
    /// microstate.add_body(Body::point(Cartesian::from([0.6, -0.6])))?;
    /// microstate.add_body(Body::point(Cartesian::from([0.6, 0.6])))?;
    /// let d = 0.1;
    /// let translate = Translate::with_maximum_distance(d.try_into()?);
    /// let mut translate_sweep = ParallelSweep::new(1.0.try_into()?, translate);
    ///
    /// let hamiltonian = PairwiseCutoff(HardSphere { diameter: 1.0 });
    /// let macrostate = Isothermal { temperature: 1.0 };
    ///
    /// translate_sweep.tune_default(&microstate, &hamiltonian, &macrostate);
    ///
    /// translate_sweep.apply(&mut microstate, &hamiltonian, &macrostate);
    ///
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn tune(
        &mut self,
        microstate: &Microstate<B, S, X, C>,
        hamiltonian: &H,
        macrostate: &MA,
        target_acceptance: OpenUnitIntervalNumber,
        samples: usize,
        steps: usize,
    ) {
        tune_local_trial(
            &mut self.local_trial,
            microstate,
            hamiltonian,
            macrostate,
            target_acceptance,
            samples,
            steps,
        );
    }
}

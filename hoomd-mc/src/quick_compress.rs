// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `QuickCompress`
use rand::distr::{Distribution, Uniform};

use hoomd_geometry::{MapPoint, Scale, Volume};
use hoomd_interaction::TotalEnergy;
use hoomd_microstate::{
    Body, Microstate, SiteKey, Tagged, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::Position,
};
use hoomd_spatial::PointUpdate;
use hoomd_utility::valid::{OpenUnitIntervalNumber, PositiveReal};

/// Track the state of a given `QuickCompress` instance.
#[derive(Default, Clone, Debug, PartialEq)]
enum State<C> {
    /// `apply` has not yet been called.
    #[default]
    Startup,
    /// Waiting for all overlaps to be removed.
    ///
    /// The first field indicates whether the compression that led to this state
    /// achieved the target volume. The second field stores a clone of the last
    /// boundary known to `QuickCompress`.
    Waiting(bool, C),
    /// Compressing the system
    Compressing,
    /// The system is at the target volume and all overlaps have been removed.
    ///
    /// The field stores a clone of the final boundary.
    Complete(C),
}

/// Quickly compress a microstate to a target volume.
///
/// [`QuickCompress`] allows you to *quickly* compress a microstate to a
/// target volume. It does so by *breaking detailed balance*, so you should
/// use it only during the initialization phase of your simulation where
/// you prepare a microstate for later equilibration. It works best with the
/// [`OverlapPenalty`] potential that allows sites to overlap a small amount and
/// for trial moves to partially reduce that overlap.
///
/// As a **algorithm**, [`QuickCompress`] is more than just a trial move, it
/// stores internal state to track its progress. Therefore, you should only use
/// a given [`QuickCompress`] on one [`Microstate`].
///
/// When not complete, [`apply`] takes the following steps:
/// 1. Check the total energy of the given Hamiltonian.
/// 2. If the total energy is less than or equal to zero *and* the microstate
///    boundary's volume does not match [`target_volume`], compress (or
///    expand) the boundary a small amount toward the target.  Reject any
///    trial move that increases the per-site energy of the system more than
///    [`maximum_energy_per_site`]. Accept in all other cases.
///
/// When the target volume is reached *and* the total energy is less than
/// or equal to 0, [`QuickCompress`] transitions to the complete state.
/// When complete, [`is_complete`] returns `true` and [`apply`] returns
/// immediately.
///
/// The generic type names are:
/// * `C`: The [`boundary`](hoomd_microstate::boundary) condition type.
///
/// [`apply`]: Self::apply
/// [`maximum_energy_per_site`]: Self::maximum_energy_per_site
/// [`target_volume`]: Self::target_volume
/// [`is_complete`]: Self::is_complete
/// [`OverlapPenalty`]: hoomd_interaction::univariate::OverlapPenalty
///
/// # Example
///
/// ```
/// use hoomd_geometry::shape::Rectangle;
/// use hoomd_mc::QuickCompress;
/// use hoomd_microstate::boundary::Closed;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let quick_compress = QuickCompress::<Closed<Rectangle>>::with_target_volume(
///     1000.0.try_into()?,
/// );
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct QuickCompress<C> {
    /// Final boundary volume after the compression algorithm completes.
    target_volume: PositiveReal,

    /// Per-site energy threshold at which to reject compression moves.
    maximum_energy_per_site: f64,

    /// Current state of the method.
    state: State<C>,

    /// Maximum value of delta.
    maximum_delta: OpenUnitIntervalNumber,
}

impl<C> QuickCompress<C> {
    /// Construct a new [`QuickCompress`] with the given target volume.
    ///
    /// The returned [`QuickCompress`] will compress (or expand) the
    /// microstate until the boundary reaches `target_volume`.
    ///
    /// * [`maximum_energy_per_site`] defaults to `25.0`.
    /// * [`maximum_delta`] defaults to `0.01`.
    ///
    /// These defaults are suitable for use with the default
    /// [`OverlapPenalty`] parameters applied to typical hard-particle
    /// systems.
    ///
    /// [`maximum_energy_per_site`]: Self::maximum_energy_per_site
    /// [`maximum_delta`]: Self::maximum_delta
    /// [`OverlapPenalty`]: hoomd_interaction::univariate::OverlapPenalty
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_mc::QuickCompress;
    /// use hoomd_microstate::boundary::Closed;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let quick_compress = QuickCompress::<Closed<Rectangle>>::with_target_volume(
    ///     1000.0.try_into()?,
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
    #[expect(
        clippy::missing_panics_doc,
        reason = "Panic would occur due to a bug in hoomd-rs."
    )]
    pub fn with_target_volume(target_volume: PositiveReal) -> Self {
        Self {
            target_volume,
            maximum_energy_per_site: 25.0,
            state: State::default(),
            maximum_delta: 0.01
                .try_into()
                .expect("hard-coded constant should be in the open unit interval"),
        }
    }

    /// Check if the compression has completed.
    ///
    /// `is_complete` will return `true` when the microstate has reached the
    /// target volume *and* the total energy of the system was less than or
    /// equal to zero during a call to [`apply`]. After that, `is_complete`
    /// will continue to return `true` even when the system energy changes.
    ///
    /// [`apply`]: Self::apply
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_mc::QuickCompress;
    /// use hoomd_microstate::boundary::Closed;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let quick_compress = QuickCompress::<Closed<Rectangle>>::with_target_volume(
    ///     1000.0.try_into()?,
    /// );
    ///
    /// assert!(!quick_compress.is_complete());
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn is_complete(&self) -> bool {
        matches!(self.state, State::Complete(_))
    }

    /// Get the target volume
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_mc::QuickCompress;
    /// use hoomd_microstate::boundary::Closed;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let quick_compress = QuickCompress::<Closed<Rectangle>>::with_target_volume(
    ///     1000.0.try_into()?,
    /// );
    ///
    /// let target_volume = quick_compress.target_volume();
    ///
    /// assert_eq!(target_volume.get(), 1000.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn target_volume(&self) -> PositiveReal {
        self.target_volume
    }

    /// Set the target volume.
    ///
    /// Setting a new value will transition a complete [`QuickCompress`] to
    /// an active state.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_mc::QuickCompress;
    /// use hoomd_microstate::boundary::Closed;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut quick_compress =
    ///     QuickCompress::<Closed<Rectangle>>::with_target_volume(
    ///         1000.0.try_into()?,
    ///     );
    ///
    /// quick_compress.set_target_volume(500.0.try_into()?);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn set_target_volume(&mut self, target_volume: PositiveReal)
    where
        C: Clone,
    {
        // Callers will expect `QuickCompress::compress` to seamlessly move
        // toward the new target. How to do that depends on the current state.
        // We cannot always switch to `Startup` because a caller *might* call
        // `set_target_volume` every time they call `compress` and that would
        // result in no progress.
        if self.target_volume != target_volume {
            self.state = match self.state {
                State::Startup | State::Complete(_) => State::Startup,
                State::Waiting(_, ref last_known_boundary) => {
                    State::Waiting(false, last_known_boundary.clone())
                }
                State::Compressing => State::Compressing,
            };
        }

        self.target_volume = target_volume;
    }

    /// Get the maximum energy per site.
    ///
    /// The largest energy per site allowed during [`apply`].
    ///
    /// [`apply`]: Self::apply
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_mc::QuickCompress;
    /// use hoomd_microstate::boundary::Closed;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let quick_compress = QuickCompress::<Closed<Rectangle>>::with_target_volume(
    ///     1000.0.try_into()?,
    /// );
    ///
    /// let maximum_energy_per_site = quick_compress.maximum_energy_per_site();
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn maximum_energy_per_site(&self) -> f64 {
        self.maximum_energy_per_site
    }

    /// A mutable reference to the maximum energy per site.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_mc::QuickCompress;
    /// use hoomd_microstate::boundary::Closed;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut quick_compress =
    ///     QuickCompress::<Closed<Rectangle>>::with_target_volume(
    ///         1000.0.try_into()?,
    ///     );
    ///
    /// *quick_compress.maximum_energy_per_site_mut() = 10.0;
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn maximum_energy_per_site_mut(&mut self) -> &mut f64 {
        &mut self.maximum_energy_per_site
    }

    /// Get the maximum value of delta.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_mc::QuickCompress;
    /// use hoomd_microstate::boundary::Closed;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let quick_compress = QuickCompress::<Closed<Rectangle>>::with_target_volume(
    ///     1000.0.try_into()?,
    /// );
    ///
    /// let maximum_delta = quick_compress.maximum_delta();
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn maximum_delta(&self) -> OpenUnitIntervalNumber {
        self.maximum_delta
    }

    /// A mutable reference to the maximum value of delta.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_mc::QuickCompress;
    /// use hoomd_microstate::boundary::Closed;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut quick_compress =
    ///     QuickCompress::<Closed<Rectangle>>::with_target_volume(
    ///         1000.0.try_into()?,
    ///     );
    ///
    /// *quick_compress.maximum_delta_mut() = 0.001.try_into()?;
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn maximum_delta_mut(&mut self) -> &mut OpenUnitIntervalNumber {
        &mut self.maximum_delta
    }

    /// Apply the quick compress algorithm to a microstate.
    ///
    /// When the total energy of the microstate (given by `hamiltonian`) is less
    /// than or equal to 0, `apply` scales the microstate's boundary toward
    /// the [`target_volume`] by an amount $` 1 \pm \delta `$ where $` \delta `$
    /// is a randomly sampled value between 0 and [`maximum_delta`].
    ///
    /// `should_map_body` controls how bodies in the microstate are transformed.
    /// When `should_map_body` returns `true`, the body's position will be mapped
    /// from the initial boundary to the final (via [`MapPoint`]). When
    /// `should_map_body` returns `false`, the body position is wrapped
    /// into the boundary (via [`Wrap`]).
    ///
    /// `apply` then evaluates the total energy change between the initial
    /// and compressed microstates. To avoid introducing too much stress
    /// into the system, it rejects moves where the total change in energy
    /// per site is greater than [`maximum_energy_per_site`].
    ///
    /// Combine `apply` with local trial moves that translate and/or rotate
    /// bodies by small amounts to relieve the stress caused by inserting
    /// overlapping sites. Without local moves, the quick compress algorithm
    /// will not achieve the target volume.
    ///
    /// [`target_volume`]: Self::target_volume
    /// [`maximum_delta`]: Self::maximum_delta
    /// [`maximum_energy_per_site`]: Self::maximum_energy_per_site
    ///
    /// # Example
    ///
    /// Hard spheres
    /// ```
    /// use anyhow::anyhow;
    ///
    /// use hoomd_geometry::{Volume, shape::Rectangle};
    /// use hoomd_interaction::{
    ///     PairwiseCutoff,
    ///     pairwise::Isotropic,
    ///     univariate::{Expanded, OverlapPenalty},
    /// };
    /// use hoomd_mc::{QuickCompress, Sweep, Translate, Trial};
    /// use hoomd_microstate::{
    ///     Body, Microstate, boundary::Periodic, property::Point,
    /// };
    /// use hoomd_simulation::macrostate::Isothermal;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let rectangle = Rectangle::with_equal_edges(16.0.try_into()?);
    ///
    /// let mut quick_compress = QuickCompress::with_target_volume(200.0.try_into()?);
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
    /// let macrostate = Isothermal { temperature: 1.0 };
    /// let mut microstate = Microstate::builder()
    ///     .boundary(Periodic::new(1.0, rectangle)?)
    ///     .try_build()?;
    /// for j in 0..8 {
    ///     let y = -8.0 + j as f64 * 2.0;
    ///     for i in 0..8 {
    ///         let x = -8.0 + i as f64 * 2.0;
    ///         microstate.add_body(Body::point(Cartesian::from([x, y])))?;
    ///     }
    /// }
    ///
    /// while(!quick_compress.is_complete()) {
    ///     quick_compress.apply(&mut microstate, &pairwise_cutoff, |_| true);
    ///     translate_sweep.apply(&mut microstate, &pairwise_cutoff, &macrostate);
    ///     microstate.increment_step();
    ///
    ///     if microstate.step() > 10_000 {
    ///         let current = microstate.boundary().volume();
    ///         let target = quick_compress.target_volume();
    ///         let step = microstate.step();
    ///         return Err(anyhow!(
    ///             "Achieved volume {current} after {step} steps. The target was {target}."
    ///         ));
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[expect(
        clippy::missing_panics_doc,
        reason = "Panic would occur due to a bug in hoomd-rs."
    )]
    #[inline]
    pub fn apply<P, B, S, X, H, F>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        hamiltonian: &H,
        should_map_body: F,
    ) where
        P: Copy,
        B: Clone + Position<Position = P> + Transform<S>,
        S: Clone + Position<Position = P> + Default,
        X: Clone + PointUpdate<P, SiteKey>,
        H: TotalEnergy<Microstate<B, S, X, C>>,
        C: Clone + MapPoint<P> + Wrap<B> + Wrap<S> + GenerateGhosts<S> + Volume + Scale + PartialEq,
        Microstate<B, S, X, C>: Clone,
        F: Fn(&Tagged<Body<B, S>>) -> bool,
    {
        let mut rng = microstate.counter().make_rng();
        microstate.increment_substep();

        match self.state {
            State::Startup => {
                self.state = State::Waiting(false, microstate.boundary().clone());
            }
            State::Complete(ref final_boundary) if final_boundary == microstate.boundary() => {}
            State::Complete(_) => {
                // While it is not intended that callers change the microstate's
                // boundary outside `QuickCompress`, it is a possibility. Should
                // a caller do so, they would be surprised that `QuickCompress`
                // failed to compress toward the target.
                self.state = State::Waiting(false, microstate.boundary().clone());
            }
            State::Compressing => {
                let delta_distribution = Uniform::new(0.0, self.maximum_delta.get())
                    .expect("maximum_delta should be greater than 0.0");
                let delta = delta_distribution.sample(&mut rng);
                let current_volume = microstate.boundary().volume();

                let trial_volume = if self.target_volume.get() > current_volume {
                    (current_volume * (1.0 + delta)).min(self.target_volume.get())
                } else {
                    (current_volume * (1.0 - delta)).max(self.target_volume.get())
                };

                let trial_boundary = microstate.boundary().scale_volume(
                    (trial_volume / current_volume)
                        .try_into()
                        .expect("both volumes should be positive"),
                );
                let Ok(trial_microstate) =
                    microstate.clone_with_boundary(trial_boundary, should_map_body)
                else {
                    return;
                };
                let delta_energy = hamiltonian.delta_energy_total(microstate, &trial_microstate);

                if delta_energy > self.maximum_energy_per_site * microstate.sites().len() as f64 {
                    // The trial state is too strained. Remain in the `Compressing` state
                    // to try again on the next call.
                } else {
                    // The trial state is valid. Transition to the waiting state and indicate
                    // whether this trial achieved the target.
                    self.state = State::Waiting(
                        trial_volume == self.target_volume.get(),
                        trial_microstate.boundary().clone(),
                    );
                    *microstate = trial_microstate;
                }
            }
            State::Waiting(is_final, ref last_known_boundary) => {
                if hamiltonian.total_energy(microstate) <= 0.0 {
                    if is_final && last_known_boundary == microstate.boundary() {
                        self.state = State::Complete(microstate.boundary().clone());
                    } else {
                        self.state = State::Compressing;
                    }
                }
            }
        }
    }
}

// TODO: test

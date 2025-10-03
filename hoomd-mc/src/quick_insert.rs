// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `QuickInsert`

use super::Count;
use hoomd_interaction::{DeltaEnergyInsert, TotalEnergy};
use hoomd_microstate::{
    Body, Microstate, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::Position,
};

use rand::distr::Distribution;

/// Track the state of a given `QuickInsert` instance.
#[derive(Debug, PartialEq)]
enum State {
    /// Inserting bodies or performing trial moves to separate them.
    Running,
    /// All bodies have been inserted and all overlaps removed.
    Complete,
}

/// Quickly add bodies to the microstate with random configurations.
///
/// [`QuickInsert`] allows you to *quickly* insert many bodies into the microstate.
/// It does so by *breaking detailed balance*, so you should use it only during
/// the initialization phase of your simulation where you prepare a microstate
/// for later equilibration. The [`QuickInsert`] protocol is an alternate to the
/// `QuickCompress` protocol with the advantage that you can keep the boundary and
/// any of your barriers fixed while randomly inserting particles. The disadvantage
/// is that [`QuickInsert`] cannot achieve densities as high as `QuickCompress`.
///
/// [`QuickInsert`] works only with hard particle potentials that go to infinity
/// when overlapping. It works best with the [`OverlapPenalty`] potential that
/// allows sites to overlap a small amount and for trial moves to partially reduce
/// that overlap.
///
/// As a **protocol**, [`QuickInsert`] is more than just a trial move. A
/// [`QuickInsert`] instance stores internal state to track its progress. Therefore,
/// you should only use a given [`QuickInsert`] on one [`Microstate`]. After
/// initialization, a [`QuickInsert`] knows the *target* number of bodies it should
/// add a distribution that places those bodies in the simulation boundary. New
/// [`QuickInsert`] instances start in the running state.
///
/// When you [`apply`] a running [`QuickInsert`] to a microstate, it:
/// 1. Checks the total energy of the given Hamiltonian.
/// 2. If the total energy is zero *and there are still bodies to insert*, generate
///    a random body and attempt to insert it into the microstate. Reject any
///    insertion that would result in an infinite energy. Accept in all other cases.
/// 3. Repeat step 2 until inserted bodies overlap with others `allowed_overlaps`
///    times, the target number of bodies have been inserted, or a total of `target`
///    attempts have been made during this call, whichever comes first.
///
/// When *both* `target` bodies have been inserted *and* the energy is
/// 0, [`QuickInsert`] transitions to the complete state. When complete,
/// [`is_complete`] returns `true` and [`apply`] does nothing.
///
/// For spherical particles, [`QuickInsert`] combined with [`OverlapPenalty`]
/// can achieve a packing fraction of 56% in 3D and 72% in 2D. You might achieve
/// slightly higher densities if you are willing to run many steps, though
/// `QuickCompress` is a better solution.
///
/// The generic type names are:
/// * `D`: The body distribution.
///
/// [`apply`]: Self::apply
/// [`is_complete`]: Self::is_complete
/// [`OverlapPenalty`]: hoomd_interaction::pairwise::OverlapPenalty
///
/// # Example
///
/// ```
/// use hoomd_geometry::shape::Rectangle;
/// use hoomd_mc::{QuickInsert, UniformIn};
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
/// let mut quick_insert = QuickInsert::new(distribution, 256);
/// # Ok(())
/// # }
/// ```
pub struct QuickInsert<D> {
    /// Sample random bodies to insert.
    distribution: D,

    /// Total number of particles to insert.
    target: usize,

    /// Maximum number of overlapping inserts allowed.
    allowed_overlaps: usize,

    /// Count of insertions completed
    inserted: usize,

    /// Current stage of the method.
    state: State,
}

impl<D> QuickInsert<D> {
    /// Build a new quick insert protocol.
    ///
    /// After construction, the `QuickInsert` starts in a running state. On
    /// successive calls to `apply`, it will attempt to insert `target` bodies into
    /// the microstate randomly sampled from the given `distribution`. The default
    /// number of `allowed_overlaps` is `target/8`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_mc::{QuickInsert, UniformIn};
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
    /// let quick_insert = QuickInsert::new(distribution, 256);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn new(distribution: D, target: usize) -> Self {
        Self {
            distribution,
            target,
            allowed_overlaps: (target / 8).max(1),
            inserted: 0,
            state: State::Running,
        }
    }

    /// Check if the quick insert protocol is complete.
    ///
    /// `QuickInsert` completes after it has inserted all `target` bodies **and**
    /// the total energy of the system is less than or equal to 0. When using the
    /// recommended `OverlapPenalty` potential, this means that that there are no
    /// overlaps between any particles.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_mc::{QuickInsert, UniformIn};
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
    /// let quick_insert = QuickInsert::new(distribution, 256);
    ///
    /// let complete = quick_insert.is_complete();
    /// assert!(!complete);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.state == State::Complete
    }

    /// The target number of bodies to insert.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_mc::{QuickInsert, UniformIn};
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
    /// let quick_insert = QuickInsert::new(distribution, 256);
    ///
    /// let target = quick_insert.target();
    /// assert_eq!(target, 256);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn target(&self) -> usize {
        self.target
    }

    /// Apply the quick insert protocol to a microstate.
    ///
    /// Combine [`QuickInsert::apply`] with local trial moves that translate and/or
    /// rotate bodies by small amounts to relieve the stress caused by inserting
    /// overlapping sites.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Rectangle;
    /// use hoomd_interaction::{
    ///     CutoffPair,
    ///     pairwise::{Expanded, Isotropic, OverlapPenalty},
    /// };
    /// use hoomd_mc::{QuickInsert, Sweep, Translate, Trial, UniformIn};
    /// use hoomd_microstate::{
    ///     Body, MicrostateBuilder, boundary::Periodic, property::Point,
    /// };
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rectangle = Rectangle::with_equal_edges(10.0.try_into()?);
    ///
    /// let distribution = UniformIn {
    ///     boundary: rectangle.clone(),
    ///     template_sites: vec![Point::default()],
    /// };
    /// let mut quick_insert = QuickInsert::new(distribution, 256);
    ///
    /// let translate = Translate::with_maximum_distance(0.1.try_into()?);
    /// let translate_sweep = Sweep(translate);
    ///
    /// let cutoff_pair = CutoffPair {
    ///     r_cut: 1.0,
    ///     evaluator: Isotropic(Expanded {
    ///         delta: 1.0,
    ///         f: OverlapPenalty::default(),
    ///     }),
    /// };
    ///
    /// let mut microstate =
    ///     MicrostateBuilder::with_boundary(Periodic::new(1.0, rectangle)?)
    ///         .bodies([Body::point(Cartesian::from([0.0, 0.0]))])
    ///         .try_build()?;
    ///
    /// quick_insert.apply(&mut microstate, &cutoff_pair);
    ///
    /// translate_sweep.apply(&mut microstate, &cutoff_pair, &1.0);
    ///
    /// assert!(microstate.bodies().len() > 1);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn apply<P, B, S, C, H>(
        &mut self,
        microstate: &mut Microstate<B, S, C>,
        hamiltonian: &H,
    ) -> Count
    where
        B: Position<Position = P> + Transform<S>,
        S: Position<Position = P> + Default,
        D: Distribution<Body<B, S>>,
        H: DeltaEnergyInsert<B, S, C> + TotalEnergy<Microstate<B, S, C>>,
        C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
    {
        let mut count = Count::default();

        // Perform no work at all if already complete.
        if self.is_complete() {
            return count;
        }

        let energy = hamiltonian.total_energy(microstate);

        // The quick insert protocol is not complete until the energy has reached 0.
        if energy <= 0.0 && self.inserted >= self.target {
            self.state = State::Complete;
            return count;
        }

        // Scaling the number of insertion attempts with the target number of insertions
        // is a good way to ensure that there are sufficient attempts on each call to
        // apply. Larger boxes will naturally get more insertion attempts. At the same
        // time, we need to limit the total strain caused by the insertions. Count
        // the number of insertions that cause overlaps and exit early when there
        // are too many.
        if energy <= 0.0 {
            let mut rng = microstate.counter().make_rng();
            let mut insertions_with_overlaps = 0;

            for _ in 0..self.target {
                let new_body = self.distribution.sample(&mut rng);

                let delta_energy = hamiltonian.delta_energy_insert(microstate, &new_body);
                if delta_energy.is_finite() && microstate.add_body(new_body).is_ok() {
                    count.accepted += 1;
                    self.inserted += 1;

                    if delta_energy > 0.0 {
                        insertions_with_overlaps += 1;
                    }

                    if self.inserted == self.target
                        || insertions_with_overlaps >= self.allowed_overlaps
                    {
                        break;
                    }
                } else {
                    count.rejected += 1;
                }
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
    use hoomd_geometry::shape::Rectangle;
    use hoomd_interaction::{
        CutoffPair,
        pairwise::{Boxcar, Isotropic},
    };
    use hoomd_microstate::{MicrostateBuilder, boundary::Closed, property::Point};
    use hoomd_vector::Cartesian;

    #[test]
    fn hard_spheres() {
        let sigma = 1.0;
        let epsilon = f64::INFINITY;
        let kt = 1.0;

        let hamiltonian = CutoffPair {
            r_cut: sigma,
            evaluator: Isotropic(Boxcar {
                left: 0.0,
                right: sigma,
                epsilon,
            }),
        };

        let translate =
            Translate::with_maximum_distance(0.1.try_into().expect("hard-coded value is non-zero"));
        let translate_sweep = Sweep(translate);

        let rectangle = Closed(Rectangle::with_equal_edges(
            6.0.try_into().expect("hard-coded value is non-zero"),
        ));

        let mut microstate = MicrostateBuilder::with_boundary(rectangle.clone())
            .bodies(vec![Body::point(Cartesian::from([0.0, 0.0]))])
            .try_build()
            .expect("hard-coded point is in the boundary");

        let distribution = UniformIn {
            boundary: rectangle,
            template_sites: vec![Point::new([0.0, 0.0].into())],
        };
        let mut quick_insert = QuickInsert::new(distribution, 10);

        assert_eq!(quick_insert.target, 10);
        assert_eq!(quick_insert.state, State::Running);

        for _ in 0..100 {
            quick_insert.apply(&mut microstate, &hamiltonian);
            if quick_insert.is_complete() {
                break;
            }
        }

        translate_sweep.apply(&mut microstate, &hamiltonian, &kt);

        assert_eq!(quick_insert.inserted, 10);
        assert_eq!(quick_insert.state, State::Complete);
        assert!(quick_insert.is_complete());
        assert_eq!(microstate.bodies().len(), 11);
        assert_eq!(hamiltonian.total_energy(&microstate), 0.0);
    }
}

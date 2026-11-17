// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`PinOrientations`]

use hoomd_microstate::{
    Body, Microstate,
    boundary::GenerateMinimumImageCandidates,
    property::{Orientation, Position},
};
use hoomd_vector::{Metric, RotationalDistance};

use crate::{DeltaEnergyOne, MaximumInteractionRange, TotalEnergy};

/// Connect each body orientation to a reference with a harmonic spring.
///
/// ```math
/// U_\mathrm{body} = \begin{cases}
/// \min_j \frac{1}{2} k (\Delta \theta_j)^2 & n_\mathrm{references} > 0 \\
/// 0 & n_\mathrm{references} = 0 \\
/// \end{cases}
/// ```
///
/// Apply a harmonic external potential to to the angle between the body's
/// orientation and a given reference orientation. Take the minimum over all
/// symmetric references *j*. $`\Delta \theta_j`$ is the smallest angle that
/// rotates the body to reference *j* (as given by `R`'s
/// [`RotationalDistance`] implementation).
///
/// If there is no reference orientation for a body, the energy is 0.
///
/// Use [`PinOrientations`] for Frenkel-Ladd free energy computations or any
/// time you want to loosely pin body orientations in place. To fix body
/// body orientations exactly at the initial condition, do not apply rotational
/// trial moves.
///
/// [`PinOrientations`] stores references in *body tag order*. It is thus not
/// suitable for simulations that remove bodies from the microstate. If you
/// need to pin orientations *and* remove bodies, write a custom external
/// potential that uses a custom `reference` stored in the body properties.
pub struct PinOrientations<R> {
    /// The spring constant $`[\mathrm{energy}]`$.
    pub k: f64,
    /// Set of references orientations for each body (in tag order).
    pub reference: Vec<Vec<R>>,
}

impl<R> PinOrientations<R> {
    /// Compute the energy for a given body.
    ///
    /// ```math
    /// U_\mathrm{body} = \begin{cases}
    /// \min_j \frac{1}{2} k (\Delta \theta_j)^2 & n_\mathrm{references} > 0 \\
    /// 0 & n_\mathrm{references} = 0 \\
    /// \end{cases}
    /// ```
    ///
    /// The body must exist in the microstate with the given `body_tag`.
    /// The `body_properties` may match those in the microstate or may be
    /// modified (i.e. a trial move).
    #[inline]
    pub fn body_energy<B>(&self, body_tag: usize, body_properties: &B) -> f64
    where
        R: RotationalDistance,
        B: Orientation<Rotation = R>,
    {
        self.reference[body_tag]
            .iter()
            .map(|reference_orientation| {
                0.5 * self.k
                    * body_properties
                        .orientation()
                        .rotational_distance(reference_orientation)
                        .powi(2)
            })
            .reduce(f64::min)
            .unwrap_or(0.0)
    }
}

impl<B, S, X, C, R> DeltaEnergyOne<B, S, X, C> for PinOrientations<R>
where
    R: RotationalDistance,
    B: Orientation<Rotation = R>,
{
    #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
        let tagged_body = &initial_microstate.bodies()[body_index];
        let body_tag = tagged_body.tag;

        let energy_initial = self.body_energy(body_tag, &tagged_body.item.properties);
        let energy_final = self.body_energy(body_tag, &final_body.properties);

        energy_final - energy_initial
    }
}

impl<B, S, X, C, R> TotalEnergy<Microstate<B, S, X, C>> for PinOrientations<R>
where
    R: RotationalDistance,
    B: Orientation<Rotation = R>,
{
    #[inline]
    fn total_energy(&self, microstate: &Microstate<B, S, X, C>) -> f64 {
        microstate
            .bodies()
            .iter()
            .map(|tagged_body| self.body_energy(tagged_body.tag, &tagged_body.item.properties))
            .sum()
    }
}

impl<R> MaximumInteractionRange for PinOrientations<R> {
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        // External interactions are not applied between pairs of particles.
        0.0
    }
}

/// Connect each body position to a reference with a harmonic spring.
///
/// ```math
/// U_\mathrm{body} = \begin{cases}
/// \min_j \frac{1}{2} k (\Delta \vec{r}_j)^2 & n_\mathrm{references} > 0 \\
/// 0 & n_\mathrm{references} = 0 \\
/// \end{cases}
/// ```
///
/// Apply a harmonic external potential to to the separation distance between a
/// body's position and a reference position. Take the minimum over all
/// symmetric references *j* given by any periodic boundary conditions
/// in effect.
///
/// Use [`PinPositions`] for Frenkel-Ladd free energy computations or any
/// time you want to loosely pin body positions in place. To fix body
/// body positions exactly at the initial condition, do not apply translational
/// trial moves.
///
/// [`PinPositions`] stores references in *body tag order*. It is thus not
/// suitable for simulations that remove bodies from the microstate. If you
/// need to pin positions *and* remove bodies, write a custom external
/// potential that uses a custom `reference` stored in the body properties.
pub struct PinPositions<V> {
    /// The spring constant $`[\mathrm{energy}] \cdot [\mathrm{length}]^{-2}`$.
    pub k: f64,
    /// Reference position for each body (in tag order).
    pub reference: Vec<V>,
}

impl<P> PinPositions<P>
where
    P: Metric,
{
    /// Compute the energy for a given body.
    ///
    /// ```math
    /// U_\mathrm{body} = \begin{cases}
    /// \min_j \frac{1}{2} k (\Delta \vec{r}_j)^2 & n_\mathrm{references} > 0 \\
    /// 0 & n_\mathrm{references} = 0 \\
    /// \end{cases}
    /// ```
    ///
    /// The body must exist in the microstate with the given `body_tag`.
    /// The `body_properties` may match those in the microstate or may be
    /// modified (i.e. a trial move).
    #[inline]
    pub fn body_energy<B, S, C>(&self, body_tag: usize, body: &Body<B, S>, boundary: &C) -> f64
    where
        B: Position<Position = P>,
        C: GenerateMinimumImageCandidates<B>,
    {
        let reference = &self.reference[body_tag];
        let ghosts = boundary.generate_minimum_image_candidates(&body.properties);
        std::iter::once(body.properties.position())
            .chain(
                ghosts
                    .iter()
                    .map(hoomd_microstate::property::Position::position),
            )
            .map(|position| 0.5 * self.k * position.distance_squared(reference))
            .reduce(f64::min)
            .unwrap_or(0.0)
    }
}

impl<B, S, X, C, P> DeltaEnergyOne<B, S, X, C> for PinPositions<P>
where
    B: Position<Position = P>,
    C: GenerateMinimumImageCandidates<B>,
    P: Metric,
{
    #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
        let tagged_body = &initial_microstate.bodies()[body_index];
        let body_tag = tagged_body.tag;
        let boundary = initial_microstate.boundary();

        let energy_initial = self.body_energy(body_tag, &tagged_body.item, boundary);
        let energy_final = self.body_energy(body_tag, final_body, boundary);

        energy_final - energy_initial
    }
}

impl<B, S, X, C, P> TotalEnergy<Microstate<B, S, X, C>> for PinPositions<P>
where
    B: Position<Position = P>,
    C: GenerateMinimumImageCandidates<B>,
    P: Metric,
{
    #[inline]
    fn total_energy(&self, microstate: &Microstate<B, S, X, C>) -> f64 {
        let boundary = microstate.boundary();
        microstate
            .bodies()
            .iter()
            .map(|tagged_body| self.body_energy(tagged_body.tag, &tagged_body.item, boundary))
            .sum()
    }
}

impl<V> MaximumInteractionRange for PinPositions<V> {
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        // External interactions are not applied between pairs of particles.
        0.0
    }
}

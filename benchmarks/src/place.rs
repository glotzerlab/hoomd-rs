// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement particle placement methods.

use hoomd_spatial::{PointUpdate, PointsNearBall, WithSearchRadius};
use log::{debug, trace};
use rand::distr::Distribution;

use hoomd_geometry::{Volume, shape::Hypercuboid};
use hoomd_interaction::{
    DeltaEnergyInsert, DeltaEnergyOne, TotalEnergy
};
use hoomd_mc::{LocalTrial, QuickCompress, QuickInsert, Rotate, Sweep, Translate, Trial, UniformIn};
use hoomd_microstate::{
    Body, Microstate, SiteKey, Transform,
    boundary::{GenerateGhosts, Periodic},
    property::{Orientation, Position},
};
use hoomd_simulation::macrostate::Isothermal;
use hoomd_vector::Cartesian;

/// Place `n` single site bodies in the microstate.
///
/// Inserts `n` bodies, then compresses to the given number density using the given
/// `insert_hamiltonian`. Bodies are translated only.
///
/// # Errors
/// Returns an error when the microstate cannot be constructed.
#[inline]
pub fn place_single_site_point_bodies<B, S, const D: usize, H, X>(
    n: usize,
    number_density: f64,
    maximum_interaction_range: f64,
    overlap_penalty_hamiltonian: &H,
) -> anyhow::Result<Microstate<B, S, X, Periodic<Hypercuboid<D>>>>
where
    B: Default + Position<Position = Cartesian<D>> + Transform<S> + Copy,
    S: Default + Position<Position = Cartesian<D>> + Copy,
    UniformIn<S, Periodic<Hypercuboid<D>>>: Distribution<Body<B, S>>,
    Periodic<Hypercuboid<D>>: GenerateGhosts<S> + Volume,
    X: PointsNearBall<Cartesian<D>, SiteKey>
        + PointUpdate<Cartesian<D>, SiteKey>
        + WithSearchRadius
        + Clone,
    H: DeltaEnergyInsert<B, S, X, Periodic<Hypercuboid<D>>>
     + DeltaEnergyOne<B, S, X, Periodic<Hypercuboid<D>>>
     + TotalEnergy<Microstate<B, S, X,Periodic<Hypercuboid<D>>>>,
{
    let initial_number_density = 0.7 * number_density;
    let initial_box_length = (n as f64 / initial_number_density).powf(1.0 / (D as f64));
    let final_box_volume = n as f64 / number_density;
    let macrostate = Isothermal { temperature: 1.0 };

    debug!("Initializing {n} bodies in {D} dimensions with number density {number_density}...");

    let cell_list = X::with_search_radius(maximum_interaction_range.try_into()?);
    let boundary = Periodic::new(
        maximum_interaction_range,
        Hypercuboid::<D>::with_equal_edges(initial_box_length.try_into()?),
    )?;

    let mut microstate = Microstate::builder()
        .spatial_data(cell_list)
        .boundary(boundary)
        .try_build()?;

    let translate = Translate::with_maximum_distance(0.1.try_into()?);
    let mut translate_sweep = Sweep(translate);

    let distribution = UniformIn {
        boundary: microstate.boundary().clone(),
        template_sites: vec![S::default()],
    };
    let mut quick_insert = QuickInsert::new(distribution, n);
    let mut quick_compress =
        QuickCompress::with_target_volume(final_box_volume.try_into()?);

    while !quick_compress.is_complete() {
        if quick_insert.is_complete() {
            quick_compress.apply(&mut microstate, overlap_penalty_hamiltonian, |_| true);
        } else {
            quick_insert.apply(&mut microstate, overlap_penalty_hamiltonian);
        }

        translate_sweep.apply(&mut microstate, overlap_penalty_hamiltonian, &macrostate);
        microstate.increment_step();

        if microstate.step().is_multiple_of(100) {
            if quick_insert.is_complete() {
                trace!(
                    "Step {}: rho = {} / {}",
                    microstate.step(),
                    n as f64 / microstate.boundary().volume(),
                    n as f64 / final_box_volume,
                );
            } else {
                trace!(
                    "Step {}: N = {} / {}",
                    microstate.step(),
                    microstate.sites().len(),
                    n
                );
            }
        }
    }

    Ok(microstate)
}

/// Place `n` single site, orientable bodies in the microstate.
///
/// Inserts `n` bodies, then compresses to the given number density using the given
/// `insert_hamiltonian`. Bodies are translated and rotated.
///
/// # Errors
/// Returns an error when the microstate cannot be constructed.
#[inline]
pub fn place_single_site_orientable_bodies<B, S, R, const D: usize, H, X>(
    n: usize,
    number_density: f64,
    maximum_interaction_range: f64,
    insert_hamiltonian: &H,
) -> anyhow::Result<Microstate<B, S, X, Periodic<Hypercuboid<D>>>>
where
    B: Default + Position<Position = Cartesian<D>> + Orientation<Rotation = R> + Transform<S> + Copy,
    S: Default + Position<Position = Cartesian<D>> + Copy,
    UniformIn<S, Periodic<Hypercuboid<D>>>: Distribution<Body<B, S>>,
    Periodic<Hypercuboid<D>>: GenerateGhosts<S> + Volume,
    Rotate<R>: LocalTrial<B> + Clone,
    X: PointsNearBall<Cartesian<D>, SiteKey>
        + PointUpdate<Cartesian<D>, SiteKey>
        + WithSearchRadius
        + Clone,
    H: DeltaEnergyInsert<B, S, X, Periodic<Hypercuboid<D>>>
     + DeltaEnergyOne<B, S, X, Periodic<Hypercuboid<D>>>
     + TotalEnergy<Microstate<B, S, X,Periodic<Hypercuboid<D>>>>,
{
    let initial_number_density = 0.7 * number_density;
    let initial_box_length = (n as f64 / initial_number_density).powf(1.0 / (D as f64));
    let final_box_volume = n as f64 / number_density;
    let macrostate = Isothermal { temperature: 1.0 };

    debug!("Initializing {n} bodies in {D} dimensions with number density {number_density}...");

    let cell_list = X::with_search_radius(maximum_interaction_range.try_into()?);
    let boundary = Periodic::new(
        maximum_interaction_range,
        Hypercuboid::<D>::with_equal_edges(initial_box_length.try_into()?),
    )?;

    let mut microstate = Microstate::builder()
        .spatial_data(cell_list)
        .boundary(boundary)
        .try_build()?;

    let translate = Translate::with_maximum_distance(0.1.try_into()?);
    let mut translate_sweep = Sweep(translate);

    let rotate = Rotate::with_maximum_rotation(0.1.try_into()?);
    let mut rotate_sweep = Sweep(rotate.clone());

    let distribution = UniformIn {
        boundary: microstate.boundary().clone(),
        template_sites: vec![S::default()],
    };
    let mut quick_insert = QuickInsert::new(distribution, n);
    let mut quick_compress =
        QuickCompress::with_target_volume(final_box_volume.try_into()?);

    while !quick_compress.is_complete() {
        if quick_insert.is_complete() {
            quick_compress.apply(&mut microstate, insert_hamiltonian, |_| true);
        } else {
            quick_insert.apply(&mut microstate, insert_hamiltonian);
        }

        translate_sweep.apply(&mut microstate, insert_hamiltonian, &macrostate);
        rotate_sweep.apply(&mut microstate, insert_hamiltonian, &macrostate);
        microstate.increment_step();

        if microstate.step().is_multiple_of(100) {
            if quick_insert.is_complete() {
                trace!(
                    "Step {}: rho = {} / {}",
                    microstate.step(),
                    n as f64 / microstate.boundary().volume(),
                    n as f64 / final_box_volume,
                );
            } else {
                trace!(
                    "Step {}: N = {} / {}",
                    microstate.step(),
                    microstate.sites().len(),
                    n
                );
            }
        }
    }

    Ok(microstate)
}

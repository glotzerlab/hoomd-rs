// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implementations that enable dynamic switch between styles of
ChIMES transformation and smoothing function. Also provide a
assembler to construct complete ChIMES potential functional.
 */
use crate::potential::{Chimes2b, ChimesPenalty, CubicSmooth, TersoffSmooth};
use crate::transformation::{
    DirectTransformation, InverseTransformation, MorseTransformation, Transformation,
};
use hoomd_interaction::univariate::{UnivariateEnergy, UnivariateForce};

/// Enum to encapsulate different ChIMES transformation style.
#[derive(Clone)]
pub enum ChimesTransformation {
    /// See [`MorseTransformation`].
    Morse(MorseTransformation),
    /// See [`InverseTransformation`].
    Inverse(InverseTransformation),
    /// See [`DirectTransformation`]
    Direct(DirectTransformation),
}

impl Transformation for ChimesTransformation {
    fn s(&self, r: &f64) -> f64 {
        match self {
            ChimesTransformation::Morse(t) => t.s(r),
            ChimesTransformation::Inverse(t) => t.s(r),
            ChimesTransformation::Direct(t) => t.s(r),
        }
    }

    fn ds_dr(&self, r: &f64) -> f64 {
        match self {
            ChimesTransformation::Morse(t) => t.ds_dr(r),
            ChimesTransformation::Inverse(t) => t.ds_dr(r),
            ChimesTransformation::Direct(t) => t.ds_dr(r),
        }
    }
}

/// Enum to encapsulate different ChIMES smoothing functions.
#[derive(Clone)]
pub enum ChimesSmoothing<F: Transformation, const N: usize> {
    /// See [`CubicSmooth`].
    Cubic(CubicSmooth<Chimes2b<F, N>>),
    /// See [`TersoffSmooth`].
    Tersoff(TersoffSmooth<Chimes2b<F, N>>),
}

impl<F: Transformation, const N: usize> UnivariateEnergy for ChimesSmoothing<F, N> {
    fn energy(&self, r: f64) -> f64 {
        match self {
            ChimesSmoothing::Cubic(s) => s.energy(r),
            ChimesSmoothing::Tersoff(s) => s.energy(r),
        }
    }
}
impl<F: Transformation, const N: usize> UnivariateForce for ChimesSmoothing<F, N> {
    fn force(&self, r: f64) -> f64 {
        match self {
            ChimesSmoothing::Cubic(s) => s.force(r),
            ChimesSmoothing::Tersoff(s) => s.force(r),
        }
    }
}

/// Represents a two-body ChIMES potential for a specific pair type.
#[derive(Clone)]
pub struct ChimesTwobPotential<const N: usize> {
    /// A string represents particle type one in a pair.
    pub type1: String,
    /// A string represents particle type two in a pair.
    pub type2: String,
    /// The Chebyshev expansion part of ChIMES two-body potential.
    /// See [`ChimesSmoothing`] and [`ChimesTransformation`].
    pub chimes: ChimesSmoothing<ChimesTransformation, N>,
    /// See [`ChimesPenalty`].
    pub penalty: ChimesPenalty,
    /// Single particle energy.
    pub energy_shifting: f64,
}

impl<const N: usize> UnivariateEnergy for ChimesTwobPotential<N> {
    fn energy(&self, r: f64) -> f64 {
        self.penalty.energy(r) + self.chimes.energy(r) + self.energy_shifting
    }
}

impl<const N: usize> UnivariateForce for ChimesTwobPotential<N> {
    fn force(&self, r: f64) -> f64 {
        self.penalty.force(r) + self.chimes.force(r)
    }
}

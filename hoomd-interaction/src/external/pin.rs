use hoomd_microstate::{Body, Microstate, property::Orientation};
use hoomd_vector::{Angle, Quaternion};

use crate::DeltaEnergyOne;

pub struct PinOrientations<R> {
    pub k: f64,
    pub reference: Vec<Vec<R>>,
}

pub trait RotationDelta {
    fn delta(reference_orientation: &Self) -> f64;
}

impl RotationDelta for Quaternion {
    #[inline]
    fn delta(reference_orientation: &Quaternion) -> f64 {
        // TODO
    }
}

impl RotationDelta for Angle {
    #[inline]
    fn delta(reference_orientation: &Angle) -> f64 {
        // TODO
    }
}

impl<R> PinOrientations<R>
{
    pub fn body_energy<B, S>(&self, body_tag: usize, body: &Body<B, S>) -> f64
    where
        R: RotationDelta,
        B: Orientation<Rotation = R>,
    {
        self.reference[body_tag].iter()
            .map(|reference_orientation| 0.5 * self.k * body.orientation().delta(reference_orientation).powi(2))
            .reduce(|acc, x| acc.min(x))
            .unwrap_or(0.0)
    }
}

impl<B, S, X, C, R> DeltaEnergyOne<B, S, X, C> for PinOrientations<R>
where
    R: RotationDelta,
    B: Orientation<Rotation = R>,
{
        #[inline]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64 {
        let tagged_body = initial_microstate.bodies()[body_index];
        let body_tag = tagged_body.tag;
        
        let energy_initial = self.body_energy(body_tag, tagged_body.item);
        let energy_final = self.body_energy(body_tag, final_body);

        energy_final - energy_initial
    }
}

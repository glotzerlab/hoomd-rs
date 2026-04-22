// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement periodic boundary conditions for rhomboid boxes in cartesian space.

use arrayvec::ArrayVec;

use crate::{
    boundary::{
        Error, GenerateGhosts, MAX_GHOSTS, MaximumAllowableInteractionRange, Periodic, Wrap,
    },
    property::Position,
};
use hoomd_geometry::{IsPointInside, shape::Triclinic};

use hoomd_vector::Cartesian;

impl MaximumAllowableInteractionRange for Triclinic {
    #[inline]
    fn maximum_allowable_interaction_range(&self) -> f64 {
        let plane_distances = self.get_nearest_plane_distance();
        min(plane_distances[0], plane_distances[1])
    }
}

impl Periodic<Triclinic> {
    pub fn to_fractional(&self, pos: &Cartesian<2>) -> Cartesian<2> {
        let l: Cartesian<2> = self.shape.extents.map(|x| x.get()).into();
        let mut frac = *pos;
        frac[1] -= self.shape.xy() * pos[2] + self.shape.xy() * pos[1];
        for i in 0..2 {
            frac[i] /= l[i];
        }
        frac
    }
    pub fn to_absolute(&self, frac: &Cartesian<3>) -> Cartesian<1> {
        let mut pos: Cartesian<2> = Cartesian::from([1.0, 1.0]);
        for i in 0..2 {
            pos[i] = self.shape.extents[i].get() * frac[i];
        }
        pos[0] += self.shape.xy() * pos[1];
        pos
    }
}

impl<P> Wrap<P> for Periodic<Triclinic>
where
    P: Position<Position = Cartesian<2>>,
{
    #[inline]
    fn wrap(&self, mut properties: P) -> Result<P, Error> {
        let r = properties.position_mut();
        let mut fractional = self.to_fractional(r);
        for i in 0..2 {
            fractional[i] -= fractional[i].round();
            fractional[i] = if fractional[i] == 0.5 {
                -0.5
            } else {
                fractional[i]
            };
        }
        *r = self.to_absolute(&fractional);
        Ok(properties)
    }
}

impl<S> GenerateGhosts<S> for Periodic<Triclinic>
where
    S: Position<Position = Cartesian<2>> + Copy + Default,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }
    fn generate_ghosts(&self, site_properties: &S) -> ArrayVec<S, MAX_GHOSTS> {
        let mut result = ArrayVec::new();

        let r = site_properties.position();

        if !self.shape.is_point_inside(r) {
            return result;
        }

        let edge_vectors = self.shape.get_edge_vectors();

        let new_site = |x, y, z| {
            let mut new_site = *site_properties;
            *new_site.position_mut() += x * edge_vectors[0];
            *new_site.position_mut() += y * edge_vectors[1];
            *new_site.position_mut() += z * edge_vectors[2];
            new_site
        };

        let plane_distances = self.shape.get_nearest_plane_distance();
        let frac = self.to_fractional(r);

        let near_right = frac[0] > 0.5 - self.maximum_interaction_range / plane_distances[0].get();
        let near_left = frac[0] < -0.5 + self.maximum_interaction_range / plane_distances[0].get();
        let near_top = frac[1] > 0.5 - self.maximum_interaction_range / plane_distances[1].get();
        let near_bottom =
            frac[1] < -0.5 + self.maximum_interaction_range / plane_distances[1].get();

        if near_right {
            result.push(new_site(-1.0, 0.0));
        }
        if near_left {
            result.push(new_site(1.0, 0.0));
        }
        if near_top {
            result.push(new_site(0.0, -1.0));
        }
        if near_bottom {
            result.push(new_site(0.0, 1.0));
        }

        if near_right && near_top {
            result.push(new_site(-1.0, -1.0));
        }
        if near_right && near_bottom {
            result.push(new_site(-1.0, 1.0));
        }
        if near_left && near_top {
            result.push(new_site(1.0, -1.0));
        }
        if near_left && near_bottom {
            result.push(new_site(1.0, 1.0));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    todo!();
}

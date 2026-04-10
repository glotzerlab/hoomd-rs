// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement periodic boundary conditions for cuboids in cartesian space.

use tinyvec::ArrayVec;

use crate::{
    boundary::{
        Error, GenerateGhosts, MAX_GHOSTS, MaximumAllowableInteractionRange, Periodic, Wrap,
    },
    property::Position,
};
use hoomd_geometry::{IsPointInside, shape::Triclinic};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Cartesian;
use hoomd_geometry::{IsPointInside, shape::Hyperparallelepiped};
use hoomd_linear_algebra::{MatMul, matrix::Matrix, matrix::Matrix33, matrix::qr};
use hoomd_vector::{Cartesian, Cross, InnerProduct};
use tinyvec::ArrayVec;

impl<const N: usize> MaximumAllowableInteractionRange for Triclinic {
     #[inline]
    fn maximum_allowable_interaction_range(&self) -> f64 {
     let minimum_l = self
            .edge_lengths //TODO: Change this to L_i's
            .iter()
            .map(PositiveReal::get)
            .reduce(f64::min)
            .expect("cuboid should have dimension 1 or greater");
        minimum_l / 2.0
    }
}

impl<P, const N: usize> Wrap<P> for Periodic<Triclinic>
where
    P: Position<Vector = Cartesian<N>>,
{
    #[inline]
    fn wrap(&self, properties: P) -> Result<P, Error> {
        todo!();
    }
}

impl<S> GenerateGhosts<S> for Periodic<Triclinic>
where
    S: Position<Vector = Cartesian<3>> + Copy + Default,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range
    }

    #[inline]
    /// Place periodic images of sites near the edge of the periodic boundary.
    ///
    /// For triclinic boxes, `generate_ghosts` places ghosts near the 6 faces, 12 edges,
    /// and 8 vertices of the box.
    fn generate_ghosts(&self, site_properties: &S) -> ArrayVec<S, MAX_GHOSTS> {
        let mut result = ArrayVec::new();

        let r = site_properties.position();
        let max = self.shape.maximal_extents();
        let min = self.shape.minimal_extents();

        if !self.shape.is_point_inside(r) {
            return result;
        }

        let new_site = |x, y, z| {
            let mut new_site = *site_properties;
            new_site.position_mut()[0] += x * self.shape.edge_lengths[0].get(); // Todo: convert tiltfactor/extents
            new_site.position_mut()[1] += y * self.shape.edge_lengths[1].get();
            new_site.position_mut()[2] += z * self.shape.edge_lengths[2].get();
            new_site
        };

        let near_left = r[0] < min[0] + self.maximum_interaction_range;
        let near_right = r[0] > max[0] - self.maximum_interaction_range;
        let near_top = r[1] > max[1] - self.maximum_interaction_range;
        let near_bottom = r[1] < min[1] + self.maximum_interaction_range;
        let near_front = r[2] > max[2] - self.maximum_interaction_range;
        let near_back = r[2] < min[2] + self.maximum_interaction_range;

        if near_right {
            result.push(new_site(-1.0, 0.0, 0.0));
        }
        if near_left {
            result.push(new_site(1.0, 0.0, 0.0));
        }
        if near_top {
            result.push(new_site(0.0, -1.0, 0.0));
        }
        if near_bottom {
            result.push(new_site(0.0, 1.0, 0.0));
        }
        if near_front {
            result.push(new_site(0.0, 0.0, -1.0));
        }
        if near_back {
            result.push(new_site(0.0, 0.0, 1.0));
        }

        if near_right && near_top {
            result.push(new_site(-1.0, -1.0, 0.0));
        }
        if near_right && near_bottom {
            result.push(new_site(-1.0, 1.0, 0.0));
        }
        if near_right && near_front {
            result.push(new_site(-1.0, 0.0, -1.0));
        }
        if near_right && near_back {
            result.push(new_site(-1.0, 0.0, 1.0));
        }
        if near_left && near_top {
            result.push(new_site(1.0, -1.0, 0.0));
        }
        if near_left && near_bottom {
            result.push(new_site(1.0, 1.0, 0.0));
        }
        if near_left && near_front {
            result.push(new_site(1.0, 0.0, -1.0));
        }
        if near_left && near_back {
            result.push(new_site(1.0, 0.0, 1.0));
        }

        if near_top && near_front {
            result.push(new_site(0.0, -1.0, -1.0));
        }
        if near_bottom && near_front {
            result.push(new_site(0.0, 1.0, -1.0));
        }
        if near_top && near_back {
            result.push(new_site(0.0, -1.0, 1.0));
        }
        if near_bottom && near_back {
            result.push(new_site(0.0, 1.0, 1.0));
        }

        if near_right && near_top && near_front {
            result.push(new_site(-1.0, -1.0, -1.0));
        }
        if near_right && near_top && near_back {
            result.push(new_site(-1.0, -1.0, 1.0));
        }
        if near_right && near_bottom && near_front {
            result.push(new_site(-1.0, 1.0, -1.0));
        }
        if near_right && near_bottom && near_back {
            result.push(new_site(-1.0, 1.0, 1.0));
        }
        if near_left && near_top && near_front {
            result.push(new_site(1.0, -1.0, -1.0));
        }
        if near_left && near_top && near_back {
            result.push(new_site(1.0, -1.0, 1.0));
        }
        if near_left && near_bottom && near_front {
            result.push(new_site(1.0, 1.0, -1.0));
        }
        if near_left && near_bottom && near_back {
            result.push(new_site(1.0, 1.0, 1.0));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    todo()!
}

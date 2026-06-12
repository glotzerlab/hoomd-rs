use arrayvec::ArrayVec;
use hoomd_geometry::{Error as GeometryError, IsPointInside, MapPoint, Scale, Volume};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Cartesian;
use rand::distr::Distribution as RandDistribution;
use rand::distr::Uniform;
use serde::{Deserialize, Serialize};
use std::array;

use super::super::{GenerateGhosts, MaximumAllowableInteractionRange, Wrap};
use super::Periodic;
use crate::property::Position;

/// A 4D hypercube (tesseract) with equal edge lengths.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Octachoron {
    /// Edge length of the tesseract.
    pub edge_length: PositiveReal,
}

impl Octachoron {
    /// Construct an octachoron with the given edge length.
    pub fn with_edge_length(edge_length: PositiveReal) -> Self {
        Self { edge_length }
    }

    fn half(&self) -> f64 {
        self.edge_length.get() / 2.0
    }
}

impl Volume for Octachoron {
    #[inline]
    fn volume(&self) -> f64 {
        self.edge_length.get().powi(4)
    }
}

impl Scale for Octachoron {
    #[inline]
    fn scale_length(&self, v: PositiveReal) -> Self {
        Self {
            edge_length: (self.edge_length.get() * v.get())
                .try_into()
                .expect("scaled edge length should be positive"),
        }
    }

    #[inline]
    fn scale_volume(&self, v: PositiveReal) -> Self {
        let new_edge = self.edge_length.get() * v.get().powf(0.25);
        Self {
            edge_length: new_edge
                .try_into()
                .expect("scaled edge length should be positive"),
        }
    }
}

impl IsPointInside<Cartesian<4>> for Octachoron {
    #[inline]
    fn is_point_inside(&self, point: &Cartesian<4>) -> bool {
        let half = self.half();
        point.coordinates.iter().all(|&c| -half <= c && c < half)
    }
}

impl MapPoint<Cartesian<4>> for Octachoron {
    #[inline]
    fn map_point(&self, point: Cartesian<4>, other: &Self) -> Result<Cartesian<4>, GeometryError> {
        if !self.is_point_inside(&point) {
            return Err(GeometryError::PointOutsideShape);
        }
        let scale = other.edge_length.get() / self.edge_length.get();
        let half_other = other.half();
        Ok(Cartesian::from(array::from_fn(|i| {
            (scale * point[i]).clamp(-half_other, half_other.next_down())
        })))
    }
}

impl MaximumAllowableInteractionRange for Octachoron {
    #[inline]
    fn maximum_allowable_interaction_range(&self) -> f64 {
        self.half()
    }
}

impl<P> Wrap<P> for Periodic<Octachoron>
where
    P: Position<Position = Cartesian<4>>,
{
    #[inline]
    fn wrap(&self, properties: P) -> Result<P, super::super::Error> {
        let mut properties = properties;
        let r = properties.position_mut();
        let l = self.shape().edge_length.get();
        for c in r.coordinates.iter_mut() {
            let lambda = *c / l;
            let lambda = lambda - lambda.round();
            let lambda = if lambda == 0.5 { -0.5 } else { lambda };
            *c = lambda * l;
        }
        Ok(properties)
    }
}

impl<S> GenerateGhosts<S> for Periodic<Octachoron>
where
    S: Position<Position = Cartesian<4>> + Copy + Default,
{
    #[inline]
    fn maximum_interaction_range(&self) -> f64 {
        self.maximum_interaction_range()
    }

    #[inline]
    fn generate_ghosts(&self, site_properties: &S) -> ArrayVec<S, { super::super::MAX_GHOSTS }> {
        let mut result = ArrayVec::new();
        let r = site_properties.position();
        let l = self.shape().edge_length.get();
        let half = l / 2.0;
        let mir = self.maximum_interaction_range();

        if !(r.coordinates.iter().all(|&c| -half <= c && c < half)) {
            return result;
        }

        // For each dimension, determine the fixed ghost offset (if near a face)
        // and collect the near-dimension bitmask.
        let mut near_mask = 0u32;
        let mut dim_offset = [0.0_f64; 4];
        for i in 0..4 {
            if r[i] > half - mir {
                near_mask |= 1 << i;
                dim_offset[i] = -l;
            } else if r[i] < -half + mir {
                near_mask |= 1 << i;
                dim_offset[i] = l;
            }
        }

        // Enumerate all non-empty subsets of the near dimensions.
        let mut subset = near_mask;
        while subset != 0 {
            let mut ghost = *site_properties;
            let pos = ghost.position_mut();
            for i in 0..4 {
                if subset & (1 << i) != 0 {
                    pos[i] += dim_offset[i];
                }
            }
            result.push(ghost);
            subset = (subset - 1) & near_mask;
        }

        result
    }
}

impl RandDistribution<Cartesian<4>> for Octachoron {
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Cartesian<4> {
        let half = self.half();
        Cartesian::from(array::from_fn(|_| {
            Uniform::new(-half, half)
                .expect("edge length should be positive")
                .sample(rng)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::property::Point;

    #[test]
    fn test_volume() -> Result<(), Box<dyn std::error::Error>> {
        let oct = Octachoron::with_edge_length(2.0.try_into()?);
        assert_eq!(oct.volume(), 16.0);
        Ok(())
    }

    #[test]
    fn test_is_point_inside() -> Result<(), Box<dyn std::error::Error>> {
        let oct = Octachoron::with_edge_length(2.0.try_into()?);
        assert!(oct.is_point_inside(&Cartesian::from([0.0, 0.0, 0.0, 0.0])));
        assert!(oct.is_point_inside(&Cartesian::from([-1.0, 0.0, 0.0, 0.0])));
        assert!(!oct.is_point_inside(&Cartesian::from([1.0, 0.0, 0.0, 0.0])));
        assert!(!oct.is_point_inside(&Cartesian::from([2.0, 0.0, 0.0, 0.0])));
        Ok(())
    }

    #[test]
    fn test_ghosts_face_only() -> Result<(), Box<dyn std::error::Error>> {
        let oct = Octachoron::with_edge_length(10.0.try_into()?);
        let periodic = Periodic::new(2.0, oct)?;
        let site = Point::new(Cartesian::from([4.5, 0.0, 0.0, 0.0]));
        let ghosts = periodic.generate_ghosts(&site);
        // Near +x face only: 1 face ghost
        assert_eq!(ghosts.len(), 1);
        Ok(())
    }

    #[test]
    fn test_ghosts_corner() -> Result<(), Box<dyn std::error::Error>> {
        let oct = Octachoron::with_edge_length(10.0.try_into()?);
        let periodic = Periodic::new(2.0, oct)?;
        // Near all 4 positive faces
        let site = Point::new(Cartesian::from([4.5, 4.5, 4.5, 4.5]));
        let ghosts = periodic.generate_ghosts(&site);
        assert_eq!(ghosts.len(), 15);
        Ok(())
    }

    #[test]
    fn test_ghosts_interior() -> Result<(), Box<dyn std::error::Error>> {
        let oct = Octachoron::with_edge_length(10.0.try_into()?);
        let periodic = Periodic::new(2.0, oct)?;
        let site = Point::new(Cartesian::from([0.0, 0.0, 0.0, 0.0]));
        let ghosts = periodic.generate_ghosts(&site);
        assert!(ghosts.is_empty());
        Ok(())
    }

    #[test]
    fn test_wrap() -> Result<(), Box<dyn std::error::Error>> {
        let oct = Octachoron::with_edge_length(10.0.try_into()?);
        let periodic = Periodic::new(2.0, oct)?;
        let site = Point::new(Cartesian::from([6.0, 0.0, 0.0, 0.0]));
        let wrapped = periodic.wrap(site)?;
        assert!(periodic.shape().is_point_inside(wrapped.position()));
        Ok(())
    }
}

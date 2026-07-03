use crate::shape::Hypersphere;
use hoomd_vector::{Cartesian, InnerProduct, Rotate, Vector};
use thiserror::Error;

#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum PenetrationError {
    /// A penetration is only well-defined if the shapes intersect
    #[error("Shapes do not intersect.")]
    DoesNotIntersect,
}

pub trait ShapePenetration<const N: usize, S, P: Vector + InnerProduct, R: Rotate<P>> {
    /// .
    const TOLERANCE: f64 = 1e-12;

    /// Return the vector that translates A such that it no longer overlaps B.
    ///
    /// Sep function will come up with an initial guess
    fn penetration_vector(
        &self,
        other: &S,
        v_a: P,
        o_a: R,
        v_b: P,
        o_b: R,
    ) -> Result<P, PenetrationError>;

    // /// .
    // fn penetration_vector_from_guess<A, B>(a: &A, b: &B, guess: &V) -> Result<V, PenetrationError>;
}

impl<const N: usize, R: Rotate<Cartesian<N>>> ShapePenetration<N, Hypersphere<N>, Cartesian<N>, R>
    for Hypersphere<N>
{
    fn penetration_vector(
        &self,
        other: &Hypersphere<N>,
        v_a: Cartesian<N>,
        _o_a: R,
        v_b: Cartesian<N>,
        _o_b: R,
    ) -> Result<Cartesian<N>, PenetrationError> {
        let v_center_center = v_b - v_a;
        let d_center_center = v_center_center.norm();
        let radii_sum = (self.radius + other.radius).get();
        if d_center_center <= radii_sum {
            return Ok(v_center_center * (d_center_center.recip() * (d_center_center - radii_sum)));
        }
        Err(PenetrationError::DoesNotIntersect)
    }
}

#[cfg(test)]
mod tests {
    use approxim::assert_relative_eq;
    use hoomd_vector::{Cartesian, Versor};
    use std::assert_matches;

    use crate::{
        separate::{PenetrationError, ShapePenetration},
        shape::Hypersphere,
    };

    #[test]
    fn spheres_penetrate_along_z() -> anyhow::Result<()> {
        let a = Hypersphere::<3>::with_radius(0.5.try_into()?);
        let b = Hypersphere::<3>::with_radius(2.5.try_into()?);
        let origin = Cartesian::<3>::default();
        let q = Versor::default();

        let displacement = [0.0, 0.0, 3.001].into();
        assert_matches!(
            ShapePenetration::penetration_vector(&a, &b, origin, q, displacement, q),
            Err(PenetrationError::DoesNotIntersect)
        );

        let displacement = [0.0, 0.0, 2.999].into();
        let penetration = ShapePenetration::penetration_vector(&a, &b, origin, q, displacement, q)
            .expect("Should overlap.");
        assert_relative_eq!(penetration, Cartesian::from([0.0, 0.0, -0.001]));
        Ok(())
    }
}

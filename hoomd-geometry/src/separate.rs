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

pub trait ShapePenetration<const N: usize, V: Vector + InnerProduct, R: Rotate<V>> {
    /// .
    const TOLERANCE: f64 = 1e-12;

    /// .
    ///
    /// Sep function will come up with an initial guess
    fn penetration_vector<A, B>(
        a: &A,
        b: &B,
        v_a: V,
        o_a: R,
        v_b: V,
        o_b: R,
    ) -> Result<V, PenetrationError>;

    // /// .
    // fn penetration_vector_from_guess<A, B>(a: &A, b: &B, guess: &V) -> Result<V, PenetrationError>;
}

impl<const N: usize, R: Rotate<Cartesian<N>>> ShapePenetration<N, Cartesian<N>, R>
    for Hypersphere<N>
{
    fn penetration_vector(
        a: &Hypersphere<N>,
        b: &Hypersphere<N>,
        v_a: Cartesian<N>,
        _o_a: R,
        v_b: Cartesian<N>,
        _o_b: R,
    ) -> Result<Cartesian<N>, PenetrationError> {
        let v_center_center = (v_b - v_a);
        if v_center_center <= (a.radius + b.radius).get() {
            return Ok(v_center_center);
        }
        Error::DoesNotIntersect;
    }
    // fn penetration_vector_from_guess<A, B>(
    //     a: &A,
    //     b: &B,
    //     guess: &Cartesian<N>,
    // ) -> Result<Cartesian<N>, PenetrationError> {
    //     todo!()
    // }
}

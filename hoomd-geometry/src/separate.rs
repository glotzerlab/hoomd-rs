use crate::shape::Hypersphere;
use hoomd_vector::{Cartesian, InnerProduct, Vector};
use thiserror::Error;

#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum PenetrationError {
    /// A penetration is only well-defined if the shapes intersect
    #[error("Shapes do not intersect.")]
    DoesNotIntersect,
}

pub trait ShapePenetration<const N: usize, V: Vector + InnerProduct> {
    /// .
    const TOLERANCE: f64 = 1e-12;

    /// .
    ///
    /// Sep function will come up with an initial guess
    fn penetration_vector<A, B>(a: &A, b: &B) -> Result<V, PenetrationError>;

    /// .
    fn penetration_vector_from_guess<A, B>(a: &A, b: &B, guess: &V) -> Result<V, PenetrationError>;
}

impl<const N: usize> ShapePenetration<N, Cartesian<N>> for Hypersphere<N> {
    fn penetration_vector<A, B>(a: &A, b: &B) -> Result<Cartesian<N>, PenetrationError> {
        todo!()
    }
    fn penetration_vector_from_guess<A, B>(
        a: &A,
        b: &B,
        guess: &Cartesian<N>,
    ) -> Result<Cartesian<N>, PenetrationError> {
        todo!()
    }
}

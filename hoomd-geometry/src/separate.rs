use hoomd_vector::{Cartesian, InnerProduct, Vector};

pub trait SeparationDistance<const N: usize, V: Vector + InnerProduct> {
    /// .
    const TOLERANCE: f64;

    /// .
    ///
    /// Sep function will come up with an initial guess
    fn separating_vector<A, B>(a: &A, v: &B) -> V;

    /// .
    fn separating_vector_from_guess<A, B>(a: &A, b: &B, guess: &V);
}

/*!
Modifier structs to provide additional information via encapsulation.

[`Sphero`] rounds a geometry with some radius, and [`Centered`] provides a notion of the
origin of a particular shape.
*/

use hoomd_vector::Cartesian;

/**
Round a `Shape` with some radius.
*/
pub struct Sphero<S> {
    /// The struct to be rounded. This is typically a `Shape`, but does not have to be
    pub shape: S,
    /// The radius of the rounding sphere
    pub rounding_radius: f64,
}

/**
Provide an origin to a `Shape` struct.
*/
#[derive(Clone)]
pub struct Centered<S, const N: usize> {
    /// The struct to be centered. This is typically a `Shape`, but does not have to be
    pub shape: S,
    /// The center of mass of the struct
    pub centroid: Cartesian<N>,
}

// impl<B, S> Intersects<S> for Centered<B> where B: Intersects<S> {

// }

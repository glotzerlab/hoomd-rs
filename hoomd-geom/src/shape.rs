/// General traits for shapes
use hoomd_vector::Vector;

///
pub trait Volume {
    /// The N-hypervolume of a geometry
    #[must_use]
    fn volume(&self) -> f64;
}

///
trait Particle {} // In different crate! - should be copyable (array based?)

pub trait Convex {
    /// Whether a shape is convex.
    #[must_use]
    #[inline]
    fn is_convex(&self) -> bool {
        true
    }
    // Can be used to enable Xenocollide or something? provides a nice trait bound
}

/// A generalization of properties that are well defined for arbitrary shapes.
pub trait Shape<const N: usize> {
    /// TODO
    type V: Vector;
    /// Center of mass
    fn centroid(&self) -> Self::V;
}

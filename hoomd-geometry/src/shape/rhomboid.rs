// TODO: Oblique?

use hoomd_utility::valid::PositiveReal;

/// An axis-aligned parallelogram defined by a 2 x 2 upper triangular matrix.
struct Rhomboid {
    extents: [PositiveReal; 2],
    xy: f64,
}

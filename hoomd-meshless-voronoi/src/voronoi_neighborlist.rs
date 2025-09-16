/** TODO: documentation

nearest neighbors from rtree
*/

pub use crate::local::GeneratorHyperbolic;
use crate::rtree_nn::{build_rtree, build_rtree_hyperbolic, nn_iter, nn_iter_hyperbolic};
use glam::DVec3;
#[cfg(feature = "rayon")]
use rayon::prelude::*;


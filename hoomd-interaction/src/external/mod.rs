// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! External interactions.

mod constant_force;
pub use constant_force::ConstantForce;

mod constant_torque;
pub use constant_torque::ConstantTorque;

/// Linear potential based on position.
///
/// Computes a Linear external potential at a point in space relative to the plane
/// origin `p`, plane normal `n`, and the interaction strength `alpha`.
///
/// ```math
/// U = \alpha \cdot \hat{n} \cdot ( \vec{r} - \vec{p} )
/// ```
///
/// # Generics
///
/// * `V`: The type used to represent the position and normal vectors.
///
/// # Example
///
/// Basic usage:
///
/// ```
/// use hoomd_interaction::external::Linear;
/// use hoomd_vector::{Cartesian, Unit};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let Linear = Linear {
///     alpha: 2.0,
///     plane_origin: [0.0, -10.0].into(),
///     plane_normal: [0.0, 1.0].try_into()?,
/// };
/// # Ok(())
/// # }
/// ```
pub type Linear<V> = ConstantForce<V>;

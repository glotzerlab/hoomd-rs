use crate::{IntersectsAt, Shape, SupportFn, Volume};
use hoomd_vector::{Cartesian, Cross, Rotate, Rotation, Vector};

/// Get a vector perpendicular to a 2-vector
#[inline]
pub fn perp(v: Cartesian<2>) -> Cartesian<2> {
    Cartesian::from([-v[1], v[0]])
}

/// Functor for Support Function calculations
struct SupportFunctor<'a, const N: usize, R: Copy + Rotation + Rotate<Cartesian<N>>, T: SupportFn> {
    /// Support-function shape A
    sa: &'a T,
    /// Support-function shape B
    sb: &'a T,
    /// Vector separating A and B
    v_ij: &'a Cartesian<N>,
    /// Relative orientation between A and B
    q_ij: &'a R,
}

/// Composite support function
#[inline]
fn composite_support<const N: usize, R: Copy + Rotation + Rotate<Cartesian<N>>>(
    sa: &impl SupportFn,
    sb: &impl SupportFn,
    v_ij: &Cartesian<N>,
    q_ij: &R, // RotationMatrix derives copy, so this should always be valid
    n: Cartesian<N>,
) -> Cartesian<N> {
    // TODO: this should hold state of the components, so you only need to pass in n
    // For now, this is ok
    // Support point of b in the direction of vij
    // "translation/rotation formula comes from pg 168 of "Games Programming Gems 7""
    // Formula is dimension agnostic: q @ sb.support(q_inverse @ n) + v_ij
    let sb_n = q_ij.rotate(&sb.support(&q_ij.inverted().rotate(&n))) + *v_ij;

    sb_n - sa.support(&-n)
}

/// Xenocollide in 2 dimensions. For now, hard coded to 2
#[inline]
fn collide<R: Rotate<Cartesian<2>> + Rotation + Copy>(
    sa: &impl SupportFn,
    sb: &impl SupportFn,
    v_ij: &Cartesian<2>, // Probably ok to take ownership?
    q_ij: &R,
) -> bool {
    // Phase 1: Portal discovery

    // 1a: Determine whether the origin lies in B⊖A, given only the support mapping
    // 1b: Obtain a point that lies deep in B⊖A:
    let v0 = *v_ij; // self.centroid()-other.centroid() in extrinsic coords

    // 1c: Construct a normal pointing from p to the origin: this is just p̂?
    // Find support point in this direction
    // Find a candidate portal
    // let v1 = sa.support(&-v0); // negative, to ensure ||v1|| > 0
    let v1 = composite_support(sa, sb, v_ij, q_ij, -v0); // negative, to ensure ||v1|| > 0

    // 1d. We construct a ray that is perpendicular to the line between the
    // support just discovered and the interior point. There are two choices for this
    // ray, one for each side of the line segment. We choose the ray that lies on the
    // same side of the segment as the origin. We use this ray to find a second support
    // point on the surface of B–A. NOTE: more choices in 3+dimensions?

    // v_perp is on the same side as the origin if v1.dot(v_perp) < 0
    let mut v_perp_v1v0 = (v1 - v0).perp();
    if v1.dot(&v_perp_v1v0) > 0.0 {
        v_perp_v1v0 = -v_perp_v1v0;
    }

    let v2 = sb.support(&v_perp_v1v0);

    // Now we have three points, which form a frustum (angle in 2d). The origin lies
    // somewhere within this frustum
    // 1f. Create a line segment between our two support points: called the portal, as
    // the origin ray (v0?) must pass through the segment on the way to the origin
    let p0 = v2 - v1; // TODO

    // Phase 2: Portal Refinement

    // Figure 1f: If the origin lies on the same side of the portal as the
    // interior point, then it lies within the dotted triangle, and must therefore lie
    // within B–A. When this is the case, we terminate with a hit. In this example, the
    // point lies on the outside of the portal, so the algorithm continues.

    // If dot product between
    let mut p0_perp = p0.perp();

    // Ensure p0_perp is facing away from our interior point
    if (v1 - v0).dot(&p0_perp) < 0.0 {
        p0_perp = -p0_perp;
    }

    // Check if origin is inside the initial portal. MUST be >= to cover exact overlaps.
    if v1.dot(&p0_perp) >= 0.0 {
        return true; // TODO: why is this >0.0? origin should be facing away?
    }
    // TODO: mock the diagram in python to verify signs, etc.

    // 1g: We construct a normal perpendicular to the portal, pointing away from the
    // interior. We use this normal (p0_perp) to obtain a third support point on the
    // surface of B–A. If the origin lies outside of the support line formed by the
    // point and the normal, we know that the origin lies outside of B–A. In this case,
    // the point lies on the inside of the support line, so the algorithm continue
    // let v3

    false
}

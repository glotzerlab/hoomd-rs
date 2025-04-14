use crate::{IntersectsAt, Shape, SupportFn, Volume};
use hoomd_vector::{Angle, Cartesian, Cross, Rotate, Rotation, Vector};

// /// Get a vector perpendicular to a 2-vector
// #[inline]
// pub fn perp(v: Cartesian<2>) -> Cartesian<2> {
//     Cartesian::from([-v[1], v[0]])
// }

const XENOCOLLIDE_2D_MAX_ITER: usize = 16;

/// Stateful function for support function calculations on Minkowski differences.
struct SupportFunctor<
    'a,
    const N: usize,
    R: Copy + Rotation + Rotate<Cartesian<N>>,
    T: SupportFn<Cartesian<N>>,
> {
    /// Support-function shape A
    sa: &'a T,
    /// Support-function shape B
    sb: &'a T,
    /// Vector separating A and B
    v_ij: &'a Cartesian<N>,
    /// Relative orientation between A and B
    q_ij: &'a R,
}

impl<const N: usize, T: SupportFn<Cartesian<N>>, R: Copy + Rotation + Rotate<Cartesian<N>>>
    SupportFunctor<'_, N, R, T>
{
    /// Compute the support function on the Minkowski difference of two shapes.
    #[inline]
    fn composite_support(&self, n: Cartesian<N>) -> Cartesian<N> {
        // Support point of b in the direction of vij
        // 'translation/rotation formula comes from pg 168 of "Games Programming Gems 7"'
        // Dimension-agnostic formula: r @ sb.support(r_inverse @ n) + v_ij
        let sb_n = self
            .q_ij
            .rotate(&self.sb.support(&self.q_ij.inverted().rotate(&n)))
            + *self.v_ij;

        sb_n - self.sa.support(&-n) // eq. 2.5.6 in GPG7
    }
}

/// Xenocollide in 2 dimensions. For now, hard coded to 2
#[allow(dead_code)] // TODO: temp
#[inline]
fn collide<R: Rotate<Cartesian<2>> + Rotation + Copy, T: SupportFn<Cartesian<2>>>(
    sa: &T,
    sb: &T,
    v_ij: &Cartesian<2>, // Probably ok to take ownership?
    q_ij: &R,
) -> bool {
    let tol_multiplier = 10_000f64;
    let s = SupportFunctor { sa, sb, v_ij, q_ij };

    // Phase 1: Portal discovery
    // Obtain a point lying deep within B⊖A
    let v0 = *v_ij; // self.centroid()-other.centroid() in extrinsic coords

    // Find the support point in the direction of the origin ray
    let mut v1 = s.composite_support(-v0); // negative, to ensure ||v1|| > 0

    // v_perp is on the same side as the origin if v1.dot(v_perp) < 0
    let mut v_perp_v1v0 = (v1 - v0).perp();
    if v1.dot(&v_perp_v1v0) > 0.0 {
        v_perp_v1v0 = -v_perp_v1v0;
    }

    // Support point perpendicular to plane containing the origin, v0, and v1
    let mut v2 = s.composite_support(v_perp_v1v0);

    // 2. Portal Refinement
    // Now we have three points which form our portal

    let mut count = 0usize;
    loop {
        count += 1;

        // Vector normal to the portal segment, facing away from the interior point
        let mut v_perp_v2v1 = (v2 - v1).perp();
        if (v1 - v0).dot(&v_perp_v2v1) < 0.0 {
            v_perp_v2v1 = -v_perp_v2v1;
        }

        // Check if origin is inside or overlapping the initial portal
        if v1.dot(&v_perp_v2v1) >= 0.0 {
            return true;
        }

        // Support point in the direction of the portal
        let v3 = s.composite_support(v_perp_v2v1);

        // If the origin is outside the support plane, return false (no overlap)
        if v3.dot(&v_perp_v2v1) < 0.0 {
            return false;
        }

        // TODO: Tolerance check. Do we need this with f64?
        // let d = ((v3 - v1) - project(v3 - v1, v2 - v1)) * tol_multiplier;

        // Choose new portal, which may either be v3v2 or v1v3
        let mut v_perp_v3v0 = (v3 - v0).perp();
        // make v_perp_v3v0 point toward v1
        if (v1 - v3).dot(&v_perp_v3v0) < 0.0 {
            v_perp_v3v0 = -v_perp_v3v0;
        }
        if v3.dot(&v_perp_v3v0) < 0.0 {
            // Origin is on the v1 side, so new portal is v3v1
            v2 = v3;
        } else {
            // Origin is on the v2 side, so new portal is v3v2
            v1 = v3;
        }

        // TODO: does this need to be a loop?

        println!("LOOPING # {count}");
        if count >= XENOCOLLIDE_2D_MAX_ITER {
            return true;
        }
    }

    // If the origin lies on the same side of the portal as the interior point, then it
    // lies within the dotted triangle, and must therefore lie within B⊖A. When this is
    // the case, we terminate with a hit.

    // TODO: mock the diagram in python to verify signs, etc.

    // 1g: We construct a normal perpendicular to the portal, pointing away from the
    // interior. We use this normal (p0_perp) to obtain a third support point on the
    // surface of B–A. If the origin lies outside of the support line formed by the
    // point and the normal, we know that the origin lies outside of B–A. In this case,
    // the point lies on the inside of the support line, so the algorithm continue
    // let v3 = s.composite_support(p0_perp);

    // If origin is outside the support plane, we are not overlapping
    // if v3.dot(&p0_perp) < 0.0 {
    //     return false;
    // }
}

#[cfg(test)]
#[allow(clippy::used_underscore_binding)]
mod tests {
    use super::*;
    use rstest::*;

    use crate::Sphere;

    #[rstest(
        v => [[0.1, 0.1], [999.9, 0.0], [0.0, 5.123], [0.0, 5.124]]
    )]
    fn test_discs_collide(v: [f64; 2]) {
        let (s0, s1) = (Sphere::<2>::from(1.0), Sphere::<2>::from(4.123));
        let theta = &Angle::from(0.0);

        let overlaps = collide(&s0, &s1, &v.into(), theta);
        println!("{v:?}, {overlaps}");

        assert_eq!(overlaps, s0.intersects_at(&s1, &v.into(), theta),);
    }
}

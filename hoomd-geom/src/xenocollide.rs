// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use crate::{IntersectsAt, Shape, SupportFn, Volume};
use hoomd_vector::{Angle, Cartesian, Cross, Rotate, Rotation, Vector};

// /// Get a vector perpendicular to a 2-vector
// #[inline]
// pub fn perp(v: Cartesian<2>) -> Cartesian<2> {
//     Cartesian::from([-v[1], v[0]])
// }

/// Maximum allowed iterations for Xenocollide in 2D
const XENOCOLLIDE_2D_MAX_ITER: usize = 1024;
/// Maximum allowed iterations for Xenocollide in 3D
const XENOCOLLIDE_3D_MAX_ITER: usize = 1024;

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
#[inline]
pub fn collide2d<R: Rotate<Cartesian<2>> + Rotation + Copy, T: SupportFn<Cartesian<2>>>(
    sa: &T,
    sb: &T,
    v_ij: &Cartesian<2>, // Probably ok to take ownership?
    q_ij: &R,
) -> bool {
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

        if count >= XENOCOLLIDE_2D_MAX_ITER {
            return true;
        }
    }
}

/// Minkowski Portal Refinement-based collision detection in 3d
#[inline]
pub fn collide3d<R: Rotate<Cartesian<3>> + Rotation + Copy, T: SupportFn<Cartesian<3>>>(
    sa: &T,
    sb: &T,
    v_ij: &Cartesian<3>, // Probably ok to take ownership?
    q_ij: &R,
) -> bool {
    let precision_tol = 2e-15; // Set fixed tol, rather than rounding-radius based
    let s = SupportFunctor { sa, sb, v_ij, q_ij };

    // Phase 1: Portal discovery
    // Obtain a point lying deep within B⊖A
    let v0 = *v_ij; // self.centroid()-other.centroid() in extrinsic coords

    // find_candidate_portal()

    // Support point in the direction of the origin ray
    let mut v1 = s.composite_support(-v0); // negative, to ensure ||v1|| > 0

    // Equivalent to v1 . (v1-v0) <= 0 by convexity
    if v1.dot(&v0) > 0.0 {
        return false; // Origin is outside the v1 support plane
    }

    // Direction perpendicular to v0, v1 plane
    let n = v1.cross(&v0);

    // Cross product is zero if v0,v1 colinear with origin, but we have already
    // determined origin is within v1 support plane. If origin is on a line between
    // v1 and v0, particles overlap. We assume precision_tol has units l**2
    if n.into_iter().all(|x| x.abs() < precision_tol) {
        return true;
    }

    // Support point perpendicular to plane containing the origin, v0, and v1
    let mut v2 = s.composite_support(n);

    if v2.dot(&n) < 0.0 {
        return false; // Origin lies outside the v2 support plane
    }

    // Support point perpendicular to plane containing interior point and first 2 supports
    let mut n = (v2 - v0).cross(&(v1 - v0));

    // Maintain known handedness of the portal
    if n.dot(&v0) >= 0.0 {
        (v1, v2) = (v2, v1);
        n = -n;
    }

    // while origin_ray_does_not_intersect_candidate()
    let mut intersects = false;
    let mut count = 0usize;
    loop {
        count += 1;

        if count >= XENOCOLLIDE_3D_MAX_ITER {
            return true;
        }

        let v3 = s.composite_support(n);
        if v3.dot(&n) <= 0.0 {
            return false; // Origin is outside the v3 support plane
        }

        // If origin lies on the opposite side of the plane from our third support
        // point, use the outer facing plane normal.
        // Check the v3, v0, v1 plane for validity
        if v1.cross(&v3).dot(&v0) < 0.0 {
            v2 = v3; // Preserve handedness
            n = (v1 - v0).cross(&(v2 - v0));
            continue; // Continue iterating to find a valid portal
        }
        if v3.cross(&v2).dot(&v0) < 0.0 {
            v1 = v3; // Preserve handedness
            n = (v1 - v0).cross(&(v2 - v0));
            continue;
        }
        break false; // If we've made it this far, we've found a valid portal
    }

    // TODO: continue
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    use crate::{Cuboid, Sphere};

    #[rstest(
        v => [[0.1, 0.1], [999.9, 0.0], [0.0, 5.123], [0.0, 5.123_000_000_000_001]]
    )]
    fn test_discs_collide(v: [f64; 2]) {
        let (s0, s1) = (Sphere::<2>::from(1.0), Sphere::<2>::from(4.123));
        let theta = &Angle::from(0.0);

        let overlaps = collide2d(&s0, &s1, &v.into(), theta);

        assert_eq!(overlaps, s0.intersects_at(&s1, &v.into(), theta),);
    }

    #[rstest(
        v => [[0.1, 0.1], [999.9, 0.0], [0.0, 5.123], [0.0, 5.123_000_000_000_001]],
        rect => [[1.0, 1.0], [999.0, 0.1], [1.0, 2.0*4.623]]
    )]
    fn test_aabrs_collide(v: [f64; 2], rect: [f64; 2]) {
        let c0 = Cuboid::from(rect);
        let c1 = Cuboid::from([1.0; 2]);
        let theta = &Angle::from(0.0);

        let overlaps = collide2d(&c0, &c1, &v.into(), theta);
        assert_eq!(overlaps, c0.intersects_at(&c1, &v.into(), theta),);
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Implementations of the Xenocollide collision detection algorithm.

`collide2d` and `collide3d` allow for intersections between arbitrary geometries that
define the `SupportMapping<Cartesian<2|3>>` trait.
*/
use crate::SupportMapping;
use hoomd_vector::{Cartesian, Cross, Rotate, Rotation, RotationMatrix, Vector};

/// Maximum allowed iterations for Xenocollide in 2D
const XENOCOLLIDE_2D_MAX_ITER: usize = 1024;
/// Maximum allowed iterations for Xenocollide in 3D
const XENOCOLLIDE_3D_MAX_ITER: usize = 1024;

/// Stateful function for support function calculations on Minkowski differences.
struct SupportFunctor<
    'a,
    const N: usize,
    A: SupportMapping<Cartesian<N>>,
    B: SupportMapping<Cartesian<N>>,
> {
    /// Support-function shape A
    sa: &'a A,
    /// Support-function shape B
    sb: &'a B,
    /// Vector separating A and B
    v_ij: &'a Cartesian<N>,
    /// Relative orientation between A and B
    q_ij: RotationMatrix<N>,
    /// Inverse of relative orientation between A and B
    q_ij_inv: RotationMatrix<N>,
}

impl<'a, const N: usize, A: SupportMapping<Cartesian<N>>, B: SupportMapping<Cartesian<N>>>
    SupportFunctor<'_, N, A, B>
{
    /// Compute the support function on the Minkowski difference of two shapes.
    #[inline]
    fn composite_support_mapping(&self, n: Cartesian<N>) -> Cartesian<N> {
        // Support point of b in the direction of vij
        // 'translation/rotation formula comes from pg 168 of "Games Programming Gems 7"'
        // Dimension-agnostic formula: r @ sb.support_mapping(r_inverse @ n) + v_ij
        // Applying rotation takes ~24% of total runtime in collide3d simplex3
        let sb_n = self
            .q_ij
            .rotate(&self.sb.support_mapping(&self.q_ij_inv.rotate(&n)))
            + *self.v_ij;
        sb_n - self.sa.support_mapping(&-n) // eq. 2.5.6 in GPG7
    }
    /// Create a new `SupportFunctor` from a Rotation that can be converted into a `RotationMatrix`
    #[inline]
    fn new<R: Rotation + Copy>(
        sa: &'a A,
        sb: &'a B,
        v_ij: &'a Cartesian<N>,
        r: R,
    ) -> SupportFunctor<'a, N, A, B>
    where
        RotationMatrix<N>: From<R>,
    {
        let q_ij = RotationMatrix::<N>::from(r);
        let q_ij_inv = RotationMatrix::<N>::from(r.inverted());
        SupportFunctor {
            sa,
            sb,
            v_ij,
            q_ij,
            q_ij_inv,
        }
    }
}

/// Xenocollide in 2 dimensions. For now, hard coded to 2
#[inline]
pub fn collide2d<
    R: Copy + Rotation,
    A: SupportMapping<Cartesian<2>>,
    B: SupportMapping<Cartesian<2>>,
>(
    sa: &A,
    sb: &B,
    v_ij: &Cartesian<2>,
    q_ij: &R,
) -> bool
where
    RotationMatrix<2>: From<R>,
{
    let tol_multiplier = 10000.0;
    let tol = 1e-16 * tol_multiplier;
    let s = SupportFunctor::new(sa, sb, v_ij, *q_ij);

    // Phase 1: Portal discovery
    // Obtain a point lying deep within B⊖A
    let v0 = *v_ij; // self.centroid()-other.centroid() in extrinsic coords

    // Find the support point in the direction of the origin ray
    let mut v1 = s.composite_support_mapping(-v0); // negative, to ensure ||v1|| > 0

    // v_perp is on the same side as the origin if v1.dot(v_perp) < 0
    let mut v_perp_v1v0 = (v1 - v0).perpendicular();
    if v1.dot(&v_perp_v1v0) > 0.0 {
        v_perp_v1v0 = -v_perp_v1v0;
    }

    // Support point perpendicular to plane containing the origin, v0, and v1
    let mut v2 = s.composite_support_mapping(v_perp_v1v0);

    // 2. Portal Refinement
    // Now we have three points which form our portal

    let mut count = 0usize;
    loop {
        count += 1;

        // Vector normal to the portal segment, facing away from the interior point
        let mut v_perp_v2v1 = (v2 - v1).perpendicular();
        if (v1 - v0).dot(&v_perp_v2v1) < 0.0 {
            v_perp_v2v1 = -v_perp_v2v1;
        }

        // Check if origin is inside or overlapping the initial portal
        if v1.dot(&v_perp_v2v1) >= 0.0 {
            return true;
        }

        // Support point in the direction of the portal
        let v3 = s.composite_support_mapping(v_perp_v2v1);

        // If the origin is outside the support plane, return false (no overlap)
        if v3.dot(&v_perp_v2v1) < 0.0 {
            return false;
        }

        // Tolerance check. NOTE: may not always be necessary?
        let d = (v3 - v1) - (v3 - v1).project(&(v2 - v1)) * tol_multiplier;
        if d.norm_squared() < tol.powi(2) * v3.norm_squared() {
            return true;
        }

        // Choose new portal, which may either be v3v2 or v1v3
        let mut v_perp_v3v0 = (v3 - v0).perpendicular();
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
#[inline(never)]
pub fn collide3d<
    R: Rotation + Copy,
    A: SupportMapping<Cartesian<3>>,
    B: SupportMapping<Cartesian<3>>,
>(
    sa: &A,
    sb: &B,
    v_ij: &Cartesian<3>, // Probably ok to take ownership?
    q_ij: &R,
) -> bool
where
    RotationMatrix<3>: From<R>,
{
    let precision_tol = 2e-12; // Set fixed tol, rather than rounding-radius based
    let root_tol = 4e-8;
    let tol_multiplier = 10_000.0;

    if v_ij.into_iter().all(|x| x.abs() < root_tol) {
        // Interior point is at the origin => shapes overlap
        return true;
    }

    let s = SupportFunctor::new(sa, sb, v_ij, *q_ij);

    // Phase 1: Portal discovery
    // Obtain a point lying deep within B⊖A
    let v0 = *v_ij; // self.centroid()-other.centroid() in extrinsic coords

    // find_candidate_portal()

    // Support point in the direction of the origin ray
    let mut v1 = s.composite_support_mapping(-v0); // negative, to ensure ||v1|| > 0

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
    let mut v2 = s.composite_support_mapping(n);

    if v2.dot(&n) < 0.0 {
        return false; // Origin lies outside the v2 support plane
    }

    // Support point perpendicular to plane containing interior point and first 2 supports
    let mut n = (v1 - v0).cross(&(v2 - v0));
    // Maintain known handedness of the portal
    if n.dot(&v0) > 0.0 {
        (v1, v2) = (v2, v1);
        n = -n;
    }

    // while origin_ray_does_not_intersect_candidate()
    let mut count = 0usize;
    let mut v3 = loop {
        count += 1;

        if count >= XENOCOLLIDE_3D_MAX_ITER {
            return true;
        }

        let v3 = s.composite_support_mapping(n);
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
        break v3; // If we've made it this far, we've found a valid portal
    };

    count = 0;
    loop {
        count += 1;

        // Outer-facing normal of the current portal
        n = (v2 - v1).cross(&(v3 - v1));

        // Check if origin is inside (or overlapping) the portal
        if n.dot(&v1) >= 0.0 {
            // We already know that the origin lies within 3 of the faces of our portal
            // simplex. If it lies within the final face, it lies within B⊖A
            return true;
        }

        // Support point in direction of outer-facing normal of portal
        // This point helps us determine how far outside the portal the origin lies
        let v4 = s.composite_support_mapping(n);

        // If the origin is outside the support plane, it cannot lie inside B⊖A
        if n.dot(&v4) < 0.0 {
            return false;
        }

        // TODO: tolerance checks?
        n = (v2 - v1).cross(&(v3 - v1));
        let mut d = ((v4 - v1) * tol_multiplier).dot(&n);
        // let R = 1.0; // Average circumsphere diameter of the two shapes
        let tol = precision_tol * tol_multiplier * n.norm();

        // First, check if v4 is on plane (v2, v1, v3)
        if d.abs() < tol {
            // No more refinement possible, but not intersection detected
            return false;
        }
        // Second, check if origin is on plane (v2, v1, v3) and has been missed by other checks
        d = (v1 * tol_multiplier).dot(&n);
        if d.abs() < tol {
            return true;
        }

        // Choose a new portal. Two of its edges will be from the planes (v4,v0,v1),
        // (v4,v0,v2), (v4,v0,v3). Find which two have the origin on the same side.

        /* Comment inherited from HOOMD source code:
        "MEI: As I understand this statement, I don't believe it is correct. An _inside_
        needs to be defined and used. The only way I can think to do this is to consider
        all three pairs of planes to find which pair has the origin between them. Need
        to better understand and document this. The following code was directly adapted
        from example code."

        Test origin against the three planes that separate the new portal candidates
        Note:  We're taking advantage of the triple product identities here
        as an optimization
               (v1 % v4) * v0 == v1 * (v4 % v0) > 0 if origin inside (v1, v4, v0)
               (v2 % v4) * v0 == v2 * (v4 % v0) > 0 if origin inside (v2, v4, v0)
               (v3 % v4) * v0 == v3 * (v4 % v0) > 0 if origin inside (v3, v4, v0)
        */
        let v_perp_v4v0 = v4.cross(&v0);

        // Compiles to the same code as the original if-else, despite the extra dot
        #[expect(
            clippy::match_same_arms,
            reason = "Clearly illustrate translation from c."
        )]
        match (
            v_perp_v4v0.dot(&v1) > 0.0,
            v_perp_v4v0.dot(&v2) > 0.0,
            v_perp_v4v0.dot(&v3) > 0.0,
        ) {
            (true, true, _) => v1 = v4,   // Inside  v1 && inside  v2 => eliminate v1
            (true, false, _) => v3 = v4,  // Inside  v1 && OUTside v2 => eliminate v3
            (false, _, true) => v2 = v4,  // OUTside v1 && inside  v3 => eliminate v2
            (false, _, false) => v1 = v4, // OUTside v1 && OUTside v3 => eliminate v1
        }
        if count >= XENOCOLLIDE_3D_MAX_ITER {
            return true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntersectsAt;
    use rstest::*;

    use crate::shape::{Cuboid, Sphere};
    use hoomd_vector::{Angle, Versor};

    #[rstest(
        v => [[0.1, 0.1], [999.9, 0.0], [0.0, 5.123_f64.next_down()], [0.0, 5.123_000_001]],
        r => [0.001, 1.0, 4.123, 99.05],
        o_ij => [
            Angle::default(),
            Angle::from(std::f64::consts::PI / 3.0),
            Angle::from(1.234)
        ],
    )]
    fn test_discs_collide(v: [f64; 2], r: f64, o_ij: Angle) {
        let (s0, s1) = (Sphere::<2>::from(1.0), Sphere::<2>::from(r));

        let overlaps = collide2d(&s0, &s1, &v.into(), &o_ij);

        assert_eq!(overlaps, s0.intersects_at(&s1, &v.into(), &o_ij));
    }
    #[rstest(
        v => [[0.1, 0.1, 0.1], [999.9, 0.0, -10.9], [0.0, 5.123, 0.0], [0.0, 0.0, 5.123_000_001]],
        r => [0.001, 1.0, 4.123, 99.05],
        o_ij => [
            Versor::default(),
            Versor::from_axis_angle(
                [1.0, 0.0, 0.0].try_into().unwrap(), std::f64::consts::FRAC_PI_2
            ),
            Versor::from_axis_angle([0.0, 1.0, 0.0].try_into().unwrap(), 0.1234)
        ]
    )]
    fn test_spheres_collide(v: [f64; 3], r: f64, o_ij: Versor) {
        let (s0, s1) = (Sphere::<3>::from(1.0), Sphere::<3>::from(r));
        let overlaps = collide3d(&s0, &s1, &v.into(), &o_ij);

        assert_eq!(
            overlaps,
            s0.intersects_at(&s1, &v.into(), &o_ij),
            "Xenocollide result did not match standard implementation!"
        );
    }

    #[rstest(
        v => [[0.1, 0.1], [999.9, 0.0], [0.0, 5.123], [0.0, 5.123_000_000_000_001]],
        rect => [[1.0, 1.0], [999.0, 0.1], [1.0, 2.0*4.623]]
    )]
    fn test_aabrs_collide(v: [f64; 2], rect: [f64; 2]) {
        let c0 = Cuboid::from(rect);
        let c1 = Cuboid::from([1.0; 2]);
        let theta = Angle::from(0.0);

        let overlaps = collide2d(&c0, &c1, &v.into(), &theta);
        assert_eq!(
            overlaps,
            c0.intersects_at(&c1, &v.into(), &Angle::default())
        );
    }
    #[rstest(
        v => [[0.1, 2.1, 0.1], [999.9, 0.0, 0.05], [0.0, 5.123, 0.0], [0.0, 5.123_000_000_001, 0.0]],
        aabb => [[1.0, 1.0, 1.0], [999.0, 0.1, 0.5], [1.0, 2.0*4.623, 5.0]]
    )]
    fn test_aabbs_collide(v: [f64; 3], aabb: [f64; 3]) {
        let c0 = Cuboid::from(aabb);
        let c1 = Cuboid::from([1.0; 3]);
        let theta = Versor::identity();

        let overlaps = collide3d(&c0, &c1, &v.into(), &theta);
        assert_eq!(
            overlaps,
            c0.intersects_at(&c1, &v.into(), &Versor::default())
        );
    }
}

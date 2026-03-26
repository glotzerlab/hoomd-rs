// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implementations of the Xenocollide collision detection algorithm.
//!
//! [`collide2d`] and [`collide3d`] test for intersections between arbitrary geometries
//! that implement the [`SupportMapping<Cartesian<2|3>>`](`crate::SupportMapping`) trait.
//!
//! # Example
//!
//! In general, Xenocollide should be used via the [`IntersectsAt`](`crate::IntersectsAt`)
//! trait. However, the raw xenocollide methods can be used where needed.
//! ```
//! use hoomd_geometry::{IntersectsAt, shape::Circle, xenocollide::collide2d};
//! use hoomd_vector::Angle;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let (s0, s1) = (
//!     Circle {
//!         radius: 1.0.try_into()?,
//!     },
//!     Circle {
//!         radius: 2.0.try_into()?,
//!     },
//! );
//! let displacement = [3.0; 2].into();
//! assert_eq!(
//!     collide2d(&s0, &s1, &displacement, &Angle::default()),
//!     s0.intersects_at(&s1, &displacement, &Angle::default())
//! );
//! # Ok(())
//! # }
//! ```
use crate::SupportMapping;
use hoomd_vector::{Cartesian, Cross, InnerProduct, Rotate, RotationMatrix};

/// Maximum allowed iterations for Xenocollide in 2D
const XENOCOLLIDE_2D_MAX_ITER: usize = 1024;
/// Maximum allowed iterations for Xenocollide in 3D
const XENOCOLLIDE_3D_MAX_ITER: usize = 1024;

/// Stateful type that efficiently computes repeated Minkowski differences.
struct MinkowskiDifference<
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
    MinkowskiDifference<'_, N, A, B>
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

    /// Create a new `MinkowskiDifference`
    #[inline]
    fn new<R>(
        sa: &'a A,
        sb: &'a B,
        v_ij: &'a Cartesian<N>,
        r: R,
    ) -> MinkowskiDifference<'a, N, A, B>
    where
        R: Copy,
        RotationMatrix<N>: From<R>,
    {
        let q_ij = RotationMatrix::<N>::from(r);
        let q_ij_inv = q_ij.inverted();
        MinkowskiDifference {
            sa,
            sb,
            v_ij,
            q_ij,
            q_ij_inv,
        }
    }
}

/// Detect collision between two convex 2D objects via Minkowski Portal Refinement.
#[inline]
pub fn collide2d<R: Copy, A: SupportMapping<Cartesian<2>>, B: SupportMapping<Cartesian<2>>>(
    sa: &A,
    sb: &B,
    v_ij: &Cartesian<2>,
    q_ij: &R,
) -> bool
where
    RotationMatrix<2>: From<R>,
{
    const TOLERANCE: f64 = 1e-16;
    let s = MinkowskiDifference::new(sa, sb, v_ij, *q_ij);

    // Phase 1: Portal discovery
    // Obtain a point lying deep within B⊖A
    let v0 = *v_ij; // self.centroid()-other.centroid() in extrinsic coords

    // TODO: This is unsafe for types like `ConvexPolytope`. Users can construct
    // such types where the origin is outside the shape. Option 1: Validate
    // the vertices when constructing `ConvexPolytope` Option 2: Implement
    // a `SomePointInside` trait that must always return a point inside the
    // shape. For `ConvexPolytope` this could be the mean of the vertices.
    // `ConvexPolytope` makes vertices private, so this point could be
    // precomputed and result in no performance impact to xenocollide.

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

    let mut count = 0_usize;
    loop {
        count += 1;

        // Vector normal to the portal segment, facing away from the interior point
        let mut v_perp_v2v1 = (v2 - v1).perpendicular();
        if (v1 - v0).dot(&v_perp_v2v1) < 0.0 {
            v_perp_v2v1 = -v_perp_v2v1;
        }

        // Check if origin is inside or overlapping the initial portal
        if v1.dot(&v_perp_v2v1) >= 0.0 {
            // FUTURE: `v1.dot(&v_perp_v2v1) / v_perp_v2v1.norm()` is the approximate overlap distance
            return true;
        }

        // Support point in the direction of the portal
        let v3 = s.composite_support_mapping(v_perp_v2v1);

        // If the origin is outside the support plane, return false (no overlap)
        if v3.dot(&v_perp_v2v1) < 0.0 {
            return false;
        }

        // are we within an epsilon of the surface of the shape? If yes, done (overlap)
        let d = (v3 - v1) - (v3 - v1).project(&(v2 - v1));
        if d.norm_squared() < TOLERANCE * v3.norm_squared() {
            // FUTURE: `d.norm()` is the approximate overlap distance
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
            // FUTURE: `TOLERANCE` is the approximate overlap distance
            return true;
        }
    }
}

/// Detect collision between two convex 3D objects via Minkowski Portal Refinement.
#[inline(never)]
pub fn collide3d<R, A, B>(
    sa: &A,
    sb: &B,
    v_ij: &Cartesian<3>, // Probably ok to take ownership?
    q_ij: &R,
) -> bool
where
    A: SupportMapping<Cartesian<3>>,
    B: SupportMapping<Cartesian<3>>,
    R: Copy,
    RotationMatrix<3>: From<R>,
{
    const TOLERANCE: f64 = 2e-12;

    if v_ij.into_iter().all(|x| x.abs() < TOLERANCE) {
        // Interior point is at the origin => shapes overlap
        return true;
    }

    let s = MinkowskiDifference::new(sa, sb, v_ij, *q_ij);

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

    // Cross product is zero if v0,v1 collinear with origin, but we have already
    // determined origin is within v1 support plane. If origin is on a line between
    // v1 and v0, particles overlap.
    if n.into_iter().all(|x| x.abs() < TOLERANCE) {
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
    let mut count = 0_usize;
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

        // Are we within an epsilon of the surface of the shape? If yes, done, one way or another.
        n = (v2 - v1).cross(&(v3 - v1));
        let mut d = (v4 - v1).dot(&n);

        // Scale the tolerance with the size of the shapes.
        let tolerance = TOLERANCE * n.norm();

        // First, check if v4 is on plane (v2, v1, v3)
        if d.abs() < tolerance {
            // No more refinement possible, but not intersection detected
            return false;
        }
        // Second, check if origin is on plane (v2, v1, v3) and has been missed by other checks
        d = v1.dot(&n);
        if d.abs() < tolerance {
            return true;
        }

        // Choose a new portal. Two of its edges will be from the planes (v4,v0,v1),
        // (v4,v0,v2), (v4,v0,v3). Find which two have the origin on the same side.

        /* Test origin against the three planes that separate the new portal candidates
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

    use crate::shape::{Circle, Hypercuboid, Hypersphere};
    use hoomd_utility::valid::PositiveReal;
    use hoomd_vector::{Angle, Rotation, Versor};

    #[rstest(
        v => [[0.1, 0.1], [999.9, 0.0], [0.0, 5.123_f64.next_down()], [0.0, 5.123_000_001]],
        radius => [0.001, 1.0, 4.123, 99.05],
        o_ij => [
            Angle::default(),
            Angle::from(std::f64::consts::PI / 3.0),
            Angle::from(1.234)
        ],
    )]
    fn test_discs_collide(v: [f64; 2], radius: f64, o_ij: Angle) {
        let (s0, s1) = (
            Hypersphere {
                radius: 1.0.try_into().expect("test value is a positive real"),
            },
            Circle {
                radius: radius.try_into().expect("test value is a positive real"),
            },
        );

        let overlaps = collide2d(&s0, &s1, &v.into(), &o_ij);

        assert_eq!(overlaps, s0.intersects_at(&s1, &Cartesian::from(v), &o_ij));
    }
    #[rstest(
        v => [[0.1, 0.1, 0.1], [999.9, 0.0, -10.9], [0.0, 5.123, 0.0], [0.0, 0.0, 5.123_000_001]],
        radius => [0.001, 1.0, 4.123, 99.05],
        o_ij => [
            Versor::default(),
            Versor::from_axis_angle(
                [1.0, 0.0, 0.0].try_into().unwrap(), std::f64::consts::FRAC_PI_2
            ),
            Versor::from_axis_angle([0.0, 1.0, 0.0].try_into().unwrap(), 0.1234)
        ]
    )]
    fn test_spheres_collide(v: [f64; 3], radius: f64, o_ij: Versor) {
        let (s0, s1) = (
            Hypersphere {
                radius: 1.0.try_into().expect("test value is a positive real"),
            },
            Hypersphere::<3> {
                radius: radius.try_into().expect("test value is a positive real"),
            },
        );
        let overlaps = collide3d(&s0, &s1, &v.into(), &o_ij);

        assert_eq!(
            overlaps,
            s0.intersects_at(&s1, &Cartesian::from(v), &o_ij),
            "Xenocollide result did not match standard implementation!"
        );
    }

    #[rstest(
        v => [[0.1, 0.1], [999.9, 0.0], [0.0, 5.123], [0.0, 5.123_000_000_000_001]],
        rect => [[1.0.try_into().expect("test value is a positive real"), 1.0.try_into().expect("test value is a positive real")], [999.0.try_into().expect("test value is a positive real"), 0.1.try_into().expect("test value is a positive real")], [1.0.try_into().expect("test value is a positive real"), (2.0*4.623).try_into().expect("test value is a positive real")]]
    )]
    fn test_aabrs_collide(v: [f64; 2], rect: [PositiveReal; 2]) {
        let c0 = Hypercuboid { edge_lengths: rect };
        let c1 = Hypercuboid {
            edge_lengths: [1.0.try_into().expect("test value is a positive real"); 2],
        };
        let theta = Angle::from(0.0);

        let overlaps = collide2d(&c0, &c1, &v.into(), &theta);
        assert_eq!(overlaps, c0.intersects_aligned(&c1, &v.into()));
    }
    #[rstest(
        v => [[0.1, 2.1, 0.1], [999.9, 0.0, 0.05], [0.0, 5.123, 0.0], [0.0, 5.123_000_000_001, 0.0]],
        aabb => [[1.0.try_into().expect("test value is a positive real"), 1.0.try_into().expect("test value is a positive real"), 1.0.try_into().expect("test value is a positive real")], [999.0.try_into().expect("test value is a positive real"), 0.1.try_into().expect("test value is a positive real"), 0.5.try_into().expect("test value is a positive real")], [1.0.try_into().expect("test value is a positive real"), (2.0*4.623).try_into().expect("test value is a positive real"), 5.0.try_into().expect("test value is a positive real")]]

    )]
    fn test_aabbs_collide(v: [f64; 3], aabb: [PositiveReal; 3]) {
        let c0 = Hypercuboid { edge_lengths: aabb };
        let c1 = Hypercuboid {
            edge_lengths: [1.0.try_into().expect("test value is a positive real"); 3],
        };
        let theta = Versor::identity();

        let overlaps = collide3d(&c0, &c1, &v.into(), &theta);
        assert_eq!(overlaps, c0.intersects_aligned(&c1, &v.into()));
    }
}

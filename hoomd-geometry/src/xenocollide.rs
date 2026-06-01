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

/// Maximum allowed iterations for Xenocollide portal refinement.
const XENOCOLLIDE_MAX_ITER: usize = 1024;

/// Result of portal discovery.
pub(crate) enum Discovery<const N: usize> {
    /// Collision result already determined during discovery.
    Known(bool),
    /// Portal discovered — proceed to refinement.
    Found([Cartesian<N>; N]),
}

/// Dimension-specific operations for Minkowski Portal Refinement.
///
/// This trait encapsulates the operations that differ between MPR in various dimensions
/// - Tolerance for convergence check
/// - Portal discovery (can be done directly in 2D, requires search in 3D and up)
/// - Portal vertex replacement logic
pub(crate) trait MinkowskiPortalRefinement<const N: usize> {
    /// Dimension-specific convergence tolerance.
    const TOLERANCE: f64;

    /// Compute the outward-facing normal to the portal, oriented away from `interior`.
    fn outward_normal(portal: &[Cartesian<N>; N], interior: &Cartesian<N>) -> Cartesian<N>;

    /// Discover the initial portal for MPR refinement.
    fn discover_portal<A, B>(s: &MinkowskiDifference<N, A, B>, v0: &Cartesian<N>) -> Discovery<N>
    where
        A: SupportMapping<Cartesian<N>>,
        B: SupportMapping<Cartesian<N>>;

    /// Check convergence tolerance.
    ///
    /// Returns `Some(bool)` if the tolerance check determines the result, `None` to continue.
    fn tolerance_check(
        portal: &[Cartesian<N>; N],
        v_new: &Cartesian<N>,
        normal: &Cartesian<N>,
    ) -> Option<bool>;

    /// Choose which portal vertex to replace with the new support point.
    fn replace_vertex(interior: &Cartesian<N>, portal: &mut [Cartesian<N>; N], v_new: Cartesian<N>);
}

impl MinkowskiPortalRefinement<2> for Cartesian<2> {
    const TOLERANCE: f64 = 1e-16;

    #[inline]
    fn outward_normal(portal: &[Cartesian<2>; 2], interior: &Cartesian<2>) -> Cartesian<2> {
        let mut n = (portal[1] - portal[0]).perpendicular();
        if (portal[0] - *interior).dot(&n) < 0.0 {
            n = -n;
        }
        n
    }

    #[inline]
    fn discover_portal<A: SupportMapping<Cartesian<2>>, B: SupportMapping<Cartesian<2>>>(
        s: &MinkowskiDifference<2, A, B>,
        v0: &Cartesian<2>,
    ) -> Discovery<2> {
        // Find the support point in the direction of the origin ray
        let v1 = s.composite_support_mapping(-*v0);

        // v_perp is on the same side as the origin if v1.dot(v_perp) < 0
        let mut v_perp_v1v0 = (v1 - *v0).perpendicular();
        if v1.dot(&v_perp_v1v0) > 0.0 {
            v_perp_v1v0 = -v_perp_v1v0;
        }

        // Support point perpendicular to plane containing the origin, v0, and v1
        let v2 = s.composite_support_mapping(v_perp_v1v0);

        // NOTE: this assumes the origin is within the shape. This assumption matches
        // HOOMD-Blue, but is important to note regardless.

        Discovery::Found([v1, v2])
    }

    #[inline]
    fn tolerance_check(
        portal: &[Cartesian<2>; 2],
        v_new: &Cartesian<2>,
        _normal: &Cartesian<2>,
    ) -> Option<bool> {
        // In 2D, we either find a valid vertex or require further search to be sure
        let d = (*v_new - portal[0]) - (*v_new - portal[0]).project(&(portal[1] - portal[0]));
        if d.norm_squared() < Self::TOLERANCE * v_new.norm_squared() {
            return Some(true);
        }
        None
    }

    #[inline]
    fn replace_vertex(
        interior: &Cartesian<2>,
        portal: &mut [Cartesian<2>; 2],
        v_new: Cartesian<2>,
    ) {
        let mut v_perp = (v_new - *interior).perpendicular();
        // Orient toward portal[0]
        if (portal[0] - v_new).dot(&v_perp) < 0.0 {
            v_perp = -v_perp;
        }
        if v_new.dot(&v_perp) < 0.0 {
            // Origin is on the portal[0] side — replace portal[1]
            portal[1] = v_new;
        } else {
            // Origin is on the portal[1] side — replace portal[0]
            portal[0] = v_new;
        }
    }
}

impl MinkowskiPortalRefinement<3> for Cartesian<3> {
    const TOLERANCE: f64 = 2e-12;

    #[inline]
    fn outward_normal(portal: &[Cartesian<3>; 3], interior: &Cartesian<3>) -> Cartesian<3> {
        let e1 = portal[1] - portal[0];
        let e2 = portal[2] - portal[0];
        let mut n = e1.cross(&e2);
        if (portal[0] - *interior).dot(&n) < 0.0 {
            n = -n;
        }
        n
    }

    /// Discover a 2-simplex portal such that the origin passes through the portal face.
    #[inline]
    fn discover_portal<A: SupportMapping<Cartesian<3>>, B: SupportMapping<Cartesian<3>>>(
        s: &MinkowskiDifference<3, A, B>,
        v0: &Cartesian<3>,
    ) -> Discovery<3> {
        // Interior point at origin implies overlap
        if v0.into_iter().all(|x| x.abs() < Self::TOLERANCE) {
            return Discovery::Known(true);
        }
        // Support point in the direction of the origin ray
        let mut v1 = s.composite_support_mapping(-*v0);

        // Equivalent to v1 . (v1-v0) <= 0 by convexity
        if v1.dot(v0) > 0.0 {
            return Discovery::Known(false);
        }

        // Direction perpendicular to v0, v1 plane
        let n = v1.cross(v0);

        // Cross product is zero if v0,v1 collinear with origin, but we have already
        // determined the origin is within the v1 support plane.
        // If the origin is on a line between v1 and v0, particles overlap.
        if n.into_iter().all(|x| x.abs() < Self::TOLERANCE) {
            return Discovery::Known(true);
        }

        // Support point perpendicular to plane containing the origin, v0, and v1
        let mut v2 = s.composite_support_mapping(n);

        if v2.dot(&n) < 0.0 {
            return Discovery::Known(false);
        }

        // Support point perpendicular to plane containing interior point and first 2 supports
        let mut n = (v1 - *v0).cross(&(v2 - *v0));
        // Maintain known handedness of the portal
        if n.dot(v0) > 0.0 {
            (v1, v2) = (v2, v1);
            n = -n;
        }

        // while origin_ray_does_not_intersect_candidate()
        let mut count = 0_usize;
        let v3 = loop {
            count += 1;

            if count >= XENOCOLLIDE_MAX_ITER {
                return Discovery::Known(true);
            }

            let v3 = s.composite_support_mapping(n);
            if v3.dot(&n) <= 0.0 {
                return Discovery::Known(false);
            }

            // If origin lies on the opposite side of the plane from our third support
            // point, use the outer facing plane normal.
            // Check the v3, v0, v1 plane for validity
            if v1.cross(&v3).dot(v0) < 0.0 {
                v2 = v3; // Preserve handedness
                n = (v1 - *v0).cross(&(v2 - *v0));
                continue;
            }
            if v3.cross(&v2).dot(v0) < 0.0 {
                v1 = v3; // Preserve handedness
                n = (v1 - *v0).cross(&(v2 - *v0));
                continue;
            }
            break v3;
        };

        Discovery::Found([v1, v2, v3])
    }

    #[inline]
    fn tolerance_check(
        portal: &[Cartesian<3>; 3],
        v_new: &Cartesian<3>,
        normal: &Cartesian<3>,
    ) -> Option<bool> {
        let tolerance = Self::TOLERANCE * normal.norm(); // Handle non-unit shapes

        // Check if v_new is on the portal plane: if so, no more refinement is possible
        let d = (*v_new - portal[0]).dot(normal);
        if d.abs() < tolerance {
            return Some(false);
        }
        // Check if origin is on the portal plane: if so, intersection detected
        let d = portal[0].dot(normal);
        if d.abs() < tolerance {
            return Some(true);
        }
        None
    }

    #[inline]
    fn replace_vertex(
        interior: &Cartesian<3>,
        portal: &mut [Cartesian<3>; 3],
        v_new: Cartesian<3>,
    ) {
        let [v1, v2, v3] = *portal;
        // Test origin against the three planes that separate the new portal candidates
        // using the triple product identities as an optimization:
        //   (v1 % v4) * v0 == v1 * (v4 % v0) > 0 if origin inside (v1, v4, v0)
        //   (v2 % v4) * v0 == v2 * (v4 % v0) > 0 if origin inside (v2, v4, v0)
        //   (v3 % v4) * v0 == v3 * (v4 % v0) > 0 if origin inside (v3, v4, v0)
        let v_perp = v_new.cross(interior);

        #[expect(
            clippy::match_same_arms,
            reason = "Clearly illustrate translation from c."
        )]
        match (
            v_perp.dot(&v1) > 0.0,
            v_perp.dot(&v2) > 0.0,
            v_perp.dot(&v3) > 0.0,
        ) {
            (true, true, _) => portal[0] = v_new, // Inside  v1 && inside  v2 => eliminate v1
            (true, false, _) => portal[2] = v_new, // Inside  v1 && OUTside v2 => eliminate v3
            (false, _, true) => portal[1] = v_new, // OUTside v1 && inside  v3 => eliminate v2
            (false, _, false) => portal[0] = v_new, // OUTside v1 && OUTside v3 => eliminate v1
        }
    }
}

/// Stateful type that efficiently computes repeated Minkowski differences.
pub(crate) struct MinkowskiDifference<
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

/// Detect collision between two convex N-dimensional objects via Minkowski Portal Refinement.
///
/// This is the generic implementation underlying [`collide2d`] and [`collide3d`].
/// Dimension-specific operations (portal discovery, normal computation, tolerance
/// checks, vertex replacement) are resolved at compile time via the [`Mpr`] trait.
#[inline]
pub(crate) fn collide<const N: usize, R, A, B>(
    sa: &A,
    sb: &B,
    v_ij: &Cartesian<N>,
    q_ij: &R,
) -> bool
where
    A: SupportMapping<Cartesian<N>>,
    B: SupportMapping<Cartesian<N>>,
    R: Copy,
    RotationMatrix<N>: From<R>,
    Cartesian<N>: MinkowskiPortalRefinement<N>,
{
    let s = MinkowskiDifference::new(sa, sb, v_ij, *q_ij);
    let v0 = *v_ij;

    // Phase 1: Portal discovery
    let mut portal = match Cartesian::<N>::discover_portal(&s, &v0) {
        Discovery::Found(p) => p,
        Discovery::Known(r) => return r,
    };

    // Phase 2: Portal refinement
    // The loop is the same in general dimension, but the outward facing normal function
    // depends on the (n-1)-ary cross product (perp in 2d, cross in 3d)
    // See https://ncatlab.org/nlab/show/cross+product#counary for further details on
    // this operation
    let mut count = 0_usize;
    loop {
        count += 1;

        let normal = Cartesian::<N>::outward_normal(&portal, &v0);

        // Hit test: origin is enclosed by the portal
        if portal[0].dot(&normal) >= 0.0 {
            return true;
        }

        // Support query in the direction of the portal normal
        let v_new = s.composite_support_mapping(normal);

        // Miss test: origin is outside the support plane
        if v_new.dot(&normal) < 0.0 {
            return false;
        }

        // Tolerance check (dimension-specific)
        if let Some(result) = Cartesian::<N>::tolerance_check(&portal, &v_new, &normal) {
            return result;
        }

        // Face test and vertex replacement (dimension-specific)
        Cartesian::<N>::replace_vertex(&v0, &mut portal, v_new);

        if count >= XENOCOLLIDE_MAX_ITER {
            return true;
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
    collide::<2, R, A, B>(sa, sb, v_ij, q_ij)
}

/// Detect collision between two convex 3D objects via Minkowski Portal Refinement.
#[inline(never)]
pub fn collide3d<R, A, B>(sa: &A, sb: &B, v_ij: &Cartesian<3>, q_ij: &R) -> bool
where
    A: SupportMapping<Cartesian<3>>,
    B: SupportMapping<Cartesian<3>>,
    R: Copy,
    RotationMatrix<3>: From<R>,
{
    collide::<3, R, A, B>(sa, sb, v_ij, q_ij)
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

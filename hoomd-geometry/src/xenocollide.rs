// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implementations of the Xenocollide collision detection algorithm.
//!
//! > [!IMPORTANT]
//! > This implementation of Xenocollide *requires* that geometries contain the origin
//! > in their local frame. If this assumption is violated, the algorithm can produce
//! > incorrect results!
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
enum Discovery<const N: usize> {
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
trait MinkowskiPortalRefinement<const N: usize> {
    /// Dimension-specific convergence tolerance.
    const TOLERANCE: f64;

    /// Compute the outward-facing normal to the portal (facing toward the surface of the minkowski difference)
    fn outward_normal(portal: &[Cartesian<N>; N], interior: &Cartesian<N>) -> Cartesian<N>;

    /// Discover the initial portal for MPR refinement.
    ///
    /// The initial portal will be an (N-1)-simplex, through which the ray `-v0` must
    /// pass on its way to the origin.
    fn discover_portal<A, B>(s: &MinkowskiDifference<N, A, B>, v0: &Cartesian<N>) -> Discovery<N>
    where
        A: SupportMapping<Cartesian<N>>,
        B: SupportMapping<Cartesian<N>>;

    /// Check whether the portal has reached the surface of the Minkowski
    /// difference within numerical precision (in which case we can determine overlap).
    ///
    /// Returns `Some(result)` if convergence is detected, or `None` if
    /// refinement can continue.
    fn tolerance_check(
        portal: &[Cartesian<N>; N],
        v_new: &Cartesian<N>,
        normal: &Cartesian<N>,
    ) -> Option<bool>;

    /// Narrow the portal by replacing one vertex with a new support point.
    ///
    /// The portal vertices and `v_new` form an N-simplex whose interior face
    /// is the current portal. The origin ray enters through this face and must
    /// exit through one of the outer faces. This method identifies which outer
    /// face the ray exits through and replaces the portal vertex opposite that
    /// face with `v_new`, ensuring that the origin ray always passes through the portal
    ///
    /// Returns `true` when the origin is known to lie inside the N-simplex
    /// {``v_new``, *portal}. This simplex is guaranteed to be in the Minkowski
    /// difference, so this guarantees we found an overlap.
    fn narrow_portal(
        interior: &Cartesian<N>,
        portal: &mut [Cartesian<N>; N],
        v_new: Cartesian<N>,
    ) -> bool;
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
    fn narrow_portal(
        interior: &Cartesian<2>,
        portal: &mut [Cartesian<2>; 2],
        v_new: Cartesian<2>,
    ) -> bool {
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
        false
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
    fn narrow_portal(
        interior: &Cartesian<3>,
        portal: &mut [Cartesian<3>; 3],
        v_new: Cartesian<3>,
    ) -> bool {
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
        false
    }
}

/// Find a vector perpendicular to the plane spanned by `a` and `b` in R^4.
///
/// `counary_cross` requires 3 inputs but during early portal discovery we only have
/// 2 vectors. This function uses a standard basis vector as the 3rd input — at least
/// one of the 4 basis vectors is guaranteed to be non-coplanar with any 2 given vectors.
#[inline]
fn perp_to_2d_subspace(a: Cartesian<4>, b: Cartesian<4>, tol: f64) -> Option<Cartesian<4>> {
    for i in 0..4 {
        let mut e = [0.0f64; 4];
        e[i] = 1.0;
        let n = Cartesian::<4>::counary_cross(&[a, b, e.into()]);
        if n.into_iter().any(|x| x.abs() > tol) {
            return Some(n);
        }
    }
    None
}

impl MinkowskiPortalRefinement<4> for Cartesian<4> {
    const TOLERANCE: f64 = 2e-12;

    #[inline]
    fn outward_normal(portal: &[Cartesian<4>; 4], interior: &Cartesian<4>) -> Cartesian<4> {
        let e1 = portal[1] - portal[0];
        let e2 = portal[2] - portal[0];
        let e3 = portal[3] - portal[0];
        let mut n = Self::counary_cross(&[e1, e2, e3]);
        if (portal[0] - *interior).dot(&n) < 0.0 {
            n = -n;
        }
        n
    }

    fn discover_portal<A: SupportMapping<Cartesian<4>>, B: SupportMapping<Cartesian<4>>>(
        s: &MinkowskiDifference<4, A, B>,
        v0: &Cartesian<4>,
    ) -> Discovery<4> {
        // Interior point at origin implies overlap
        if v0.into_iter().all(|x| x.abs() < Self::TOLERANCE) {
            return Discovery::Known(true);
        }

        // v1: first support point
        let mut v1 = s.composite_support_mapping(-*v0);
        if v1.dot(v0) > 0.0 {
            return Discovery::Known(false);
        }

        // v2: support in direction perpendicular to the v0-v1 plane
        let Some(n) = perp_to_2d_subspace(*v0, v1, Self::TOLERANCE) else {
            return Discovery::Known(true); // v0, v1, origin collinear
        };

        let mut v2 = s.composite_support_mapping(n);
        if v2.dot(&n) < 0.0 {
            return Discovery::Known(false);
        }

        // v3: support in direction perpendicular to the v0-v1-v2 subspace
        let Some(mut n) = perp_to_2d_subspace(v1 - *v0, v2 - *v0, Self::TOLERANCE) else {
            return Discovery::Known(true);
        };
        // Orient normal away from v0 (maintain handedness)
        if n.dot(v0) > 0.0 {
            (v1, v2) = (v2, v1);
            n = -n;
        }
        let v3 = s.composite_support_mapping(n);
        if v3.dot(&n) <= 0.0 {
            return Discovery::Known(false);
        }

        // Recompute portal normal from 3 edge vectors
        let mut n = Self::counary_cross(&[v1 - *v0, v2 - *v0, v3 - *v0]);
        // Coplanar v1, v2, v3 with v0 produce a zero normal — origin lies on a flat
        // boundary region which we treat as overlap
        if n.into_iter().all(|x| x.abs() < Self::TOLERANCE) {
            return Discovery::Known(true);
        }
        if n.dot(v0) > 0.0 {
            n = -n;
        }

        // v4-validation loop
        // The 3 face checks use the 4D quadruple product:
        //   counary_cross(&[v_a, v_b, v4]).dot(&v0) = det([v_a, v_b, v4, v0])
        // Negative means origin is on the wrong side of that face.
        let mut v3 = v3;
        let mut count = 0_usize;
        let v4 = loop {
            count += 1;
            if count >= XENOCOLLIDE_MAX_ITER {
                return Discovery::Known(true);
            }

            let v4 = s.composite_support_mapping(n);
            if v4.dot(&n) <= 0.0 {
                return Discovery::Known(false);
            }

            // Each face test is a 4×4 determinant. For axis-aligned shapes, the
            // support vertices are frequently *coplanar* with `v0`, so the determinant
            // lands within a few ULP of zero. `< 0.0` then fires on floating-point
            // noise, replacing a vertex and bouncing the portal between states until
            // `MAX_ITER` bails out with a false-positive overlap.
            //
            // A determinant within `det_band` of zero means the origin lies ON the
            // face (within precision), which we treat as enclosed. Only a clearly-
            // negative determinant triggers a vertex replacement.
            let det_tol = Self::TOLERANCE * v1.norm() * v2.norm() * v3.norm() * v0.norm();

            // Face (v1, v2, v4) — opposite v3
            if Self::counary_cross(&[v1, v2, v4]).dot(v0) < -det_tol {
                v3 = v4;
                n = Self::counary_cross(&[v1 - *v0, v2 - *v0, v3 - *v0]);
                if n.dot(v0) > 0.0 {
                    n = -n;
                }
                continue;
            }
            // Face (v2, v3, v4) — opposite v1
            if Self::counary_cross(&[v2, v3, v4]).dot(v0) < -det_tol {
                v1 = v4;
                n = Self::counary_cross(&[v1 - *v0, v2 - *v0, v3 - *v0]);
                if n.dot(v0) > 0.0 {
                    n = -n;
                }
                continue;
            }
            // Face (v1, v4, v3) — opposite v2
            if Self::counary_cross(&[v1, v4, v3]).dot(v0) < -det_tol {
                v2 = v4;
                n = Self::counary_cross(&[v1 - *v0, v2 - *v0, v3 - *v0]);
                if n.dot(v0) > 0.0 {
                    n = -n;
                }
                continue;
            }
            break v4;
        };

        Discovery::Found([v1, v2, v3, v4])
    }

    #[inline]
    fn tolerance_check(
        portal: &[Cartesian<4>; 4],
        v_new: &Cartesian<4>,
        normal: &Cartesian<4>,
    ) -> Option<bool> {
        let tolerance = Self::TOLERANCE * normal.norm();

        let d = (*v_new - portal[0]).dot(normal);
        if d.abs() < tolerance {
            return Some(false);
        }
        let d = portal[0].dot(normal);
        if d.abs() < tolerance {
            return Some(true);
        }
        None
    }
    #[inline]
    fn narrow_portal(
        interior: &Cartesian<4>,
        portal: &mut [Cartesian<4>; 4],
        v_new: Cartesian<4>,
    ) -> bool {
        // The 4-simplex [portal[0..4] , v_new] has 5 tetrahedral facets.
        // The entry face is the current portal (portal[0..4]).
        // The 4 exit facets each contain v_new and all portal vertices except portal[i]
        //
        // The origin ray (from `v0` through the origin and beyond) enters through the
        // portal and must exit through one of the four remaining facets, each of which
        // contains `v_new` and all but one portal vertex. Note that each point on the
        // origin ray is expressable as `λ*v_0 + 0`. λ=1 is at exactly the deep point
        // (v0), at λ=0 the ray touches the origin, and for all 1>=λ>0, the ray is
        // marching outward toward the origin (and for λ<0 we have moved past the
        // origin). Once we pass the origin, we can identify which facet we leave
        // through by searching the ray in barycentric coordinates: the weight of a
        // selected vertex is positive inside the simplex, and then *0 exactly when the
        // ray exits the facet opposite that vertex*. If we can identify which vertex's
        // weight reaches 0 first, we've narrowed the portal successfully.
        //
        // Locating where a ray meets a facet's hyperplane can be done in the standard
        // way: with a facet normal n_i oriented toward portal[i], the crossing sits at
        // `λ_i = (v_new . n_i) / (v_0 . n_i)`. A candidate hyperplane is reachable only
        // when the denominator is greater than zero.
        // (see https://stackoverflow.com/a/23976134/21897583 for an example)
        //
        // The key thing: since marching outward means decreasing λ, the first
        // hyperplane we encounter after entering the simplex is the reachable candidate
        // with the *largest* λ_i. All other weights are positive at this point, meaning
        // the crossing lies *inside* the exit facet (not just its hyperplane). This
        // facet is the exit, and will serve as the new portal in the next iteration.
        //
        // The sign of the largest λ_i tells us where the exit lies: positive means the
        // ray exits before reaching the origin (refinement continues), while zero or
        // negative means the origin lies on or inside the simplex and therefore inside
        // the Minkowski difference. Installing that facet is still correct (as the ray
        // crosses it), and the next iteration's hit test reports the overlap.

        let mut exit_face: Option<(usize, f64)> = None;
        for i in 0..4 {
            let edges = std::array::from_fn(|k| portal[(i + k + 1) % 4] - v_new);
            let n = Self::counary_cross(&edges);
            // Skip degenerate facets. |n|**2 has units of length^6, so we match that
            let scale = portal[i].norm_squared().max(v_new.norm_squared()).sqrt();
            if n.norm_squared() < Self::TOLERANCE.powi(2) * scale.powi(6) {
                continue;
            }
            // Orient the normal toward portal[i], the vertex this candidate would replace
            let n = if (portal[i] - v_new).dot(&n) < 0.0 {
                -n
            } else {
                n
            };
            let denom = interior.dot(&n);
            if denom <= 0.0 {
                continue; // The origin ray can't reach this facet -> can't be the exit
            }
            let lambda = v_new.dot(&n) / denom;
            if exit_face.is_none_or(|(_, best_lambda)| lambda > best_lambda) {
                exit_face = Some((i, lambda));
            }
        }
        match exit_face {
            // The first exit lies at or beyond the origin, so the origin is on (or
            // inside) the portal+v_new simplex. That simplex is contained in Minkowski
            // Minkowski difference, so we can immediately exit -> overlap.
            Some((_, lambda)) if lambda <= 0.0 => true,
            Some((i, _)) => {
                portal[i] = v_new;
                false
            }
            // No facet was reachable: every candidate was degenerate or numerically
            // rejected. (An origin strictly inside the simplex would instead yield a
            // reachable facet with lambda <= 0, above). Fail safe as an overlap.
            None => true,
        }
    }
}

/// Stateful type that efficiently computes repeated Minkowski differences.
pub struct MinkowskiDifference<
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
/// checks, vertex replacement) are resolved at compile time via the
/// [`MinkowskiPortalRefinement`] trait.
#[inline]
fn collide<const N: usize, R, A, B>(sa: &A, sb: &B, v_ij: &Cartesian<N>, q_ij: &R) -> bool
where
    A: SupportMapping<Cartesian<N>>,
    B: SupportMapping<Cartesian<N>>,
    R: Copy,
    RotationMatrix<N>: From<R>,
    Cartesian<N>: MinkowskiPortalRefinement<N>,
{
    let s = MinkowskiDifference::new(sa, sb, v_ij, *q_ij);
    let v0 = *v_ij;

    // Portal discovery
    let mut portal = match Cartesian::<N>::discover_portal(&s, &v0) {
        Discovery::Found(p) => p,
        Discovery::Known(r) => return r,
    };

    // Portal refinement
    // The loop is the same in general dimension, but the outward facing normal function
    // depends on the (N-1)-ary cross product (perp in 2d, cross in 3d)
    // See https://ncatlab.org/nlab/show/cross+product#counary for further details on
    // this operation
    let mut count = 0_usize;
    loop {
        count += 1;

        let normal = Cartesian::<N>::outward_normal(&portal, &v0);

        // If the portal's (N-1)-simplex is degenerate, the normal is near zero and the
        // rest of the algorithm does not make sense (so we fail safe -> overlap)
        if normal.norm_squared()
            < Cartesian::<N>::TOLERANCE * Cartesian::<N>::TOLERANCE * v0.norm_squared()
        {
            return true;
        }

        // Hit test: is the origin enclosed by the portal?
        if portal[0].dot(&normal) >= 0.0 {
            return true;
        }

        // Support query in the direction of the portal normal
        let v_new = s.composite_support_mapping(normal);

        // Miss test: is the origin outside the support plane?
        if v_new.dot(&normal) < 0.0 {
            return false;
        }

        // Can we numerically distinguish the portal face and the support plane?
        if let Some(result) = Cartesian::<N>::tolerance_check(&portal, &v_new, &normal) {
            return result;
        }

        // Face test and vertex replacement (dimension-specific). A `true` return
        // confirms that the origin lie inside the portal (and therefore shapes overlap)
        if Cartesian::<N>::narrow_portal(&v0, &mut portal, v_new) {
            return true;
        }

        if count >= XENOCOLLIDE_MAX_ITER {
            return true;
        }
    }
}

/// Detect collision between two convex 2D objects via Minkowski Portal Refinement.
#[inline(never)]
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

/// Detect collision between two convex 4D objects via Minkowski Portal Refinement.
#[inline(never)]
pub fn collide4d<R, A, B>(sa: &A, sb: &B, v_ij: &Cartesian<4>, q_ij: &R) -> bool
where
    A: SupportMapping<Cartesian<4>>,
    B: SupportMapping<Cartesian<4>>,
    R: Copy,
    RotationMatrix<4>: From<R>,
{
    collide::<4, R, A, B>(sa, sb, v_ij, q_ij)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntersectsAt;
    use rstest::*;

    use crate::shape::{Circle, ConvexPolytope, Hypercuboid, Hypersphere};
    use hoomd_utility::valid::PositiveReal;
    use hoomd_vector::{Angle, Rotation, Versor};
    use rand::{RngExt, SeedableRng, rngs::StdRng};

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

    #[rstest(
        v => [
            [0.1, 0.1, 0.1, 0.1],
            [999.9, 0.0, 0.0, -10.9],
            [0.0, 5.123, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.123_000_001],
        ],
        radius => [0.001, 1.0, 4.123],
    )]
    fn test_4d_spheres_collide(v: [f64; 4], radius: f64) {
        let (s0, s1) = (
            Hypersphere {
                radius: 1.0.try_into().expect("test value is a positive real"),
            },
            Hypersphere::<4> {
                radius: radius.try_into().expect("test value is a positive real"),
            },
        );
        let q_ij = RotationMatrix::<4>::default();

        let overlaps = collide4d(&s0, &s1, &v.into(), &q_ij);

        assert_eq!(
            overlaps,
            s0.intersects_at(&s1, &Cartesian::from(v), &q_ij),
            "4D Xenocollide result did not match standard implementation!"
        );
    }

    #[rstest(
        v => [
            [0.1, 0.1, 0.1, 0.1],
            [999.9, 0.0, 0.0, 0.05],
            [0.0, 5.123, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.123_000_000_001],
        ],
        tesseract => [
            [1.0.try_into().expect("test value is a positive real"); 4],
            [999.0.try_into().expect("test value is a positive real"), 0.1.try_into().expect("test value is a positive real"), 0.5.try_into().expect("test value is a positive real"), 1.0.try_into().expect("test value is a positive real")],
        ],
    )]
    fn test_tesseracts_collide(v: [f64; 4], tesseract: [PositiveReal; 4]) {
        let c0 = Hypercuboid {
            edge_lengths: tesseract,
        };
        let c1 = Hypercuboid {
            edge_lengths: [1.0.try_into().expect("test value is a positive real"); 4],
        };
        let q_ij = RotationMatrix::<4>::default();

        let overlaps = collide4d(&c0, &c1, &v.into(), &q_ij);
        assert_eq!(overlaps, c0.intersects_aligned(&c1, &v.into()));
    }

    /// Sweep two unit hypercubes from overlapping to separated along a diagonal direction,
    /// verifying collide4d matches the analytical result at every step.
    #[rstest(
        direction => [
            [1.0_f64, 1.0, 1.0, 1.0],
            [2.0_f64, 1.0, 1.0, 1.0],
            [1.0_f64, 1.0, 1.0, 3.0],
            [1.0_f64, 2.0, 3.0, 1.0],
        ],
    )]
    fn test_4d_hypercuboid_diagonal_separation_sweep(direction: [f64; 4]) {
        let one: PositiveReal = 1.0.try_into().unwrap();
        let c0 = Hypercuboid {
            edge_lengths: [one; 4],
        };
        let c1 = Hypercuboid {
            edge_lengths: [one; 4],
        };

        // Half-edge-lengths: 0.5 each, sum = 1.0 each.
        // Critical t = min_i(1.0 / |dir_i|)
        let critical_t = (0..4)
            .map(|i| 1.0 / direction[i].abs())
            .reduce(f64::min)
            .unwrap();

        let t_start = critical_t - 0.001;
        let t_end = critical_t + 0.001;
        let steps = 10_000;
        let dt = (t_end - t_start) / f64::from(steps);

        let q_ij = RotationMatrix::<4>::default();

        for step in 0..=steps {
            let t = t_start + dt * f64::from(step);
            let d: Cartesian<4> = direction.map(|d_i| d_i * t).into();
            let expected = c0.intersects_aligned(&c1, &d);
            let result = collide4d(&c0, &c1, &d, &q_ij);
            assert_eq!(
                result, expected,
                "Mismatch at step {step}, t = {t:.12}, critical_t = {critical_t:.12}"
            );
        }
    }

    #[rstest(seed => [0_usize, 1, 2, 7, 42, 2024])]
    fn test_4d_hypercuboid_random_near_boundary(seed: usize) {
        let one: PositiveReal = 1.0.try_into().unwrap();
        let c0 = Hypercuboid {
            edge_lengths: [one; 4],
        };
        let c1 = Hypercuboid {
            edge_lengths: [one; 4],
        };
        let q_ij = RotationMatrix::<4>::default();

        let mut rng = StdRng::seed_from_u64(seed as u64);
        let shell = 1e-9_f64;
        let samples = 100_000_usize;
        for _ in 0..samples {
            // Uniform direction in [-1, 1]^4 (scaling is absorbed by `t0`).
            let dir: Cartesian<4> = rng.random();
            let max_abs = dir.into_iter().map(f64::abs).fold(0.0_f64, f64::max);
            if max_abs < f64::EPSILON {
                continue;
            }
            let t0 = 1.0 / max_abs;
            let t = t0 + (rng.random::<f64>() * 2.0 - 1.0) * shell;
            let d = dir * t;
            let expected = c0.intersects_aligned(&c1, &d);
            let result = collide4d(&c0, &c1, &d, &q_ij);
            assert_eq!(
                result, expected,
                "near-boundary mismatch: dir = {dir:?}, t = {t}, d = {d:?}"
            );
        }
    }

    /// Two unit hypercubes are separated by `d = [2t, t, t, t]`; the analytical boundary
    /// is `t = 0.5` (`|d_0| = 1`). Just past it (e.g. `t = 0.5000158`) the
    /// shapes do not overlap, but a coplanar portal previously drove MPR into a cycle
    /// that falesely reported overlap.
    #[test]
    fn test_4d_hypercuboid_diagonal_2111_near_boundary() {
        let one: PositiveReal = 1.0.try_into().unwrap();
        let c0 = Hypercuboid {
            edge_lengths: [one; 4],
        };
        let c1 = Hypercuboid {
            edge_lengths: [one; 4],
        };
        let q_ij = RotationMatrix::<4>::default();

        for t in [
            0.5_f64,
            0.5 + 5e-6,
            0.5 + 1.58e-5, // Exact failing case from previous code
            0.5 + 3e-5,
            0.5 + 1e-4,
            0.5 - 1e-4,
        ] {
            let d: Cartesian<4> = [2.0 * t, t, t, t].into();
            let expected = c0.intersects_aligned(&c1, &d);
            let result = collide4d(&c0, &c1, &d, &q_ij);
            assert_eq!(result, expected, "t = {t}, d = {d:?}");
        }
    }

    /// Sweep two tesseracts (vertex-based `ConvexPolytope::hypercube()`) along
    /// axis-aligned directions, crossing the Minkowski-difference boundary at a cubical
    /// facet *center*.
    #[rstest(
        direction => [
            [1.0_f64, 0.0, 0.0, 0.0],
            [0.0_f64, 1.0, 0.0, 0.0],
            [0.0_f64, 0.0, 1.0, 0.0],
            [0.0_f64, 0.0, 0.0, 1.0],
            [1.0_f64, 1.0, 0.0, 0.0],
            [0.0_f64, 1.0, 1.0, 0.0],
            [0.0_f64, 0.0, 1.0, 1.0],
            [2.0_f64, 1.0, 0.0, 0.0],
        ],
    )]
    fn test_4d_hypercube_polytope_axis_separation_sweep(direction: [f64; 4]) {
        let c0 = ConvexPolytope::<4, 16>::hypercube();
        let c1 = ConvexPolytope::<4, 16>::hypercube();

        // Half-edge 0.5 each; collide4d reports overlap (incl. touching) iff |d[i]| <= 1.0
        // for all i. Critical t = min_i(1.0 / |dir_i|); zero components contribute infinity.
        let critical_t = (0..4)
            .map(|i| 1.0 / direction[i].abs())
            .reduce(f64::min)
            .unwrap();

        let t_start = critical_t - 0.001;
        let t_end = critical_t + 0.001;
        let steps = 10_000;
        let dt = (t_end - t_start) / f64::from(steps);

        let q_ij = RotationMatrix::<4>::default();

        for step in 0..=steps {
            let t = t_start + dt * f64::from(step);
            let d: Cartesian<4> = direction.map(|d_i| d_i * t).into();
            let expected = (0..4).all(|i| d[i].abs() <= 1.0);
            let result = collide4d(&c0, &c1, &d, &q_ij);
            assert_eq!(
                result, expected,
                "Mismatch at step {step}, t = {t:.12}, critical_t = {critical_t:.12}"
            );
        }
    }

    /// Stress-test collide4d with two hyperspheres near their overlap boundary.
    ///
    /// Generates random displacement directions uniformly on S^3 using random
    /// unit quaternions ([`Versor`], Muller/Marsaglia Method 19) and random
    /// radii in a thin shell of width 1e-3 centered on the overlap radius.
    #[rstest(
        r0 => [1.0, 0.5, 3.7],
        r1 => [1.0, 2.0, 0.8],
        seed => [0, 1, 42],
    )]
    fn test_4d_hypersphere_shell_overlap(r0: f64, r1: f64, seed: u64) {
        let boundary = r0 + r1;
        let shell_half_width = 5e-10;
        let n_samples = 10_000;

        let s0 = Hypersphere::<4> {
            radius: r0.try_into().unwrap(),
        };
        let s1 = Hypersphere::<4> {
            radius: r1.try_into().unwrap(),
        };
        let q_ij = RotationMatrix::<4>::default();

        let mut rng = StdRng::seed_from_u64(seed);

        for i in 0..n_samples {
            // Random unit quaternion → uniform direction on S^3
            let q: Versor = rng.random();
            let q = q.get();
            let dir: Cartesian<4> = [q.scalar, q.vector[0], q.vector[1], q.vector[2]].into();

            let r = boundary - shell_half_width + rng.random::<f64>() * 2.0 * shell_half_width;
            let d = dir * r;

            let analytical = r < boundary;
            let result = collide4d(&s0, &s1, &d, &q_ij);
            assert_eq!(
                result, analytical,
                "Mismatch at sample {i}, |d| = {r:.12}, boundary = {boundary:.12}"
            );
        }
    }

    /// Construct a 4-simplex from 5 vertices drawn from c·{±1}⁴.
    ///
    /// 9 of the 10 edges have length 2c√2; the v₃–v₄ edge has length 4c.
    fn pentachoron(c: f64) -> ConvexPolytope<4, 8> {
        ConvexPolytope::<4, 8>::with_vertices([
            Cartesian::from([c, c, c, c]),
            Cartesian::from([c, -c, -c, c]),
            Cartesian::from([-c, c, -c, c]),
            Cartesian::from([-c, -c, c, c]),
            Cartesian::from([c, c, -c, -c]),
        ])
        .unwrap()
    }

    /// Sweep two simplices along vertex-to-vertex (edge) directions.
    ///
    /// For two identical simplices with vertices {`v_i`}, a sweep along the
    /// edge direction `d0 = v_a − v_b` has an exact analytical boundary:
    /// the t* at intersection equals the edge length `|v_a − v_b|` (9 edges have
    /// length 2c*sqrt(2), 1 edge has length 4c).
    #[rstest(
        edge => [
            [0.0_f64, 2.0, 2.0, 0.0],   // v0 - v1
            [2.0_f64, 0.0, 2.0, 0.0],   // v0 - v2
            [2.0_f64, 2.0, 0.0, 0.0],   // v0 - v3
            [0.0_f64, 0.0, 2.0, 2.0],   // v0 - v4
            [2.0_f64, -2.0, 0.0, 0.0],  // v1 - v2
            [2.0_f64, 0.0, -2.0, 0.0],  // v1 - v3
            [0.0_f64, -2.0, 0.0, 2.0],  // v1 - v4
            [0.0_f64, 2.0, -2.0, 0.0],  // v2 - v3
            [-2.0_f64, 0.0, 0.0, 2.0],  // v2 - v4
            [-2.0_f64, -2.0, 2.0, 2.0], // v3 - v4
        ],
        c => [1.0, 0.5, 2.0],
    )]
    fn test_4d_pentachoron_edge_sweep(edge: [f64; 4], c: f64) {
        let s0 = pentachoron(c);
        let s1 = pentachoron(c);

        let edge: Cartesian<4> = edge.map(|e| e * c).into();
        let edge_length = edge.norm();
        let direction = edge / edge_length;

        let t_start = edge_length - 1.0 / 3.0;
        let t_end = edge_length + 0.01;
        let steps = 10_000;
        let dt = (t_end - t_start) / f64::from(steps);

        let q_ij = RotationMatrix::<4>::default();

        for step in 0..=steps {
            let t = t_start + dt * f64::from(step);
            let d = direction * t;
            let analytical = t < edge_length;
            let result = collide4d(&s0, &s1, &d, &q_ij);
            assert_eq!(
                result, analytical,
                "Mismatch at step {step}, t = {t:.12}, edge_length = {edge_length:.12}"
            );
        }
    }
}

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! A tetrahedron in three dimensions. This struct should be viewed as a prototype for
more complex geometries in addition to its standalone functionality.
*/
use std::{array, fmt};

use hoomd_vector::{Cartesian, Cross, Rotate, RotationMatrix, Vector};
use itertools::Itertools;

use crate::{IntersectsAt, SupportFn};

/// The simplest three-dimensional geometry. This struct has a wide range of defined
/// functionality and is useful as a primitive in a variety of contexts.
#[derive(Clone, Copy, Debug)]
pub struct Simplex3 {
    /// Vertices of the simplex
    vertices: [Cartesian<3>; 4], // NOT public, to force orientation on construction
}

impl SupportFn<Cartesian<3>> for Simplex3 {
    #[inline]
    #[expect(
        clippy::expect_used,
        reason = "Support function of a vector containing NaN is undefined."
    )]
    fn support(&self, n: &Cartesian<3>) -> Cartesian<3> {
        let dots = self.vertices.map(|v| v.dot(n));
        self.vertices[dots
            .iter()
            // position_max_by consumes ~9% of Xenocollide runtime!
            .position_max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .expect("Took dot product with NaN!")]
    }
}

impl From<[Cartesian<3>; 4]> for Simplex3 {
    #[inline]
    fn from(vertices: [Cartesian<3>; 4]) -> Self {
        Simplex3 { vertices }.orient()
    }
}
impl From<[[f64; 3]; 4]> for Simplex3 {
    #[inline]
    fn from(arrs: [[f64; 3]; 4]) -> Self {
        Simplex3 {
            vertices: arrs.map(Cartesian::from),
        }
        .orient()
    }
}

impl Default for Simplex3 {
    /// A uniform tetrahedron, centered on the origin with edges bisecting half of
    /// the faces of a cube on their diagonals.
    #[inline]
    fn default() -> Self {
        // Oriented by construction.
        Simplex3 {
            vertices: [
                [1.0, 1.0, 1.0].into(),
                [1.0, -1.0, -1.0].into(),
                [-1.0, 1.0, -1.0].into(),
                [-1.0, -1.0, 1.0].into(),
            ],
        }
    }
}

impl fmt::Display for Simplex3 {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Simplex {{ [{}] }}",
            self.vertices
                .iter()
                .map(Cartesian::to_string)
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}
impl Simplex3 {
    /// Translate a simplex via rowwise addition of a Cartesian3
    #[inline]
    #[must_use]
    pub fn translate_by(&mut self, rhs: &Cartesian<3>) -> Self {
        Self {
            vertices: self.vertices.map(|v| v + *rhs),
        }
    }

    /// The center of mass of the tetrahedron.
    #[inline]
    #[must_use]
    pub fn centroid(&self) -> Cartesian<3> {
        array::from_fn::<_, 3, _>(|i| self.vertices.iter().fold(0.0, |acc, v| acc + v[i])).into()
    }

    /// Get the edges of the tetrahedron as edge endpoint coordinates. In vertex index
    /// form, this returns values in the order [(1, 0), (2, 0), (3, 0), (2, 1), (3, 2)]
    #[inline]
    #[must_use]
    pub fn get_edges(&self) -> [[Cartesian<3>; 2]; 5] {
        [
            [self.b(), self.a()],
            [self.c(), self.a()],
            [self.d(), self.a()],
            [self.c(), self.b()],
            [self.d(), self.b()],
        ]
    }

    /// Edge vectors, in the same order as `get_edges` and pointing left to right.
    #[inline]
    #[must_use]
    pub fn get_edge_vectors(&self) -> [Cartesian<3>; 5] {
        self.get_edges().map(|[l, r]| l - r)
    }

    #[inline]
    #[must_use]
    /// 0th vertex of the tetrahedron
    pub(crate) fn a(&self) -> Cartesian<3> {
        self.vertices[0]
    }
    #[inline]
    #[must_use]
    /// 1st vertex of the tetrahedron
    pub(crate) fn b(&self) -> Cartesian<3> {
        self.vertices[1]
    }
    #[inline]
    #[must_use]
    /// 2nd vertex of the tetrahedron
    pub(crate) fn c(&self) -> Cartesian<3> {
        self.vertices[2]
    }
    #[inline]
    #[must_use]
    /// 3rd vertex of the tetrahedron
    pub(crate) fn d(&self) -> Cartesian<3> {
        self.vertices[3]
    }
    /// Return the vertices of an oriented tetrahedron. Users should call ``orient_in_place``
    #[inline]
    pub(crate) fn orient(&self) -> Simplex3 {
        let dp = (self.d() - self.a()).dot(&((self.b() - self.a()).cross(&(self.c() - self.a()))));
        if dp < 0.0 {
            Simplex3 {
                vertices: self.vertices,
            }
        } else {
            Simplex3 {
                vertices: [self.a(), self.c(), self.b(), self.d()],
            }
        }
    }
}
/// Check if plane ``P_i`` defined by 4 coordinates and containing face ``i`` is a
/// separating plane (or conversely, if the face normal is a separating axis)
#[inline]
#[must_use]
fn check_face_on_p_is_separating(aff: &[f64; 4]) -> u8 {
    aff.iter().enumerate().fold(
        0u8,
        |acc, (i, &x)| {
            if x > 0.0 { acc | (1 << i) } else { acc }
        },
    )
}

/// Check whether plane ``P_j`` parallel to a face on q is a separating plane.
#[inline]
#[must_use]
fn check_face_on_q_is_separating(aff: &[f64; 4]) -> bool {
    aff.iter().all(|&x| x > 0.0)
}

/// Check that no point in our projected vertices lies in the (-, -) quadrant
/// If a point satisfying that condition is found, there exists a separating line.
#[expect(
    clippy::too_many_arguments,
    reason = "Internal function not exposed to users."
)]
#[inline]
#[must_use]
fn edge_test(
    ma: u8,
    mb: u8,
    a: u8,
    b: u8,
    i: usize,
    j: usize,
    ea: &[f64; 4],
    eb: &[f64; 4],
) -> bool {
    let cp = (ea[j] * eb[i]) - (ea[i] * eb[j]);
    // NOTE: original paper used cp > | < 0.0, but we want to detect exact edge-edge overlaps.
    ((ma & a) != 0 && (mb & b) != 0 && (cp >= 0.0)) || (ma & b) != 0 && (mb & a) != 0 && (cp <= 0.0)
}

/// Separating edge cases used in ``check_edge_is_separating``
const _SEPARATING_EDGE_CASES: [(u8, u8, usize, usize); 6] = [
    (1, 2, 0, 1),
    (1, 4, 0, 2),
    (1, 8, 0, 3),
    (2, 4, 1, 2),
    (2, 8, 1, 3),
    (4, 8, 2, 3),
];

/// Check if there exists a separating plane containing the edge e shared by faces
/// f0 and f1 of the tetrahedron.
#[inline]
#[must_use]
fn check_edge_is_separating(aff_a: &[f64; 4], aff_b: &[f64; 4], ma: u8, mb: u8) -> bool {
    if (ma | mb) != 15 {
        return false; // If there is a vertex of b contained in the (-, -) quadrant
    }
    // exclude the vertices in (+,+) quadrant
    let xa = ma & (ma ^ mb);
    let xb = mb & (ma ^ mb);

    if _SEPARATING_EDGE_CASES
        .iter()
        .any(|&(a, b, i, j)| edge_test(xa, xb, a, b, i, j, aff_a, aff_b))
    {
        return false;
    }

    // There exists a separating plane supported by the edge shared by f0 and f1
    true
}

impl<R: Rotate<Cartesian<3>> + Copy> IntersectsAt<Simplex3, Cartesian<3>, R> for Simplex3
where
    RotationMatrix<3>: From<R>,
{
    /**

    Original C code of algorithm:
    <https://web.archive.org/web/20030907000716/http://www.acm.org/jgt/papers/GanovelliPonchioRocchini02/tet_a_tet.html>

    Recent Clojure reimplementation that orients shapes:
    <https://gist.github.com/postspectacular/9021724>
    */
    #[inline]
    fn intersects_at(&self, other: &Simplex3, v_ij: &Cartesian<3>, o_ij: &R) -> bool {
        let r = RotationMatrix::from(*o_ij);
        let q = Simplex3::from(other.vertices.map(|v| r.rotate(&v))).translate_by(v_ij);

        let q_deltas = q.vertices.map(|v| v - self.vertices[0]);

        // Edge difference vectors for tetrahedron p
        let mut edge_vectors_p = [Cartesian::<3>::default(); 5];
        let mut masks = [0u8; 4];
        let mut affs = [[0f64; 4]; 4];

        edge_vectors_p[0] = self.vertices[1] - self.vertices[0];
        edge_vectors_p[1] = self.vertices[2] - self.vertices[0];

        // Handle all possible SAT cases, in a performance-optimized order

        // f 0 1
        let n = edge_vectors_p[0].cross(&edge_vectors_p[1]);
        affs[0] = q_deltas.map(|v| v.dot(&n));
        let mask = check_face_on_p_is_separating(&affs[0]);
        if mask == 15 {
            return false;
        }
        masks[0] = mask;

        // f 2 0
        edge_vectors_p[2] = self.vertices[3] - self.vertices[0];
        let n = edge_vectors_p[2].cross(&edge_vectors_p[0]);

        affs[1] = q_deltas.map(|v| v.dot(&n));
        let mask = check_face_on_p_is_separating(&affs[1]);
        if mask == 15 {
            return false;
        }
        masks[1] = mask;

        // e 0 1
        if check_edge_is_separating(&affs[0], &affs[1], masks[0], masks[1]) {
            return false;
        }

        // f 1 2
        let n = edge_vectors_p[1].cross(&edge_vectors_p[2]);
        affs[2] = q_deltas.map(|v| v.dot(&n));
        let mask = check_face_on_p_is_separating(&affs[2]);
        if mask == 15 {
            return false;
        }
        masks[2] = mask;

        // e 0 2
        if check_edge_is_separating(&affs[0], &affs[2], masks[0], masks[2]) {
            return false;
        }

        // e 1 2
        if check_edge_is_separating(&affs[1], &affs[2], masks[1], masks[2]) {
            return false;
        }

        // f* 4 3
        edge_vectors_p[3] = self.vertices[2] - self.vertices[1];
        edge_vectors_p[4] = self.vertices[3] - self.vertices[1];

        let n = edge_vectors_p[4].cross(&edge_vectors_p[3]);
        affs[3] = q.vertices.map(|v| (v - self.vertices[1]).dot(&n));
        let mask = check_face_on_p_is_separating(&affs[3]);
        if mask == 15 {
            return false;
        }
        masks[3] = mask;

        // e 0|1|2 3
        if [(0, 3), (1, 3), (2, 3)]
            .iter()
            .any(|&(i, j)| check_edge_is_separating(&affs[i], &affs[j], masks[i], masks[j]))
        {
            return false;
        }

        // If (at least) a vertex of q is inside tetrahedron p
        // (vertex bounded by all 4 halfspaces)
        // TODO: do we need to check if masks are empty?
        if masks.iter().fold(0, |acc, &m| acc | m) != 15 {
            return true;
        }

        // From now on, if there is a separating plane it is parallel to a face of q

        // Difference between vertices on p and a vertex on q
        let p_deltas = self.vertices.map(|v| v - q.vertices[0]);
        let mut edge_vectors_q = [Cartesian::<3>::default(); 5];
        edge_vectors_q[0] = q.vertices[1] - q.vertices[0];
        edge_vectors_q[1] = q.vertices[2] - q.vertices[0];

        // f 0 1
        let n = edge_vectors_q[0].cross(&edge_vectors_q[1]);
        if check_face_on_q_is_separating(&p_deltas.map(|v| v.dot(&n))) {
            return false;
        }

        edge_vectors_q[2] = q.vertices[3] - q.vertices[0];

        // f 2 0
        let n = edge_vectors_q[2].cross(&edge_vectors_q[0]);
        if check_face_on_q_is_separating(&p_deltas.map(|v| v.dot(&n))) {
            return false;
        }

        // f 1 2
        let n = edge_vectors_q[1].cross(&edge_vectors_q[2]);
        if check_face_on_q_is_separating(&p_deltas.map(|v| v.dot(&n))) {
            return false;
        }
        edge_vectors_q[3] = q.vertices[2] - q.vertices[1];
        edge_vectors_q[4] = q.vertices[3] - q.vertices[1];

        // f* 4 3
        let n = edge_vectors_q[4].cross(&edge_vectors_q[3]);
        let aff = self.vertices.map(|v| (v - other.vertices[1]).dot(&n));
        if check_face_on_q_is_separating(&aff) {
            return false;
        }
        true // No separating planes -> intersection!
    }
}
#[cfg(test)]
mod tests {
    use crate::xenocollide::collide3d;

    use super::*;
    use hoomd_vector::{Quaternion, Rotation, Versor};

    use rstest::rstest;

    #[test]
    fn test_compute_mask() {
        let arrays = (0u8..=15).map(|i| {
            (
                i,
                [
                    f64::from(i & 1),
                    f64::from((i >> 1) & 1),
                    f64::from((i >> 2) & 1),
                    f64::from((i >> 3) & 1),
                ],
            )
        });
        arrays.for_each(|(i, arr)| assert_eq!(check_face_on_p_is_separating(&arr), i));
    }

    #[rstest(
        ma, mb, ea, eb,
        case(
            0, 2,
            [0.0, -100_000.0, 0.0, -500_000.0],
            [0.0, 50_000.0, -500_000.0, 0.0]
        ),
        case(
            6, 2,
            [0.0, 1_025_000.0, 500_000.0, -500_000.0],
            [0.0, 50_000.0, -500_000.0, 0.0]
        ),
        case(
            0, 8,
            [0.0, -100_000.0, 0.0, -500_000.0],
            [-500_000.0, -1_475_000.0, -500_000.0, 500_000.0]
        ),
        case(
            6, 8,
            [0.0, 1_025_000.0, 500_000.0, -500_000.0],
            [-500_000.0, -1_475_000.0, -500_000.0, 500_000.0]
        ),
        case(
            2, 8,
            [0.0, 50_000.0, -500_000.0, 0.0],
            [-500_000.0, -1_475_000.0, -500_000.0, 500_000.0]
        ),
        case(
            0, 0,
            [0.0, 0.0, 0.0, -50_000.0],
            [-500_005.0, -500_000.0, -1_000_000.0, -612_500.0]
        ),
        case(
            0, 0,
            [0.0, 0.0, 0.0, -50_000.0],
            [0.0, -500_000.0, 0.0, -225_000.0]
        ),
        case(
            0, 0,
            [-500_005.0, -500_000.0, -1_000_000.0, -612_500.0],
            [0.0, -500_000.0, 0.0, -225_000.0]
        )
    )]
    fn test_edge_a(ma: u8, mb: u8, ea: [f64; 4], eb: [f64; 4]) {
        // let t = Simplex3::default();
        assert!(!check_edge_is_separating(&ea, &eb, ma, mb));
    }
    #[rstest(
        n=> [[0.0, 0.0, -5000.0], [-5000.0, 5000.0, 1250.0], [0.0, -10000.0, 2500.0]],
    )]
    fn test_face_a(n: [f64; 3]) {
        let deltas: [Cartesian<3>; 4] = [
            [0.0, 0.0, 0.0],
            [-200.0, 0.0, 20.0],
            [-50.0, 50.0, 0.0],
            [150.0, 25.0, 100.0],
        ]
        .map(Cartesian::from);
        let aff = deltas.map(|v| v.dot(&n.into()));
        let result = check_face_on_p_is_separating(&aff);
        let expected_result = match n {
            [0.0, 0.0, -5000.0] => (0, [0.0, -100_000.0, 0.0, -500_000.0]),
            [-5000.0, 5000.0, 1250.0] => (6, [0.0, 1_025_000.0, 500_000.0, -500_000.0]),
            [0.0, -10_000.0, 2500.0] => (2, [0.0, 50_000.0, -500_000.0, 0.0]),
            _ => unreachable!(),
        };
        assert_eq!((result, aff), expected_result);
    }

    #[test]
    fn test_tetisect() {
        let mut p = Simplex3::from([
            [0.0, 0.0, 0.0],
            [50.0, 50.0, 0.0],
            [100.0, 0.0, 0.0],
            [50.0, 25.0, 100.0],
        ]);
        let mut q = Simplex3::from([
            [0.0, 0.0, 0.0],
            [-50.0, 50.0, 0.0],
            [-200.0, 0.0, 20.0],
            [150.0, 25.0, 100.0],
        ]);

        // First test case is constructed to intersect
        assert!(p.intersects_at(&q, &Cartesian::default(), &Versor::identity()));

        let p_centered = p.translate_by(&-p.centroid());
        let q_centered = q.translate_by(&-q.centroid());

        assert_eq!(
            p.intersects_at(&q, &Cartesian::default(), &Versor::identity()),
            collide3d(
                &p_centered,
                &q_centered,
                &(q.centroid() - p.centroid()),
                &Versor::identity()
            )
        );

        let mut q_nooverlap = Simplex3::from([
            [100.001, 0.0, 0.0],
            [150.0, 50.0, 0.0],
            [200.0, 0.0, 0.0],
            [150.0, 25.0, 10.0],
        ]);

        // Second test case is constructed to NOT intersect
        assert!(!p.intersects_at(&q_nooverlap, &Cartesian::default(), &Versor::identity()));
        let q_nooverlap_centered = q_nooverlap.translate_by(&-p.centroid());
        assert_eq!(
            p.intersects_at(&q_nooverlap, &Cartesian::default(), &Versor::identity()),
            collide3d(
                &p_centered,
                &q_nooverlap_centered,
                &(q_nooverlap.centroid() - p.centroid()),
                &Versor::identity()
            )
        );
    }

    /*
    #[test] // TODO: is this correct? Xenocollide converges, but tetAtet does not!
    fn test_percolation_collisions() {
        const N_SAMPLES: usize = 500_000;
        const VOLUME: f64 = 1.0 / 3.0;
        const AREA: f64 = 3.464_101_615_137_755; // 2.0 * sqrt(3)
        const CURVATURE: f64 = 0.645_065_353_795_142;
        const BOX_L: f64 = 20.0;
        const BOX_VOLUME: f64 = BOX_L * BOX_L * BOX_L;

        let expected_value = (2.0 * (VOLUME + AREA * CURVATURE)) / BOX_VOLUME;

        // Assume 1000x1000x1000 box, which has bo

        let mut rng = StdRng::seed_from_u64(1);
        let (mut t0, mut t1) = (Simplex3::default(), Simplex3::default());
        t0 = t0.translate_by(&-t0.centroid());
        t1 = t1.translate_by(&-t1.centroid());
        t1 = t1.translate_by(&[1.1, 1.1, 1.1].into()); // Just barely not touching

        let (mut n_acc, mut n_rej) = (0, 0);
        for i in 0..N_SAMPLES {
            let mut t_test = Simplex3::default();
            t_test = t_test.translate_by(&-t_test.centroid());

            let theta: Versor = rng.random();
            let v_ij = Cartesian::from([
                rng.random_range((-BOX_L / 2.0)..(BOX_L / 2.0)),
                rng.random_range((-BOX_L / 2.0)..(BOX_L / 2.0)),
                rng.random_range((-BOX_L / 2.0)..(BOX_L / 2.0)),
            ]);
            // if t0.intersects_at(&t_test, &v_ij, &theta) {
            if collide3d(&t0, &t_test, &v_ij, &theta) {
                n_rej += 1;
            } else {
                n_acc += 1;
            }

            // TODO: assert two methods are equal
        }
        println!(
            "expected {}, got {}",
            expected_value,
            n_rej as f64 / (n_acc + n_rej) as f64
        );
    }
    */
    #[rstest(
        v_ij, o_ij, overlaps,
        case::perfect_overlap(
            [0.0, 0.0, 0.0].into(),
            Versor::identity(),
            true,
        ),
        case::particle_at_infinity(
            [f64::INFINITY, 0.0, 0.0].into(),
            Versor::identity(),
            false,
        ),
        case::particle_at_negative_infinity(
            [f64::NEG_INFINITY, 0.0, 0.0].into(),
            Versor::identity(),
            false,
        ),
        case::tip_tip_intersection_exact(
            [2.0, 2.0, 2.0].into(),
            Versor::from_axis_angle([1.0, 0.0, 0.0].try_into().unwrap(), std::f64::consts::FRAC_PI_2),
            true,
        ),
        case::tip_tip_intersection_imprecise(
            [1.999_999_999, 1.999_999_999, 1.999_999_999].into(),
            Versor::from_axis_angle([1.0, 0.0, 0.0].try_into().unwrap(), std::f64::consts::FRAC_PI_2),
            true,
        ),
        case::tip_tip_intersection_nooverlap(
            [2.000_000_001, 2.000_000_001, 2.000_000_001].into(),
            Versor::from_axis_angle([1.0, 0.0, 0.0].try_into().unwrap(), std::f64::consts::FRAC_PI_2),
            false,
        ),
        case::unrotated_tip_tip_intersection_exact(
            [2.0, 2.0, 0.0].into(),
            Versor::identity(),
            true,
        ),
        case::unrotated_tip_tip_intersection_imprecise(
            [1.999_999_999, 1.999_999_999, 0.0].into(),
            Versor::identity(),
            true,
        ),
        case::unrotated_tip_tip_intersection_nooverlap(
            [2.000_000_001, 2.000_000_001, 0.0].into(),
            Versor::identity(),
            false,
        ),
        // NOTE: as above, this type of overlap was not counted in the original paper
        case::tip_edge_intersection_exact(
            [1.0, 1.0, 2.0].into(),
            Versor::default(),
            true,
        ),
        case::tip_edge_intersection_imprecise(
            [1.0, 1.0, 1.999_999_999].into(),
            Versor::default(),
            true,
        ),
        case::tip_edge_intersection_nooverlap(
            [1.0, 1.0, 2.000_000_001].into(),
            Versor::default(),
            false,
        ),
        case::parallel_edge_edge_intersection_exact(
            [1.0, 1.0, 2.0].into(),
            // Hard-coded quaternion required to prevent error accumulation
            Quaternion::from([2.0_f64.sqrt() / 2.0, 2.0_f64.sqrt()/2.0, 0.0, 0.0]).to_versor_unchecked(),
            true,
        ),
        case::parallel_edge_edge_intersection_imprecise(
            [1.0, 1.0, 1.999_999_999].into(),
            Versor::from_axis_angle([1.0, 0.0, 0.0].try_into().unwrap(), std::f64::consts::FRAC_PI_2),
            true,
        ),
        case::parallel_edge_edge_intersection_nooverlap(
            [1.0, 1.0, 2.000_000_001].into(),
            Versor::from_axis_angle([1.0, 0.0, 0.0].try_into().unwrap(), std::f64::consts::FRAC_PI_2),
            false,
        ),
        case::orthogonal_edge_edge_intersection_exact(
            [1.0, 0.0, 2.0].into(),
            Versor::identity(),
            true,
        ),
        case::orthogonal_edge_edge_intersection_imprecise(
            [1.0, 0.0, 1.999_999_999].into(),
            Versor::identity(),
            true,
        ),
        case::orthogonal_edge_edge_intersection_nooverlap(
            [1.0, 0.0, 2.000_000_001].into(),
            Versor::identity(),
            false,
        ),
        case::nonorthogonal_edge_edge_intersection_exact(
            [1.0, 0.0, 2.0].into(),
            Versor::from_axis_angle([0.0, 0.0, 1.0].try_into().unwrap(), std::f64::consts::PI / 3.1),
            true,
        ),
        case::nonorthogonal_edge_edge_intersection_imprecise(
            [1.0, 0.0, 1.999_999_999].into(),
            Versor::from_axis_angle([0.0, 0.0, 1.0].try_into().unwrap(), std::f64::consts::PI / 3.1),
            true,
        ),
        case::nonorthogonal_edge_edge_intersection_nooverlap(
            [1.0, 0.0, 2.000_000_001].into(),
            Versor::from_axis_angle([0.0, 0.0, 1.0].try_into().unwrap(), std::f64::consts::PI / 3.1),
            false,
        ),
        // Overlapping portions of the shape exactly align faces and edges
        case::partial_aligned_overlap_exact(
            [0.0, 1.0, -1.0].into(),
            Versor::identity(),
            true,
        ),
        case::partial_aligned_overlap_imprecise(
            [0.0, 1.0, -0.999_999_999].into(),
            Versor::identity(),
            true,
        ),
        // Top edges are parallel, shapes partially within one another
        case::partial_parallel_overlap(
            [0.0, 0.0, -1.0].into(),
            Versor::identity(),
            true,
        ),
        // Vertex of p passes through q and lies on an edge
        case::vertex_into_edge_shallow_exact(
            [0.0, 1.0, 2.0].into(),
            Versor::from_axis_angle([1.0, 0.0, 0.0].try_into().unwrap(), std::f64::consts::FRAC_PI_4),
            true,
        ),
        case::vertex_into_edge_shallow_imprecise(
            [0.0, 0.999_999_999, 2.0].into(),
            Versor::from_axis_angle([1.0, 0.0, 0.0].try_into().unwrap(), std::f64::consts::FRAC_PI_4),
            true,
        ),
        case::vertex_into_edge_deep_exact(
            [0.0, 1.0, 1.0].into(),
            Versor::from_axis_angle([1.0, 0.0, 0.0].try_into().unwrap(), std::f64::consts::FRAC_PI_4),
            true,
        ),
        case::vertex_into_edge_deep_imprecise(
            [0.0, 0.999_999_999, 1.0].into(),
            Versor::from_axis_angle([1.0, 0.0, 0.0].try_into().unwrap(), std::f64::consts::FRAC_PI_4),
            true,
        ),
        // Degenerate case that fails for both HOOMD-Blue and tet-a-tet
        /*
        case::vertex_face_imprecise(
            [1.0, 1.0, 2.0].into(),
            Versor::from_axis_angle([1.0, 0.0, 0.0].try_into().unwrap(), std::f64::consts::FRAC_PI_4),
            true,
        ),
        */
        case::vertex_face_nooverlap(
            [1.2765, -1.2765, 1.2765].into(),
            Versor::from_axis_angle([1.0, 1.0, 0.0].try_into().unwrap(), std::f64::consts::FRAC_PI_2),
            false,
        ),
        case::vertex_face_near_exact(
            [1.275, -1.275, 1.275].into(),
            Versor::from_axis_angle([1.0, 1.0, 0.0].try_into().unwrap(), std::f64::consts::FRAC_PI_2),
            true,
        ),
    )]
    fn test_tetrahedron_overlap_param(
        #[values("intersects_at", "xenocollide")] method: &str,
        v_ij: Cartesian<3>,
        o_ij: Versor,
        overlaps: bool,
    ) {
        let p = Simplex3::default();
        let q = Simplex3::default();

        let result = if method == "intersects_at" {
            p.intersects_at(&q, &v_ij, &o_ij)
        } else {
            collide3d(&p, &q, &v_ij, &o_ij)
        };

        assert_eq!(
            result, overlaps,
            "Method `{method}` gave wrong overlap result.",
        );
    }
}

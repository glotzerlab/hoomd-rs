use hoomd_vector::{Cartesian, Cross, Rotate, Vector};

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
    #[allow(clippy::expect_used)]
    fn support(&self, n: &Cartesian<3>) -> Cartesian<3> {
        let n = *n / n.norm(); // TODO: does this need to be normalized?
        *self
            .vertices
            .iter()
            .max_by(|a, b| {
                a.dot(&n)
                    .partial_cmp(&b.dot(&n))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("Support function not valid with 0 vertices!")
    }
}

impl From<[Cartesian<3>; 4]> for Simplex3 {
    #[inline]
    fn from(vertices: [Cartesian<3>; 4]) -> Self {
        let s = Simplex3 { vertices };
        s.orient()
    }
}
impl From<[[f64; 3]; 4]> for Simplex3 {
    #[inline]
    fn from(arrs: [[f64; 3]; 4]) -> Self {
        let s = Simplex3 {
            vertices: arrs.map(Cartesian::from),
        };
        s.orient()
    }
}

impl Default for Simplex3 {
    /// A uniform tetrahedron.
    #[inline]
    fn default() -> Self {
        let s = Simplex3 {
            vertices: [
                [1.0, 0.0, 0.0].into(),
                [0.0, 1.0, 0.0].into(),
                [0.0, 0.0, 1.0].into(),
                [1.0, 1.0, 1.0].into(),
            ],
        };
        s.orient()
    }
}
impl Simplex3 {
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

    /// Edge vectors, in the same order as get_edges and pointing left to right.
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
    /// Orient the vertices of a simplex such that the fourth vertex is on the opposite
    /// side of the plane defined by the first three points.
    #[inline]
    fn orient_in_place(&mut self) {
        *self = self.orient();
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
pub fn check_face_is_separating(
    aff: [f64; 4], // deltas: &[Cartesian<3>; 4],
                   // n: &Cartesian<3>,
) -> (u8, bool) {
    let mask = aff.iter().enumerate().fold(
        0u8,
        |acc, (i, &x)| {
            if x > 0.0 { acc | (1 << i) } else { acc }
        },
    );
    (mask, mask == 15)
}
/// Check if there exists a seperating plane containing the edge e shared by faces
/// f0 and f1 of the tetrahedron.
#[inline]
#[must_use]
pub fn check_edge_is_separating(aff_a: &[f64; 4], aff_b: &[f64; 4], ma: u8, mb: u8) -> bool {
    false // TODO: fill out function!
}

impl<R: Rotate<Cartesian<3>>> IntersectsAt<Simplex3, Cartesian<3>, R> for Simplex3 {
    #[inline]
    fn intersects_at(&self, other: &Simplex3, r_ij: &Cartesian<3>, o_ij: &R) -> bool {
        let q_deltas = other.vertices.map(|q| q - self.vertices[0]);
        // TODO: update to use extrinsic coodinates, and orient
        // Edge difference vectors for tetrahedron p
        let mut edge_vectors_p = [Cartesian::<3>::default(); 5];
        let mut masks = [0u8; 4];
        let mut affs = [[0f64; 4]; 4];

        edge_vectors_p[0] = self.vertices[1] - self.vertices[0];
        edge_vectors_p[1] = self.vertices[2] - self.vertices[0];

        // Handle all possible SAT cases, in a performance-optimized order

        // f 0 1
        affs[0] = q_deltas.map(|v| v.dot(&edge_vectors_p[0].cross(&edge_vectors_p[1])));
        let (mask, is_sep) = check_face_is_separating(affs[0]);
        if is_sep {
            return false;
        }
        masks[0] = mask;

        // f 2 0
        edge_vectors_p[2] = self.vertices[3] - self.vertices[0];

        affs[1] = q_deltas.map(|v| v.dot(&edge_vectors_p[0].cross(&edge_vectors_p[1])));
        let (mask, is_sep) = check_face_is_separating(affs[1]);
        if is_sep {
            return false;
        }
        masks[1] = mask;

        // e 0 1
        if check_edge_is_separating(&affs[0], &affs[1], masks[0], masks[1]) {
            return false;
        }

        // f 1 2
        affs[2] = q_deltas.map(|v| v.dot(&edge_vectors_p[1].cross(&edge_vectors_p[2])));
        let (mask, is_sep) = check_face_is_separating(affs[2]);
        if is_sep {
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

        // f 4 3
        edge_vectors_p[3] = self.vertices[2] - self.vertices[1];
        edge_vectors_p[4] = self.vertices[3] - self.vertices[1];

        affs[3] = q_deltas.map(|v| v.dot(&edge_vectors_p[4].cross(&edge_vectors_p[3])));
        let (mask, is_sep) = check_face_is_separating(affs[3]);
        if is_sep {
            return false;
        }
        masks[3] = mask;

        // e 0|1|2 3
        for (i, j) in [(0, 3), (1, 3), (2, 3)] {
            if check_edge_is_separating(&affs[i], &affs[j], masks[i], masks[j]) {
                return false;
            }
        }

        // If any of the masks is NOT full, there is not separating plane
        if masks.iter().any(|&m| m != 15) {
            return true;
        }

        // From now on, if there is a separating plane it is parallel to a face of q

        // Difference between vertices on p and a vertex on q
        let p_deltas = self.vertices.map(|v| v - other.vertices[0]);
        let mut edge_vectors_q = [Cartesian::<3>::default(); 5];
        edge_vectors_q[0] = other.vertices[1] - other.vertices[0];
        edge_vectors_q[1] = other.vertices[2] - other.vertices[0];

        false
    }
}

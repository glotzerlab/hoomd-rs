// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement [`Triclinic`]

use std::array;

use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, InnerProduct};
use serde::{Deserialize, Serialize};
use std::ops::Mul;

use rand::{
    Rng,
    distr::{Distribution, Uniform},
};

use crate::{IsPointInside, MapPoint, Scale, SupportMapping, Volume, shape::Hyperparallelepiped};

/// A hexahedron with three pairs of parallel faces described by an upper-triangular matrix.
///
/// A triclinic box is a parallelepiped defined by three edge vectors that may be
/// non-orthogonal. It is characterized by three extents $`(L_x, L_y, L_z)`$ and three
/// tilt factors $`(xy, xz, yz)`$ that describe the shearing of the box. The tilt factors
/// describe the ratio of the length of the components of the basis vector to the extent in
/// the corresponding direction. That is, the edges of the box are spanned by the vectors:
/// ```math
///  \vec{a}_1 = \left(L_x,0,0\right) \qquad \vec{a}_2 = \left(xyL_y,L_y,0\right) \qquad \vec{a}_3 = \left(xzL_z,yzL_z,L_z\right)
/// ```
///
/// The box is centered at the origin, $`(0,0,0)`$.
///
/// # Example
///
/// Basic construction and methods:
/// ```
/// use hoomd_geometry::shape::Triclinic;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let triclinic =
///     Triclinic { extents: [10.0.try_into()?, 12.0.try_into()?, 14.0.try_into()?], tilt_factors: [1.0, 0.5, -0.2]};
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Triclinic {
    /// The extents of each edge of the triclinic box. [lx, ly, lz]
    pub extents: [PositiveReal; 3],
    /// The tilt factors that define the shear of the box.
    ///
    /// `[xy, xz, yz]` where, for example, $`xy`$ is the ratio of the $`y`$-component of basis vector $`\vec{a}_2`$
    /// and the extent in the $`y`$-direction, $`L_y`$.
    pub tilt_factors: [f64; 3],
}

impl Triclinic {
    /// Construct a triclinic box from a general parallelepiped.
    ///
    /// Computes the triclinic parameters from a parallelepiped with edge vectors $`\vec{u}_i`$ by computing
    /// applying the transformation formulas:
    /// ```math
    ///     a_{2x} = \frac{\vec{u}_1\cdot \vec{u}_2}{v_1}, \qquad a_{3x} = \frac{\vec{u}_1\cdot \vec{u}_3}{u_1} \\[3pt]
    ///     L_x = u_1, \qquad L_y = \sqrt{u_2^2 - a_{2x}^2}, \qquad L_z = \vec{u}_3 \cdot \frac{\vec{u}_1 \times \vec{u}_2}{\left|\vec{u}_1 \times \vec{u}_2 \right|}\\[2pt]
    ///     xy = \frac{a_{2x}}{L_y}, \qquad xz = \frac{a_{3x}}{L_z}, \qquad yz = \frac{\vec{u}_2\cdot \vec{u}_3 - a_{2x} a_{3x}}{L_yL_z}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::shape::{Hyperparallelepiped, Triclinic};
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let parallelepiped = Hyperparallelepiped::new([
    ///     Cartesian::from([-1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]),
    ///     Cartesian::from([2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0]),
    ///     Cartesian::from([2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0]),
    /// ]); // Rotated unit cube
    /// let triclinic = Triclinic::from_parallelepiped(&parallelepiped);
    /// assert_relative_eq!(triclinic.lx().get(), 1.0, epsilon = 1e-8);
    /// assert_relative_eq!(triclinic.xy(), 0.0, epsilon = 1e-8);
    /// # Ok(())
    /// # }
    /// ```
    /// # Panics
    ///
    /// Panics if the computed box dimensions are not positive.
    #[inline]
    pub fn from_parallelepiped(parallelepiped: &Hyperparallelepiped<3>) -> Self {
        let v1 = parallelepiped.edge_vectors[0];
        let v2 = parallelepiped.edge_vectors[1];
        let v3 = parallelepiped.edge_vectors[2];

        let v1_mag = v1.norm();
        let v2_mag = v2.norm();
        let v2_dot_v1 = v2.dot(&v1);
        let v3_dot_v1 = v3.dot(&v1);
        let v3_dot_v2 = v3.dot(&v2);

        let lx = v1_mag;
        let a2x = v2_dot_v1 / v1_mag;
        let ly = (v2_mag * v2_mag - a2x * a2x).sqrt();
        let cross_v1_v2 = Cartesian::from([
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        ]);
        let cross_mag = cross_v1_v2.norm();
        let lz = v3.dot(&cross_v1_v2) / cross_mag;
        let a3x = v3_dot_v1 / v1_mag;
        let xy = a2x / ly;
        let xz = a3x / lz;
        let yz = (v3_dot_v2 - a2x * a3x) / (ly * lz);

        Self {
            extents: [
                lx.try_into().expect("lx must be positive"),
                ly.try_into().expect("ly must be positive"),
                lz.try_into().expect("lz must be positive"),
            ],
            tilt_factors: [xy, xz, yz],
        }
    }

    /// Returns the box extent in the x-direction (lx)
    #[inline]
    pub fn lx(&self) -> PositiveReal {
        self.extents[0]
    }

    /// Returns the box extent in the y-direction (ly)
    #[inline]
    pub fn ly(&self) -> PositiveReal {
        self.extents[1]
    }

    /// Returns the box extent in the z-direction (lz)
    #[inline]
    pub fn lz(&self) -> PositiveReal {
        self.extents[2]
    }

    /// Returns the xy tilt factor
    #[inline]
    pub fn xy(&self) -> f64 {
        self.tilt_factors[0]
    }

    /// Returns the xz tilt factor
    #[inline]
    pub fn xz(&self) -> f64 {
        self.tilt_factors[1]
    }

    /// Returns the yz tilt factor
    #[inline]
    pub fn yz(&self) -> f64 {
        self.tilt_factors[2]
    }

    /// Convert a real space (absolute) position to fractional coordinates.
    ///
    /// We can express a
    /// general point in space $`\vec{r}`$ in terms of the basis vectors $`a_i`$,
    /// ```math
    ///     \vec{r} = s_1 \vec{a}_1 + s_2 \vec{a}_2 + s_3 \vec{a}_3.
    /// ```
    /// The vector, $`\vec{s}=(s_1, s_2, s_3)`$ then gives the fractional coordinates
    /// of this point. This can be expressed as $`\vec{r} = \mathbf{A} \vec{s}`$, where
    /// $`\mathbf{A}`$ is the matrix with columns equal to the box vectors. That is,
    /// $`\vec{s} = \mathbf{A}^{-1} \vec{r}`$. Geometrically, we can view $`\mathbf{A}^{-1}`$
    /// as linearly shearing the triclinic box to a unit cube centered at the origin.
    /// For a triclinic box, the transformation can be written as
    /// ```math
    /// \begin{align*}
    ///     s_1 &= \frac{r_1-(xy)r_2-(xz-yz\cdot xy) r_3}{L_x}\\
    ///     s_2 &= \frac{r_2-(yz) r_3}{L_y}\\
    ///     s_3 &= \frac{r_3}{L_z}.\\
    /// \end{align*}
    /// ```
    ///
    /// Each fractional coordinate is in the range $`[-0.5, 0.5)`$.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Triclinic;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic = Triclinic { extents: [2.0.try_into()?, 2.0.try_into()?, 2.0.try_into()?], tilt_factors: [0.0, 0.0, 0.0]};
    ///
    /// let pos = Cartesian::from([1.0, 0.0, 0.0]);
    /// let frac = triclinic.fractional(&pos);
    /// assert_eq!(frac, Cartesian::from([0.5, 0.0, 0.0]));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn fractional(&self, absolute: &Cartesian<3>) -> Cartesian<3> {
        let l: Cartesian<3> = self.extents.map(|x| x.get()).into();
        let mut fractional = *absolute;
        fractional[0] -= (self.xz() - self.yz() * self.xy()) * absolute[2] + self.xy() * absolute[1];
        fractional[1] -= self.yz() * absolute[2];
        for i in 0..3 {
            fractional[i] /= l[i];
        }
        fractional
    }

    /// Convert fractional coordinates to absolute position.
    ///
    /// This is the inverse operation of `fractional`, $`\vec{r} = \mathbf{A} \vec{s}`$.
    /// Namely,
    /// ```math
    /// \begin{align*}
    ///     r_1 &= L_x s_1 + xyL_y s_2 + xzL_z s_3\\
    ///     r_2 &= L_y s_2 + yz L_z s_3\\
    ///     r_3 &= L_z s_3.\\
    /// \end{align*}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Triclinic;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic = Triclinic { extents: [2.0.try_into()?, 2.0.try_into()?, 2.0.try_into()?], tilt_factors: [0.0, 0.0, 0.0] };
    ///
    /// let frac = Cartesian::from([0.5, 0.0, 0.0]);
    /// let pos = triclinic.absolute(&frac);
    /// assert_eq!(pos, Cartesian::from([1.0, 0.0, 0.0]));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn absolute(&self, fractional: &Cartesian<3>) -> Cartesian<3> {
        let mut pos: Cartesian<3> = Cartesian::from([1.0, 1.0, 1.0]);
        for i in 0..3 {
            pos[i] = self.extents[i].get() * fractional[i];
        }
        pos[0] += self.xy() * pos[1] + self.xz() * pos[2];
        pos[1] += self.yz() * pos[2];
        pos
    }

    /// Get the edge vectors of the triclinic box.
    ///
    /// Returns the three basis vectors [`a_1`, `a_2`, `a_3`] that span the box edges.
    /// These vectors are computed from the extents and tilt factors.
    /// ```math
    /// \begin{align*}
    ///     \vec{a}_1 &= \left( L_x, 0, 0 \right) \\
    ///     \vec{a}_2 &= \left( xy L_y, L_y, 0 \right) \\
    ///     \vec{a}_3 &= \left( xz L_z, yz L_z, L_z \right)
    /// \end{align*}
    /// ```
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Triclinic;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic = Triclinic { extents: [2.0.try_into()?, 3.0.try_into()?, 4.0.try_into()?], tilt_factors: [0.5, 0.0, 0.0]};
    /// let edges = triclinic.edge_vectors();
    ///
    /// assert_eq!(edges[0], Cartesian::from([2.0, 0.0, 0.0]));
    /// assert_eq!(edges[1], Cartesian::from([1.5, 3.0, 0.0]));
    /// assert_eq!(edges[2], Cartesian::from([0.0, 0.0, 4.0]));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn edge_vectors(&self) -> [Cartesian<3>; 3] {
        let mut edge_vectors = [Cartesian::<3>::default(); 3];
        edge_vectors[0] = [self.lx().get(), 0., 0.].into();
        edge_vectors[1] = [self.ly().get() * self.xy(), self.ly().get(), 0.].into();
        edge_vectors[2] = [
            self.lz().get() * self.xz(),
            self.lz().get() * self.yz(),
            self.lz().get(),
        ]
        .into();
        edge_vectors
    }

    /// Get the box angles $`\alpha`$, $`\beta`$, $`\gamma`$ in radians.
    ///
    /// Returns the angles between the basis vectors:
    /// - $`\alpha`$: angle between vectors $`\vec{a}_2`$ and $`\vec{a}_3`$
    /// - $`\beta`$: angle between vectors  $`\vec{a}_1`$ and  $`\vec{a}_3`$
    /// - $`\gamma`$: angle between vectors $`\vec{a}_1`$ and $`\vec{a}_2`$
    ///
    /// The angles are computed using the tilt factors via:
    /// ```math
    /// \begin{align*}
    ///     \cos\gamma &= \cos(\angle\vec a_1, \vec a_2) =
    ///         \frac{xy}{\sqrt{1+xy^2}} \\
    ///     \cos\beta &= \cos(\angle\vec a_1, \vec a_3) =
    ///         \frac{xz}{\sqrt{1+xz^2+yz^2}} \\
    ///     \cos\alpha &= \cos(\angle\vec a_2, \vec a_3) =
    ///         \frac{xy \cdot xz + yz}{\sqrt{1+xy^2} \sqrt{1+xz^2+yz^2}}
    /// \end{align*}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::shape::Triclinic;
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic =
    ///     Triclinic { extents: [10.0.try_into()?, 10.0.try_into()?, 10.0.try_into()?], tilt_factors: [0.0, 0.0, 0.0]};
    /// let angles = triclinic.box_angles();
    ///
    /// assert_relative_eq!(angles[0], PI / 2.0);
    /// assert_relative_eq!(angles[1], PI / 2.0);
    /// assert_relative_eq!(angles[2], PI / 2.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn box_angles(&self) -> [f64; 3] {
        let xy = self.xy();
        let xz = self.xz();
        let yz = self.yz();

        let cos_gamma = xy / (1.0 + xy * xy).sqrt();
        let cos_beta = xz / (1.0 + xz * xz + yz * yz).sqrt();
        let cos_alpha =
            (xy * xz + yz) / ((1.0 + xy * xy).sqrt() * (1.0 + xz * xz + yz * yz).sqrt());

        [cos_alpha.acos(), cos_beta.acos(), cos_gamma.acos()]
    }

    /// Get the perpendicular distances between parallel faces of the triclinic box.
    ///
    /// Returns [`d_x`, `d_y`, `d_z`] where `d_i` is the width in direction i.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Triclinic;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic = Triclinic { extents: [2.0.try_into()?, 4.0.try_into()?, 6.0.try_into()?], tilt_factors: [0.0, 0.0, 0.0]};
    /// let distances = triclinic.nearest_plane_distance();
    ///
    /// assert_eq!(distances[0].get(), 2.0);
    /// assert_eq!(distances[1].get(), 4.0);
    /// assert_eq!(distances[2].get(), 6.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[expect(
        clippy::missing_panics_doc,
        reason = "Panic would occur due to a bug in hoomd-rs."
    )]
    pub fn nearest_plane_distance(&self) -> [PositiveReal; 3] {
        // Since V = A_ih_i, h_i = V/A_i. V = det(a_1, a_2, a_3), A = |a_j x a_k|.
        let mut dist = [PositiveReal::default(); 3];
        dist[0] = self.lx()
            / (f64::sqrt(
                1.0 + self.xy() * self.xy() + (self.xy() * self.yz() - self.xz()).powi(2),
            ))
            .try_into()
            .expect("nearest-plane distance must be positive");
        dist[1] = self.ly()
            / (f64::sqrt(1.0 + self.yz() * self.yz()))
                .try_into()
                .expect("nearest-plane distance must be positive");
        dist[2] = self.lz();
        dist
    }

    /// Represent the triclinic box in the GSD box format.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Triclinic;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic = Triclinic { extents: [5.0.try_into()?, 5.0.try_into()?, 6.0.try_into()?], tilt_factors: [1.5, 1.2, -1.0]};
    ///
    /// let gsd_box = triclinic.to_gsd_box();
    /// assert_eq!(gsd_box, [5.0, 5.0, 6.0, 1.5, 1.2, -1.0]);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    pub fn to_gsd_box(&self) -> [f64; 6] {
        [
            self.extents[0].get(),
            self.extents[1].get(),
            self.extents[2].get(),
            self.tilt_factors[0],
            self.tilt_factors[1],
            self.tilt_factors[2],
        ]
    }
}

impl Volume for Triclinic {
    /// Compute the volume of the triclinic box.
    ///
    /// The volume is computed as the product of the three extents:
    /// ```math
    /// V = L_x \cdot L_y \cdot L_z
    /// ```
    ///
    /// # Example
    /// ```
    /// use hoomd_geometry::{Volume, shape::Triclinic};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic =
    ///     Triclinic { extents: [10.0.try_into()?, 12.0.try_into()?, 14.0.try_into()?], tilt_factors: [1.0, 0.5, -0.2]};
    ///
    /// assert_eq!(triclinic.volume(), 1680.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn volume(&self) -> f64 {
        self.extents
            .iter()
            .map(PositiveReal::get)
            .reduce(f64::mul)
            .expect("N should be >= 1")
    }
}

impl SupportMapping<Cartesian<3>> for Triclinic {
    /// Compute the support point of the triclinic box in a given direction.
    ///
    /// Mathematically:
    /// ```math
    ///  \mathbf{A}^{-1} \vec{v_i} \cdot \mathbf{A}^T \vec{n} = \vec{v_i}^T (\mathbf{A}^{-T} \mathbf{A}^{T}) \vec{n} = \vec{v_i} \cdot \vec{n}.
    /// ```
    /// The vertices of the scaled box ($`\mathbf{A}^{-1} \vec{v_i}`$) have components $`\pm 0.5`$, so we can determine the correct vertex using the sign of $`\mathbf{A}^{T}n`$.
    #[inline]
    fn support_mapping(&self, n: &Cartesian<3>) -> Cartesian<3> {
        let d = Cartesian::from([
            self.lx().get() * n[0],
            self.ly().get() * (self.xy() * n[0] + n[1]),
            self.lz().get() * (self.xz() * n[0] + self.yz() * n[1] + n[2]),
        ]);
        let s = Cartesian::from([
            d[0].signum() * 0.5,
            d[1].signum() * 0.5,
            d[2].signum() * 0.5,
        ]);
        self.absolute(&s)
    }
}

impl IsPointInside<Cartesian<3>> for Triclinic {
    /// Check whether a Cartesian point lies inside the triclinic box.
    ///
    /// A point $`\vec{r} = (x, y, z)`$ is inside if it can be expressed in fractional
    /// coordinates $`\vec{s} = (s_1, s_2, s_3)`$ with all components in the range $`[-0.5, 0.5)`$.
    /// The test is performed by checking the inequalities:
    /// ```math
    /// \begin{align*}
    /// |z| &< L_z / 2 \\
    /// |y - yz \cdot z| &< L_y / 2 \\
    /// |x - (xz - xy \cdot yz) \cdot z - xy \cdot y| &< L_x / 2
    /// \end{align*}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{IsPointInside, shape::Triclinic};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic = Triclinic { extents: [6.0.try_into()?, 8.0.try_into()?, 10.0.try_into()?], tilt_factors: [0.5, 0.0, 0.0]};
    ///
    /// assert!(triclinic.is_point_inside(&[1.0, 1.0, 1.0].into()));
    /// assert!(!triclinic.is_point_inside(&[4.0, 1.0, 1.0].into()));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn is_point_inside(&self, point: &Cartesian<3>) -> bool {
        let [x, y, z] = point.coordinates;
        if z.abs() >= self.lz().get() / 2.0 {
            return false;
        }

        if (y - self.yz() * z).abs() >= self.ly().get() / 2.0 {
            return false;
        }

        if (x - (self.xz() - self.xy() * self.yz()) * z - self.xy() * y).abs()
            >= self.lx().get() / 2.0
        {
            return false;
        }
        true
    }
}

impl Scale for Triclinic {
    /// Construct a scaled triclinic box by scaling edge lengths.
    ///
    /// The resulting triclinic's extents $` L_\mathrm{new} `$ are
    /// the original's $` L `$ scaled by $` v `$:
    /// ```math
    /// L_\mathrm{new} = v L
    /// ```
    ///
    /// The shear factors (tilt factors) remain unchanged. The centroid remains at the origin.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::Triclinic};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic =
    ///     Triclinic { extents: [10.0.try_into()?, 12.0.try_into()?, 14.0.try_into()?], tilt_factors: [1.0, 0.5, -0.2]};
    ///
    /// let scaled = triclinic.scale_length(2.0.try_into()?);
    ///
    /// assert_eq!(scaled.lx().get(), 20.0);
    /// assert_eq!(scaled.ly().get(), 24.0);
    /// assert_eq!(scaled.lz().get(), 28.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_length(&self, v: PositiveReal) -> Self {
        Self {
            extents: array::from_fn(|i| self.extents[i] * v),
            tilt_factors: self.tilt_factors,
        }
    }

    /// Construct a scaled triclinic box by scaling volume.
    ///
    /// The resulting triclinic's extents $` L_\mathrm{new} `$ are scaled by $` v^\frac{1}{3} `$:
    /// ```math
    /// L_\mathrm{new} = v^\frac{1}{3} L
    /// ```
    ///
    /// The shear factors (tilt factors) remain unchanged. The centroid remains at the origin.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::Triclinic};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic = Triclinic { extents: [5.0.try_into()?, 5.0.try_into()?, 6.0.try_into()?], tilt_factors: [1.5, 1.2, -1.0]};
    ///
    /// let scaled_triclinic = triclinic.scale_volume(8.0.try_into()?);
    ///
    /// assert_eq!(scaled_triclinic.lx().get(), 10.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_volume(&self, v: PositiveReal) -> Self {
        let v = v
            .get()
            .cbrt()
            .try_into()
            .expect("v^{1/3} should be a positive real");
        self.scale_length(v)
    }
}

impl Distribution<Cartesian<3>> for Triclinic {
    /// Generate points uniformly distributed in the triclinic box.
    ///
    /// # Example
    ///
    /// ```
    /// use rand::{SeedableRng, distr::Distribution, rngs::StdRng};
    ///
    /// use hoomd_geometry::{IsPointInside, shape::Triclinic};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic = Triclinic { extents: [6.0.try_into()?, 8.0.try_into()?, 10.0.try_into()?], tilt_factors: [0.5, 0.0, 0.0]};
    /// let mut rng = StdRng::seed_from_u64(1);
    ///
    /// let point = triclinic.sample(&mut rng);
    /// assert!(triclinic.is_point_inside(&point));
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Cartesian<3> {
        let uniform = Uniform::new(-0.5, 0.5).expect("");
        let x = uniform.sample(rng);
        let y = uniform.sample(rng);
        let z = uniform.sample(rng);

        self.absolute(&Cartesian::from([x, y, z]))
    }
}

impl MapPoint<Cartesian<3>> for Triclinic {
    #[inline]
    fn map_point(&self, point: Cartesian<3>, other: &Self) -> Result<Cartesian<3>, crate::Error> {
        let fractional = self.fractional(&point);
        let mapped_coords = other.absolute(&fractional);
        Ok(mapped_coords)
    }
}

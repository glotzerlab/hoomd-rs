// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::array;

use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, InnerProduct};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::ops::Mul;

use rand::{
    Rng,
    distr::{Distribution, Uniform},
};

use crate::{IsPointInside, MapPoint, Scale, SupportMapping, Volume, shape::Hyperparallelepiped};

/// A triclinic box shape with arbitrary tilt factors.
///
/// A triclinic box is a parallelepiped defined by three edge vectors that may be
/// non-orthogonal. It is characterized by three extents $`(L_x, L_y, L_z)`$ and three
/// tilt factors $`(xy, xz, yz)`$ that describe the shearing of the box.
///
/// The box is centered at the origin, with the centroid at $`(0,0,0)`$.
///
/// # Construction
///
/// Triclinic boxes can most easily be constructed using the `from_box_vector` method,
/// which takes an array of 6 values: `[lx, ly, lz, xy, xz, yz]`. It can also be generated
/// from a 3D parallelepiped using the `from_parallelepiped` method.
///
/// # Examples
///
/// Basic construction and methods:
/// ```
/// use hoomd_geometry::{Volume, shape::Triclinic};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let triclinic = Triclinic::from_box_vector([10.0, 12.0, 14.0, 1.0, 0.5, -0.2]);
/// assert_eq!(triclinic.volume(), 1680.0);
///
/// let extents = triclinic.extents.map(|x| x.get());
/// assert_eq!(extents, [10.0, 12.0, 14.0]);
/// # Ok(())
/// # }
/// ```
///
/// Checking if a point is inside the triclinic box:
/// ```
/// use hoomd_geometry::{IsPointInside, shape::Triclinic};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let triclinic = Triclinic::from_box_vector([6.0, 8.0, 10.0, 0.5, 0.0, 0.0]);
///
/// assert!(triclinic.is_point_inside(&[1.0, 1.0, 1.0].into()));
/// assert!(!triclinic.is_point_inside(&[4.0, 1.0, 1.0].into()));
/// # Ok(())
/// # }
/// ```
///
/// Scaling the triclinic box:
/// ```
/// use hoomd_geometry::{Scale, shape::Triclinic};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let triclinic = Triclinic::from_box_vector([10.0, 12.0, 14.0, 1.0, 0.5, -0.2]);
///
/// let scaled = triclinic.scale_length(2.0.try_into()?);
/// assert_eq!(scaled.lx().get(), 20.0);
/// assert_eq!(scaled.ly().get(), 24.0);
/// assert_eq!(scaled.lz().get(), 28.0);
/// # Ok(())
/// # }
/// ```
#[serde_as]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Triclinic {
    /// The extents of each edge of the triclinic box. [lx, ly, lz]
    #[serde_as(as = "[_; 3]")]
    pub extents: [PositiveReal; 3],
    /// The tilt factors that define the shear of the box.
    /// [xy, xz, yz] where, for example, $`xy`$ is the ratio of the $`y`$-component of basis vector $`\vec{a}_2`$
    /// and the extent in the $`y`$-direction, $`L_y`$.
    #[serde_as(as = "[_; 3]")]
    pub tilt_factors: [f64; 3],
}

impl Triclinic {
    /// Returns the box extent in the x-direction (lx)
    #[inline]
    #[allow(non_snake_case)]
    pub fn lx(&self) -> PositiveReal {
        self.extents[0]
    }

    /// Returns the box extent in the y-direction (ly)
    #[inline]
    #[allow(non_snake_case)]
    pub fn ly(&self) -> PositiveReal {
        self.extents[1]
    }

    /// Returns the box extent in the z-direction (lz)
    #[inline]
    #[allow(non_snake_case)]
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

    #[inline]
    fn matmul(&self, v: [f64; 3]) -> [f64; 3] {
        [
            self.lx().get() * v[0]
                + self.ly().get() * self.xy() * v[1]
                + self.lz().get() * self.xz() * v[2],
            self.ly().get() * v[1] + self.lz().get() * self.yz() * v[2],
            self.lz().get() * v[2],
        ]
    }

    #[inline]
    fn matmul_inv(&self, v: [f64; 3]) -> [f64; 3] {
        [
            1.0 / self.lx().get()
                * (v[0] - self.xy() * v[1] - (self.xz() + self.xy() * self.yz()) * v[2]),
            1.0 / self.ly().get() * (v[1] - self.yz() * v[2]),
            1.0 / self.lz().get() * v[2],
        ]
    }

    /// Construct a triclinic box from box dimensions.
    ///
    /// The dimensions array should contain [lx, ly, lz, xy, xz, yz] where:
    /// - lx, ly, lz are the box extents (must be positive)
    /// - xy, xz, yz are the tilt factors
    ///
    /// # Panics
    ///
    /// Panics if any of lx, ly, lz are not positive.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Triclinic;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic = Triclinic::from_box_vector([10.0, 12.0, 14.0, 1.0, 0.5, -0.2]);
    /// assert_eq!(triclinic.lx().get(), 10.0);
    /// assert_eq!(triclinic.xy(), 1.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_box_vector(box_dimensions: [f64; 6]) -> Self {
        Self {
            extents: [
                box_dimensions[0]
                    .try_into()
                    .expect("Extent lx must be positive"),
                box_dimensions[1]
                    .try_into()
                    .expect("Extent ly must be positive"),
                box_dimensions[2]
                    .try_into()
                    .expect("Extent lz must be positive"),
            ],
            tilt_factors: [box_dimensions[3], box_dimensions[4], box_dimensions[5]],
        }
    }
    /// Construct a triclinic box from a general parallelepiped.
    ///
    /// Computes the triclinic parameters from a parallelepiped by computing
    /// the edge vectors and applying the transformation formulas:
    /// ```math
    ///     a_{2x} = \frac{\vec{v}_1\cdot \vec{v}_2}{v_1}, \qquad a_{3x} = \frac{\vec{v}_1\cdot \vec{v}_3}{v_1} \\[3pt]
    ///     L_x = v_1, \qquad L_y = \sqrt{v_2^2 - a_{2x}^2}, \qquad L_z = \vec{v}_3 \cdot \frac{\vec{v}_1 \times \vec{v}_2}{\left|\vec{v}_1 \times \vec{v}_2 \right|}\\[2pt]
    ///     xy = \frac{a_{2x}}{L_y}, \qquad xz = \frac{a_{3x}}{L_z}, \qquad yz = \frac{\vec{v}_2\cdot \vec{v}_3 - a_{2x} a_{3x}}{L_yL_z}
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::{Triclinic, Hyperparallelepiped};
    /// use hoomd_vector::Cartesian;
    /// use approxim::assert_relative_eq;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let parallelepiped = Hyperparallelepiped::new([
    ///     Cartesian::from([-1.0/3.0, 2.0/3.0, 2.0/3.0]),
    ///     Cartesian::from([2.0/3.0, -1.0/3.0, 2.0/3.0]),
    ///     Cartesian::from([2.0/3.0, 2.0/3.0, -1.0/3.0]),
    /// ]); // Rotated unit cube
    /// let triclinic = Triclinic::from_parallelepiped(parallelepiped);
    /// assert_relative_eq!(triclinic.lx().get(), 1.0, epsilon = 1e-8);
    /// assert_relative_eq!(triclinic.xy(), 0.0, epsilon = 1e-8);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_parallelepiped(parallelepiped: Hyperparallelepiped<3>) -> Self {
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

    /// Get the edge vectors of the triclinic box.
    ///
    /// Returns the three basis vectors [a_1, a_2, a_3] that span the box edges.
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
    /// let triclinic = Triclinic::from_box_vector([2.0, 3.0, 4.0, 0.5, 0.0, 0.0]);
    /// let edges = triclinic.get_edge_vectors();
    ///
    /// assert_eq!(edges[0], Cartesian::from([2.0, 0.0, 0.0]));
    /// assert_eq!(edges[1], Cartesian::from([1.5, 3.0, 0.0])); // xy * ly = 0.5 * 3.0
    /// assert_eq!(edges[2], Cartesian::from([0.0, 0.0, 4.0]));
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_edge_vectors(&self) -> [Cartesian<3>; 3] {
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
    /// The angles are computed using the tilt factors according to:
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
    /// use hoomd_geometry::shape::Triclinic;
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic = Triclinic::from_box_vector([10.0, 10.0, 10.0, 0.0, 0.0, 0.0]);
    /// let angles = triclinic.get_box_angles();
    ///
    /// // For orthogonal box, all angles should be 90 degrees
    /// assert!((angles[0] - PI/2.0).abs() < 1e-10);
    /// assert!((angles[1] - PI/2.0).abs() < 1e-10);
    /// assert!((angles[2] - PI/2.0).abs() < 1e-10);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_box_angles(&self) -> [f64; 3] {
        let xy = self.xy();
        let xz = self.xz();
        let yz = self.yz();

        let cos_gamma = xy / (1.0 + xy * xy).sqrt();
        let cos_beta = xz / (1.0 + xz * xz + yz * yz).sqrt();
        let cos_alpha =
            (xy * xz + yz) / ((1.0 + xy * xy).sqrt() * (1.0 + xz * xz + yz * yz).sqrt());

        [cos_alpha.acos(), cos_beta.acos(), cos_gamma.acos()]
    }

    /// Get the perpendicualar distances between parallel faces of the triclinic box.
    ///
    /// For a triclinic box, the distance between parallel faces is not simply
    /// the extent since the box is sheared.
    ///
    /// Returns [d_x, d_y, d_z] where d_i is the width in direction i.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Triclinic;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic = Triclinic::from_box_vector([2.0, 2.0, 2.0, 0.0, 0.0, 0.0]);
    /// let distances = triclinic.get_nearest_plane_distance();
    ///
    /// // For orthogonal box, distances are just extents/2
    /// assert_eq!(distances[0].get(), 2.0);
    /// assert_eq!(distances[1].get(), 2.0);
    /// assert_eq!(distances[2].get(), 2.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_nearest_plane_distance(&self) -> [PositiveReal; 3] {
        // Since V = A_ih_i, h_i = V/A_i. V = det(a_1, a_2, a_3), A = |a_j x a_k|.
        let mut dist = [PositiveReal::default(); 3];
        dist[0] = self.lx()
            / (f64::sqrt(
                1.0 + self.xy() * self.xy() + (self.xy() * self.yz() - self.xz()).powi(2),
            ))
            .try_into()
            .unwrap();
        dist[1] = self.ly() / (f64::sqrt(1.0 + self.yz() * self.yz())).try_into().unwrap();
        dist[2] = self.lz();
        dist
    }
}

impl Volume for Triclinic {
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
    /// Calculate the point furthest from the center in a given direction
    #[inline]
    fn support_mapping(&self, n: &Cartesian<3>) -> Cartesian<3> {
        let mut iter = n
            .into_iter()
            .zip(self.extents)
            .map(|(n_i, l_i)| l_i.get() / 2.0 * n_i.signum());
        array::from_fn(|_| iter.next().unwrap_or_default()).into()
    }
}

impl Triclinic {
    /// Represent the triclinic box in the GSD box format.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::shape::Triclinic;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic =
    ///     Triclinic::from_box_vector([5.0, 5.0, 6.0, 1.5, 1.2, -1.0]);
    ///
    /// let gsd_box = triclinic.to_gsd_box();
    /// assert_eq!(gsd_box, [5.0, 5.0, 6.0, 1.5, 1.2, -1.0]);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    #[must_use]
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

impl IsPointInside<Cartesian<3>> for Triclinic {
    fn is_point_inside(&self, point: &Cartesian<3>) -> bool {
        let [x, y, z] = point.coordinates;
        if z.abs() >= self.lz().get() / 2.0 {
            return false;
        };

        if (y - self.yz() * z).abs() >= self.ly().get() / 2.0 {
            return false;
        };

        if (x - (self.xz() - self.xy() * self.yz()) * z - self.xy() * y).abs()
            >= self.lx().get() / 2.0
        {
            return false;
        }
        true
    }
}

impl Scale for Triclinic {
    /// Construct a scaled triclinic box.
    ///
    /// The resulting triclinic's extents $` L_\mathrm{new} `$ are
    /// the original's $` L `$ scaled by $` v `$:
    /// ```math
    /// L_\mathrm{new} = v L
    /// ```
    ///
    /// The centroid remains at the origin.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::Triclinic};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic =
    ///     Triclinic::from_box_vector([5.0, 5.0, 6.0, 1.5, 1.2, -1.0]);
    ///
    /// let scaled_triclinic = triclinic.scale_length(0.5.try_into()?);
    ///
    /// assert_eq!(scaled_triclinic.lx().get(), 2.5);
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

    /// Construct a scaled triclinic box.
    ///
    /// The resulting triclinic's extents $` L_\mathrm{new} `$ are
    /// the original's $` L `$ scaled by $` v^\frac{1}{3} `$:
    /// ```math
    /// L_\mathrm{new} = v^\frac{1}{3} L
    /// ```
    ///
    /// The centroid remains at the origin.
    ///
    /// # Examples
    ///
    /// ```
    /// use hoomd_geometry::{Scale, shape::Triclinic};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let triclinic =
    ///     Triclinic::from_box_vector([5.0, 5.0, 6.0, 1.5, 1.2, -1.0]);
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
    /// let triclinic = Triclinic::from_box_vector([6.0, 8.0, 10.0, 0.5, 0.0, 0.0]);
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

        let scaled_x =
            self.lx().get() * x + self.xy() * self.ly().get() * y + self.xz() * self.lz().get() * z;
        let scaled_y = self.ly().get() * y + self.yz() * self.lz().get() * z;
        let scaled_z = self.lz().get() * z;

        Cartesian {
            coordinates: [scaled_x, scaled_y, scaled_z],
        }
    }
}

impl MapPoint<Cartesian<3>> for Triclinic {
    fn map_point(&self, point: Cartesian<3>, other: &Self) -> Result<Cartesian<3>, crate::Error> {
        let fractional = self.matmul_inv(point.coordinates);
        let mapped_coords = other.matmul(fractional);
        Ok(Cartesian::from(mapped_coords))
    }
}

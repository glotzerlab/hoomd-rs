// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::array;

use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Cartesian;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::ops::Mul;

use rand::{
    Rng,
    distr::{Distribution, Uniform},
};

use crate::{IsPointInside, Scale, SupportMapping, Volume};

#[serde_as]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]

/// TODO:
pub struct Triclinic {
    /// The extents of each edge of the cuboid. [L_x, L_y, L_z]
    #[serde_as(as = "[_; 3]")]
    pub extents: [PositiveReal; 3],
    /// The tilt factor ij is the ratio of the component of a basis vector a_i in the
    /// jth axis to its extent in the ith direction [xy,xz,yz].
    #[serde_as(as = "[_; 3]")]
    pub tilt_factors: [f64; 3],
}

impl Triclinic {
    /// Returns the box extent in the x-direction (Lx)
    #[inline]
    #[allow(non_snake_case)]
    pub fn Lx(&self) -> PositiveReal {
        self.extents[0]
    }

    /// Returns the box extent in the y-direction (Ly)
    #[inline]
    #[allow(non_snake_case)]
    pub fn Ly(&self) -> PositiveReal {
        self.extents[1]
    }

    /// Returns the box extent in the z-direction (Lz)
    #[inline]
    #[allow(non_snake_case)]
    pub fn Lz(&self) -> PositiveReal {
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

    pub fn with_box_dimensions(box_dimensions: [f64; 6]) -> Self {
        Self {
            extents: [
                box_dimensions[0]
                    .try_into()
                    .expect("Extent Lx must be positive"),
                box_dimensions[1]
                    .try_into()
                    .expect("Extent Ly must be positive"),
                box_dimensions[2]
                    .try_into()
                    .expect("Extent Lz must be positive"),
            ],
            tilt_factors: [box_dimensions[3], box_dimensions[4], box_dimensions[5]],
        }
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
    ///     Triclinic::with_box_dimensions([5.0, 5.0, 6.0, 1.5, 1.2, -1.0]);
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
        if z.abs() >= self.Lz().get() / 2.0 {
            return false;
        };

        if (y - self.yz() * z).abs() >= self.Ly().get() / 2.0 {
            return false;
        };

        if (x - (self.xz() - self.xy() * self.yz()) * z - self.xy() * y).abs()
            >= self.Lx().get() / 2.0
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
    ///     Triclinic::with_box_dimensions([5.0, 5.0, 6.0, 1.5, 1.2, -1.0]);
    ///
    /// let scaled_triclinic = triclinic.scale_length(0.5.try_into()?);
    ///
    /// assert_eq!(scaled_triclinic.edge_lengths[0].get(), 2.5);
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
    ///     Triclinic::with_box_dimensions([5.0, 5.0, 6.0, 1.5, 1.2, -1.0]);
    ///
    /// let scaled_triclinic = triclinic.scale_volume(8.0.try_into()?);
    ///
    /// assert_eq!(scaled_triclinic.edge_lengths[0].get(), 10.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn scale_volume(&self, v: PositiveReal) -> Self {
        let v = v.get().cbrt();
        self.scale_length(v.try_into().expect("v^{1/3} should be a positive real"))
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
    /// let cuboid = Triclinic {
    ///     edge_lengths: [6.0.try_into()?, 8.0.try_into()?],
    /// };
    /// let mut rng = StdRng::seed_from_u64(1);
    ///
    /// let point = cuboid.sample(&mut rng);
    /// assert!(cuboid.is_point_inside(&point));
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
            self.Lx().get() * x + self.xy() * self.Ly().get() * y + self.xz() * self.Lz().get() * z;
        let scaled_y = self.Ly().get() * x + self.yz() * self.Lz().get() * z;
        let scaled_z = self.Lz().get() * z;

        Cartesian {
            coordinates: [scaled_x, scaled_y, scaled_z],
        }
    }
}

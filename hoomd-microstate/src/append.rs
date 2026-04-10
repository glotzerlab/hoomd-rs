// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `AppendMicrostate` for built-in site and boundary types.

use hoomd_geometry::shape::{Hypercuboid, Triclinic};
use hoomd_gsd::hoomd::{AppendError, Dimensions, Frame, HoomdGsdFile};
use hoomd_vector::{Angle, Cartesian, Versor};

use crate::{
    AppendMicrostate, Microstate,
    boundary::{Closed, Periodic},
    property::{OrientedPoint, Point},
};

impl<B, X> AppendMicrostate<B, Point<Cartesian<2>>, X, Closed<Hypercuboid<2>>> for HoomdGsdFile {
    #[inline]
    fn append_microstate(
        &mut self,
        microstate: &Microstate<B, Point<Cartesian<2>>, X, Closed<Hypercuboid<2>>>,
    ) -> Result<Frame<'_>, AppendError> {
        self.append_frame(microstate.step())?
            .configuration_box(microstate.boundary().0.to_gsd_box())?
            .configuration_dimensions(Dimensions::Two)?
            .particles_position(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.position)
                    .map(|p| [p[0], p[1], 0.0].into()),
            )
    }
}

impl<B, X> AppendMicrostate<B, Point<Cartesian<2>>, X, Periodic<Hypercuboid<2>>> for HoomdGsdFile {
    #[inline]
    fn append_microstate(
        &mut self,
        microstate: &Microstate<B, Point<Cartesian<2>>, X, Periodic<Hypercuboid<2>>>,
    ) -> Result<Frame<'_>, AppendError> {
        self.append_frame(microstate.step())?
            .configuration_box(microstate.boundary().shape().to_gsd_box())?
            .configuration_dimensions(Dimensions::Two)?
            .particles_position(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.position)
                    .map(|p| [p[0], p[1], 0.0].into()),
            )
    }
}

impl<B, X> AppendMicrostate<B, OrientedPoint<Cartesian<2>, Angle>, X, Closed<Hypercuboid<2>>>
    for HoomdGsdFile
{
    #[inline]
    fn append_microstate(
        &mut self,
        microstate: &Microstate<B, OrientedPoint<Cartesian<2>, Angle>, X, Closed<Hypercuboid<2>>>,
    ) -> Result<Frame<'_>, AppendError> {
        self.append_frame(microstate.step())?
            .configuration_box(microstate.boundary().0.to_gsd_box())?
            .configuration_dimensions(Dimensions::Two)?
            .particles_position(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.position)
                    .map(|p| [p[0], p[1], 0.0].into()),
            )?
            .particles_orientation(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.orientation.theta)
                    .map(|a| {
                        Versor::from_axis_angle(
                            [0.0, 0.0, 1.0]
                                .try_into()
                                .expect("hard-coded vector can be normalized"),
                            a,
                        )
                    }),
            )
    }
}

impl<B, X> AppendMicrostate<B, OrientedPoint<Cartesian<2>, Angle>, X, Periodic<Hypercuboid<2>>>
    for HoomdGsdFile
{
    #[inline]
    fn append_microstate(
        &mut self,
        microstate: &Microstate<B, OrientedPoint<Cartesian<2>, Angle>, X, Periodic<Hypercuboid<2>>>,
    ) -> Result<Frame<'_>, AppendError> {
        self.append_frame(microstate.step())?
            .configuration_box(microstate.boundary().shape().to_gsd_box())?
            .configuration_dimensions(Dimensions::Two)?
            .particles_position(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.position)
                    .map(|p| [p[0], p[1], 0.0].into()),
            )?
            .particles_orientation(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.orientation.theta)
                    .map(|a| {
                        Versor::from_axis_angle(
                            [0.0, 0.0, 1.0]
                                .try_into()
                                .expect("hard-coded vector can be normalized"),
                            a,
                        )
                    }),
            )
    }
}

impl<B, X> AppendMicrostate<B, Point<Cartesian<3>>, X, Closed<Hypercuboid<3>>> for HoomdGsdFile {
    #[inline]
    fn append_microstate(
        &mut self,
        microstate: &Microstate<B, Point<Cartesian<3>>, X, Closed<Hypercuboid<3>>>,
    ) -> Result<Frame<'_>, AppendError> {
        self.append_frame(microstate.step())?
            .configuration_box(microstate.boundary().0.to_gsd_box())?
            .configuration_dimensions(Dimensions::Three)?
            .particles_position(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.position),
            )
    }
}

impl<B, X> AppendMicrostate<B, Point<Cartesian<3>>, X, Periodic<Hypercuboid<3>>> for HoomdGsdFile {
    #[inline]
    fn append_microstate(
        &mut self,
        microstate: &Microstate<B, Point<Cartesian<3>>, X, Periodic<Hypercuboid<3>>>,
    ) -> Result<Frame<'_>, AppendError> {
        self.append_frame(microstate.step())?
            .configuration_box(microstate.boundary().shape().to_gsd_box())?
            .configuration_dimensions(Dimensions::Three)?
            .particles_position(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.position),
            )
    }
}

impl<B, X> AppendMicrostate<B, Point<Cartesian<3>>, X, Periodic<Triclinic>> for HoomdGsdFile {
    #[inline]
    fn append_microstate(
        &mut self,
        microstate: &Microstate<B, Point<Cartesian<3>>, X, Periodic<Triclinic>>,
    ) -> Result<Frame<'_>, AppendError> {
        self.append_frame(microstate.step())?
            .configuration_box(microstate.boundary().shape().to_gsd_box())?
            .configuration_dimensions(Dimensions::Three)?
            .particles_position(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.position),
            )
    }
}

impl<B, X> AppendMicrostate<B, OrientedPoint<Cartesian<3>, Versor>, X, Closed<Hypercuboid<3>>>
    for HoomdGsdFile
{
    #[inline]
    fn append_microstate(
        &mut self,
        microstate: &Microstate<B, OrientedPoint<Cartesian<3>, Versor>, X, Closed<Hypercuboid<3>>>,
    ) -> Result<Frame<'_>, AppendError> {
        self.append_frame(microstate.step())?
            .configuration_box(microstate.boundary().0.to_gsd_box())?
            .configuration_dimensions(Dimensions::Three)?
            .particles_position(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.position),
            )?
            .particles_orientation(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.orientation),
            )
    }
}

impl<B, X> AppendMicrostate<B, OrientedPoint<Cartesian<3>, Versor>, X, Periodic<Hypercuboid<3>>>
    for HoomdGsdFile
{
    #[inline]
    fn append_microstate(
        &mut self,
        microstate: &Microstate<
            B,
            OrientedPoint<Cartesian<3>, Versor>,
            X,
            Periodic<Hypercuboid<3>>,
        >,
    ) -> Result<Frame<'_>, AppendError> {
        self.append_frame(microstate.step())?
            .configuration_box(microstate.boundary().shape().to_gsd_box())?
            .configuration_dimensions(Dimensions::Three)?
            .particles_position(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.position),
            )?
            .particles_orientation(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.orientation),
            )
    }
}

impl<B, X> AppendMicrostate<B, OrientedPoint<Cartesian<3>, Versor>, X, Periodic<Triclinic>>
    for HoomdGsdFile
{
    #[inline]
    fn append_microstate(
        &mut self,
        microstate: &Microstate<B, OrientedPoint<Cartesian<3>, Versor>, X, Periodic<Triclinic>>,
    ) -> Result<Frame<'_>, AppendError> {
        self.append_frame(microstate.step())?
            .configuration_box(microstate.boundary().shape().to_gsd_box())?
            .configuration_dimensions(Dimensions::Three)?
            .particles_position(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.position),
            )?
            .particles_orientation(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.orientation),
            )
    }
}

#[cfg(test)]
mod test {
    use assert2::assert;
    use std::f64::consts::PI;
    use tempfile::tempdir;

    use super::*;
    use crate::Body;
    use hoomd_geometry::shape::Rectangle;
    use hoomd_gsd::file_layer::{GsdFile, Mode};

    #[test]
    fn point_closed_rectangle_2d() -> anyhow::Result<()> {
        let boundary = Closed(Rectangle {
            edge_lengths: [12.0.try_into()?, 18.0.try_into()?],
        });

        let microstate = Microstate::builder()
            .boundary(boundary)
            .bodies([
                Body::point(Cartesian::from([1.0, 0.0])),
                Body::point(Cartesian::from([-1.0, 2.0])),
            ])
            .step(1234)
            .try_build()?;

        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        hoomd_gsd_file.append_microstate(&microstate)?;

        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path, Mode::Read)?;

        assert!(gsd_file.n_frames() == 1);

        let step = gsd_file.iter_scalars::<u64>(0, "configuration/step")?;
        itertools::assert_equal(step, [1234]);

        let positions = gsd_file.iter_arrays::<f32, 3>(0, "particles/position")?;
        itertools::assert_equal(positions, [[1.0, 0.0, 0.0], [-1.0, 2.0, 0.0]]);

        let dimensions = gsd_file.iter_scalars::<u8>(0, "configuration/dimensions")?;
        itertools::assert_equal(dimensions, [2]);

        let box_ = gsd_file.iter_scalars::<f32>(0, "configuration/box")?;
        itertools::assert_equal(box_, [12.0_f32, 18.0, 0.0, 0.0, 0.0, 0.0]);

        Ok(())
    }

    #[test]
    fn point_periodic_rectangle_2d() -> anyhow::Result<()> {
        let boundary = Periodic::new(
            0.0,
            Rectangle {
                edge_lengths: [12.0.try_into()?, 18.0.try_into()?],
            },
        )?;

        let microstate = Microstate::builder()
            .boundary(boundary)
            .bodies([
                Body::point(Cartesian::from([1.0, 0.0])),
                Body::point(Cartesian::from([-1.0, 2.0])),
            ])
            .step(1234)
            .try_build()?;

        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        hoomd_gsd_file.append_microstate(&microstate)?;

        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path, Mode::Read)?;

        assert!(gsd_file.n_frames() == 1);

        let step = gsd_file.iter_scalars::<u64>(0, "configuration/step")?;
        itertools::assert_equal(step, [1234]);

        let positions = gsd_file.iter_arrays::<f32, 3>(0, "particles/position")?;
        itertools::assert_equal(positions, [[1.0, 0.0, 0.0], [-1.0, 2.0, 0.0]]);

        let dimensions = gsd_file.iter_scalars::<u8>(0, "configuration/dimensions")?;
        itertools::assert_equal(dimensions, [2]);

        let box_ = gsd_file.iter_scalars::<f32>(0, "configuration/box")?;
        itertools::assert_equal(box_, [12.0_f32, 18.0, 0.0, 0.0, 0.0, 0.0]);

        Ok(())
    }

    #[test]
    fn oriented_point_closed_rectangle_2d() -> anyhow::Result<()> {
        let boundary = Closed(Rectangle {
            edge_lengths: [12.0.try_into()?, 18.0.try_into()?],
        });

        let site = OrientedPoint {
            position: Cartesian::from([0.0, 0.0]),
            orientation: Angle::default(),
        };
        let a = OrientedPoint {
            position: Cartesian::from([1.0, 0.0]),
            orientation: Angle::from(PI / 2.0),
        };
        let b = OrientedPoint {
            position: Cartesian::from([-1.0, 2.0]),
            orientation: Angle::from(PI),
        };
        let body_a = Body {
            properties: a,
            sites: [site].into(),
        };
        let body_b = Body {
            properties: b,
            sites: [site].into(),
        };

        let microstate = Microstate::builder()
            .boundary(boundary)
            .bodies([body_a, body_b])
            .step(1234)
            .try_build()?;

        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        hoomd_gsd_file.append_microstate(&microstate)?;

        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path, Mode::Read)?;

        assert!(gsd_file.n_frames() == 1);

        let step = gsd_file.iter_scalars::<u64>(0, "configuration/step")?;
        itertools::assert_equal(step, [1234]);

        let positions = gsd_file.iter_arrays::<f32, 3>(0, "particles/position")?;
        itertools::assert_equal(positions, [[1.0, 0.0, 0.0], [-1.0, 2.0, 0.0]]);

        assert!(
            gsd_file
                .iter_arrays::<f32, 4>(0, "particles/orientation")?
                .count()
                == 2
        );

        let dimensions = gsd_file.iter_scalars::<u8>(0, "configuration/dimensions")?;
        itertools::assert_equal(dimensions, [2]);

        let box_ = gsd_file.iter_scalars::<f32>(0, "configuration/box")?;
        itertools::assert_equal(box_, [12.0_f32, 18.0, 0.0, 0.0, 0.0, 0.0]);

        Ok(())
    }

    #[test]
    fn oriented_point_periodic_rectangle_2d() -> anyhow::Result<()> {
        let boundary = Periodic::new(
            0.0,
            Rectangle {
                edge_lengths: [12.0.try_into()?, 18.0.try_into()?],
            },
        )?;

        let site = OrientedPoint {
            position: Cartesian::from([0.0, 0.0]),
            orientation: Angle::default(),
        };
        let a = OrientedPoint {
            position: Cartesian::from([1.0, 0.0]),
            orientation: Angle::from(PI / 2.0),
        };
        let b = OrientedPoint {
            position: Cartesian::from([-1.0, 2.0]),
            orientation: Angle::from(PI),
        };
        let body_a = Body {
            properties: a,
            sites: [site].into(),
        };
        let body_b = Body {
            properties: b,
            sites: [site].into(),
        };

        let microstate = Microstate::builder()
            .boundary(boundary)
            .bodies([body_a, body_b])
            .step(1234)
            .try_build()?;

        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        hoomd_gsd_file.append_microstate(&microstate)?;

        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path, Mode::Read)?;

        assert!(gsd_file.n_frames() == 1);

        let step = gsd_file.iter_scalars::<u64>(0, "configuration/step")?;
        itertools::assert_equal(step, [1234]);

        let positions = gsd_file.iter_arrays::<f32, 3>(0, "particles/position")?;
        itertools::assert_equal(positions, [[1.0, 0.0, 0.0], [-1.0, 2.0, 0.0]]);

        assert!(
            gsd_file
                .iter_arrays::<f32, 4>(0, "particles/orientation")?
                .count()
                == 2
        );

        let dimensions = gsd_file.iter_scalars::<u8>(0, "configuration/dimensions")?;
        itertools::assert_equal(dimensions, [2]);

        let box_ = gsd_file.iter_scalars::<f32>(0, "configuration/box")?;
        itertools::assert_equal(box_, [12.0_f32, 18.0, 0.0, 0.0, 0.0, 0.0]);

        Ok(())
    }

    #[test]
    fn point_closed_cuboid_3d() -> anyhow::Result<()> {
        let boundary = Closed(Hypercuboid {
            edge_lengths: [12.0.try_into()?, 18.0.try_into()?, 24.0.try_into()?],
        });

        let microstate = Microstate::builder()
            .boundary(boundary)
            .bodies([
                Body::point(Cartesian::from([1.0, 0.0, 4.0])),
                Body::point(Cartesian::from([-1.0, 2.0, -2.0])),
            ])
            .step(1234)
            .try_build()?;

        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        hoomd_gsd_file.append_microstate(&microstate)?;

        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path, Mode::Read)?;

        assert!(gsd_file.n_frames() == 1);

        let step = gsd_file.iter_scalars::<u64>(0, "configuration/step")?;
        itertools::assert_equal(step, [1234]);

        let positions = gsd_file.iter_arrays::<f32, 3>(0, "particles/position")?;
        itertools::assert_equal(positions, [[1.0, 0.0, 4.0], [-1.0, 2.0, -2.0]]);

        let dimensions = gsd_file.iter_scalars::<u8>(0, "configuration/dimensions")?;
        itertools::assert_equal(dimensions, [3]);

        let box_ = gsd_file.iter_scalars::<f32>(0, "configuration/box")?;
        itertools::assert_equal(box_, [12.0_f32, 18.0, 24.0, 0.0, 0.0, 0.0]);

        Ok(())
    }

    #[test]
    fn point_periodic_cuboid_3d() -> anyhow::Result<()> {
        let boundary = Periodic::new(
            0.0,
            Hypercuboid {
                edge_lengths: [12.0.try_into()?, 18.0.try_into()?, 24.0.try_into()?],
            },
        )?;

        let microstate = Microstate::builder()
            .boundary(boundary)
            .bodies([
                Body::point(Cartesian::from([1.0, 0.0, 4.0])),
                Body::point(Cartesian::from([-1.0, 2.0, -2.0])),
            ])
            .step(1234)
            .try_build()?;

        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        hoomd_gsd_file.append_microstate(&microstate)?;

        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path, Mode::Read)?;

        assert!(gsd_file.n_frames() == 1);

        let step = gsd_file.iter_scalars::<u64>(0, "configuration/step")?;
        itertools::assert_equal(step, [1234]);

        let positions = gsd_file.iter_arrays::<f32, 3>(0, "particles/position")?;
        itertools::assert_equal(positions, [[1.0, 0.0, 4.0], [-1.0, 2.0, -2.0]]);

        let dimensions = gsd_file.iter_scalars::<u8>(0, "configuration/dimensions")?;
        itertools::assert_equal(dimensions, [3]);

        let box_ = gsd_file.iter_scalars::<f32>(0, "configuration/box")?;
        itertools::assert_equal(box_, [12.0_f32, 18.0, 24.0, 0.0, 0.0, 0.0]);

        Ok(())
    }

    #[test]
    fn oriented_point_closed_cuboid_3d() -> anyhow::Result<()> {
        let boundary = Closed(Hypercuboid {
            edge_lengths: [12.0.try_into()?, 18.0.try_into()?, 24.0.try_into()?],
        });

        let site = OrientedPoint {
            position: Cartesian::default(),
            orientation: Versor::default(),
        };
        let a = OrientedPoint {
            position: Cartesian::from([1.0, 0.0, 4.0]),
            orientation: Versor::from_axis_angle([1.0, 0.0, 0.0].try_into()?, PI / 2.0),
        };
        let b = OrientedPoint {
            position: Cartesian::from([-1.0, 2.0, -2.0]),
            orientation: Versor::from_axis_angle([0.0, 1.0, 0.0].try_into()?, PI / 2.0),
        };
        let body_a = Body {
            properties: a,
            sites: [site].into(),
        };
        let body_b = Body {
            properties: b,
            sites: [site].into(),
        };

        let microstate = Microstate::builder()
            .boundary(boundary)
            .bodies([body_a, body_b])
            .step(1234)
            .try_build()?;

        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        hoomd_gsd_file.append_microstate(&microstate)?;

        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path, Mode::Read)?;

        assert!(gsd_file.n_frames() == 1);

        let step = gsd_file.iter_scalars::<u64>(0, "configuration/step")?;
        itertools::assert_equal(step, [1234]);

        let positions = gsd_file.iter_arrays::<f32, 3>(0, "particles/position")?;
        itertools::assert_equal(positions, [[1.0, 0.0, 4.0], [-1.0, 2.0, -2.0]]);

        assert!(
            gsd_file
                .iter_arrays::<f32, 4>(0, "particles/orientation")?
                .count()
                == 2
        );

        let dimensions = gsd_file.iter_scalars::<u8>(0, "configuration/dimensions")?;
        itertools::assert_equal(dimensions, [3]);

        let box_ = gsd_file.iter_scalars::<f32>(0, "configuration/box")?;
        itertools::assert_equal(box_, [12.0_f32, 18.0, 24.0, 0.0, 0.0, 0.0]);

        Ok(())
    }

    #[test]
    fn oriented_point_periodic_cuboid_3d() -> anyhow::Result<()> {
        let boundary = Periodic::new(
            0.0,
            Hypercuboid {
                edge_lengths: [12.0.try_into()?, 18.0.try_into()?, 24.0.try_into()?],
            },
        )?;

        let site = OrientedPoint {
            position: Cartesian::from([0.0, 0.0, 0.0]),
            orientation: Versor::default(),
        };
        let a = OrientedPoint {
            position: Cartesian::from([1.0, 0.0, 4.0]),
            orientation: Versor::from_axis_angle([1.0, 0.0, 0.0].try_into()?, PI / 2.0),
        };
        let b = OrientedPoint {
            position: Cartesian::from([-1.0, 2.0, -2.0]),
            orientation: Versor::from_axis_angle([0.0, 1.0, 0.0].try_into()?, PI / 2.0),
        };
        let body_a = Body {
            properties: a,
            sites: [site].into(),
        };
        let body_b = Body {
            properties: b,
            sites: [site].into(),
        };

        let microstate = Microstate::builder()
            .boundary(boundary)
            .bodies([body_a, body_b])
            .step(1234)
            .try_build()?;

        let tmp_dir = tempdir()?;
        let path = tmp_dir.path().join("test.gsd");
        let mut hoomd_gsd_file = HoomdGsdFile::create(path.clone())?;
        hoomd_gsd_file.append_microstate(&microstate)?;

        drop(hoomd_gsd_file);

        let gsd_file = GsdFile::open(path, Mode::Read)?;

        assert!(gsd_file.n_frames() == 1);

        let step = gsd_file.iter_scalars::<u64>(0, "configuration/step")?;
        itertools::assert_equal(step, [1234]);

        let positions = gsd_file.iter_arrays::<f32, 3>(0, "particles/position")?;
        itertools::assert_equal(positions, [[1.0, 0.0, 4.0], [-1.0, 2.0, -2.0]]);

        assert!(
            gsd_file
                .iter_arrays::<f32, 4>(0, "particles/orientation")?
                .count()
                == 2
        );

        let dimensions = gsd_file.iter_scalars::<u8>(0, "configuration/dimensions")?;
        itertools::assert_equal(dimensions, [3]);

        let box_ = gsd_file.iter_scalars::<f32>(0, "configuration/box")?;
        itertools::assert_equal(box_, [12.0_f32, 18.0, 24.0, 0.0, 0.0, 0.0]);

        Ok(())
    }
}

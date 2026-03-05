// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `AppendMicrostate` for built-in site and boundary types.

use hoomd_geometry::shape::Hypercuboid;
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

use hoomd_geometry::shape::{Hypercuboid, Hypersphere};
use hoomd_gsd::hoomd::{AppendError, Frame, HoomdGsdFile};
use hoomd_vector::Cartesian;

use crate::{
    AppendMicrostate, Microstate,
    boundary::{Closed, Periodic},
    property::Point,
};

impl<B, X> AppendMicrostate<B, Point<Cartesian<2>>, X, Closed<Hypercuboid<2>>> for HoomdGsdFile {
    #[inline]
    fn append_microstate(
        &mut self,
        microstate: &Microstate<B, Point<Cartesian<2>>, X, Closed<Hypercuboid<2>>>,
    ) -> Result<Frame<'_>, AppendError> {
        self.append_frame(microstate.step())?
            .configuration_box(microstate.boundary().0.to_gsd_box())?
            .configuration_dimensions(2)?
            .particles_position(
                microstate
                    .iter_sites_tag_order()
                    .map(|s| s.properties.position)
                    .map(|p| [p[0], p[1], 0.0].into()),
            )
    }
}

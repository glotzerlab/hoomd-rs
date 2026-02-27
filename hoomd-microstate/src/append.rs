use hoomd_geometry::shape::{Hypercuboid, Hypersphere};
use hoomd_gsd::hoomd::{HoomdGsdFile, AppendError, Frame};
use hoomd_vector::Cartesian;

use crate::{AppendMicrostate, Microstate, boundary::{Closed, Periodic}, property::Point};


impl<B, X> AppendMicrostate<B, Point<Cartesian<2>>, X, Closed<Hypercuboid<2>>> for HoomdGsdFile {
    #[inline]
    fn append_microstate(&mut self, microstate: &Microstate<B, Point<Cartesian<2>>, X, Closed<Hypercuboid<2>>>) -> Result<Frame<'_>, AppendError> {
        let box_values = [microstate.boundary().0.edge_lengths[0].get(),
            microstate.boundary().0.edge_lengths[1].get(),
            0.0,
            0.0,
            0.0,
            0.0,
        ];

        // TODO: Sites in tag order!
        
        self.append_frame(microstate.step())?
            .configuration_box(box_values)?
            .configuration_dimensions(2)?
            .particles_position(microstate.sites().iter().map(|s| [s.properties.position[0], s.properties.position[1], 0.0].into()))
    }
}

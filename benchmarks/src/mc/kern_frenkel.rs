// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use hoomd_spatial::{PointUpdate, PointsInBall, VecCell};
use hoomd_vector::{Cartesian, InnerProduct, Rotate, Vector, Versor};
use hoomd_simulation::{macrostate::Isothermal, Simulation};
use hoomd_microstate::{boundary::{GenerateGhosts, Periodic}, property::{OrientedPoint, Point}, Microstate, MicrostateBuilder, SiteKey};
use hoomd_geometry::shape::Hypercuboid;
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_interaction::{pairwise::{angular_mask::Patch, AngularMask, Anisotropic, AnisotropicEnergy, Boxcar, Isotropic, IsotropicEnergy}, CutoffPair};

struct Interaction<V> {
    angular_mask: AngularMask<Boxcar, V>,
}

pub struct KernFrenkel<X> {
    microstate: Microstate<OrientedPoint<Cartesian<3>, Versor>, OrientedPoint<Cartesian<3>, Versor>, X, Periodic<Hypercuboid<3>>>,
    translate_sweep: Sweep<Translate<Cartesian<3>>>,
    hamiltonian: CutoffPair<Anisotropic<Interaction<Cartesian<3>>>>,
    macrostate: Isothermal,
}

impl<V, R> AnisotropicEnergy<V, R> for Interaction<V> where
V: Vector + InnerProduct,
R: Rotate<V> + Into<R::Matrix> + Copy,
{
    fn energy(&self, r_ij: &V, o_ij: &R) -> f64 {
        if r_ij.norm_squared() <= 1.0 {
            f64::INFINITY
        } else {
            self.angular_mask.energy(r_ij, o_ij)
        }
    }
}

// TODO: need a hard shape + energy interaction so these benchmarks don't need
// custom ones. Will also be useful in a patchy particle tutorial.

impl<X> Simulation for KernFrenkel<X> where
X: PointsInBall<Cartesian<3>, SiteKey> + PointUpdate<Cartesian<3>, SiteKey>,
Periodic<Hypercuboid<3>>: GenerateGhosts<Point<Cartesian<3>>>,
{
    fn advance(&mut self) -> anyhow::Result<()> {
        self.translate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
        );
        self.microstate.increment_step();

        // TODO: rotation moves

        Ok(())
    }

    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

impl KernFrenkel<VecCell<SiteKey, 3>> where
Periodic<Hypercuboid<3>>: GenerateGhosts<Point<Cartesian<3>>>,
{
    pub fn with_microstate<X>(microstate: &Microstate<OrientedPoint<Cartesian<3>, Versor>, OrientedPoint<Cartesian<3>, Versor>, X, Periodic<Hypercuboid<3>>>) -> anyhow::Result<Self> {
        let maximum_interaction_range = 1.5;

        let translate = Translate::with_maximum_distance(0.1.try_into()?);
        let translate_sweep = Sweep(translate);

        let boxcar = Boxcar {
            epsilon: -0.2,
            left: 0.0,
            right: 1.5,
        };
        let masks = [Patch {
            director: [0.0, 0.0, 1.0].try_into()?,
            cos_delta: (0.5f64).cos(),
        }];
        let angular_mask = AngularMask::new(boxcar, masks);
    
        let hamiltonian = CutoffPair {
            r_cut: maximum_interaction_range,
            evaluator: Anisotropic(Interaction { angular_mask, }),
        };    
    
        let cell_list = VecCell::new(maximum_interaction_range, 1);
        let boundary = Periodic::new(maximum_interaction_range,
            microstate.boundary().shape().clone())?;
        let microstate = MicrostateBuilder::with_spatial_data_and_boundary(cell_list, boundary)
            .bodies(microstate.bodies().iter().map(|b| b.item.clone()))
            .try_build()?;

        Ok(Self {
            microstate,
            translate_sweep,
            hamiltonian,
            macrostate: Isothermal { temperature: 1.0 },
        })
    }
}

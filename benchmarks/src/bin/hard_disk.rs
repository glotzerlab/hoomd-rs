use log::info;

use hoomd_vector::Cartesian;
use hoomd_simulation::{macrostate::Isothermal, Simulation};
use hoomd_spatial::CellList;
use hoomd_microstate::{boundary::Periodic, property::Point, Microstate, SiteKey};
use hoomd_geometry::shape::Rectangle;
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_interaction::{CutoffPairOverlap, pairwise::AlwaysTrue};

type PositionVector = Cartesian<2>;
type BodyProperties = Point<PositionVector>;
type SiteProperties = Point<PositionVector>;

struct HardDisk {
    microstate: Microstate<BodyProperties, SiteProperties, CellList<SiteKey, 2>, Periodic<Rectangle>>,
    translate_sweep: Sweep<Translate<PositionVector>>,
    hamiltonian: CutoffPairOverlap<AlwaysTrue>,
    macrostate: Isothermal,
}

impl Simulation for HardDisk {
    fn advance(&mut self) -> anyhow::Result<()> {
        self.translate_sweep.apply(
            &mut self.microstate,
            &self.hamiltonian,
            &self.macrostate,
        );
        self.microstate.increment_step();

        Ok(())
    }

    fn step(&self) -> u64 {
        self.microstate.step()
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let n = 4096;
    let number_density = 0.8;
    let sigma = 1.0;
    
    info!("Hard disk benchmark: {} disks at number density {}", n, number_density);

    let microstate = benchmarks::place_hard_hyperspheres::<BodyProperties, SiteProperties, _>(n, number_density)?;

    let translate = Translate::with_maximum_distance((sigma * 0.1).try_into()?);
    let translate_sweep = Sweep(translate);

    let hamiltonian = CutoffPairOverlap {
        r_cut: sigma,
        evaluator: AlwaysTrue,
    };    

    let mut hard_disk = HardDisk {
        microstate,
        translate_sweep,
        hamiltonian,
        macrostate: Isothermal { temperature: 1.0 },
    };

    benchmarks::benchmark(&mut hard_disk, 1000, 1000, 10)?;

    Ok(())
}

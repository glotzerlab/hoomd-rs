# The Simulation Model

## Initialize from a State Point

The file `src/model.rs` implements the simulation model similarly to all the
previous tutorials (see [Applying Interactions] for a full explanation). One
slight difference is in the initialization. Instead of hard-coding simulation
parameters, the simulation in the workflow template initializes using parameters
from the state point:
```rust,ignore
pub fn new(state_point: StatePoint) -> anyhow::Result<Self> {
    let macrostate = Isothermal {
        temperature: state_point.temperature,
    };

    let hamiltonian = PairwiseCutoff(Isotropic {
        interaction: LennardJones {
            epsilon: state_point.epsilon,
            sigma: state_point.sigma,
        },
        r_cut: 2.5 * state_point.sigma,
    });

    let initial_box_volume = state_point.n as f64 / INITIAL_NUMBER_DENSITY;
    let initial_box_edge_length = initial_box_volume.cbrt();
    let cuboid = Cuboid::with_equal_edges(initial_box_edge_length.try_into()?);
    let periodic_cuboid = Periodic::new(hamiltonian.maximum_interaction_range(), cuboid)?;

    let vec_cell = VecCell::builder()
        .nominal_search_radius(hamiltonian.maximum_interaction_range().try_into()?)
        .build();
    let microstate = Microstate::builder()
        .seed(state_point.replicate)
        .boundary(periodic_cuboid)
        .spatial_data(vec_cell)
        .try_build()?;

    // ...
```

> [!TIP]
> Use a `replicate` state point field as the random number seed (as shown here)
> to execute many different realizations of the same state point parameters.

## Constant Parameters

In the [row]/[signac] workflow model, it is important that every simulation can
be constructed only as a function of the state point. At the same time, there
are usually some parameters that are a fixed part of the simulation protocol.
Define those as constants:
```rust,ignore
const INITIAL_NUMBER_DENSITY: f64 = 0.2;
const INITIAL_MAXIMUM_DISTANCE: f64 = 0.1;
const RELAX_STEPS: u64 = 10_000;
```

## Serialization

As with the state point, notice that all types in `model.rs` also
include `#[derive(Serialize, Deserialize)]`. On the next page, the
`simulate_one` method will make use of these traits.


> [!TIP]
> To implement your own simulation model, replace the struct fields and
> method implementations.

[row]: https://row.readthedocs.io
[signac]: https://signac.readthedocs.io
[Applying Interactions]: ../mc-tutorial/applying-interactions.html#the-simulation-model

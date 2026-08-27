# The State Point

The file `src/state_point.rs` defines a struct that holds the [signac] state
point. The template executes a Lennard-Jones fluid simulation. The state point
sets the number of beads, the potential parameters, the temperature (in units of
energy), the number density, and the replicate index.

```rust,ignore
use serde::{Deserialize, Serialize};

/// Model parameters that describe a single simulation.
#[derive(Serialize, Deserialize)]
pub struct StatePoint {
    pub n: usize,
    pub epsilon: f64,
    pub sigma: f64,
    pub temperature: f64,
    pub number_density: f64,
    pub replicate: u32,
}
```

The `#[derive(Serialize, Deserialize)]` line calls [serde] macros that
automatically implement `Serialize` and `Deserialize` for the struct.
`bin/populate_workspace.rs` uses the `Serialize` trait indirectly through
[hoomd-workspace]'s `add` method to create several state point directories in
the workspace:
```rust,ignore
use hoomd_workflow::StatePoint;

fn main() -> anyhow::Result<()> {
    hoomd_workspace::add(&StatePoint {
        n: 1_000,
        epsilon: 1.0,
        sigma: 1.0,
        temperature: 1.0,
        number_density: 0.4,
        replicate: 0,
    })?;

    hoomd_workspace::add(&StatePoint {
        n: 1_000,
        epsilon: 1.0,
        sigma: 1.0,
        temperature: 1.0,
        number_density: 0.4,
        replicate: 1,
    })?;

    hoomd_workspace::add(&StatePoint {
        n: 1_000,
        epsilon: 1.0,
        sigma: 1.0,
        temperature: 0.7,
        number_density: 0.4,
        replicate: 0,
    })?;

    Ok(())
}
```


To execute it, run:
```shell
$ target/release/populate_workspace
```

`populate_workspace` wrote these state points to JSON files in the `workspace`
directory. Explor the `workspace` directory and inspect the `signac_statepoint.json`
files you find. They should match the structs written by `populate_workspace.rs`.
The workspace created by `hoomd-workspace` is compatible with the [signac] Python
framework.

In `simulate.rs` (explained later), the workflow reads the state point using
the `state_point` function in [hoomd-workspace], which uses the `Deserialize`
trait to read the JSON file and parse the struct fields:
```rust,ignore
let state_point: StatePoint = hoomd_workspace::state_point(directory)
    .context("could not read state point")?
    .ok_or(anyhow!("state point not found"))?;
```

> [!TIP]
> To implement your own simulation model, replace the struct fields with
> ones that match your state point.

[signac]: https://signac.readthedocs.io
[serde]: https://docs.rs/serde
[hoomd-workspace]: ../../api/hoomd_workspace/index.html

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

Signac stores these state points in JSON files in the `workspace` directory.
To populate that directory with three example state points, execute:
```shell
$ python populate_workspace.py
```

You might be wondering what the `#[derive(Serialize, Deserialize)]` is for.
In `simulate.rs` (explained later), the workflow reads the state point using
[serde_json]:
```rust,ignore
let state_point_bytes = fs::read("signac_statepoint.json")?;
let state_point: StatePoint = serde_json::from_slice(&state_point_bytes)?;
```

`#[derive(Serialize, Deserialize)]` invokes a macro that *automatically*
implements the `Serialize` and `Deserialize` traits for our type.

> [!TIP]
> To implement your own simulation model, replace the struct fields with
> ones that match your state point.

[signac]: https://signac.readthedocs.io
[serde_json]: https://docs.rs/serde_json

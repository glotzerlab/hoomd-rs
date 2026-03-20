# NVE Simulation of a Lennard-Jones Fluid

<script type="module">
import init from './nve-lj-fluid.js'
{{#include ../../scripts/init-wasm-canvas.js}}
</script>
{{#include ../../scripts/canvas.html}}

## Overview

This tutorial demonstrates how to set up and run a **classical microcanonical (NVE) simulation** of a Lennard-Jones (LJ) fluid using the molecular dynamics (MD) modules of `hoomd-rs`.

We will:

- Create a cubic periodic box filled with Lennard-Jones particles
- Use the velocity Verlet integrator (`ConstantVolume`)
- Apply the Bussi thermostat during equilibration (NVT phase)
- Switch to pure NVE integration after equilibration
- Include long-range correction (LRC) for the truncated LJ potential
- Remove center-of-mass momentum and angular momentum drift
- Print basic thermodynamic quantities every 10,000 steps during production

* Objective: Learn how to combine integrators, thermostats, force computation, and momentum correction in an MD workflow.
* File: `hoomd-rs/examples/md-tutorial/nve-lj-fluid.rs`
* Run (interactively – if Bevy visualization is enabled):
  ```shell
  cargo run --release --features "bevy" --example nve-lj-fluid
  ```

## Dynamic Bodies and Sites
Following the **Applying-Interactions** tutorial in the `mc-tutorial`, we again use point particles in the three-dimension with each **body** now has extended properties to perform MD simulation, including the momentum, mass and net force. Specifically, that means
each **body** has `DynamicsPoint<Cartesian<3>>` for its **body properties** type (`B`),
and a single **site** at the origin (*in the body reference frame*) which has
`Point<Cartesian<3>>` for its **site properties** (`S`) type.
### Note
- `DynamicsPoint<Cartesian<3>>`: contains position, momentum, net force, and mass
- `Point<Cartesian<3>>`: just position (single site at the origin of the body frame)

The `DynamicsPoint` type is provided by [`hoomd-microstate`] and is the most common choice when doing classical MD with point particles or rigid bodies without internal degrees of freedom.

## Use Declarations

```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:use}}
```

## The Simulation Model

Here is the type that holds the simulation model:
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:simulation_struct}}
```

## Construct the Simulation Model

The `new()` method constructs a new simulation model:
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:simulation_new}}
```

### Parameters

Assign all the model parameters in one code block so that they are easy to modify:
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:parameters}}
```

Here, we choose to use reduced Lennard-Jones units: $` \epsilon=1 `$, $`\sigma=1`$, $`m = 1`$ => temperature is in units of $`k_BT/\epsilon`$, time in units of $`\sigma \sqrt{(m/\epsilon)}`$. 

We will initialize a system with `n x n x n` particles and then run  `eq_steps` in the NVT ensemble to equilibrate the system at temperature $`T^*=`$ `temperature_lj` at the fixed volume density $`\rho^*=`$ `density_lj`, followed by a production run in the NVE ensemble. We use a time step of $`\delta t ^*= `$ `dt_lj` and a temperature damping time constant of $`\tau^*=`$ `tau_lj` for the thermostating. The LJ potential is truncated at the `r_cut_lj`.

### Boundary and Spatial Data Structure
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:boundary}}
```
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:spatial_data}}
```
> [!IMPORTANT]
> The nominal search radius passed to `VecCell::builder()` must be at least as large as the largest cutoff used in any pairwise interaction.
Here we use `r_cut`, so we set the search radius accordingly. In *hoomd-rs*, it is *YOUR responsibility* to determine the appropriate `r_cut`.

### Lennard-Jones Potential
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:pair_force}}
```
We use the 12-6 LJ potential truncated at `r_cut` and wrapped in `PairwiseCutoff` and `Rigid`. 

Although there are no rigid bodies here, `Rigid` is the standard wrapper when sites belong to bodies and when net force calculation on each **body** is needed. `Rigid` type sums over all forces acting on **sites** that constitutes the **body**.

### Long-Range Correction (LRC)
Because we truncate the LJ potential, we add the standard mean-field long-range correction to the potential energy per particle:

The correction can be calculated as:
```math
U_\mathrm{LRC} = \frac{1}{2} 4 \pi \rho \int_{r_\mathrm{cut}}^{\infty} r^2 U_\mathrm{LJ}(r) dr
```
where $`\rho`$ is the number density `density`.

```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:energy_lrc}}
```

### Initialize Positions
We initialize the particle position as a simple cubic crystal with the lattice constant of `space`, which will result in the number density `density`.

```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:particle_positions}}
```

### Initialize Momentums
We first draw random momentums from the Maxwell–Boltzmann distribution at temperature `kt`, then remove center-of-mass linear and angular momentum to avoid net drift of the system.
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:particle_momenta}}
```

### Integrator and Thermostat
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:integrator}}
```
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:thermostat}}
```
We use the Bussi (stochastic velocity rescaling) thermostat during the equilibration phase.

### Simulation Phase Tracking
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:phase}}
```
We define two phases:

- `Equilibrate`: NVT simulation that equilibrates the system.
- `SampleNVE`: production NVE run.

## Implement Simulation
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:impl_simulation}}
```

### Advance the simulation
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:advance}}
```

### NVT Equilibration Phase
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:nvt}}
```
We run velocity Verlet with the thermostat for `eq_step` steps, then switch to NVE.

#### First-Half Integration
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:first_half_integration}}
```

#### Net Force Update
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:update_force}}
```

#### Second-Half Integration
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:second_half_integration}}
```

#### Simulation Phase Shifting
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:state_transition}}
```

### NVE Production Phase
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:nve}}
```
In the production phase we:

- Disable the thermostat (pass `NoThermostat`)
- Use a dummy macrostate `Isoenergy` (not used in NVE)
- Print temperature and potential energy every 10,000 steps

### Compute Thermodynamic Properties
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:properties}}
```
We calculate:

- Potential energy per particle including long-range correction.
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:potetial_energy}}
```
- Instantaneous temperature from translational kinetic energy.
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:current_temeprature}}
```

## Execute the Simulation in Batch Mode

### Implement `main()`
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:main}}
```

### Create a GSD Trajectory
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:create_gsd}}
```

### Advance the Simulation
We run a fixed 100,000 total steps of MD simulation.
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:advance}}
```

### Write Frames to the GSD File
Call `append_microstate` to write to the GSD file for every 5,000 steps.
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:append_microstate}}
```


Should see output similar to:
```
Isotherm preparation finished at step 50000.
NVE, Step 10000, kT 0.8747741585945321, Potential energy (w/ LRC) per particle -5.481555101132949
NVE, Step 20000, kT 0.890226773983914, Potential energy (w/ LRC) per particle -5.50325224785257
NVE, Step 30000, kT 0.8838402378487593, Potential energy (w/ LRC) per particle -5.493167294987841
NVE, Step 40000, kT 0.8942898500494086, Potential energy (w/ LRC) per particle -5.509090715545137
...
```

## Conclusion
You have now seen how to:

- Initialize a dense Lennard-Jones fluid on an simple cubic lattice
- Use velocity Verlet integration (`ConstantVolume`)
- Equilibrate in the NVT ensemble with the Bussi thermostat
- Switch to pure NVE production
- Apply standard momentum removal
- Compute and print basic thermodynamic observables
- Include the long-range correction for truncated LJ potentials

Navigate to the top of the page and refresh to see the simulation in action (if Bevy visualization is enabled).

## Reference Resources
Benchmark results can be found on the NIST website for comparison.

[NIST NVE Lennard-Jones fluid simulation](https://mmlapps.nist.gov/srs/LJ_PURE/md.htm)

## Complete Code
```rust,ignore
{{#rustdoc_include ../../../examples/md-tutorial/nve-lj-fluid.rs:all}}


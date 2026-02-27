# Patchy Particle Self-Assembly

<script type="module">
import init from 'https://glotzerlab.github.io/hoomd-rs/mc-tutorial/patchy-particle-self-assembly.js'
{{#include ../../scripts/init-wasm-canvas.js}}
</script>
{{#include ../../scripts/canvas.html}}

## Overview

There are many ways you can model **anisotropic bodies** in *hoomd-rs*.
This tutorial shows you how to represents **sites** that have a hard core
and two attractive patches. This system self-assembles the kagome structure
([10.1039/C0SM01494J]) using the optimal parameters given in [10.1039/D2SM01593E].
The tutorial also uses a temperature ramp to improve the quality of the
self-assembled structures and logs the temperature and potential energy
for further analysis.

[10.1039/C0SM01494J]: http://doi.org/10.1039/C0SM01494J
[10.1039/D2SM01593E]: http://doi.org/10.1039/D2SM01593E

* Objectives:
  * Explain how to model systems of hard core particles with attractive patches.
  * Describe how to vary system parameters as a function of step.
  * Show how to log the system potential energy and temperature as a function of simulation step.
  * Demonstrate the self-assembly of patchy particles.
* File: `hoomd-rs/examples/mc-tutorial/patchy-particle-self-assembly.rs`
* Run (interactively):
  ```shell
  cargo run --release --features "bevy" --example patchy-particle-self-assembly
  ```
* Run (in batch mode):
  ```shell
  cargo run --release --example patchy-particle-self-assembly
  ```

## Use Declarations

```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:use}}
```

## Type Aliases

Create type aliases for your model's *vector*, *body properties*, and *site
properties* types so that you don't need to repeat the full generic type
names throughout the code:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:type_aliases}}
```

The **sites** are in this tutorial are represented by disks with both
position and orientation. Therefore, use `OrientedPoint` for both the **body**
and **site** properties. Use `Angle` to represent rotations in 2D.

## Site Pair Interaction

The `SitePairInteraction` type combines a hard disk overlap with attractive
patches:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:site_pair_interaction}}
```
Use the provided derive macros to implement the traits `MaximumInteractionRange`
and `SitePairEnergy` so that this type may be used with `PairwiseCutoff`.

The [Hamiltonian](#hamiltonian) section explains each term in more detail.

## Construct the Simulation Model

The `new()` method constructs a new simulation model:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:simulation_new}}
```

### Parameters

Assign all the model parameters in one code block:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:parameters}}
```

`initial_packing_fraction` is the volume of the disks divided by the volume
of the simulation boundary in the initial state. Choose this value so
that disks can be placed easily in the microstate. During the `Initialize`
phase, the microstate will be compressed until it reaches the packing
fraction `target_packing_fraction`. `n_disks` is the number of disks to add,
`maximum_distance` is the largest distance a translation trial move can take
(initially), and `maximum_rotation`controls the size of the rotation trial moves
(initially). `sigma` is the disk diameter, `patch_interaction_range` is largest
distance at which the attractive interaction applies, `patch_half_angle` is
the half open angle of the patch, and `patch_energy` is the potential energy
of a pair of particles when their patches align. `initial_temperature` sets
the temperature at step 0, `final_temperature` sets the temperature at step
`ramp_steps`, and `macrostate` holds the current temperature set point (in units
of energy).

### Hamiltonian

#### Hard Disk Term

As in [Hard Disk Self-Assembly], use `HardSphere` to model the hard cores
placed at each site:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:hard_disk}}
```

[Hard Disk Self-Assembly]: hard-disk-self-assembly.md

#### Patch Term

Use `AngularMask` combined with `Boxcar` to compute the patch interactions
detailed in [10.1039/C0SM01494J]. The `Boxcar` isotropic potential places an
attractive well at all distances less than `patch_interaction_range`.
`AngularMask` modulates that potential with oriented patches.
Place two patches, one directed up and one directed down in the site's local
frame.
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:patch}}
```

#### Combined Pairwise Potential

The full Hamiltonian of the system is the sum of these two pairwise interaction terms.
You could use `hamiltonian = Hamiltonian { hard_disk, angular_mask }`
to add the terms (as demonstrated in [Applying Interactions]), but it is slightly
faster to use one `PairwiseCutoff` on a type that holds multiple
site pair interactions:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:hamiltonian}}
```

The former performs two loops over nearby sites and adds the results together
$`\left( \sum U^A_{ij} + \sum U^B_{ij} \right)`$ while the latter performs one loop and adds
terms in the loop body $`\left( \sum U^A_{ij} + U^B_{ij} \right)`$.

> [!TIP]
> Always list hard shape potentials first in struct. If the hard shapes
> overlap, *hoomd-rs* can assume that the move will be rejected and skip the
> computation of the following terms.

[Applying Interactions]: applying-interactions.md

#### Overlap Penalty Hamiltonian

This example uses `overlap_penalty_hamiltonian` when inserting disks randomly
and compressing the system to the target packing fraction. Use only the hard core
term to allow the system to arrange randomly without being hindered by the patches:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:compress_hamiltonian}}
```

### More Initialization

See the [Hard Ellipse Self-Assembly] tutorial for a complete explanation of
remaining initialization code (see also the [complete code] below).

[Hard Ellipse Self-Assembly]: hard-ellipse-self-assembly.md
[complete code]: #complete-code

## Implement `Simulation`

To implement the temperature ramp, modify `macrostate` at the start of
`advance()`.  This code implements a linear ramp from `initial_temperature` to
`final_temperature` as a function of the current simulation step:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:simulation}}
```

The remainder of the simulation code is identical to that in the
[Hard Ellipse Self-Assembly] tutorial (see also the [complete code] below).

## Log the Potential Energy and Temperature

When running simulations in batch mode, you often want to write a **log** file
for later analysis. In this system of patchy particles, the total system
potential energy indicates how many bonds have formed and therefore what
fraction of the system is part of the kagome structure.

### Log Record

Define a struct that records all quantities of interest:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:log_record}}
```

This tutorial writes the log to a [parquet] file (you may choose any file format
you like in your own projects). Parquet is a binary, column-oriented format
that preserves the full precision of every record value and can be written and read
efficiently. It is supported by R, pandas, MATLAB, and many other tools.
`#[derive(ParquetRecordWriter)]` generates code that writes each field of the
struct to a column with the same name.

[parquet]: https://parquet.apache.org/

### `main()`

The `main()` function executes when your binary in batch mode:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:main}}
```

> [!NOTE]
> This `main()` function runs in batch mode. There is a different `main()` (not
> shown here) used in the interactive example.

### Open the Log File

`ParquetLogger` from the `hoomd_utility` crate helps you write [parquet] files.
Use it to create a new parquet file:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:log_open}}
```

### Simulation Steps

To run the simulation, construct the `PatchyParticleSelfAssembly` simulation model.
Then call `advance()` many times:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:run_simulation}}
```

Write the sites to a GSD file periodically so that you can inspect the results
of the simulation.

### Write Log Records

On desired simulation steps, construct a `LogRecord` and call `parquet_logger.log`
to add it to the log file:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:write_log}}
```

> [!NOTE]
> Log records will not immediately appear in the file. `ParquetLogger` buffers
> log records in memory and writes them in batches.

### Exit `main()`

`ParquetWriter` writes all buffered log entries and closes the file when it
is dropped, which occurs automatically when `main()` returns:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:exit}}
```

## Conclusion

This tutorial showed you how to perform patchy particle self-assembly
simulations using a shape overlap potential with attractive patches.

Navigate to the top of the page and refresh to see the simulation in action
again. Notice that the disks quickly form random chains and clusters. Over
time, hexagons will appear and the kagome structure will begin to grow. Run
the simulation long enough, and the system will equilibrate to a single large
crystal as shown in [10.1039/D2SM01593E].

You can also run the example in batch mode and then open
the generated `trajectory.gsd` in [Ovito] or another visualization tool:
```shell
cargo run --release --example patchy-particle-self-assembly
```

Open the log file `patchy-particle-self-assembly.parquet` and plot it using the
tool of your choice. It will look something like this:
<script src="https://cdn.jsdelivr.net/npm/vega@6"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@6"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@7"></script>

<div id="vis" style="width: 100%"></div>

<script type="text/javascript">
  var spec = {
    $schema: 'https://vega.github.io/schema/vega-lite/v6.json',
    description: 'Potential energy versus step for patchy particle self-assembly.',
    data: {"name": "data", "url": "patchy-particle-self-assembly.csv"} ,
    "vconcat": [{
    width: "container",
    height: 200,
    "layer": [{
    mark: { type: 'line', tooltip: true},
    encoding: {
      x: {field: 'step', type: 'quantitative'},
      y: {field: 'potential_energy', type: 'quantitative'},
    }},
    {
    mark: { type: 'line', tooltip: true},
    encoding: {
      x: {field: 'step', type: 'quantitative'},
      y: {field: 'no_ramp', type: 'quantitative'},
    }},

    ]},
    {
    width: "container",
    height: 200,
    mark: { type: 'line', tooltip: true},
    encoding: {
      x: {field: 'step', type: 'quantitative'},
      y: {field: 'temperature', type: 'quantitative'},
    }}
    ],
  };
  var tooltipOptions = {
    theme: 'dark'
  };
  vegaEmbed('#vis', spec, {theme: 'dark', actions: false, tooltip: tooltipOptions} )
    .then(function (result) {
      // Access the Vega view instance (https://vega.github.io/vega/docs/api/view/) as result.view
    })
    .catch(console.error);
</script>


[Ovito]: https://www.ovito.org/

## Complete Code

```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:all}}

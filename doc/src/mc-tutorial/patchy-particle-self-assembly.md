# Patchy Particle Self-Assembly

<script type="module">
import init from 'https://glotzerlab.github.io/hoomd-rs/mc-tutorial/patchy-particle-self-assembly.js'
{{#include ../../scripts/init-wasm-canvas.js}}
</script>
{{#include ../../scripts/canvas.html}}

## Overview

There are many ways you can model **anisotropic bodies** in *hoomd-rs*.
This tutorial shows you how to represents **sites** that have a hard core
and two attractive patches. This system self-assembles the Kagome structure
([10.1039/C0SM01494J]) using the optimal parameters given in [10.1039/D2SM01593E].

[10.1039/C0SM01494J]: http://doi.org/10.1039/C0SM01494J
[10.1039/D2SM01593E]: http://doi.org/10.1039/D2SM01593E

* Objectives:
  * Explain how to model systems of hard core particles with attractive patches.
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
properties* types so that you don't need to repeat the full nested generic type
names throughout the code:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:type_aliases}}
```

The **sites** are in this tutorial are represented by disks with both
position and orientation. Therefore, use `OrientedPoint` for both the **body**
and **site** properties. Use `Angle` to represent rotations in 2D.

## Construct the Simulation Model

The `new()` method constructs a new simulation model:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:simulation_new}}
```

### Parameters

Assign all the model parameters in one code block so that they are easy to modify:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:parameters}}
```

`initial_packing_fraction` is the volume of the disks divided by the volume
of the simulation boundary in the initial state. Choose this value so
that disks can be placed easily in the microstate. During the `Initialize`
phase, the microstate will be compressed until it reaches the packing
fraction `target_packing_fraction`. `n_disks` is the number of disks to add,
`maximum_distance` is the largest distance a translation trial move can take,
and `maximum_rotation`controls the size of the rotation trial moves. `sigma`
is the disk diameter, `patch_interaction_range` is largest distance at which
the attractive interaction applies, `patch_half_angle` is the half open angle
of the patch, `patch_energy` is the potential energy of a pair of particles when
their patches align, and `macrostate` holds the temperature set point (in units
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
Place two patches, one facing up and one facing down in the site's local
frame.
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:patch}}
```

#### Combined Pairwise Potential

The full Hamiltonian of the system is the sum of these two pairwise interaction terms.
You could use `hamiltonian = (PairwiseCutoff(hard_disk), PairwiseCutoff(angular_mask))`
to add the terms (as demonstrated in [Applying Interactions]), but it is slightly
faster to use one `PairwiseCutoff` that operates on a tuple:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:hamiltonian}}
```

The former performs two loops over nearby sites and adds the results together
($` \sum U^A_{ij} + \sum U^B_{ij} `$) while the latter performs one loop and adds
terms in the loop body ($` \sum U^A_{ij} + U^B_{ij} `$).

> [!TIP]
> Always list hard shape potentials first in the tuple. If the hard shape
> overlaps, *hoomd-rs* can assume that the move will be rejected and skip the
> computation of the following terms.

[Applying Interactions]: applying-interactions.md

#### Overlap Penalty Hamiltonian

This example uses the `overlap_penalty_hamiltonian` when inserting disks randomly
and compressing the system to the target packing fraction. Use only the hard core
term to allow the system to arrange randomly without being hindered by the patches:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:compress_hamiltonian}}
```

## Initialization and Simulation

See the [Hard Ellipse Self-Assembly] tutorial for a complete explanation of
remaining code.
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:remainder}}
```

[Hard Ellipse Self-Assembly]: hard-ellipse-self-assembly.md

## Conclusion

This tutorial showed you how to perform patchy particle self-assembly
simulations using a shape overlap potential with attractive patches.

Navigate to the top of the page and refresh to see the simulation in action
again. Notice that the disks quickly form random chains and clusters. Over
time, hexagons will appear and the Kagome structure will begin to grow. After
a few hundred thousand timesteps, several distinct grains will appear. Run
the simulation long enough, and the system will equilibrate to a single large
crystal as shown in [10.1039/D2SM01593E].

You can also run the example in batch mode and then open
the generated `trajectory.gsd` in [Ovito] or another visualization tool:
```shell
cargo run --release --example patchy-particle-self-assembly
```

The next section will explain how to model interactions between sites with
different types.

[Ovito]: https://www.ovito.org/

## Complete Code

```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/patchy-particle-self-assembly.rs:all}}

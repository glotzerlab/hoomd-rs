![hoomd-rs](doc/src/images/hoomdrust-logo-horizontal.svg)

**hoomd-rs** is a collection of [Rust] crates that implement particle simulations
and related methods. It performs Monte Carlo simulations of hard shapes and
interacting particles (both isotropic and anisotropic) as well as molecular
dynamics simulations with a variety of particle interaction. **hoomd-rs**
provides public APIs for vector math, geometric primitives, spatial data
structures, energy calculations, and all other components of the simulation
that users can employ in their own analysis and simulation methods. You can use
**hoomd-rs** to create real-time interactive visualizations of simulations,
execute long-running simulations in batch mode on high performance computing
resources, and analyze the results of those simulations.

**hoomd-rs** is the spiritual successor to the Python package [HOOMD-blue].
While the two share many common features, **hoomd-rs** provides *many*
capabilities that [HOOMD-blue] cannot, such as:

* Custom per-particle attributes.
* Custom particle interactions that can *depend on custom per-particle attributes*.
* Custom vector representations, *including curved spaces*.
* Custom MC trial moves and acceptance criteria.
* Custom simulation box geometries (*including non-periodic simulation boxes*).
* Custom visual representations of simulation elements.
* Build native command line applications on *Linux*, *macOS*, and *Windows*.
* Run real-time interactive simulations on desktop platforms or embedded in a web page.

**hoomd-rs** _does not_ come with batteries included. It provides built-in
implementations only for the most commonly used methods. At the same time,
**hoomd-rs** makes it straightforward to customize everything about the
simulation while maintaining a high level of performance. Through the use of
generics, [Rust] will inline your custom code inside the innermost simulation
loops and _compile it to machine code_. In contrast, [HOOMD-blue] offers limited
opportunities for user customization with Python scripts that are _interpreted_
at runtime.

**hoomd-rs** lacks domain decomposition and GPU parallelization, so it is best
for small to moderate sized simulations or when customization is important.
[HOOMD-blue] is best for large simulations and when using models that rely only
on built-in functionality. When you need both large simulations and custom code,
write a [C++ component for HOOMD-blue].

TODO: some comment about how performance compares for simulations in the middle -
is HOOMD-blue or hoomd-rs faster on the same CPU?

## Resources

* [Documentation]: Tutorial and full Rust API reference guide.

## Examples

Random walk:
```rust
use hoomd_interaction::Zero;
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_microstate::{Body, MicrostateBuilder};
use hoomd_simulation::macrostate::Isothermal;
use hoomd_vector::Cartesian;

fn main() -> anyhow::Result<()> {
    let mut microstate = Microstate::builder()
        .bodies([Body::point(Cartesian::from([0.0, 0.0]))])
        .try_build()?;

    let translate = Translate {
        maximum_distance: 0.15.try_into()?,
    };

    let translate_sweep = Sweep(translate);

    let hamiltonian = Zero;

    let macrostate = Isothermal { temperature: 1.0 };

    for _ in 0..100 {
        translate_sweep.apply(&mut microstate, &hamiltonian, &macrostate);
        microstate.increment_step();

        println!("{}", microstate.sites()[0].properties.position);
    }

    Ok(())
}
```

The [examples] directory contains many files that demonstrate how to use
**hoomd-rs**. To see them live in your browser, navigate to the relevant
tutorial in the [hoomd-rs documentation]. The documentation also describes [how
to build the examples] on your desktop.

[HOOMD-blue]: https://hoomd-blue.readthedocs.io
[Rust]: https://www.rust-lang.org/
[C++ component for HOOMD-blue]: https://github.com/glotzerlab/hoomd-component-template/
[examples]: examples/

[Documentation]: https://glotzerlab-hoomd-rs.readthedocs-hosted.com
[hoomd-rs documentation]: https://glotzerlab-hoomd-rs.readthedocs-hosted.com
[how to build the examples]: https://glotzerlab-hoomd-rs.readthedocs-hosted.com/examples.html

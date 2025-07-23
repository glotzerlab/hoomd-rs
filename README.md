# hoomd-rs

**hoomd-rs** is a collection of [Rust] crates that implement particle simulations
and related methods. It performs Monte Carlo simulations of hard shapes and
interacting particles (both isotropic and anisotropic) as well as molecular
dynamics simulations with a variety of particle interaction. **hoomd-rs**
provides public APIs for vector math, geometric primitives, spatial data
structures, energy calculations, and all other components of the simulation
that users can employ in their own analysis and simulation methods. You can use
**hoomd-rs** to create real-time interactive visualizations of simulations or
execute long-running simulations in batch mode on high performance computing
resources.

**hoomd-rs** is the spiritual successor to the Python package [HOOMD-blue].
While the two share many common features, **hoomd-rs** (by design) provides
*many* capabilities that [HOOMD-blue] cannot, such as:

* Custom per-particle attributes.
* Custom particle interactions that can _depend on custom per-particle attributes_.
* Custom vector representations, including curved spaces.
* Custom MC trial moves and acceptance criteria.
* Custom simulation box geometries (_including non-periodic simulation boxes_).
* Custom visual representations of simulation elements.
* All custom code compiles to *optimized machine code*.
* Build command line applications on *all the platforms* that [Rust] supports.
* Run real-time interactive simulations on Linux, Mac, and Windows *natively*.

**hoomd-rs** _does not_ come with batteries included. It provides built-in
implementations only for the most commonly used methods. At the same time,
**hoomd-rs** makes it straightforward to customize everything about the
simulation while maintaining a high level of performance. Through the use of
generics, [Rust] will inline your custom code inside the innermost simulation
loops and _compile it to machine code_. In contrast, [HOOMD-blue] offers limited
opportunities for user customization with Python scripts that are _interpreted_
at runtime. At the same time, configuring an environment to build **hoomd-rs**
code takes far fewer steps than even installing a Python environment for
[HOOMD-blue]!

**hoomd-rs** lacks domain decomposition and GPU parallelization, so it is best
for small to moderate sized simulations or when customization is important.
[HOOMD-blue] is best for large simulations and when using models that rely only
on built-in functionality. When you need both large simulations and custom code,
write a [C++ component for HOOMD-blue].

TODO: some comment about how performance compares for simulations in the middle -
is HOOMD-blue or hoomd-rs faster on the same CPU?

## Resources

* **Documentation**: View the current [hoomd-rs documentation] online.

## Examples

TODO: A simple example code.

The [examples] directory contains many files that demonstrate how to use
**hoomd-rs**. Many of these examples execute live in the tutorial section of the
[hoomd-rs documentation].

The documentation also describes [how to build examples] on your desktop,
in case you want to modify any of them and see the results.

[HOOMD-blue]: https://hoomd-blue.readthedocs.io
[Rust]: https://www.rust-lang.org/
[C++ component for HOOMD-blue]: https://github.com/glotzerlab/hoomd-component-template/
[examples]: examples/

[hoomd-rs documentation]: https://glotzerlab-hoomd-rs.readthedocs-hosted.com
[how to build examples]: https://glotzerlab-hoomd-rs.readthedocs-hosted.com/examples.html

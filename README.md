# HOOMD-rs

**HOOMD-rs** is a collection of Rust crates that implement particle simulations
and related methods. It performs Monte Carlo simulations of hard shapes and
interacting particles (both isotropic and anisotropic) as well as molecular
dynamics simulations with a variety of particle interaction. **HOOMD-rs**
provides public APIs for vector math, geometric primitives, spatial data
structures, energy calculations, and all other components of the simulation
that users can employ in their own analysis and simulation methods. You can use
**HOOMD-rs** to create real-time interactive visualizations of simulations or
execute long-running simulations in batch mode on high performance computing
resources.

**HOOMD-rs** implements many of the methods available in the Python package
[HOOMD-blue] and can be customized in many ways that [HOOMD-blue] cannot,
including:

* Custom per-particle attributes.
* Custom particle interactions that can _depend on custom per-particle attributes_.
* Custom vector representations.
* Custom MC trial moves and acceptance criteria.
* Custom simulation box geometries (_including non-periodic simulation boxes_).
* Custom visual representations of simulation elements.

**HOOMD-rs** _does not_ come with batteries included. It provides built-in
implementations only for the most commonly used methods. At the same time,
**HOOMD-rs** makes it straightforward to customize everything about the
simulation while maintaining a high level of performance. Through the use of
generics, [Rust] will inline your custom code inside the innermost simulation
loops and _compile it to machine code_. In contrast, [HOOMD-blue] offers limited
opportunities for user customization with Python scripts that are _interpreted_
at runtime. At the same time, configuring an environment to build **HOOMD-rs**
code takes far fewer steps than even installing a Python environment for
[HOOMD-blue]!

**HOOMD-rs** lacks domain decomposition and GPU parallelization, so it is best
for small to moderate sized simulations or when customization is important.
[HOOMD-blue] is best for large simulations and when using models that rely only
on built-in functionality. When you need both large simulations and custom code,
write a [C++ component for HOOMD-blue].

TODO: some comment about how performance compares for simulations in the middle -
is HOOMD-blue or HOOMD-rs faster on the same CPU?

[HOOMD-blue]: https://hoomd-blue.readthedocs.io
[Rust]: https://www.rust-lang.org/
[C++ component for HOOMD-blue]: https://github.com/glotzerlab/hoomd-component-template/

## Example

TODO

## Resources

To view the documentation or execute the examples, you need to clone this
repository if you haven't already done so:
```shell
git clone git@github.com:glotzerlab/hoomd-rs
```

Then enter the `hoomd-rs` directory:
```shell
cd hoomd-rs
```

### Documentation

To build the documentation and open it in your browser, execute:
```shell
RUSTDOCFLAGS="--html-in-header katex.html" cargo doc --workspace --no-deps --open
```
You can omit the `--workspace` to build the documentation more quickly, but this
skips `hoomd-bevy`.


## More examples

Look in the [examples] directory to find more examples. 

To execute an example, run:
```shell
cargo run --release --example {example}
```
where `{example}` is the file name (without extension) of a `.rs` file in
[examples].

Examples that show real-time interactive visualizations require the `bevy`
feature:
```shell
cargo run --release --features=bevy --example {example}
```
**Bevy** will take a few minutes to build the first time you run an example.
**Cargo** will cache the build so that it takes less time to run another
example. Execute these examples locally you your laptop or desktop for the
best experience (remote desktop platforms will run these poorly, if at all).

[examples]: examples/

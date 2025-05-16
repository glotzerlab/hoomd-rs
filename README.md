# HOOMD-rs

**HOOMD-rs** is a collection of Rust crates that implement particle simulations and
related methods. It performs Monte Carlo simulations of hard shapes and interacting
particles (both isotropic and anisotropic) as well as molecular dynamics simulations
with a variety of particle interaction. **HOOMD-rs** provides public APIs for vector
math, spatial data structures, energy calculations, and all other components of the
simulation that users can employ in their own analysis and simulation methods.

**HOOMD-rs** implements many of the methods available in the Python package
[HOOMD-blue] and can be customized in many ways that [HOOMD-blue] cannot, such as:

* Custom per-particle attributes.
* Custom particle interactions that can _depend on custom per-particle attributes_.
* Custom vector representations.
* User-defined MC trial moves and acceptance criteria.
* User-defined simulation box geometries (_including non-periodic simulation boxes_).
* True 2D simulations where vectors have no _z_ component.

Users are expected to make use of these customization opportunities. **HOOMD-rs** _does
not_ come with batteries included. It provides built-in implementations only for the
most commonly used methods. Users compile their simulation code with [Rust], which will
inline user-provided code in the innermost loops allowing the resulting executables
can realize the full performance of the CPU. In contrast, [HOOMD-blue] offers limited
opportunities for user customization with Python scripts that are _interpreted_ at
runtime. Furthermore, configuring an environment to build **HOOMD-rs** code takes far
fewer steps than even installing a Python environment for [HOOMD-blue]!

**HOOMD-rs** lacks domain decomposition and GPU parallelization, so it is best for small
to moderate sized simulations or when customization is important. [HOOMD-blue] is best
for large simulations and when using models that rely only on built-in functionality.
When you need both large simulations and custom code, write a
[C++ component for HOOMD-blue].

TODO: some comment about how performance compares for simulations in the middle -
is HOOMD-blue or HOOMD-rs faster on the same CPU?

[HOOMD-blue]: https://hoomd-blue.readthedocs.io
[Rust]: https://www.rust-lang.org/
[C++ component for HOOMD-blue]: https://github.com/glotzerlab/hoomd-component-template/

## Resources

To view the documentation, clone this repository if you haven't already done so:
```shell
git clone git@github.com:glotzerlab/hoomd-rs
```

Enter the `hoomd-rs` directory:
```shell
cd hoomd-rs
```

Build the documentation and open it in your browser:
```shell
cargo doc --no-deps --open
```

## Example

## Crates

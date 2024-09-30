# HOOMD-rs

**HOOMD-rs** is a collection of Rust crates that implement particle simulations and
related methods. It performs hard particle Monte Carlo simulations of a variety of shape
classes and molecular dynamics simulations of particles with a range of pair potentials.
**HOOMD-rs** provides public APIs for vector math, spatial data structures, energy
calculations, and all other components of the simulation that users can employ in their
own analysis and simulation methods.

**HOOMD-rs** implements a subset of the methods available in the Python package
[HOOMD-blue] but can be customized in many ways that [HOOMD-blue] cannot, such as:
* Custom per-particle attributes.
* Custom particle interactions that can _depend on custom per-particle attributes_.
* User-defined MC trial moves and acceptance criteria.
* User-defined simulation box geometries (_including non-periodic simulation boxes_).
* True 2D simulations where vectors have no z component.

Users compile their simulation with Rust, so their code can realize the full
performance of the CPU. In contrast, [HOOMD-blue] offers limited opportunities for user
customization with Python scripts evaluated at runtime.

**HOOMD-rs** lacks domain decomposition and GPU parallelization, so it is best for small
to moderate sized simulations and when customization is important. [HOOMD-blue] is best
for large simulations and when using models that rely only on built-in functionality.
When you need both large simulations and custom code, write a 
[C++ component for HOOMD-blue].

TODO: some comment about how performance compares for simulations in the middle -
is HOOMD-blue or HOOMD-rs faster on the same CPU?

[HOOMD-blue]: https://hoomd-blue.readthedocs.io
[C++ component for HOOMD-blue]: https://github.com/glotzerlab/hoomd-component-template/

## Resources

## Example

## Crates

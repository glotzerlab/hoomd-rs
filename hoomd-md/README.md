![hoomd-rs](https://raw.githubusercontent.com/glotzerlab/hoomd-rs/refs/tags/1.3.0/doc/src/images/hoomdrust-logo-horizontal.svg)

[![Crates.io Version](https://img.shields.io/crates/v/hoomd-mc?color=a44300)](https://crates.io/crates/hoomd-mc)
[![Read the Docs](https://img.shields.io/readthedocs/hoomd-rs/latest.svg)](https://hoomd-rs.readthedocs.io/)
[![GitHub Repository](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/glotzerlab/hoomd-rs)
[![GitHub Discussions](https://img.shields.io/github/discussions/glotzerlab/hoomd-rs)](https://github.com/glotzerlab/hoomd-rs/discussions/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/y/glotzerlab/hoomd-rs?label=commits)](https://github.com/glotzerlab/hoomd-rs)
[![Contributors](https://img.shields.io/github/contributors-anon/glotzerlab/hoomd-rs.svg?style=flat)](https://hoomd-rs.readthedocs.io/en/latest/credits.html)
[![GitHub top language](https://img.shields.io/github/languages/top/glotzerlab/hoomd-rs?color=a44300)](https://rust-lang.org/)
[![prek](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json)](https://github.com/j178/prek)

**hoomd-rs** is a collection of [Rust] crates that implement particle
simulations and related methods. It performs Monte Carlo simulations of hard
shapes and interacting particles (both isotropic and anisotropic) as well
as molecular dynamics simulations (coming soon!) with a variety of particle
interaction. **hoomd-rs** provides public APIs for vector math, geometric
primitives, spatial data structures, energy calculations, and all other
components of the simulation that users can employ in their own analysis and
simulation methods. You can use **hoomd-rs** to create real-time interactive
visualizations of simulations, execute long-running simulations in batch mode
on high performance computing resources, and analyze the results of those
simulations.

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
**hoomd-rs** makes it straightforward for you to customize everything about
the simulation while maintaining a high level of performance. Through the use
of generics, [Rust] will inline your custom code all the way to the innermost
simulation loops and _compile it to optimized machine code_.

**hoomd-rs** lacks GPU parallelization, so it is best for small to moderate
sized simulations or when customization is important. For most typical Monte
Carlo benchmarks, **hoomd-rs** is *faster* than [HOOMD-blue] when run on a
single CPU core. The current **hoomd-rs** does not scale as well as [HOOMD-blue]
to multiple CPU cores. Future releases will improve multi-core scaling.

[HOOMD-blue] is best for large simulations of models that rely only on built-in
functionality.

## Examples

The [examples] directory contains many files that demonstrate how to use
**hoomd-rs**. To see them live in your browser, navigate to the relevant
tutorial in the [documentation]. The documentation also describes how
to run the examples locally.

## Resources

* [Documentation]: User guide and tutorial.
* [hoomd-rs discussion board](https://github.com/glotzerlab/hoomd-rs/discussions/):
  Ask the **hoomd-rs** user community for help.
* [Template workflow]: Start here to use **hoomd-rs** with [row] and [signac].

## Related tools

- [Ovito](https://www.ovito.org/):
  Visualize simulation trajectories with **Ovito**.
- [gsd](https://gsd.readthedocs.io):
  Read simulation trajectories with the **gsd** Python library.
- [freud](https://freud.readthedocs.io/):
  Analyze simulation results with the **freud** Python library.
- [signac]:
  Manage your workspace with **signac**.
- [row]: Automate your HPC workflow using **row**.

[row]: https://row.readthedocs.io
[signac]: https://signac.readthedocs.io/
[HOOMD-blue]: https://hoomd-blue.readthedocs.io
[Rust]: https://www.rust-lang.org/
[Template workflow]: https://github.com/glotzerlab/hoomd-workflow/
[examples]: https://github.com/glotzerlab/hoomd-rs/tree/1.3.0/examples

[Documentation]: https://hoomd-rs.readthedocs.io

## License

**hoomd-rs** is available under the [3-clause BSD license](LICENSE).

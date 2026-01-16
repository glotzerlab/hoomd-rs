# Introduction

![hoomd-rs logo](images/hoomdrust-logo-horizontal.svg)

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

## How to Read This Book

This documentation is written in mdBook. The [Reading Books] chapter
explains how to search, change display settings, and navigate this book.
Press `?` to see a list of keyboard shortcuts.

[HOOMD-blue]: https://hoomd-blue.readthedocs.io
[Rust]: https://www.rust-lang.org/
[Reading Books]: https://rust-lang.github.io/mdBook/guide/reading.html

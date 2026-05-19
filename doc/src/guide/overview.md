# Overview

*hoomd-rs* is not an application that you install and use. It is a coupled
set of [Rust] *crates*. Each individual crate serves a narrow purpose and,
to the extent possible, is usable on its own without the others.
The collection as a whole enables Monte Carlo and molecular dynamics
simulations of soft matter systems.

To use *hoomd-rs*, you need to write a [Rust] application that links to
one or more *hoomd-rs* crates. What you do in your application is completely
up to you. You can learn about what each crate does in its documentation.

> [!NOTE]
> Skip to the [Monte Carlo tutorials] for an introduction to *hoomd-rs*
> based on concrete example simulations.

[Monte Carlo tutorials]: ../mc-tutorial/index.md

## Interaction models

[hoomd-interaction] implements many interaction models commonly used
in soft matter research, such as the Lennard-Jones pair,
linear external, and hard shape overlap potentials. When you need something
different, implement your own interaction model via one or more of the
traits defined in [hoomd-interaction].

See the [hoomd-interaction] documentation for a more detailed introduction,
a complete list of all interaction models, and example code.

## Representing Shapes

[hoomd-geometry] defines many common 2D and 3D shapes and implements a number
of operations on them. You can use these shapes to define your simulation boundary
and site interactions. When you need a different class of shape,
implement your own via one or more of the traits defined in [hoomd-geometry].

See the [hoomd-geometry] documentation for a more detailed introduction,
a complete list of all shapes, and example code.

## Tracking the System's Microstate

[hoomd-microstate] defines the `Microstate` type which tracks all the degrees of
freedom for the *bodies* in your simulation and provides efficient methods to
access their *sites*. Many of the [hoomd-interaction] traits operate on a
system microstate, such as `TotalEnergy` which computes the total energy of
a microstate for a given interaction model.

See the [hoomd-microstate] documentation for a more detailed introduction,
a complete list of all methods, and example code.

## Monte Carlo Simulations

[hoomd-md] implements typical Monte Carlo trial moves used in soft matter
research, such as local translation and rotation moves. Each trial move is
applied to a `Microstate` ([hoomd-microstate]) and accepted or rejected based on
the given interaction model ([hoomd-interaction]). When you need a different type
of trial move for your simulation model, implement your own via one or more of the
traits defined in [hoomd-mc].

## Writing GSD Files

Use [hoomd-gsd] to read and write GSD files that store the position and orientation
of the sites in your microstate as a function of time. Use GSD files for offline
analysis and visualization after your simulation job completes.

[Rust]: https://www.rust-lang.org/
[hoomd-interaction]: ../api/hoomd_interaction/index.html
[hoomd-geometry]: ../api/hoomd_geometry/index.html
[hoomd-microstate]: ../api/hoomd_microstate/index.html
[hoomd-mc]: ../api/hoomd_mc/index.html
[hoomd-gsd]: ../api/hoomd_gsd/index.html

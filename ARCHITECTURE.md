# HOOMD-rs Code Architecture

## Organization

HOOMD-rs is organized in many individual crates so that external code does not need to
compile all of HOOMD-rs when using only a portion.

### hoomd_vector

`hoomd_vector` implements vector and quaternion math. It also defines general traits
so that users can define custom vector storage types (and associated operations).

### hoomd_rs_spatial

`hoomd_rs_spatial` implements spatial data structures, such as the AABB tree and cell
list. Use spatial data structures to quickly find particles near a point in space.

### hoomd_rs_shape

`hoomd_rs_shape` defines traits that describe shapes (both convex and non-convex) and
operations commonly applied on shapes (shape-shape overlap, distance between, etc...).

### hoomd_rs_random

Counter based random number generator.

### hoomd_rs_interactions

Pairwise interactions between particles, interactions between particles and fields,
etc... Interactions may be used with MD, MC, by user's analysis methods, and in
other cases.

### hoomd_rs_md

### hoomd_rs_mc

### and more to come...

### Crate Versioning

All crates have identical versions and therefore all crates get a version bump even when
only one crate is modified. This is not ideal, however the alternative requires overly
complex systems to handle crate-specific tags, releases, and CI. All the hoomd_rs crates
are tightly interconnected, and most releases will update more than one crate at a time.

## Design Philosophy

HOOMD-rs's design prioritizes:
* Flexibility
* Reusability
* Code readability
* Simplicity
* Efficient algorithms
* Micro-optimizations targeted at specific hardware
_in that order._

**HOOMD-rs's** big brother, [HOOMD-blue], prioritizes micro-optimizations _first_ and
makes code reusable only at the Python level.

[HOOMD-blue]: https://hoomd-blue.readthedocs.io

A **Flexible** design allows users to customize not only what operations are performed
on each simulation step, but also details of how that operation are performed and the
**data structures** that the operation works on. For example, the MC simulation
implementation should allow users to control the specific form of the trial moves
performed.

> Note: In this context, **customize** means that the user provides custom **code**
> via implementing a trait or passing a Fn, not just changing parameters.

HOOMD-rs implements all components of the simulation in a **reusable** manner so that
users can take any portion of the calculation and use it outside the context of a
specific simulation. Everywhere that is possible, HOOMD-rs implements calculations,
data structures, and traits _without_ a reference to any centralized system object.

When users inspect HOOMD-rs's implementation, they will find clearly readable and
documented methods. Complex calculations should be split into short and understandable
methods. Where possible, concerns are separated into well-defined traits that the
user can override (see **flexibility**). These traits should define a minimum number of
clearly named associated methods.

HOOMD-rs's design is intentionally **simple**. For example, it does not provide
a massive library of particle interactions, but only the few most common (e.g.
Lennard-Jones, Sphere). Users should take advantage of **flexibility** to define
custom interactions for their model. Keeping the code **simple** will reduce the time
researchers need to spend when implementing a drastically different model or simulation
method.

Minimizing the wall clock time needed to complete a simulation is also important,
but is not at the top of the list. Keeping **flexibility**, **readability**, and
**simplicity** in mind - HOOMD-rs uses appropriate algorithms and data structures
(e.g. spatial searches) everywhere needed so that simulations can execute quickly.
Domain decomposition and GPU algorithms are completely off the table as they are not
flexible, readable, or simple (use [HOOMD-blue] if you need that level of performance).
Multithreaded CPU calculations may be achievable for some simulation algorithms within
this framework, but certainly not for all. Therefore, HOOMD-rs does not **require**
that every method support multithreading by default. This section will be updated
after we evaluate the performance possible by multithreading vs the cost to simplicity
and flexibility. It may be that we need to offer a simpler serial implementation
with more flexibility in addition to a more complex multithreaded implementation with
restrictions.

HOOMD-rs rarely implements **micro-optimizations that target specific hardware**. For
example, HOOMD-rs uses an _array-of-structures_ for the simulation state because this is
vastly **simpler** than a _structure-of-arrays_ and at the same time allows users to
easily add custom per-particle attributes. The cost is slightly reduced performance
due to cache memory inefficiencies.

### Crate-specific Design

Each crate's rustdoc documentation explains its own architecture. This document focuses
on architecture common to all HOOMD-rs crates.

## Logging

HOOMD-rs writes nothing to stdout/stderr by default. It instead uses the
[Rust log crate] so that users can opt in to detailed logging when desired. HOOMD-rs
uses log levels in the following circumstances:

* `error`: _Unused_. HOOMD-rs methods return a `Result<Error>` with a detailed
  description provided in the error. The caller can log this description with `error()`
  when desired.
* `warn`: Used sparingly to warn the user of technically well-defined but likely
  invalid parameters.
* `info`: Simulation status messages - approximately once per simulation.
* `debug`: Operation status messages - several times per simulation, but much
  less often than once/timestep.
* `trace`: Detailed operation status messages - once or more per time step.

Use `env_logger` (or a similar crate) to filter messages by log level, crate, and
module.

[Rust log crate]: https://docs.rs/log/latest/log/
[env_logger]: https://docs.rs/env_logger/latest/env_logger/

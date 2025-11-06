# hoomd_spatial

This crate provides specialized spatial data structures for particle systems. It is
designed to be independent of system infrastructure while supporting both primary and
ghost sites.

## Traits

The spatial algorithms must be exposed as traits so that `Microstate` and simulation
models can at least be generic on vector types. For example, the cell list works for
Cartesian vectors while points in curved space must use an all-pairs algorithm. In the
future, these traits allow the possibility for different data structures (i.e. AABB
trees or neighbor lists) and other spatial algorithms useful in data analysis.

For a first draft, these traits are designed to enable Monte Carlo simulations.

* `PointUpdate` describes a type where points can be inserted to and removed from
  a data structure. Points are identified by key.
* `PointsNearBall` describes a type that can find all points (identified by key) that
  *may* be in a given ball.

The traits are generic on the key type. `Microstate` will use a key that identifies
primary and ghost particles by index with an enum.

## Cell List Data Structure

The cell list is the core data structure for neighbor search operations. It maps space
into a Cartesian grid of cubic cells and leverages maps for storage and lookup. The
struct is independent of spatial dimension and works only for N-dimensional Cartesian
space.

The use of maps on cell indices provides
1) Efficient memory usage even for sparse systems (occasional use of `shrink_to_fit`
   may be needed in such circumstances).
2) One implementation that works for any shape of boundary. Cells are always cubic
   and may overhang boundaries but will still correctly find all points that
   match search queries.

### Members

- **Cell Width:**
  Uses cube boxes to partition space. The cell size is user-defined and exposed as
  a parameter.
- **Primary Hash Map:**
  Maps cell indices (keys, represented as an array of cell indices) to vectors of
  particle indices (values).
- **Secondary Hash Map:**
  Provides a mapping from individual particle indices to their corresponding cell
  index.

### Methods and Operations

- **Creation:**
   - Requires only points and cell width. PBCs are implemented via ghosts provided
     by the caller.
   - Mapping of coordinates to cells is done via:
    `cell = floor(x / width)` using array from fn
   - This information gets encoded in the two Hash Maps.

- **Locate Cell:**
   - Identify the cell that contains the particle of interest based on its index.

- **Neighbor Search:**
   - Given a position in space and a ball radius cutoff, determine all sites
     *potentially* within the ball.

- **Adding a site:**
  - Insert the particle into the appropriate cell.
  - Update both hash maps accordingly.

- **Removing a site:**
  - Delete the particle entry from the relevant cell.
  - Update both hash maps accordingly.

- **Translating a site:**
  - **If the particle crosses cell boundaries:**
    - Remove it from the old cell and add it to the new one.
  - **If the particle remains in the same cell:**
    - No action is required.

## AllPairs Data Structure

For curved space or in tiny simulations, the `AllPairs` type implements a no-op spatial
data structure. Every query will always match all points. To effectively
implement `PointUpdate` and `PointsNearBall`, `AllPairs` must unfortunately store
a `HashSet` of all keys. That makes ball searches run at 1/4 the speed of a tight
for loop over all sites.

We *could* have `iter_sites_near` return some kind of hybrid iterator that stores
both types of iterators internally and uses the one appropriate to the spatial
data structure. However, it may be better to write a separate implementation of
`CutoffPair` instead: `CutoffPairAll`. `CutoffPairAll` would not call
`iter_sites_near` and would instead loop over `sites()` and `ghosts()` directly.
As a bonus, `CutoffPairAll` could iterate over sites in parallel and boost performance
for simulations on curved surfaces that have no other opportunity for parallelism.

## Neighbor iteration protocols

Profiling shows that the majority of the runtime in MC benchmarks is spent iterating
over potential neighbors. `VecCell` and `HashCell` pre-compute sorted stencils that
check the nearest cells first. Their implementation of `PointsNearBall` builds an
iterator that stores several references and counters that effectively implements an
outer iteration over nearby cells and an inner iteration over sites in the cell.
These iterators were carefully constructed and profiled to avoid extra computations
of the cell index. Overall, they attain a performance much higher than expected ---
the hoomd-rs hard disk benchmark is more than twice as fast as HOOMD-blue. Other
MC benchmarks range from 20-50% faster.

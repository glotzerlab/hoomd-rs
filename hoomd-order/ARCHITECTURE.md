# hoomd_order

The `hoomd-order` crate implements order parameters and other related calculations,
such as k-atic, Steinhardt, radial distribution functions, and spatial correlation
functions. It is the evolution of the functionality available in the [`freud`] Python
package.

## Design goals

### Generality

[`freud`] is popular because it operates directly on point sets. Where possible,
`hoomd-order` should maintain this level of generality. At the same time,
[`freud`] users have difficulties when using it directly coupled to a running
simulation as `freud` needs to compute spatial data structures that are already
present in the simulation. `hoomd-order` should solve this problem by allowing
users to operate directly on sites in a `Microstate` and thereby make use
of the existing spatial data structures. `Microstate` also implements the
ghost site logic needed to realize periodic boundary conditions.

One approach to solving this dual problem is to implement all analysis
methods in `hoomd-order` independent of `Microstate` so that they can be
used on general point sets. A higher level API layer could provide convenience
methods that operate on `Microstate`.

### Simplicity

All methods in `hoomd-order` should be the most natural possible expression
of the underlying mathematical operation. This provides complete clarity to
the caller so that there is no ambiguity over what input maps to what
(see e.g. the confusion over `query_points` and `points` in [`freud]`).

For example, the k-atic order parameter can be expressed as a function with
the signature:
```rust
fn k_atic_psi<I: IntoIterator<Item=Cartesian<2>>>(k: f64, r_i: Cartesian<2>, neighbors: I) -> Complex<f64>;
```

Analysis methods should be exposed as functions where possible and structs only when
they need to store some internal state (such as the g(r) histogram).

Note that unlike in [`freud`], the k-atic function computes an order parameter for
a single point rather than for an entire system of points. This grants the caller
the flexibility to call it only when needed for specific points without the need
to design a complicated filter/selection system.

[`freud`]: https://freud.readthedocs.io/en/latest/

## Neighbor queries

As shown in the `k_atic_psi` signature above, neighbor queries in `hoomd-order`
simply become iterators. The iterator item depends on context, sometimes it may
be only position while other methods might need position and orientation, or
position, orientation, and weight. This grants the caller infinite flexibility.
For example, they could form a chain of iterators starting with
`Microstate::iter_sites_near` that filters for specific sites, collects into a vector,
sorts the vector by distance to `r_i` and takes the first 6 items.

Some users will appreciate that flexibility while others will find it onerous.
`hoomd-order` should provide implementations of common methods, like the ball
query and the nearest *k* neighbors (within a ball). To keep some amount of generality,
the provided methods should emit `Site` items so that callers can further filter.
At the same time, there should be provided implementations that produce positions
(and positions/orientations) directly so calls to `k_atic_psi` can be a
one-liner in simple situations. Optional exclusion of the *i* site should be
handled by the neighbor query, not each analysis method.

Each query will need to hold some state (such as `r_cut`, the identity of `i`, etc...)
so they should be implemented as structs. There will not be a `NeighborQuery` trait.
Say a caller already has their neighbor positions listed in a `Vec` (e.g. from an
explicit list of bonds). When the `neighbors` argument is `IntoIterator`, callers
can simply pass in the `vec.iter()`. If `neighbors` had to implement the
`NeighborQuery` trait, then callers would have to implement that trait for many
common types. Rather, `BallQuery` and similar structs will follow a standard API
format, but otherwise remain as independent types.

Neighbor queries are the first example where `hoomd-order` provides convenience
methods that operate on `Microstate`. The provided queries will take in `&Microstate`
and use its `iter_sites_near` method.

## Site tags

The concept of site tags has already come up in the neighbor query design. This is
a feature that [`freud`] lacks (almost) entirely. `hoomd-order` will employ site tags
whenever they are needed, such as avoiding the *i* site in a neighbor query.
When tags are not necessary (such as in the base `k_atic_phi` method), `hoomd-order`
will not require them following the simplicity rule.

Other analysis methods will need the concept of a tag implicitly. For example,
the average Steinhardt order parameter needs to know a) the non-averaged order
parameter for each site tag and b) the tags of neighbors of site with a given tag.
For generality, the `average_over_neighbors` method will be general so that
it can be applied to any quantity, not just Steinhardt. Its design remains TBD
as of this writing, but might likely involve a `HashMap` that maps tags to
`q` values.

It's neighbor query is not an iterator, but a callable that produces
the neighboring tags given a tag. The neighbor queries built around `Mircostate`
can of course provide this callable. Users working outside `Microstate` will need
to implement some sort of equivalent data structures to use `average_over_neighbors`.
Note that the neighbor query customization path is different for this type of
signature. When `neighbors` is an iterator, callers can customize it by chaining.
When it is a callable, they need to implement a new callable (which can contain
the same chain as above). It might be advantageous to have only one API for this,
but forcing the concept of tags into methods like `k_atic_psi` breaks the
simplicity design goal.

In this scheme, callers will still need to build the `HashMap` directly as
the *simplicty* goal precludes analysis methods that operate on the whole
`Microstate` (unless strictly necessary). We can provide a fully working
example in the documentation for users to copy and paste.

Clustering is another example where tags will be inherent even in the general
implementation.

# hoomd_rand

## Counter based random number generators

TODO: Document which RNGs we implement and the process used to select the default.

## Seeds and counters

Like HOOMD-blue, hoomd-rs will use a counter based random number generator to provide
reproducible results, consistent performance, and support for parallel execution. The
`hoomd_rand` module provides a unified API so that other parts of hoomd-rs
can consistently initialize CBRNG types without overlaps.

Based on the usage in HOOMD-blue, methods will need a way to create RNGs with
a fixed seed layout that includes the timestep, substep, and user seed. HOOMD-blue
uses 1 to 3 counter values to generate unique random numbers along with that seed.
Those values may be particle tags, chain ids, MPI ranks, etc... The caller chooses
whatever is needed for its algorithm.

The ChaCha CBRNG is readily available in Rust. Unless it proves to be too slow, hoomd-rs
will use it. ChaCha has 32 bytes in the seed and 12 in the stream identifier. These are
separated for cryptographic reasons. There is no special distinction when used as an
RNG, so we have a total of 44 bytes to work with. The fixed part of the seed totals 16
bytes (8 for timestep, 4 for substep, and 4 for the user seed). That leaves 28 bytes (or
7 u32's) for the counter.

Even though it is excessive (no user is likely to run more than 4 billion particles in
hoomd-rs), this provides enough bytes to statically assign the bytes to 2 u64 indices
and 3 general use u32 counters. Should we need to expand the number of counters in
the future (not likely) then this implementation moves the u64 -> u32 downcast to
only one place in the codebase.

Not all algorithms will use all counters, so `hoomd_rand` implements the
builder pattern to construct the seed along these lines:

```
let rng = Counter::new(step, substep, seed)
    .indices(i, j)
    .counter(chain)
    .make_rng();
```

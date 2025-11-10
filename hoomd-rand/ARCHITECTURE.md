# hoomd_rand

## Random Number Generator Basics

At their most basic, pseudorandom number generators (PRNGs, often shortened to RNGs)
allow for the generation of streams of data that closely resemble random numbers. While
the numbers are not truly random, the statistics of good PRNGs are indistinguishable
from true randomness and the outputs can be made deterministic, an extremely valuable
feature for reproducibility. While encryption algorithms use "cryptographically secure"
random number streams (those whose input cannot be reverse-engineered from the output),
simulations can utilize non-cryptographic PRNGs without impacting correctness.
Intuitively, cryptographic security prevents the design of targeted, structured attacks
which do not resemble the random processes we explore in simulation. Good PRNGs --
including those included in this crate -- are carefully validated to ensure output
streams are uncorrelated and have a sufficiently long period. In most cases, the minimum
period length is enforced by a 64-bit counter that

## SFC64 and the alternative PRNGs

The default PRNG in hoomd-rs is `SFC64`, a chaotic PRNG with a 256-bit state. The
"chaotic" nature indicates the presence of multiple, disconnected cycles through subsets
of the full state space. While it is not tractable to precisely quantify features of
this PRNG, we can provably guarantee that the shortest cycle of `SFC64` is $`2^{67}`$
bytes, and the expected cycle length is much larger at $`2^{258}`$ bytes. This means
that each of the $`2^{192}`$ 24-byte seeds can generate at least an exabyte of data
before colliding with another stream. Users can refer to
[Melissa O'Neill's discussion](https://www.pcg-random.org/posts/random-invertible-mapping-statistics.html)
of chaotic-type PRNGs for further details.

`SFC64` was selected as the fastest portable PRNG we tested, although the `AESRand` PRNG
is somewhat faster on systems that have hardware support for AES instructions. This is
also a chaotic-type PRNG, but uses AES encode and decode to form a random invertible
mapping rather than standard operations like addition or xor. For users who require
extreme volumes of random data, `AESRand` might be valuable -- however, the 128-bit
state means limits its use in parallel contexts where more than $`2^32`$ parallel
streams are required.

The `extras` feature of this crate also includes the classic
[`ThreeFry2x64`](https://random123.com) PRNG, which has excellent statistics but
orders-of-magnitude worse performance than `SFC64` and `AESRand`. Many other PRNGs were
tested for this crate, but `SFC64` and `AESRand` are recommended as all other options
sacrifice either performance or statistical quality.

## Seeds and counters

Like HOOMD-blue, hoomd-rs will use a counter based random number generator to provide
reproducible results, consistent performance, and support for parallel execution. The
`hoomd_rand` module provides a unified API so that other parts of hoomd-rs can
consistently initialize CBRNG types without overlaps.

Based on the usage in HOOMD-blue, methods will need a way to create RNGs with a fixed
seed layout that includes the timestep, substep, and user seed. HOOMD-blue uses 1 to 3
counter values to generate unique random numbers along with that seed. Those values may
be particle tags, chain ids, MPI ranks, etc... The caller chooses whatever is needed for
its algorithm.

As described above, the SFC64 PRNG is the default in hoomd-rs. This provides

The ChaCha CBRNG is readily available in Rust. Unless it proves to be too slow, hoomd-rs
will use it. ChaCha has 32 bytes in the seed and 12 in the stream identifier. These are
separated for cryptographic reasons. There is no special distinction when used as an
RNG, so we have a total of 44 bytes to work with. The fixed part of the seed totals 16
bytes (8 for timestep, 4 for substep, and 4 for the user seed). That leaves 28 bytes (or
7 u32's) for the counter.

Even though it is excessive (no user is likely to run more than 4 billion particles in
hoomd-rs), this provides enough bytes to statically assign the bytes to 2 u64 indices
and 3 general use u32 counters. Should we need to expand the number of counters in the
future (not likely) then this implementation moves the u64 -> u32 downcast to only one
place in the codebase.

Not all algorithms will use all counters, so `hoomd_rand` implements the builder pattern
to construct the seed along these lines:

```
let rng = Counter::new(step, substep, seed)
    .indices(i, j)
    .counter(chain)
    .make_rng();
```

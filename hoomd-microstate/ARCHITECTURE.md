# hoomd_microstate

The `hoomd_microstate` crate defines traits and types to represent the system
microstate for use in simulations.

## Design goals.

`Microstate` stores a single system state including all the particles, bonds,
angles, dihedrals, etc... It must meet the needs of a wide variety of MC and MD
algorithms, be easily accessible by users for data analysis, and be serialized
to/from GSD files.

Specific goals:

* User-provided `Particle` type.
* User-defined boundary conditions.
* Iterate over particles.
* Access specific particles.
* Efficient addition/deletion of particles, bonds, angles, dihedrals, etc...
* Incremental updates of individual particles.
* Full system updates of all particles.
* User-defined additional state data.

Many MC and MD algorithms need to access particles in a localized region of
space. While the details of that design will be documented elsewhere, it
intersects with the microstate design to some degree. In MC algorithms
especially, a query around local space occurs after an incremental update
of a single particle. Users of the API (both internal and external) should
not have to manually keep spatial data structures up to date. Through the
provided update methods, `Microstate` knows when particles are moved, so
we can consolidate the update code to one place.

As of this writing, I have not yet determined whether `Microstate` should own
the spatial data structure or allow the caller to own it and link it to the
`Microstate`. I have also not determined whether one spatial data structure
is sufficient or if we should allow for multiple data structures optimized
for difference use-cases. - JAA

## Membership criteria

In statistical mechanics, the microstate is normally defined as the positions
and momenta of all particles. HOOMD-blue (the Python package) follows this
definition closely and requires that users separate "parameters" (i.e. pair
potential epsilon/sigma, particle shape, etc...) from the state of the system.
In many types of simulations (e.g. alchemy), this is a problem as some of these
parameters formally become part of the microstate.

HOOMD-rs maintains the separation of the `Microstate` from the simulation
_model_, but through custom particle properties and custom state data, HOOMD-rs
allows users to include any appropriate data in a `Microstate` instance that
custom model implementations can access. The _model_ is the collections of
methods or algorithms that advance a microstate from one **step** to the next.
_Parameters_ are values that influence the model (and therefore influence the
microstates that are sampled) but are _not_ part of the microstate itself.
Members of the microstate include all the variables that are required to
describe a single point in phase space.

For example, the volume of the system is a variable of the microstate (via the
boundary conditions), but the final volume of a box compression procedure is a
parameter of the model.

_Formally_, the simulation step is not a member of the microstate. Rather, the
microstate is a function of time. HOOMD-blue took this approach. The step was
managed entirely by the top level simulation for loop and passed everywhere by
argument. It was awkward, but somewhat worked. It became more awkward with the
introduction of counter based RNGS. The timestep was a key component of the RNG
seed, so different algorithms each needed a unique seed component defined in a
header file. Cases where the user could use the same algorithms multiple times
on a single step required an additional instance variable to ensure that they
produced uncorrelated random numbers.

HOOMD-rs will solve these problems by including both the simulation _step_ and a
_substep_ as members of `Microstate`. While this choice does not follow a formal
statistical mechanics definition, it does fit in to a more practical definition.
_Model_ implementations DO modify the step (e.g. an integrator increases the
step) when they operate on a `Microstate`. Using the counter based RNGs, the
current step also influences how a model evolves the state. The _substep_ field
solves the problem of generating unique RNG streams for each algorithm. Any
algorithm that uses RNGs will use it in the seed and then increment it. This
way, the next algorithm will use a unique seed and without requiring any special
handling by a caller. Simulations will be binary reproducible, provided the
model's algorithms are run in the same order. The substep will reset to 0 when
the step is incremented. Conveniently, this makes the step available to every
model implementation that operates on a microstate and removes the awkward need
to pass `step` as an argument to nearly every function.

The same argument can not be made so clearly for macrostate parameters like
temperature and pressure. These are emergent parameters that arise from how
the model evolves the state (e.g. the Metropolis acceptance rule ensures a
constant temperature). At the time of this writing, it is not clear whether a
`Macrostate` struct would be helpful in sharing these parameters across model
instances. That will become clear in time.

The user-chosen RNG seed, required to ensure that replicate simulations do not
use the same RNG stream, will also be part of the microstate. It does not fit
the definition above -- no model will ever change the user seed. However, all
the other parameters of RNG seeds will come from `Microstate`. It would not be
practical to provide just this one seed in another way (e.g. via a parameter to
every RNG-using method).

The step is stored in GSD files, so this including it brings `Microstate` more
in line with a GSD `Frame`. We might consider adding the user seed to GSD files
as well.

## Particles

For simplicity and to enable user-defined particle types, `Microstate` stores
particles with an array of structures and `Microstate` is generalized on the
structure type. Code that uses `Microstate` can require one or more trait
bounds to ensure that the particle has the appropriate attributes.

* `Particle` - The base `Particle` trait provides `position` and other
   bookkeeping attributes.
* `Orientable` - Provides `orientation`.
* `Dynamic`? - Mass, velocity, ... needed for MD. TODO: how to differentiate
  between dynamic point particles and dynamic orientable particles? Do we even
  need to bother with the distinction?

## Topology

TODO

## Ghost particles

When periodic boundary conditions are employed (see the next section), model
methods need a clean way to compute interactions between the real particles
in the primary image and ghost particles in periodic images. HOOMD-blue stored
no ghost particles (when not using domain decomposition) and required every
method to wrap delta r vectors back into the box. This proved to be a bad design
decision. The box wrap method is _expensive_ to compute O(N * N_neighbors)
times. For data structures like the AABB tree, that have no internal notion of
periodicity, it required 27 image offset queries on the tree (in 3D).

HOOMD-rs will solve this problem by storing ghost particles explicitly. Spatial
data structures will not need to be aware of the periodic boundary conditions,
nor will they need to be aligned with the boundaries in any way. This also
opens up the ability for very complex user-provided boundary conditions.
It will add some other management costs, but those will be O(N).

## Boundary conditions

TODO

## Particle storage

TODO: This section is a rough draft. Many of these design decisions depend a lot
on whether we require a fixed number of ghosts per particle or allow variable
ghosts. It will take some prototype testing and more thought to determine
whether fixed all-image ghosts (while simple), are a performance problem.

It is tempting to take the path of choosing a single spatial data structure
(the cell list) and store particles directly in that structure. However, that
would make updating ghost particles, finding particles by tag, and other needed
methods very complex. It would seem that the best approach is the simplest:
Soring particles in a `Vec` where the first _N_ elements are the particles in
the primary image and particles with indices greater than or equal to _N_ are
the ghost particles.

To allow efficient addition/removal of particles from this structure,
particles must have a unique `tag` that indicates their ordering in the initial
state. Removal of a particle at index _i_ would be accomplished by swapping
particle _N-1_ into the _i_ position (and updating auxiliary data structures
accordingly). This will leave a gap between the end of the primary image
particles and the ghost particles. This gap is desirable for particle addition
as adding a new particle in the gap costs O(1). When there is no gap, insertion
costs O(N) to regenerate all ghost particles. We can follow the standard trick
of doubling the gap size to amortize the cost of insertions.

One alternative to consider would be to store a separate Vec of ghost particles.
However, this would require some special handling in spatial data structures.
They would be cleaner to work on a single Vec. The gap between particles and
ghosts also adds complications there: TODO: think about this some more. The
complications in the spatial data structures might be less than the O(n) expense
of keeping no gap.

Adding a new particle also requires adding its ghosts. Removing a particle will
remove all of its ghosts. Following the same procedure as above ghost particles
can also be removed from the `Vec` efficiently. However, we will need to somehow
maintain a list of ghost indices for each particle in order to enable efficient
particle updates. Swapping the ghost particles from the end into the middle of
the array will require that this table is also updated. This is trivial if all
particles have the same number of ghosts (e.g. 26) but could be more complex if
we allow for a variable number of ghosts (e.g. only particles within r_cut of
the boundary). In either case, the cost is O(1).

Removing a particle should also either a) remove all bonds connected to that
particle or b) produce an error if the particle participates in a bond.

Incremental updates to particles must also update all of its ghosts and the
linked spatial data structures. Complete system updates are likely best off
removing all ghosts/spatial data structures and rebuilding them with the new
particles. MC will be the primary driver of incremental updates and MD will
primarily use full system updates, although this is not a strict rule.

## Rigid bodies

TODO: Consider unifying rigid body representations across MD and MC codes.
HOOMD-blue uses constituent particles in the microstate and HPMC uses union
potentials. HOOMD-rs could potentially use constituent particles for both and
manage rigid bodies directly within `Microstate`.

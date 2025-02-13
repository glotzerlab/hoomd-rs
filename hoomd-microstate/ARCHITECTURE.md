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
* Iterate over particles, possibly with built-in filtering.
* Access specific particles.
* Efficient addition/deletion of particles, bonds, angles, dihedrals, etc...
* Consider allowing custom topology types.
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
instances. However, MD integration methods DO effectively add new degrees of
freedom to the microstate through the thermostat and barostat variables.
TODO: Consider how to account for this? Store them in Microstate? Or keep the
HOOMD-blue approach of maintaining them in the struct that applies the
thermostat?

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
It will add some other management costs, but those can be manged to not
scale with the number of neighbors..

## Boundary conditions

TODO

## Particle storage

It is tempting to take the path of choosing a single spatial data structure
(the cell list) and store particles directly in that structure. However, that
would make updating ghost particles, finding particles by tag, and other needed
methods very complex. Microstate will maintain a simple approach and store
particles in a Vec. 

To allow efficient addition/removal of particles from this structure,
particles must have a unique `tag` that indicates their ordering in the initial
state. Removal of a particle at index _i_ would be accomplished by swapping
particle _N-1_ into the _i_ position (and updating auxiliary data structures
accordingly). We need to prevent callers from modifying the particle
tag along with the other attributes. The `Tagged<T>` type will store the
tag along with the particle. Read only access `tag()` will be public, but
the field itself will be `pub(crate)` to allow only this crate to set the tag
field.

Ghost particles will be stored in a separate `Vec<Option<Tagged<P>>>`. Adding
a new particle also requires adding its ghosts (to the end of the ghosts Vec).
Removing a particle will remove all of its ghosts. Ghost removal will operate
differently than above. Simply **moving** a particle can result in the addition
or removal of ghosts -- when a particle moves toward or away from a periodic
boundary. In a cost amortized neighbor list, ghosts may appear as neighbors of
multiple particles. It would not cost O(1) to remove all those. Instead, we will
allow removed ghost particles to leave a `None` sentinel at the same index in
the array to maintain O(1) particle updates with a neighbor list.

Microstate will need to maintain auxiliary data structures to maintain
O(1) updates, including a mapping from tag to index, and a list of
ghost particle indices for each particle. 

Removing a particle should also either a) remove all bonds connected to that
particle or b) produce an error if the particle participates in a bond.

Incremental updates to particles must also update all of its ghosts and the
linked spatial data structures. Complete system updates are likely best off
removing all ghosts/spatial data structures and rebuilding them with the new
particles. MC will be the primary driver of incremental updates and MD will
primarily use full system updates, although this is not a strict rule. With
amortized neighbor lists, these rebuilds will need to be coordinated with the
spatial data structure.

As of this writing, the spatial data structures have not yet been designed
for hoomd-rs. It is clear, however, that we would like the design to be
usable with or without a Microstate. Therefore, spatial data structures
will likely operate on a set of indexed particle positions. Microstate
maintains two vectors, one for real particles and one for ghosts. One
solution (not ideal) would be to use signed indices in the spatial data
structures, with signed indices for real particles and negative values
for ghosts. This would not be ideal because it could cause off-by-one
errors when accessing ghosts (there is no -0 integer). We could
keep the same idea by using a new type for the index with helper methods
to decode the index separately from the real/ghost flag (stashed in the
highest bit of the integer).

## Rigid bodies

TODO: Consider unifying rigid body representations across MD and MC codes.
HOOMD-blue uses constituent particles in the microstate and HPMC uses union
potentials. HOOMD-rs could potentially use constituent particles for both and
manage rigid bodies directly within `Microstate`.

## Update API methods

TODO: Single particle updates

Full system updates will occur via a method that takes a Fn that operates on a
mutable slice of particles. In this way, the update function can take steps such
as reinitializing all ghost particles and rebuilding spatial data structures
after calling the provided Fn.

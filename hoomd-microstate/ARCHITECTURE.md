# hoomd_microstate

The `hoomd_microstate` crate defines traits and types to represent the system
microstate for use in simulations.

## Design goals.

`Microstate` stores a single system state including all the bodies, sites,
bonds, angles, dihedrals, etc... It must meet the needs of a wide variety of
MC and MD algorithms, be easily accessible by users for data analysis, and be
serializable to/from GSD files.

Specific goals:

* User-provided types.
* User-defined boundary conditions.
* Iterate over bodies and sites, possibly with built-in filtering.
* Access specific bodies and sites.
* Efficient addition/deletion of bodies, bonds, angles, dihedrals, etc...
* Consider allowing custom topology types.
* Incremental updates of individual bodies.
* Full system updates of all bodies.
* User-defined additional state data.

Many MC and MD algorithms need to access particles in a localized region of
space. While the details of that design will be documented elsewhere, it
intersects with the microstate design to some degree. In MC algorithms
especially, a query around local space occurs after an incremental update
of a single particle. Users of the API (both internal and external) should
not have to manually keep spatial data structures up to date. Through the
provided update methods, `Microstate` knows when particles are moved, so
we can consolidate the update code to one place. To keep the implementation
simple, `Microstate` will optionally own a cell list and a pairlist.
When the cell list is present, it will always be kept up to date. The
pairlist will be most useful for MD simulations and will be regenerated
as needed with the normal amortized update scheme.

## Rigid bodies

In HOOMD-blue, rigid bodies are an ad-hoc solution that sits on top of a
flat list of particles. Users and developers alike experience many difficulties
with this scheme.

* Users don't understand why they need to set body parameters separately from
  the gsd file.
* Users do not always want to place an interaction site at the body center.
* Adding/removing bodies from the state is not straightforward.
* Users must filter out constituent particles from the MD integration method.
* Rigid bodies in HOOMD-blue only work with MD.
* MC requires the use of union potentials. These have slow performance
  because the individual union sites must be iterated over exhaustively
  as they are not present in the spatial data structure.
* Some users would prefer to place constituents directly and have the implicit
  bodies created for them. This is challenging in HOOMD-blue because the
  local reference frame around a body is linked to the particle's type.

Except for the last point (which we could consider writing a helper function to
achieve), we can solve all the above problems with a radical redesign.

1) Do away with the idea of "particles". Instead, there are *bodies* and
   *sites*. The *bodies* are the objects that have degrees of freedom (position,
   orientation, velocity, angular momentum, etc...).
2) Each *body* consists of its degrees of freedom and 1 or more *sites* in the
   local reference frame of that body. In this way, a *body* is an object that
   can stand entirely on its own - it does not need to be part of a microstate
   to make sense. Sites have a position, orientation, and any other user-defined
   parameters of interest (e.g. types).
3) `Microstate` will store and manage a flattened list of sites in the system
   reference frame. As bodies are moved (or their local reference frame is
   updated) this flat list is always kept up to date.
4) The *sites* are the objects that interact via potentials. Therefore, they are
   the objects placed in the spatial data structure. In this way, both MD and MC
   simulations can take advantage of the spatial data structure.

While this design solves many of the problems mentioned above, it brings about
new challenges.

1) The particle-oriented GSD file format is no longer a 1:1 mapping to the
   microstate. We could either implement a whole new schema from scratch or
   attempt to convert between the representations. At a minimum, we need to
   extend the schema to allow storing the local reference frame around each body
   somehow. Another option is to use a new schema for initialization/restart
   and have a compatibility layer for the old schema for use with OVITO and/or
   HOOMD-blue. Sometimes users want *only* bodies in their GSD files and
   other times, they want only the sites.
2) With the local body definition now decoupled from the particle type, *every*
   body can potentially have a unique structure. While this adds flexibility,
   it does increase the memory footprint, adds more indirect memory accesses,
   and makes it more burdensome to change the local definition of many bodies.
   Performance is not the #1 goal, so this should be OK. Any alternative
   definition (such as body definitions dependent on type) would prevent a
   `Body` type from standing on its own - it would need to reference some other
   shared data structure to obtain the body definition.
3) Making bodies a key part of the `Microstate` forces all simulation algorithms
   to deal with them. Should an algorithm be challenging to implement for
   multi-site bodies, it could be written to handle only single-site bodies
   (body == particle) and raise an error when given more than one site.
4) Similarly, users now need to think about bodies and sites when performing
   any task. Hopefully, the new nomenclature makes it easier for them to
   understand. There will be methods to iterate over all *sites* in the system
   reference frame (similar to the old particle idea). It might also be
   possible to provide a `Particle` type that represents a single site body.

## Membership criteria

In statistical mechanics, the microstate is normally defined as the positions
and momenta of all bodies. HOOMD-blue (the Python package) follows this
definition closely and requires that users separate "parameters" (i.e. pair
potential epsilon/sigma, particle shape, etc...) from the state of the system.
In many types of simulations (e.g. alchemy), this is a problem as some of these
parameters formally become part of the microstate.

HOOMD-rs maintains the separation of the `Microstate` from the simulation
_model_, but through custom body/site properties and custom state data, HOOMD-rs
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

## Bodies and sites

For simplicity and to enable user-defined body and site types, `Microstate`
is generalized on both the body and site types and stores them in arrays of
structures. Code that uses `Microstate` can require one or more trait bounds to
ensure that the particle has the appropriate attributes.

_Ideally_, the `BodyProperties` and `SiteProperties` general types allow position
only to enable simulations of isotropic systems without needing to store unused
orientations. However, Rust lacks specialization. As the implementation
evolves, we will see if this remains feasible or if we can keep orientation
in a separate trait or not.

Here are the traits that the `*Properties` types may have:

* `Position<V>` - Holds a position vector that describes the location of the
  body or site in space.
* `Orientation<R>` - Holds an orientation that describes a rotation from a
  local reference frame to the space frame of a body or site.
* `Dynamic`? - Mass, velocity, ... needed for MD. TODO: Need separate
  names to differentiate between dynamic point particles and dynamic orientable
  particles? Nominally only applies to bodies. However, some algorithms
  (e.g. DPD) need to know the velocity of individual sites. OR: Do we make
  Mass, Velocity, AngularMomentum, etc... all separate traits to allow complete
  generality when any subset is required (or not).

## Topology

TODO: Describe how topology will be stored.

## Ghost sites

When periodic boundary conditions are employed (see the next section), model
methods need a clean way to compute interactions between the real sites
in the primary image and ghost sites in periodic images. HOOMD-blue stored
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
scale with the number of neighbors.

## Boundary conditions

TODO

## Body storage

`Microstate` will store a flat vector of bodies. To support O(1) removal of
bodies, each is given a tag. This allows the use of the O(1) swap_remove
method of vec while maintaining the identity of the bodies by tag. The
`body_indices` member will identify the current index of any given body
tag to allow O(1) lookups by tag (which may be None if the body with that
tag was removed). We also need to keep track of the free body tags to recycle
when adding new bodies:

* `bodies: Vec<Tagged<B>>`
* `body_indices: Vec<Option<usize>`
* `free_body_tags: BinaryHeap<Reverse<u32>>`.

For reasons described below, tags are reused in a monotonically increasing
order.

Removing a body should also either a) remove all bonds connected to sites
of that body or b) produce an error if the body participates in a bond.

We need to prevent callers from modifying the body's tag along with the other
attributes. Therefore, `tag` cannot be a field of a body. The `Tagged<T>` type
will store the tag along with the body. Read only access `tag()` will be
public, but the field itself will be `pub(crate)` to allow only this crate to
set the tag field.

Bonds are applied between sites, not bodies. This one requirement has profound
implications on how the flat list of sites must be managed. For simplicity,
one might think to store the flat site list in the same order as the bodies.
However, in the most general case bodies might dynamically change the number
of particles from one step to the next. We also might want to apply Hilbert
curve sorting on the sites to improve cache coherency. Therefore, sites must be
stored in their own vector of tagged sites. This also provides some amount of
compatibility with hoomd schema GSD files - the sites are the particles.

Actually, there are a number of problems in allowing the number of sites to
change from one step to the next. Say we have 2 bodies each with 4 particles.
The site layout (in tag order) is initially: `B0/S0, B0/S1, B0/S2, B0/S3, B1/S0,
B1/S1, B1/S2, B1/S3`.

* First problem: Say there is a bond between site tags 2 and 5. Now the caller
  changes `B0` to have only 3 particles in an update. How do we know which site
  tag to remove? Does this affect the bond or not?
* Second problem: The GSD schema orders sites within a body in increasing tag
  order. If we recycle site tags, we will break this order - for example by
  deleting site tag 3 (remove a particle from body 0) and then reuse site tag 3
  when adding a particle to body 1.

Therefore, the body update mechanism needs to prevent the number of sites in
the body from changing. At the same time, it should be possible to update the
properties of those sites. Sites can be dynamically allocated with an unchanging
size by storing them in a `Vec<Site>` and by never providing a mutable `Body`
to callers. `Microstate` will use `assert` as needed to ensure that the number
of sites does not change. We could use `Box<[Site]>` which is not as convenient
to resize, but a caller could still replace the site pointer when given a
mutable `Body`, so this offers no additional benefits. `Body` will provide
access to sites through an accessor method that provides a slice. Callers with
a mutable `Body` can rearrange the slice, but not change the number of elements. 

The Body should retain ownership of its sites in the process, as moving
ownership to the `Microstate` will prevent `Body` from standing on its own.
Callers that wish to change the number of sites can do so be removing a body
and then inserting a new one. Provided that we recycle tags in a monotonically
increasing order, the second problem above is solved. If there are any bonds,
the caller must deal with them explicitly which solves the first.

Going one step further: say that `Body` stores the sites in heap allocated
memory. MC updates of only position or orientation should not pay the
performance penalty necessary to set *all* body properties, including the
dynamically allocated sites. In the more general problem, users might implement
MC moves that change custom body fields, such as type. This suggests that we
should present `Body` as a type, not a trait, that is generalized on the bodies
copyable properties separate from the sites:
```
struct Body<P, S> {
    properties: P,
    sites: Vec<S>,
}
```

It might have been nice (though premature) to try and optimize away the
heap storage for fixed size bodies (e.g. single site particles), but making
`Body` a trait will open up the ability for callers to directly control its
contents and introduce bugs by changing the number of sites.

## Site storage

`Microstate` stores an expanded vector of sites. For example, a state with 10
bodies each made up of 4 sites would have 40 sites in the vector. Each of these
site's properties are placed in the reference frame of the system. As with
bodies, sites are given a unique tag that cannot be modified by the caller. In
addition to the tag, each site is also given a reference back to its body.

```
struct Site<S> {
   tag: u32,
   body: u32,
   properties: S,
}
```

As with bodies, we need to track the reverse mapping of site tags back to
indices and a list of free site tags from which we pull the smallest tags first.
In addition, we need to know the site indices that belong to bodies to enable
efficient updates and removals.

* `sites: Vec<Site>`
* `site_indices: Vec<Option<usize>`
* `free_site_tags: BinaryHeap<Reverse<u32>>`
* `bodies_sites: Vec<Vec<usize>`

## Virtual sites (Future)

The hoomd-rs developers have no initial plans to support virtual sites.
However, these do appear in a number of force fields and should be possible
to realize. Virtual sites are similar to bodies. The main difference is that
a virtual site's properties are the function of the properties of *multiple*
bodies whereas a rigid body's sites are a function of only one body. Given that
the `Microstate` design already decouples sites from bodies, this should be
possible. The main challenge to solve is the addition of data structures that
track which sites are set from what function of given bodies. Such a design
will be completed when it is needed.

## Ghost site storage

Ghost sites will be stored in a separate `Vec<Option<Site>>`. Adding
a new site also requires adding its ghosts (to the end of the ghosts Vec).
Removing a site will remove all of its ghosts. Ghost removal will operate
differently than above. Simply **moving** a body can result in the addition
or removal of ghosts -- when a body moves toward or away from a periodic
boundary. In a cost amortized neighbor list, ghosts may appear as neighbors of
multiple site. It would not cost O(1) to remove all those. Instead, we will
allow removed ghost particles to leave a `None` sentinel at the same index in
the array to maintain O(1) particle updates with a neighbor list.

Microstate will need to maintain auxiliary data structures to maintain
O(1) updates such as a list of ghost site indices for each site. 

Incremental updates to bodies must also update all of their sites and sites'
ghosts AND the linked spatial data structures. Complete system updates are
likely best off removing all ghosts/spatial data structures and rebuilding them
with the new sites. MC will be the primary driver of incremental updates and MD
will primarily use full system updates, although this is not a strict rule. With
amortized neighbor lists, these rebuilds will need to be coordinated with the
spatial data structure.

As of this writing, the spatial data structures have not yet been designed
for hoomd-rs. It is clear, however, that we would like the design to be
usable with or without a Microstate. Therefore, spatial data structures
will likely operate on a set of indexed site positions. Microstate
maintains two vectors, one for real sites and one for ghosts. One
solution (not ideal) would be to use signed indices in the spatial data
structures, with signed indices for real particles and negative values
for ghosts. This would not be ideal because it could cause off-by-one
errors when accessing ghosts (there is no -0 integer). We could
keep the same idea by using a new type for the index with helper methods
to decode the index separately from the real/ghost flag (stashed in the
highest bit of the integer).

## Update API methods

TODO: Single particle updates

Full system updates will occur via a method that takes a Fn that operates on a
mutable slice of particles. In this way, the update function can take steps such
as reinitializing all ghost particles and rebuilding spatial data structures
after calling the provided Fn.

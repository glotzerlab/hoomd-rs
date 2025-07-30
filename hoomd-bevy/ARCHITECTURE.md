# hoomd_bevy

The `hoomd_bevy` crate implements a Bevy plugin that interfaces hoomd
simulations with the Bevy game engine. It's primary use-case is to support
real-time interactive examples. However, users may find it helpful for other
tasks.

## Examples and tutorials

Users consume examples and tutorials in a number of ways, including reading
the code (with descriptions) on a web page, executing the example and observing
the output, and changing the example to see what happens.

Most production uses of `hoomd-rs` will not use Bevy. Therefore, long strings
of code that interface with Bevy will be distracting. To the extent possible,
`hoomd_bevy` separates the simulation and interface code so that the web version
of the tutorial can show only pure `hoomd-rs` examples. At the same time, Bevy
forces that code be structured so that it can execute one step at a time.
This is not necessarily a bad practice, but does add extra complication over
what could be written directly in a for loop in a single `main()` function.

## The HoomdBevyPlugin

To reduce the boilerplate needed for each example, the `HoomdBevyPlugin`
implements common functionality and interfaces:

* Camera controls (2D and 3D separately).
* Simulation step and frame pacing, with steps per second limiter.
* Pause and advance by single step controls.
* A help screen describing common controls (examples can add lines if needed).
* Key bindings to hide the UI and take screenshots.
* A menu to control common settings (steps per second limit, camera speed, etc.)

Most examples will start running the simulation right away. Most users would
want to see that. Some examples might work best if the user choose some options
before starting. The plugin might be able to accommodate deferred starts with
optional resources. This is not implemented now, but can be investigated when it
is needed.

`HoomdBevyPlugin` fills the full window with the simulation and adds text
overlays when requested in the upper left, lower left, and lower right
corners of the screen. Individual examples can add messages or buttons
to the upper right when needed (e.g. to control the simulation temperature).

Individual examples must provide their own implementations of:

* A `Simulation` with methods to advance the simulation and get the current
  step.
* `setup_simulation`: Create the `Simulation` resource and set the initial
  condition.
* `sync_simulation`: Synchronize the state of the simulation with Bevy entities
  for display.
* Any example-specific UI or controls.

At this point, it is not clear where the `Simulation` trait should live. On
one hand, it is currently only needed for `hoomd_bevy` (users can write simple
scripts that store all simulation fields in locals and implement the step in the
body of a for loop). On the other hand, users will see `impl Simulation for ...`
in the web page form of the examples and will copy that format. If `Simulation`
lived in `hoomd_bevy`, users may accidentally introduce a bevy dependency
that takes a long time to build (or may not even build on HPC platforms). If
`Simulation` lived in a different crate (which one?), they may feel obligated to
use it. This design question will be resolved in time as `hoomd-rs` develops.
For now, `Simulation` will be defined in `hoomd_bevy`. That is likely to change.

`hoomd_bevy` will provide a number of convenience methods that synchronize
subsets of the simulation state (body and/or site properties) to the Bevy
world with different visual representations (disks, polygons, spheres, etc...).
Individual examples can use these to reduce the complexity of `sync_simulation`,
but are also free to implement custom methods when needed (e.g. spheres with
velocity vectors). Due to the way Bevy is structured, each of these helper methods
is paired with a `setup` method that creates the needed geometry and material assets.
[This technique](https://www.reddit.com/r/bevy/comments/1bwq9a0/plugin_system_initialization_pattern/)
enables configurable setup via pipes.

## Error handling

While most of `hoomd-rs` uses specific error enums, `hoomd-bevy` defines
interfaces that return `anyhow::Result`. Writing custom error enums for each
example would be time consuming and serve no purpose as the examples are not
part of a library that can be directly used.

`anyhow` is convenient, easy to use, and provides nicely formatted output
when it is the return value of `main`. Therefore, it is a good thing to
encourage its use in production scripts. We will need to be careful that
its use does not propagate into `hoomd-rs` library API calls.

## Web assembly

Rust and Bevy make it possible to compile examples for WebAssembly so that
users could run the entire example in a web browser ([see the Bevy examples
here](https://bevy.org/examples/)). Implementing this in `hoomd-bevy` is not a
primary goal, but may be taken on in future work.

# hoomd_workspace

The `hoomd_workspace` crate allows Rust code to interoperate with signac projects.

## Design goals

`hoomd_workspace` extremely minimal by design. The design allows Rust projects
to:

1) Create signac job directories given a state point.
2) Determine the identifier and path of a state point.

That is it. To keep things as simple as possible, `hoomd_workspace` does not define
the notion of a project or a job. The workspace is always assumed to be in the
current working directory.

## Implementation

For compatibility with the signac Python package, `hoomd_workspace` must generate
the same identifiers. `hoomd_workspace` accomplishes this for any type that
implements `Serialize` using `serde_json`, `serde-json-fmt`, and `md-5`.

### Entry

The trait `Entry` provides methods that compute the identifier and get the path
of a state point. A blanket implementation allows callers to use these methods
on any type that implements `Serialize`.

### add

The `add` method adds a state point to the workspace. It creates the `workspace`
directory if it doesn't exist and then serializes the state point to
`workspace/{identifier}/signac_statepoint.json`.

### state_point

The `state_point` method reads the state point for the given identifier.

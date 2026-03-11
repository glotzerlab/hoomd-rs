# Contributing

Contributions are welcomed via `pull requests on GitHub
<https://github.com/glotzerlab/hoomd-rs/pulls>`__. Contact the **hoomd-rs**
developers before starting work to ensure it meshes well with the planned
development direction and standards set for the project.

## Features

### Contribute widely applicable features

Contribute new features to the core **hoomd-rs** that are likely to be used by
the *majority* of users. You should publish *your own* crates for capabilities
that are specific to a small subset of users. Please respect the HOOMD brand
name and choose a name for your crate that **does NOT** start with `hoomd-`.

### Implement functionality in a general and flexible fashion

New features should be applicable to a variety of use-cases. The **hoomd-rs**
developers can assist you in designing flexible interfaces.

## Version control

### Base your work off the correct branch

Base bug fixes, new functionality, and API incompatible changes on ``trunk``.

### Propose a minimal set of related changes

All changes in a pull request should be closely related. Multiple change sets
that are loosely coupled should be proposed in separate pull requests.

### Agree to the Contributor Agreement

All contributors must agree to the Contributor Agreement before their pull
request can be merged.

## Source code

### Use a consistent style

Follow the guidelines established in the **Code style** section of the
documentation.

### Comment sparingly

Add comments in the code, but only when necessary to explain *why* a particular
block of code is structured the way it is.

### Compile without warnings

Your changes should compile without warnings.

## Tests

### Write unit tests

Add unit tests for all new functionality.

### Validity tests

The developer should run research-scale simulations using the new functionality
and ensure that it behaves as intended.

## User documentation

### Document types, traits, and methods

Fully document your code using **rustdoc** documentation comments. Include
example snippets for *everything*.

### Add developer to the credits

Update the credits documentation to list the name and affiliation of each
individual that has contributed to the code.

### Propose a change log entry

Propose a concise entry describing the change in ``CHANGELOG.md``.

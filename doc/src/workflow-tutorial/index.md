# Workflow Tutorial

By now, you have learned how to implement your simulation model using *hoomd-rs*
and have been able to run it for a few test cases. Changing `main.rs` and
executing `cargo run` for each simulation is not very practical when you want to
run tens, hundreds, or thousands+ simulations. [Row] and [signac] can help
you with that.

This tutorial assumes that you already have a basic understanding of [row] and
[signac]. If you don't, you should read their tutorials first. Here, you will
learn the best practices to integrate your *hoomd-rs* simulations with [row]
and [signac] including:

* Reading the simulation parameters from a [signac] state point.
* Writing status messages to `stdout` using the [log] crate.
* Writing simulation trajectories to a [gsd] file.
* Logging properties of the simulation to [parquet] file(s).
* Saving the simulation state to a file to continue running it in a
  later job submission.
* Managing eligible, submitted, and completed state points with [row].

As such, this tutorial is broader in scope than the previous. The example is
not a single file, but a whole Git repository. The repository is a template,
designed for you to copy and modify for your own needs. As such, *don't clone*
or fork the repository (you don't need the template's commit history). You can
create your own new repository on GitHub [from the template] (and then clone
your own repository) or you can [download the template] for use locally.

Once you have a local copy, launch a terminal and change to that directory.
Then you can build the workflow binary with:
```shell
$ cargo build --release
```

Previous tutorials demonstrated the use of `cargo run` to both **build**
*and* **execute** the binary. This is a problem on HPC platforms where
compute nodes may not have the network access needed to build and
many parallel builds might conflict.

Use `cargo build --release` to build the binary once and then you can
run it with `target/release/action`.

> [!WARNING]
> You *must rerun* `cargo build --release` any time you change one of your
> `.rs` files. Unlike scripted languages you might be familiar with,
> Rust is compiled, so you must build your changes before they take
> effect.

You are also going to need to install [row] and [signac]. You can activate the
provided [Pixi] environment (bash and zsh):
```shell
$ eval "$(pixi shell-hook)"
```
or you can install [row] and [signac] using your preferred package manager(s).

Continue reading on the next page.

[row]: https://row.readthedocs.io
[signac]: https://signac.readthedocs.io
[log]: https://docs.rs/log
[gsd]: https://gsd.readthedocs.org
[parquet]: https://parquet.apache.org/
[from the template]: https://github.com/new?template_name=hoomd-workflow&template_owner=glotzerlab
[download the template]: https://github.com/glotzerlab/hoomd-workflow/archive/refs/heads/trunk.zip
[pixi]: https://pixi.prefix.dev

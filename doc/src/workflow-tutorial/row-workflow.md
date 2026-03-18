# The Row Workflow

Now, let's put this all together and automate the submission process with
[row].

## `workflow.toml`

This [row] `workflow.toml` configuration will run `action simulate` on
all the directories in the `workspace`:
```toml
[workspace]
value_file = "signac_statepoint.json"

[default.action]
command = "target/release/action $ACTION_NAME {directories}"
launchers = ["rayon"]

[[action]]
name = "simulate"
products = ["trajectory.gsd"]
resources.walltime.per_directory = "01:00:00"
resources.threads_per_process = 1
group.maximum_size = 1
```

### Command

```toml
command = "target/release/action $ACTION_NAME {directories}"
```
The `command` tells [row] how to launch the `action` binary. Use the
`$ACTION_NAME` environment variable instead of hard-coding `simulate`
so that `command` is ready for use when you add more actions subcommands
in the future.

### Launcher

*hoomd-rs* uses [rayon] for thread-level parallelism. Set the *rayon* launcher
to correctly configure the number of threads:
```toml
launchers = ["rayon"]
```
also set `threads_per_process` to a non-default value:
```toml
resources.threads_per_process = 1
```

> [!WARNING]
> Skip either of these settings and *hoomd-rs* will attempt to use *all* cores
> on the compute node (e.g. 128) even if SLURM has locked your job to 1 core.
> The resulting resource contention will cause your simulations to run
> *extremely slowly*.

How should you choose `threads_per_process`? You need to choose it appropriately
based on how you configure your simulation model. Most of *hoomd-rs* uses
only 1 thread, so that should be the default. In the current release, only
`ParallelSweep` uses multiple threads. Benchmark and see how your model scales
before submitting a set of jobs to a cluster. It would be a waste of your time
if you requested `threads_per_process=32`, but your model ran even faster with
`threads_per_process=8`.

### Maximum Size

At this time, you must set
```toml
group.maximum_size=1
```

A future version of this template will support bundling actions on many directories
in a single cluster job.

## Execute the Workflow

Execute:
```shell
row submit -n 1
```
to execute the `simulate` action on one of the eligible directories.
When it completes, you will find `log.parquet.0`, `trajectory.gsd`, and
`model.postcard` in the directory.

To see how the action can continue from where it left off, set
```toml
resources.walltime.per_directory = "00:06:00"
```
in `workflow.toml` and then
```shell
row submit -n 1
```
again.

This time, the action should quit after 1 minute (it defaults to a 5 minute wall
time buffer) and will print:
```
...
[INFO  hoomd_workflow::simulate] Step 15000 / 100000 (15%)
[INFO  hoomd_workflow::simulate] Step 16000 / 100000 (16%)
[INFO  hoomd_workflow::simulate] Step 17000 / 100000 (17%)
[INFO  hoomd_workflow::simulate] Step 18000 / 100000 (18%)
[INFO  hoomd_workflow::simulate] Stopping simulation, wall time limit reached.
```

Now, the job directory will contain `log.parquet.0`, `trajectory.in-progress.gsd`,
and `model.postcard`. Submit again and the action will pick up right where
it left off. Each subsequent submission will generate a new `log.parquet.N` file.
When it reaches step 100,000, `trajectory.in-progress.gsd` will be renamed to
`trajectory.gsd`.

[row]: https://row.readthedocs.io
[rayon]: https://docs.rs/rayon

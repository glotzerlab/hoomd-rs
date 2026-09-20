# The `simulate` Action

The `simulate` action will be executed by [row] (typically via a job queue
on an HPC resource). [Row] determines which directories are eligible and
passes them to the `simulate` action. The command line parsing is explained
on the next page. Eventually, the `simulate_one` method is called with
a path to the directory containing the [signac] state point.

```rust
pub fn simulate_one(directory: &Path) -> anyhow::Result<()> {
```

This document highlights important sections of the code. Find the
complete code in `src/simulate.rs`.

## Serialization and Wall Time Management

Most HPC resources limit the wall time your jobs may execute. To enable
long-running simulation models, this workflow template monitors the wall time
used and *serializes* the entire simulation state to a file a few minutes before
the time is up. At that point, your HPC job ends and [row] will indicate that
the directory is eligible again. When you submit a new job, `simulate` will
*deserialize* the simulation state and continue the simulation from where it
left off.

> [!NOTE]
> This example uses the [postcard] format to store the simulation model.
> [Postcard] is a simple and efficient binary file format. You could use
> any format supported by [serde] that you like.

When called, the `simulate_one` method is given a directory. It first needs
to determine if this directory should be continued from a serialized state
or initialized from the state point. The `get_model` method implements the
necessary logic. It first checks if `model.postcard` exists. If it does,
it deserializes the simulation state and returns it. If not, it initializes
a new simulation model from the [signac] state point:
```rust,ignore
    let model_file: PathBuf = [Path::new("workspace"), directory, Path::new(MODEL_FILE)]
        .iter()
        .collect();
    match fs::read(&model_file) {
        Ok(bytes) => {
            debug!("Continuing simulation from `{}`.", model_file.display());

            postcard::from_bytes(&bytes)
                .with_context(|| format!("could not read `{}`", model_file.display()))
        }
        Err(error) => match error.kind() {
            io::ErrorKind::NotFound => {
                debug!("Constructing a new simulation model.");
                let state_point: StatePoint = hoomd_workspace::state_point(directory)
                    .context("could not read state point")?
                    .ok_or(anyhow!("state point not found"))?;
                let _ =
                    HoomdGsdFile::create(state_point.path()?.join("trajectory.in-progress.gsd"));
                LennardJonesModel::new(state_point)
            }
            _ => Err(error).with_context(|| format!("Could not read `{MODEL_FILE}`")),
        },
    }
```

`simulate_one` breaks out of the main simulation loop when the wall time is nearly
up and then serializes the simulation state to a file:
```rust,ignore
let maybe_wall_time_limit = match env::var("ACTION_WALLTIME_IN_MINUTES") {
    Ok(value) => {
        let parsed_value = value
            .parse::<f64>()
            .context("error parsing ACTION_WALLTIME_IN_MINUTES")?;
        debug!("Limiting wall time to {parsed_value} minutes.");
        Some(parsed_value * 60.0)
    }
    Err(_) => None,
};

while model.step() < TOTAL_STEPS {
    // ...
    if let Some(wall_time_limit) = maybe_wall_time_limit
        && wall_time + WALL_TIME_BUFFER > wall_time_limit
    {
        info!("Stopping simulation, wall time limit reached.");
        break;
    }
}

let out_bytes: Vec<u8> = postcard::to_stdvec(&model)?;
let mut file =
    File::create(job_directory.join(MODEL_FILE)).context("failed to create `{MODEL_FILE}`")?;
file.write_all(&out_bytes)
    .context("failed to write `{MODEL_FILE}`")?;
```

[Row] notifies the action of its wall time limit via the `ACTION_WALLTIME_IN_MINUTES`
environment variable.

## GSD Trajectory

`simulate_one` appends to a GSD trajectory for offline visualization and
analysis:

```rust,ignore
let mut gsd_file = HoomdGsdFile::open(job_directory.join("trajectory.in-progress.gsd"))
    .context("error opening trajectory.in-progress.gsd")?;
```
`open` works here because `get_model` created the GSD file.

While the simulation is active, it names the file `trajectory.gsd.in-progress`.
The [row] workflow configuration (in a later page) will use the existence of
`trajectory.gsd` as an indication that the simulation is complete (and therefore
no longer eligible). To achieve this, `simulate_one` closes the gsd file
and then renames it when the simulation has reached the target number of total
steps:

```rust,ignore
drop(gsd_file);

if model.step() == TOTAL_STEPS {
    info!("Simulation complete.");
    fs::rename(
        job_directory.join("trajectory.in-progress.gsd"),
        job_directory.join("trajectory.gsd"),
    )
    .context("failed to rename `trajectory.in-progress.gsd` to `trajectory.gsd`")?;
}
```

## Log File

[Parquet] is a convenient file format for logging because it is widely
supported and the binary format keeps the full precision of every logged value.
Unfortunately, they are difficult to use when continuing a simulation job
because [parquet] files cannot be appended to.

The solution suggested by the [parquet] developers is to create multiple files.
The `create_unique` method creates `log.parquet.0` on the first submission then
`log.parquet.1` on the second, and so on:
```rust,ignore
let mut parquet_logger =
    ParquetLogger::<LogRecord>::create_unique(job_directory.join("log.parquet"))
        .context("error creating `log.parquet`")?;
```

When you visualize or post-process the log later, you should concatenate
the data frames from all `log.parquet.*` files.

## Error Handling

You might have noticed a recurring pattern in the above code:
`.context("...")?`. The `?` operator is a shortcut that returns early whenever
the preceding code generates an error (see [Recoverable Errors with Result]
in [The Rust Programming Language] for details).

The `.context` method comes from the [anyhow] crate. Use it to describe what you
are doing that might cause an error. Without the `.context`, your program might
print `I/O error` as its entire output with no indication of what caused it. A later
page in this tutorial will show example error messages with context and show you
how to troubleshoot errors.

## Driving the Simulation and I/O

Not shown here is the standard code for advancing the simulation with `advance()`,
writing to the GSD file, and writing to the log file. See the full code in
`src/simulate.rs`. Tutorials such as [Applying Interactions] and
[Patchy Particle Self-Assembly] explain in detail how to advance simulation
models and write to log and GSD files.

[row]: https://row.readthedocs.io
[signac]: https://signac.readthedocs.io
[postcard]: https://docs.rs/postcard/
[serde]: https://serde.rs
[Applying Interactions]: ../mc-tutorial/applying-interactions.md
[Patchy Particle Self-Assembly]: ../mc-tutorial/patchy-particle-self-assembly.md
[parquet]: https://parquet.apache.org/
[Recoverable Errors with Result]: https://doc.rust-lang.org/stable/book/ch09-02-recoverable-errors-with-result.html
[The Rust Programming Language]: https://doc.rust-lang.org/stable/book/
[anyhow]: https://docs.rs/anyhow

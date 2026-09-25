# Compatibility with other crates

*hoomd-rs* interoperates with many other crates. When you use these crates in
your own project, you should match versions to avoid incompatibility errors.
For example, when *hoomd-rs* uses `parquet` 60.0 and your crate uses `parquet`
59.0, you may get an error like this:

```text
ierror[E0277]: the trait bound `for<'a> &'a [LogRecord]: parquet::record::record_writer::RecordWriter<LogRecord>` is not satisfied
  --> src/simulate.rs:73:9
   |
73 |         ParquetLogger::<LogRecord>::create_unique(job_directory.join("log.parquet"))
   |         ^^^^^^^^^^^^^^^^^^^^^^^^^^ the trait `for<'a> parquet::record::record_writer::RecordWriter<LogRecord>` is not implemented for `&'a [LogRecord]`
   |
note: there are multiple different versions of crate `parquet` in the dependency graph
  --> /home/runner/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/parquet-59.3.0/src/record/record_writer.rs:33:1
   |
33 | pub trait RecordWriter<T> {
   | ^^^^^^^^^^^^^^^^^^^^^^^^^ this is the expected trait
   |
  ::: /home/runner/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/parquet-60.0.0/src/record/record_writer.rs:33:1
   |
33 | pub trait RecordWriter<T> {
   | ------------------------- this is the found trait
```

You can *independently* use other versions of these crates without problems. Compile
errors only occur when you try to combine your usage of the crate with methods in
*hoomd-rs*.

Use these dependencies (only those needed) in your `Cargo.toml` to ensure compatibility
with this release of *hoomd-rs*:

```toml
anyhow = "1.0.100"
approxim = { version = "0.6.6", features = ["num-complex"] }
arrayvec = { version = "0.7.6", features = ["serde"] }
bevy = "0.19"
bevy_egui = "0.42.0"
log = "0.4.28"
num-complex = { version = "0.4.6", features = ["std", "serde"] }
parquet = { version = "60.0.0", default-features=false }
parquet_derive = { version = "60.0.0", default-features=false }
rand = { version = "0.10.0", default-features=false, features = ["std", "std_rng"] }
rand_distr = "0.6.0"
serde = { version = "1.0.228", features = ["derive"]}
serde_json = "1.0.150"
```

*hoomd-rs* will change these versions only in **major** releases to avoid breaking
your projects. On *ReadTheDocs*, use the version selector to view the requirements
specific to the version of *hoomd-rs* that you are using.

> [!NOTE]
> Of these crates, `bevy` and `parquet` are the most likely to be incompatible.
> They regularly make breaking releases.

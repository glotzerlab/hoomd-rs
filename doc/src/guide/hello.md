# Hello, hoomd-rs!

You don't need to install anything other than [Rust] to use *hoomd-rs*.
[Rust] comes packaged with [Cargo]. Use [Cargo] to create a
`hello-hoomd-rs` *crate*:
```shell
$ cargo new hello-hoomd-rs
$ cd hello-hoomd-rs
```

This example uses the [`hoomd-interaction`] crate. Add it to your
dependencies with:
```shell
$ cargo add hoomd-interaction
```

[Cargo] creates a "Hello, world" application by default.
Replace `src/main.rs` with:
```rust,ignore
{{#include ../../../examples/hello.rs}}
```

Compile and run the code:
```shell
$ cargo run
```
You should see a number of *compiling...* messages and then:
```shell
lennard_jones(1.5): -0.32033659427857464
```

Congratulations for making it this far! You have successfully compiled
*hoomd-rs* and used it to compute the Lennard-Jones potential at a given
*r* value. Read the following tutorials to learn how to perform Monte Carlo
and molecular dynamics simulations using *hoomd-rs*.

> [!TIP]
> `cargo run` builds in **debug** mode by default. Debug builds make it easier
> to troubleshoot problems with your code, but they run **very slowly**. When
> your code is working, build in **release** mode and it will run much faster:
> ```shell
> $ cargo run --release
> ```

> [!NOTE]
> On HPC platforms, you should run (preferably in an interactive compute job):
> ```shell
> $ cargo build --release
> ```
> You can then use the executable placed in the `target/release` directory in
> your submitted job scripts. For more details, see the [workflow tutorial].
>
> Unlike scripting languages (e.g. Python), saving changes to `main.rs` *will
> NOT take effect* until you run `cargo build --release` again.

[Cargo]: https://doc.rust-lang.org/cargo/index.html
[Rust]: https://www.rust-lang.org/
[crates.io]: https://crates.io/
[`hoomd-interaction`]: ../api/hoomd_interaction/index.html
[workflow tutorial]: ../workflow-tutorial/index.md

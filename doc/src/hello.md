# Hello, hoomd-rs!

## Rust

The first thing you need to understand about *hoomd-rs* is that it is not an
application or package that you install. Rather it is a **[Rust] crate** that you
can use to implement your simulation models.

If you are not familiar with [Rust], it is a relatively new programming language
that (in the opinion of the *hoomd-rs* developers) combines the best features of
generic languages like Python with the best features of strongly-typed compiled
languages like C++. *hoomd-rs* takes full advantage of these capabilities to
provide a simulation framework that is fully customizable while still compiling
down to machine code. [The Rust Programming Language] explains everything you
need to know about the language itself.

Before continuing, you will need to install [Rust]. On Linux, Mac, and WSL
you can install Rust with a single command:
```shell
$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
For more details, including instructions for a native Windows installation, see
the [Rust installation documentation] (make sure you install the 64-bit build).

> [!TIP]
> *hoomd-rs* works very well with *native* builds on all platforms.

## Using *hoomd-rs* in your application

At this time, *hoomd-rs* is not yet published on [crates.io]. To use it, you
need to first clone the repository:
```shell
$ git clone git@github.com:glotzerlab/hoomd-rs
```

To keep your project separate from *hoomd-rs*, the following steps create a
directory structure like this:
```text
- hoomd-rs
  - ...
- project
  - Cargo.toml
  - src
    - main.rs
```

`git clone` created `hoomd-rs` for you. To create the `project` directory, run:
```shell
$ cargo new project
$ cd project
```
Feel free to use another name in place of `project`.

Then you need to add *hoomd-rs* dependencies to your project. *hoomd-rs*
consists of many crates (see the [API documentation] for details). This example
uses [`hoomd-interaction`]:
```shell
$ cargo add --path ../hoomd-rs/hoomd-interaction hoomd-interaction
```
Repeat this command for each crate that you *directly use* in your project.
Cargo will install the dependencies of your dependencies automatically.

Replace `src/main.rs` with:
```rust,ignore
{{#include ../../examples/hello.rs}}
```

Compile and run the code:
```shell
$ cargo run
```
You should see a number of compiling... messages and then:
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
> $ RUSTFLAGS="-C target-cpu=native" cargo run --release
> ```


> [!NOTE]
> On HPC platforms, you should run:
> ```shell
> $ RUSTFLAGS="-C target-cpu=native" cargo build --release
> ```
> on the login node and and then use the executable placed in the
> `target/release` directory in your submitted job scripts.
>
> Unlike scripting languages (e.g. Python), saving changes to `main.rs` will
> not take effect until you run
> `RUSTFLAGS="-C target-cpu=native" cargo build --release` again.


[Rust]: https://www.rust-lang.org/
[Rust installation documentation]: https://www.rust-lang.org/tools/install
[The Rust Programming Language]: https://doc.rust-lang.org/stable/book/
[API documentation]: api.md
[crates.io]: https://crates.io/
[`hoomd-interaction`]: api/hoomd_interaction/index.html

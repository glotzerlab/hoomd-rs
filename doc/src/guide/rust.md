# Rust

The first thing you need to understand about *hoomd-rs* is that it is not an
application or package that you install. Rather it is a collection of **[Rust]
crates** that you can use to implement your simulation models.

If you are not familiar with [Rust], it is a relatively new programming language
that combines the best features of generic languages like Python with the best
features of strongly-typed compiled languages like C++. *hoomd-rs* takes full
advantage of these capabilities to provide a simulation framework that is fully
customizable while still compiling down to machine code. [The Rust Programming
Language] explains everything you need to know about the language itself.

Before continuing, you will need to install [Rust]. On Linux, Mac, and WSL
you can install Rust with a single command:
```shell
$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
For more details, including instructions for a native Windows installation, see
the [Rust installation documentation] (make sure you install the 64-bit build).

> [!TIP]
> *hoomd-rs* works very well with *native* builds on all platforms.

[Rust]: https://www.rust-lang.org/
[Rust installation documentation]: https://www.rust-lang.org/tools/install
[The Rust Programming Language]: https://doc.rust-lang.org/stable/book/

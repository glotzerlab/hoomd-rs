# Running the Examples

Most of the following tutorials show the example executing in the browser. Click
on the display to focus it. Click on the rest of the page to cancel that focus.

While focused, you can interact with the simulation. All the examples have a few
common controls. For example, you can press `<space>` to pause the simulation
and then `<return>` to advance it one step at a time. You can also press `<esc>`
to bring up the settings screen and then `-` to decrease the simulation speed.
Press `h` to see a help screen that shows all the keyboard controls.

Some examples provide a relatively static view while others give you more
opportunity to interact with the simulation. See the on-screen help messages of
each for details.

## Building Desktop Applications

If you want to tweak an example's code and see what happens, you will need
compile your changes.

First, clone the *hoomd-rs* repository if you have not done so already:
```shell
$ git clone git@github.com:glotzerlab/hoomd-rs
```

Then change to the repository directory:
```shell
$ cd hoomd-rs
```
You will find the example code in `examples`.

To compile and run an example as a desktop application, execute:
```shell
$ cargo run --release --features=bevy --example {example}
```
where `{example}` is the name of the example *without* the path or extension
(e.g. `random-walk`, will build and run `examples/mc-tutorial/random-walk.rs`).

The examples use the [Bevy] engine. If you get compile errors when building
`bevy` crates, you may need to [install additional software]. MacOS is the
simplest platform to configure, as you only need XCode. On Linux, you will need
to install a number of system packages depending on your distribution. You will
need Visual Studio with some additional components for native windows builds
(TODO: test). In WSL, you can try building for Linux in WSL, but be aware that
there are many limitations in WSL's support for graphics. One guide recommends
that you [cross compile for native Windows] in WSL (TODO: test).

> [!NOTE]
> These additional software dependencies are *only* needed to build examples
> with interactive displays, or your own code that uses the `hoomd-bevy` crate.
> You do not need to install additional software to build and run command line
> applications with *hoomd-rs*, you only need Rust.

## Building Browser Applications

If you are having problems building Bevy for desktop and you would still like to
compile modified examples, try building the examples for the web.

First install a few additional tools to add web support to your Rust installation:
```shell
$ rustup target install wasm32-unknown-unknown
$ cargo install wasm-bindgen-cli wasm-server-runner
```

To build and run an example, execute:
```shell
$ cargo run --features bevy --target wasm32-unknown-unknown --example {example}
```
then open the printed `http://localhost` URL in your browser.

[Bevy]: https://bevy.org/
[install additional software]: https://bevy.org/learn/quick-start/getting-started/setup/#installing-os-dependencies
[cross compile for native Windows]: https://bevy-cheatbook.github.io/platforms/windows/wsl2.html

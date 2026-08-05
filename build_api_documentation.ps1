# Build documentation with `--no-deps` so that the sidebard is not polluted
# with hundreds of extra crates and the build time is kept reasonable.
# HOWEVER: `cargo doc` fails to build packages in the correct order when
# `--no-deps` is set (e.g. it may build `hoomd-mc` before `hoomd-geometry`).
# rustdoc works by examining the current target/doc directory and adding new
# crates to what is already present. The solution is to build one crate at
# a time, buiding all dependents first.

# Stop on errors; treat unset variables as errors (PowerShell does this for variables)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Set env var for this session
$env:RUSTDOCFLAGS = "--html-in-header google_analytics.html --html-in-header katex.html"

$packages = @(
  'derive',
  'linear-algebra',
  'simulation',
  'utility',
  'rand',
  'vector',
  'gsd',
  'manifold',
  'spatial',
  'geometry',
  'microstate',
  'interaction',
  'mc',
  'bevy',
  'md'
)

foreach ($package in $packages) {
  cargo doc --package "hoomd-$package" --lib --no-deps
  Copy-Item .\katex.html "hoomd-$package\"
  Copy-Item .\README.md "hoomd-$package\"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }  # ensure non-zero exit stops the script
}
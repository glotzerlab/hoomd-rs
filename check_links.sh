#!/bin/bash

# Assemble the full documenation and check links with lychee.
# This must be run locally because GitHub Actions runners are
# often blocked by websites, leading to false positive errors.

set -euo pipefail

mdbook build doc
./build_api_documentation.sh
mkdir -p doc/book/api
mv target/doc/* doc/book/api/

lychee README.md --cache
lychee doc/book --root-dir doc/book --cache

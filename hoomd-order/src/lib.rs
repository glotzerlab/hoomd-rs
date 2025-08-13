// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Implements methods for analyzing microstate data. The basis for much of the analyses is
the `SpatialHistogram` struct, which allows the user to construct histograms from any data type
which implements `Add`.
*/

mod density;

pub use density::SpatialHistogram;

// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! TODO DOCS */

rust
struct Tag {pub id: i32}


struct CellList<const N: usize> {
    pub width: i32,
    pub origin: [i32; N],
    pub map: HashMap<[i32; N], Vec<Tag>>
}
// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Define physical models by applying partile interactions to microstates.

TODO: Expand documentation.
*/

pub mod external;

/**

*/
pub trait Energy<M> {
    /**

    */
    fn energy(&self, microstate: &M) -> f64;
}

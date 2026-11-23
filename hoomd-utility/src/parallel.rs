// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Manage the `ForkUnion` thread pool.
//!
//! [`ForkUnion`] requires that callers pass in an explicit thread pool with
//! every parallel operation. The **hoomd-rs** API will use parallel operations
//! in many inner calls and it would be onerous for the user to manually manage
//! the pool and for the API to pass them from all top level functions down to
//! where they are needed. Therefore, this module defines a single shared thread
//! pool that all methods can access through [`with_thread_pool`].
//!
//! [`ForkUnion`]: https://github.com/ashvardanian/ForkUnion
//!
//! # Example
//!
//! ```
//! use forkunion::prelude::*;
//! use hoomd_utility::parallel::with_thread_pool;
//!
//! let mut data: Vec<usize> = (0..1000).collect();
//!
//! with_thread_pool(|pool| {
//!     data.as_mut_slice()
//!         .into_par_iter()
//!         .with_schedule(pool, DynamicScheduler)
//!         .for_each(|value, _task, _at| {
//!             *value *= 2;
//!         })
//! }).expect("should be able to execute parallel work");
//! assert_eq!(data, (0..1000).map(|value| value * 2).collect::<Vec<_>>());
//! ```

use std::{
    collections::BTreeMap,
    sync::{LazyLock, Mutex},
};

use forkunion::{ComputeDomain, ThreadPool, Topology};

/// The global thread pool.
static THREAD_POOL: LazyLock<Mutex<ThreadPool>> = LazyLock::new(|| {
    let topology = Topology::new().expect("should be able to detect hardware topology");
    Mutex::new(forkunion::spawn(&topology, default_thread_count(&topology)))
});

/// Execute a closure with exclusive mutable access to the global thread pool.
///
/// **hoomd-rs** methods should always use [`with_thread_pool`] when launching
/// parallel work. By default, it initializes to the number of high
/// performance cores available.
///
/// - On Apple silicon Macs, this will choose the number of the highest performing cores.
/// - In SLURM jobs (or when otherwise constrained by `cpuset` or `cgroups`) it will
///   automatically choose the number of requested cores.
/// - In other situations, it will choose the total number of CPU cores in the system.
///
/// The pool initializes on first use. The lock is held throughout the closure
/// and released when it returns; independent callers serialize their parallel
/// operations. Returns the closure's result.
///
/// Do not call this function recursively or from a task running on the pool,
/// as doing so can deadlock.
///
/// # Panics
///
/// Panics if hardware topology detection fails or the thread pool mutex is
/// poisoned.
#[inline]
pub fn with_thread_pool<R, F: FnOnce(&mut ThreadPool) -> R>(f: F) -> R {
    let mut pool = THREAD_POOL
        .lock()
        .expect("thread pool mutex should not be poisoned");
    f(&mut pool)
}

/// Determine the default number of threads
fn default_thread_count(topology: &Topology) -> usize {
    // Some systems, such as Macs, have some high performance and some low
    // performance cores. Choose only the high performance cores by default.
    // On SLURM clusters, this implementation will also default to the number
    // of cores assigned by the scheduler.

    let mut cores_by_level = BTreeMap::new();
    for domain in 0..topology
        .compute_domains_count()
        .expect("should be able to count compute domains")
    {
        let domain = ComputeDomain(domain);
        let cores = topology
            .logical_cores_count_in(domain)
            .expect("should be able to count logical cores in the domain");
        let level = topology
            .compute_level_in(domain)
            .expect("should be able to query the compute level in the domain");

        cores_by_level
            .entry(level)
            .and_modify(|curr| *curr += cores)
            .or_insert(cores);
    }
    cores_by_level.last_entry().map_or(1, |entry| *entry.get())
}

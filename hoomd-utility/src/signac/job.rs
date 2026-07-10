// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Interoperate with signac workspaces.

use md5::{Digest, Md5};
use serde::Serialize;
use serde_json_fmt::JsonFormat;

pub struct Job;

impl Job {
    /// Compute the job id hash.
    ///
    /// Operates on any type that is serializable to JSON.
    ///
    /// # Errors
    ///
    /// Returns [`serde_json::Error`] when `state_point` cannot be serialized to JSON.
    #[inline]
    #[expect(
        clippy::missing_panics_doc,
        reason = "Panic would only occur due a bug in the code."
    )]
    pub fn compute_job_id<T: ?Sized + Serialize>(state_point: &T) -> serde_json::Result<String> {
        // This implementation is compatible with the Python implementation of signac:
        // https://github.com/glotzerlab/signac/blob/43655eeb22c25aba4ddd4421f702d5352cd29ca8/signac/job.py#L34-L54

        let mut value = serde_json::to_value(state_point)?;
        value.sort_all_objects();

        let formatted = JsonFormat::new()
            .comma(", ")
            .expect("format should be valid")
            .colon(": ")
            .expect("format should be valid")
            .format_to_string(&value)?;

        let hash = Md5::digest(formatted.as_bytes());
        Ok(hex::encode(hash))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rstest::*;

    #[derive(Serialize)]
    struct Test1 {
        a: u32,
    }

    #[derive(Serialize)]
    struct Test2 {
        b: String,
    }

    #[derive(Serialize)]
    struct Test3 {
        b: f64,
        a: Option<u32>,
    }

    #[derive(Serialize)]
    struct Test4 {
        z: Vec<f64>,
        x: i64,
        a: u32,
    }

    #[derive(Serialize)]
    struct Test5 {
        test_3: Test3,
        test_4: Test4,
        a: u64,
    }

    #[rstest]
    #[case(Test1 { a: 1}, "42b7b4f2921788ea14dac5566e6f06d0")]
    #[case(Test1 { a: 932_164}, "675a00e1b14ee1d618d783ea2205ff45")]
    #[case(Test2 { b: "some_string".into() }, "594d1ea83433eefb661113b866e6eeba")]
    #[case(Test2 { b: "another_string".into() }, "26c23ac85e2be5058fab7ca3531f5244")]
    #[case(Test3 { b: 7.897_231_4, a: None }, "2ab6264db9442f72fc975d63d1eea743")]
    #[case(Test3 { b: 7.897_231_4, a: Some(63) }, "6e9713313e9c7e6746eee47934d3f59e")]
    #[case(Test4 { z: vec![1.125, -4.25, 8.9375], x: -12, a: 18 }, "00d5437b248864b98a24dc9a96dc083c")]
    #[case(Test4 { z: vec![], x: -204, a: 0 }, "293e5fe23ff59e75d3ff9241c596670a")]
    #[case(Test5 { test_3: Test3 { b: 7.897_231_4, a: None }, test_4: Test4 { z: vec![], x: -204, a: 0 },
        a: 2_u64.pow(42) }, "c0efafb8312c9d9be48dffb560b36422")]
    fn test_compute_job_id<T: Serialize>(
        #[case] state_point: T,
        #[case] job_id: &str,
    ) -> anyhow::Result<()> {
        assert_eq!(Job::compute_job_id(&state_point)?, job_id);

        Ok(())
    }
}

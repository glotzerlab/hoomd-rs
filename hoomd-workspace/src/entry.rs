// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement `Entry`

use std::path::{Path, PathBuf};

use md5::{Digest, Md5};
use serde::Serialize;
use serde_json_fmt::JsonFormat;
use super::Error;

/// Create the Python-compatible JSON formatted string.
///
/// Sort keys and add spaces after `:` and `,`.
pub(crate) fn formatted<T: ?Sized + Serialize>(state_point: &T) -> Result<String, Error> {
        // This implementation is compatible with the Python implementation of signac:
        // https://github.com/glotzerlab/signac/blob/43655eeb22c25aba4ddd4421f702d5352cd29ca8/signac/job.py#L34-L54

        let mut value = serde_json::to_value(state_point).map_err(Error::Serialize)?;
        value.sort_all_objects();

        JsonFormat::new()
            .comma(", ")
            .expect("format should be valid")
            .colon(": ")
            .expect("format should be valid")
            .format_to_string(&value).map_err(Error::Format)
    }

/// Compute properties of entries in the workspace
///
/// Each entry in the workspace (called a "job" in *signac*) is uniquely identified
/// by a state point and stored in the [`path`] `workspace/{identifier}`. The state
/// point contains all the parameters (name and value) that are needed to uniquely
/// identify the point. The state point type (`Self`) must be serializable to JSON.
/// The entry's [`identifier`] is the *md5* hash of the JSON representation of the
/// state point.
///
/// [`Entry`] is implemented for all types that implement [`Serialize`].
///
/// # Example
///
/// ```
/// use hoomd_workspace::Entry;
/// use serde::Serialize;
///
/// #[derive(Serialize)]
/// struct MyStatePoint {
///     temperature: f64,
///     pressure: f64,
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let my_state_point = MyStatePoint { temperature: 1.0, pressure: 2.0 };
/// # Ok(())
/// # }
/// ```
///
/// [`path`]: Self::path
/// [`identifier`]: Self::identifier
pub trait Entry {
    /// Compute the state point's identifier.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_workspace::Entry;
    /// use serde::Serialize;
    ///
    /// #[derive(Serialize)]
    /// struct MyStatePoint {
    ///     temperature: f64,
    ///     pressure: f64,
    /// }
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let my_state_point = MyStatePoint { temperature: 1.0, pressure: 2.0 };
    ///
    /// let identifier = my_state_point.identifier()?;
    ///
    /// assert_eq!(identifier, "bb97883a3a70ccfc0840d49a8c794342");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// * [`Error::Serialize`] when `serde_json` cannot serialize `self` to JSON.
    /// * [`Error::Format`] when `serde_json` cannot format the JSON as a string.
    fn identifier(&self) -> Result<String, Error>;

    /// Compute the path to the state point.
    ///
    /// # Example
    ///
    /// ```
    /// use std::path::Path;
    /// use hoomd_workspace::Entry;
    /// use serde::Serialize;
    ///
    /// #[derive(Serialize)]
    /// struct MyStatePoint {
    ///     temperature: f64,
    ///     pressure: f64,
    /// }
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let my_state_point = MyStatePoint { temperature: 1.0, pressure: 2.0 };
    ///
    /// let path = my_state_point.path()?;
    ///
    /// assert_eq!(path, Path::new("workspace").join("bb97883a3a70ccfc0840d49a8c794342"));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// * [`Error::Serialize`] when `serde_json` cannot serialize `self` to JSON.
    /// * [`Error::Format`] when `serde_json` cannot format the JSON as a string.
    #[inline]
    fn path(&self) -> Result<PathBuf, Error> {
        Ok(Path::new("workspace").join(self.identifier()?))
    } 
}

impl<T> Entry for T
where
    T: ?Sized + Serialize
    {
    #[inline]
    fn identifier(&self) -> Result<String, Error> {
        let formatted = formatted(&self)?;
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
    fn test_identifier<T: Serialize>(
        #[case] state_point: T,
        #[case] job_id: &str,
    ) -> anyhow::Result<()> {
        assert_eq!(state_point.identifier()?, job_id);

        Ok(())
    }

    #[test]
    fn test_path() -> anyhow::Result<()> {
        let state_point = Test1 { a: 1};

        assert_eq!(state_point.path()?, Path::new("workspace/42b7b4f2921788ea14dac5566e6f06d0"));

        Ok(())
    }
}

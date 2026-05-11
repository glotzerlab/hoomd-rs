use std::path::Path;

use anyhow::Context;
use hoomd_gsd::file_layer::GsdFile;

/// Maximum buffer size (in bytes) for a single type shape JSON string.
const TYPE_SHAPE_BUFFER_SIZE: usize = 1024;

/// Maximum buffer size (in bytes) for a particle type name.
const TYPE_NAME_BUFFER_SIZE: usize = 64;

/// Create a GSD file and write type metadata in frame 0.
///
/// Write `particles/types` and `particles/type_shapes` so that visualization
/// tools (e.g. Ovito) can render the correct shapes for each particle type.
/// The GSD readers fall back to frame 0 for these metadata chunks, so they only
/// need to be written once.
pub fn create_with_type_shapes(
    path: &Path,
    type_names: &[&str],
    type_shapes: &[&str],
) -> anyhow::Result<()> {
    let mut gsd_file = GsdFile::create(path, "hoomd-workflow", "hoomd", (1, 4))
        .with_context(|| {
            format!("error creating GSD file at {}", path.display())
        })?;

    write_type_names(&mut gsd_file, type_names)?;
    write_type_shapes(&mut gsd_file, type_shapes)?;

    gsd_file.end_frame().context("error ending frame")?;

    Ok(())
}

fn write_type_names(
    gsd_file: &mut GsdFile,
    type_names: &[&str],
) -> anyhow::Result<()> {
    let max_len = type_names.iter().map(|s| s.len()).max().unwrap_or(0);
    anyhow::ensure!(
        max_len < TYPE_NAME_BUFFER_SIZE,
        "type name is too long ({max_len} bytes, max {})",
        TYPE_NAME_BUFFER_SIZE - 1,
    );

    gsd_file
        .write_arrays(
            "particles/types",
            type_names.iter().map(|s| -> [u8; TYPE_NAME_BUFFER_SIZE] {
                std::array::from_fn(|i| {
                    if i < s.len() { s.as_bytes()[i] } else { 0 }
                })
            }),
        )
        .context("error writing particles/types")?;

    Ok(())
}

fn write_type_shapes(
    gsd_file: &mut GsdFile,
    type_shapes: &[&str],
) -> anyhow::Result<()> {
    let max_len = type_shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    anyhow::ensure!(
        max_len < TYPE_SHAPE_BUFFER_SIZE,
        "type shape JSON is too long ({max_len} bytes, max {})",
        TYPE_SHAPE_BUFFER_SIZE - 1,
    );

    gsd_file
        .write_arrays(
            "particles/type_shapes",
            type_shapes.iter().map(|s| -> [i8; TYPE_SHAPE_BUFFER_SIZE] {
                std::array::from_fn(|i| {
                    if i < s.len() {
                        s.as_bytes()[i] as i8
                    } else {
                        0
                    }
                })
            }),
        )
        .context("error writing particles/type_shapes")?;

    Ok(())
}

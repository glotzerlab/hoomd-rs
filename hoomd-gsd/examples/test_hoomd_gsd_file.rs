use hoomd_gsd::hoomd::HoomdGsdFile;

fn main() -> anyhow::Result<()> {
    let mut hoomd_gsd_file = HoomdGsdFile::create("test.gsd")?;
    hoomd_gsd_file.append_frame(1000)?
        .particles_position([[0.0, 1.0, 2.0].into(), [3.0, 6.0, 12.0].into()]);
    hoomd_gsd_file.append_frame(2000)?;
    hoomd_gsd_file.append_frame(3000)?;
    Ok(())
}

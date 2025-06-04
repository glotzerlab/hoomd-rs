use hoomd_gsd::file_layer::{GsdFile, Mode};

fn main() -> anyhow::Result<()> {
    let mut gsd_file = GsdFile::open("test.gsd", Mode::Read)?;

    println!("{:?}", gsd_file);
    println!("{:?}", gsd_file.read_array::<f32>(0, "particles/position")?);
    println!("{:?}", gsd_file.read_array::<u64>(0, "configuration/step")?);



    let file2 = GsdFile::create_new("test2.gsd", "app", "schema", (1,0))?;
    println!("{:?}", file2);
    
    Ok(())
}

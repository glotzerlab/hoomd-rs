/*! This is an example
*/

use hoomd_gsd::file_layer::{GsdFile, Mode};

fn main() -> anyhow::Result<()> {
    let mut gsd_file = GsdFile::open("test.gsd", Mode::Read)?;

    println!("{:?}", gsd_file);
    println!("{:?}", gsd_file.read_array::<u64>(0, "configuration/step")?);
    println!("{:?}", gsd_file.read_array::<f32>(0, "configuration/box")?);

    println!("");


    let mut file2 = GsdFile::create("test2.gsd", "app", "schema", (1,0))?;
    println!("{:?}", file2);

    file2.write_array::<f32>("a", 1, &[1.0, 2.0, 3.0])?;
    file2.write_array::<f32>("b", 3, &[1.0, 2.0, 3.0])?;
    file2.end_frame()?;
    file2.write_array::<f32>("b", 1, &[2.0, 4.0, 6.0])?;
    file2.write_array::<f32>("a", 3, &[2.0, 4.0, 3.0])?;
    file2.end_frame()?;
    file2.write_array::<f32>("b", 1, &[2.0, 4.0, 6.0])?;
    file2.write_array::<f32>("a", 3, &[2.0, 4.0, 3.0])?;
    file2.end_frame()?;
    
    println!("{:?}", file2);
    drop(file2);

    println!("");

    let mut file3 = GsdFile::open("test2.gsd", Mode::Read)?;
    println!("{:?}", file3);
    println!("{:?}", file3.read_array::<f32>(0, "a")?);
    println!("{:?}", file3.read_array::<f32>(0, "b")?);
    println!("{:?}", file3.read_array::<f32>(1, "a")?);
    println!("{:?}", file3.read_array::<f32>(1, "b")?);
    println!("{:?}", file3.read_array::<f32>(2, "a")?);
    println!("{:?}", file3.read_array::<f32>(2, "b")?);
    
    
    Ok(())
}

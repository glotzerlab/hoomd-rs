#![allow(clippy::print_stdout, reason = "Demonstration purposes")]

/*! This is an example
*/

use hoomd_interaction::{
    CutoffPair, Single,
    external::Linear,
    pairwise::{Boxcar, Isotropic},
};
use hoomd_mc::{Sweep, Translate, Trial};
use hoomd_microstate::{Body, Microstate, MicrostateBuilder, boundary::Square, property::Point};
use hoomd_vector::Cartesian;

use bevy::prelude::*;

fn main() -> anyhow::Result<()> {
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins,
    ))
    .insert_resource(ClearColor(Color::oklch(0.3, 0.0, 0.0)))
    .add_systems(Startup, setup_scene);
    
    app.run();

    Ok(())
}

/// Assets that represent a Disk in the scene.
#[derive(Resource)]
struct Disk {
    /// The disk's mesh.
    mesh: AssetId<Mesh>,
    /// The disk's color.
    color: AssetId<ColorMaterial>,
}

/// Set up the bevy scene.
fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let projection = Projection::Orthographic(OrthographicProjection {
       scaling_mode: bevy::render::camera::ScalingMode::FixedVertical { viewport_height: 10.0 },
       ..OrthographicProjection::default_2d()
    });

    commands.spawn((Camera2d,
                    projection)
                );

    let mesh = meshes.add(Circle::new(0.5));
    let color = materials.add(Color::oklch(0.64, 0.14, 256.71));
    commands.insert_resource(Disk { mesh: mesh.id(), color: color.id() });

    commands.spawn((
        Mesh2d(mesh),
        MeshMaterial2d(color),
        Transform::from_xyz(
            0.0,
            0.0,
            0.0,
        ),
    ));    
}

// /// Run the simulation
// fn run(mut terminal: DefaultTerminal) -> Result<(), Box<dyn std::error::Error>> {
//     const FRAME_TIME: Duration = Duration::from_millis(10);

//     let mut microstate = MicrostateBuilder::with_boundary(Square {
//         l: 10.0.try_into()?,
//     })
//     .try_build()?;

//     let boxcar = Boxcar {
//         epsilon: 1000.0,
//         left: 0.0,
//         right: 1.0,
//     };
//     let evaluator = Isotropic(boxcar);
//     let cutoff_pair = CutoffPair {
//         r_cut: 1.0,
//         evaluator,
//     };

//     let linear = Single(Linear {
//         alpha: 10.0,
//         plane_origin: Cartesian::default(),
//         plane_normal: [0.0, 1.0].try_into()?,
//     });

//     let hamiltonian = (cutoff_pair, linear);

//     let kt = 1.0;
//     let d = 0.15;

//     let translate = Translate {
//         maximum_distance: d.try_into()?,
//     };
//     let translate_sweep = Sweep { local: translate };

//     loop {
//         let time = Instant::now();
//         terminal.draw(|frame| render(frame, &microstate))?;

//         if poll(Duration::from_millis(0))? && matches!(event::read()?, Event::Key(_)) {
//             break Ok(());
//         }

//         if microstate.step() % 100 == 0 {
//             microstate.add_body(Body::point([0.0, 4.5].into()))?;
//         }

//         translate_sweep.apply(&mut microstate, &hamiltonian, &kt);
//         microstate.increment_step();

//         let elapsed = time.elapsed();
//         if elapsed < FRAME_TIME {
//             sleep(FRAME_TIME - time.elapsed());
//         }
//     }
// }

// /// Render the system state.
// fn render(
//     frame: &mut Frame,
//     microstate: &Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Square>,
// ) {
//     let l = microstate.boundary().l.get();

//     let canvas = Canvas::default()
//         .block(Block::bordered().title(format!("Box fill (step: {})", microstate.step())))
//         .marker(Marker::Braille)
//         .paint(|ctx| {
//             for site in microstate.sites() {
//                 ctx.draw(&Circle {
//                     x: site.properties.position[0],
//                     y: site.properties.position[1],
//                     radius: 0.5,
//                     color: Color::Yellow,
//                 });
//             }
//         })
//         .x_bounds([-l / 2.0, l / 2.0])
//         .y_bounds([-l / 2.0, l / 2.0]);

//     let horizontal = Layout::horizontal([frame.area().height * 2]).flex(Flex::Center);
//     let [area] = horizontal.areas(frame.area());

//     frame.render_widget(canvas, area);
// }

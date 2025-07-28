#import bevy_sprite::{
    mesh2d_functions as mesh_functions,
    mesh2d_view_bindings::view,
}

struct DiskMaterial {
    outline_color: vec4<f32>,
    outline_width: f32,
    texture_scale: f32,
}

@group(2) @binding(0) var<uniform> disk_material: DiskMaterial;
@group(2) @binding(1) var image_color_texture: texture_2d<f32>;
@group(2) @binding(2) var image_color_sampler: sampler;
@group(2) @binding(3) var<storage, read> background_colors: array<vec4<f32>>;

// Modify the mesh2d vertex shader to look up a per instance background color.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) background_color: vec4<f32>,
}

struct Vertex {
    @builtin(instance_index) instance_index: u32,
#ifdef VERTEX_POSITIONS
    @location(0) position: vec3<f32>,
#endif
#ifdef VERTEX_NORMALS
    @location(1) normal: vec3<f32>,
#endif
#ifdef VERTEX_UVS
    @location(2) uv: vec2<f32>,
#endif
#ifdef VERTEX_TANGENTS
    @location(3) tangent: vec4<f32>,
#endif
#ifdef VERTEX_COLORS
    @location(4) color: vec4<f32>,
#endif
};

///// Vertex shader

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    // Adapted from: bevy/crates/bevy_sprite/src/mesh2d/mesh2d.wgsl
    var out: VertexOutput;

#ifdef VERTEX_UVS
    out.uv = vertex.uv;
#endif

#ifdef VERTEX_POSITIONS
    var world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    out.world_position = mesh_functions::mesh2d_position_local_to_world(
        world_from_local,
        vec4<f32>(vertex.position, 1.0)
    );
    out.position = mesh_functions::mesh2d_position_world_to_clip(out.world_position);
#endif

    let tag = mesh_functions::get_tag(vertex.instance_index);
    out.background_color = background_colors[tag % arrayLength(&background_colors)];
    return out;
}

///// Fragment shader


@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let texture_scale = disk_material.texture_scale;
    let outline_color = disk_material.outline_color;
    let outline_width = disk_material.outline_width;
    let background_color = in.background_color;

    let r = distance(in.uv, vec2<f32>(0.5));

    let radius = 0.5;

    if r > radius {
        discard;
    }

    // Sample the scaled texture.
    let scaled_uv = in.uv * texture_scale - vec2<f32>(texture_scale) / 2.0 + vec2<f32>(0.5);
    let image_color = textureSample(image_color_texture,
                                    image_color_sampler,
                                    scaled_uv);
    // Blend the texture with the background.
    var color = image_color * background_color;

    // Fill with the background color outside the scaled texture.
    var texture_valid = true;
    if scaled_uv.r < 0 || scaled_uv.r > 1 {
        color = background_color;
    }
    if scaled_uv.g < 0 || scaled_uv.g > 1 {
        color = background_color;
    }

    return select(outline_color, vec4<f32>(color.rgb, 1.0), r <= radius - outline_width);
}

#import bevy_sprite::{
    mesh2d_functions as mesh_functions,
    mesh2d_view_bindings::view,
}

// webgl2 does not support storage buffers. Use a uniform buffer of fixed size instead.

struct DiskMaterial {
    outline_color: vec4<f32>,
    outline_width: f32,
    #ifdef WEBGL2
    n_background_colors: u32,
    #endif
}

@group(2) @binding(0) var<uniform> disk_material: DiskMaterial;
#ifdef WEBGL2
@group(2) @binding(1) var<uniform> background_colors: array<vec4<f32>, 1024>;
#else
@group(2) @binding(1) var<storage, read> background_colors: array<vec4<f32>>;
#endif

// Modify the mesh2d vertex shader to look up a per instance background color.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) background_color: vec4<f32>,
    @location(3) @interpolate(flat) aspect: f32,
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

#ifdef VERTEX_POSITIONS
    var world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    out.world_position = mesh_functions::mesh2d_position_local_to_world(
        world_from_local,
        vec4<f32>(vertex.position, 1.0)
    );
    out.position = mesh_functions::mesh2d_position_world_to_clip(out.world_position);
#endif

#ifdef VERTEX_UVS
    // Scale the UV coordinates to have the same aspect ratio as the rectangle.
    // This is based on the assumption that the mesh rendered for each ellipse
    // is a square that has been scaled appropriately.
    let a = length(world_from_local[0]);
    let b = length(world_from_local[1]);
    let aspect = b/a;
    out.uv = vec2<f32>(vertex.uv.x - 0.5, (vertex.uv.y - 0.5) * aspect);
    out.aspect = aspect;
#endif

    let tag = mesh_functions::get_tag(vertex.instance_index);
    #ifdef WEBGL2
    let n_background_colors = disk_material.n_background_colors;
    #else
    let n_background_colors = arrayLength(&background_colors);
    #endif
    out.background_color = background_colors[tag % n_background_colors];
    return out;
}

///// Fragment shader

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let outline_color = disk_material.outline_color;
    let outline_width = disk_material.outline_width;
    let color = in.background_color;
    let aspect = in.aspect;

    let r = signed_distance_from_ellipse(in.uv, vec2<f32>(0.5, aspect/2.0));

    if r > 0 {
        discard;
    }

    return select(outline_color, vec4<f32>(color.rgb, 1.0), r <= -outline_width);
}

///// Distance from a point to an ellipse
// adapted from: https://github.com/0xfaded/ellipse_demo/issues/1
fn signed_distance_from_ellipse( p: vec2<f32>, e: vec2<f32> ) -> f32
{
    let pAbs: vec2<f32> = abs(p);
    let ei: vec2<f32> = 1.0 / e;
    let e2: vec2<f32> = e*e;
    let ve: vec2<f32> = ei * vec2<f32>(e2.x - e2.y, e2.y - e2.x);

    var t = vec2<f32>(0.70710678118654752, 0.70710678118654752);
    for (var i = 0; i < 3; i++) {
        let v: vec2<f32> = ve*t*t*t;
        let u: vec2<f32> = normalize(pAbs - v) * length(t * e - v);
        let w: vec2<f32> = ei * (v + u);
        t = normalize(saturate(w));
    }

    let nearestAbs: vec2<f32> = t * e;
    let dist: f32 = length(pAbs - nearestAbs);

    if dot(pAbs, pAbs) < dot(nearestAbs, nearestAbs) {
        return -dist;
    }

    return dist;
}

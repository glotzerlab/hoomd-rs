#import bevy_sprite::{
    mesh2d_functions as mesh_functions,
    mesh2d_view_bindings::view,
}

struct HyperbolicPolygonMaterial {
    outline_color: vec4<f32>,
    outline_width: f32,
    texture_scale: f32,
    n_sides: f32,
}

@group(2) @binding(0) var<uniform> background_color: vec4<f32>;
@group(2) @binding(1) var<uniform> outline_color: vec4<f32>;
@group(2) @binding(2) var<uniform> outline_width: f32;
@group(2) @binding(3) var<uniform> texture_scale: f32;

@group(2) @binding(4) var image_color_texture: texture_2d<f32>;
@group(2) @binding(5) var image_color_sampler: sampler;
@group(2) @binding(6) var<storage, read> n_sides: f32;

struct VertexOutput {
    // this is `clip position` when the struct is used as a vertex stage output
    // and `frag coord` when used as a fragment stage input
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    #ifdef VERTEX_TANGENTS
    @location(3) world_tangent: vec4<f32>,
    #endif
    #ifdef VERTEX_COLORS
    @location(4) color: vec4<f32>,
    #endif
    @location(5) @interpolate(flat) disk_center: vec2<f32>,
    @location(6) @interpolate(flat) disk_radius: f32,
    @location(7) @interpolate(flat) n_sides: f32,
    @location(8) @interpolate(flat) angle: f32,
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

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
#ifdef VERTEX_UVS
    out.uv = vertex.uv;
#endif

#ifdef VERTEX_POSITIONS
    var world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    out.disk_center = world_from_local[3].xy;
    out.disk_radius = world_from_local[2].z;
    out.angle = world_from_local[3].z;
    out.world_position = mesh_functions::mesh2d_position_local_to_world(
        world_from_local,
        vec4<f32>(vertex.position, 1.0)
    );
    out.position = mesh_functions::mesh2d_position_world_to_clip(out.world_position);
    out.n_sides = n_sides;
#endif

#ifdef VERTEX_NORMALS
    out.world_normal = mesh_functions::mesh2d_normal_local_to_world(vertex.normal, vertex.instance_index);
#endif

#ifdef VERTEX_TANGENTS
    out.world_tangent = mesh_functions::mesh2d_tangent_local_to_world(
        world_from_local,
        vertex.tangent
    );
#endif

#ifdef VERTEX_COLORS
    out.color = vertex.color;
#endif
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let pos1 = in.disk_center;
    let pos2 = in.world_position.xy;
    let n = 4.0;
    let arg = 1 + 2*dot((pos1 - pos2),(pos1 - pos2))/((1-dot(pos1,pos1))*(1-dot(pos2,pos2)));
    let r = acosh(arg)/2.0;
    let a = sqrt(pow(pos1.x,2.0) + pow(pos1.y, 2.0));
    let theta = atan2(pos1.y, pos1.x);
    let c = pos2.x * cos(theta) + pos2.y * sin(theta);
    let d = - pos2.x * sin(theta) + pos2.y * cos(theta);

    let world_angle = in.angle;

    let real = (c + pow(a,2.0)* c  - a * (1.0 + pow(c, 2.0) + pow(d, 2.0)))/ (1.0 - 2.0 * a * c + pow(a,2.0)*(pow(c,2.0) + pow(d,2.0)));
    let imag = (d - pow(a,2.0) * d)/(1.0  - 2.0*a*c + pow(a,2.0) * (pow(c, 2.0) + pow(d, 2.0)));
    let angle = atan2(imag, real) - world_angle;
    let phi = angle - (2.0 * 3.1415926/n)* floor(angle/(2.0 * 3.1415926/n))  - (3.1415926/n);

    let radius = in.disk_radius * cos(3.1415926/n)/cos(phi);

    if r > radius {
        discard;
    }

    // Sample the scaled texture.
    let scaled_uv = in.uv * texture_scale - vec2<f32>(texture_scale) / 2.0 + vec2<f32>(0.5);
    let image_color = textureSample(image_color_texture,
                                    image_color_sampler,
                                    scaled_uv);
    // Blend the texture with the background.
    var color = mix(background_color.rgb, image_color.rgb, image_color.a);

    // Fill with the background color outside the scaled texture.
    var texture_valid = true;
    if scaled_uv.r < 0 || scaled_uv.r > 1 {
        color = background_color.rgb;
    }
    if scaled_uv.g < 0 || scaled_uv.g > 1 {
        color = background_color.rgb;
    }

    //return(vec4<f32>(0, 1-r,1-r/2,1));
    return select(outline_color, vec4<f32>(color, 1.0), r <= radius - outline_width);
}

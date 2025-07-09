#import bevy_sprite::mesh2d_vertex_output::VertexOutput

@group(2) @binding(0) var<uniform> background_color: vec4<f32>;
@group(2) @binding(1) var<uniform> outline_color: vec4<f32>;
@group(2) @binding(2) var<uniform> outline_width: f32;

@group(2) @binding(3) var image_color_texture: texture_2d<f32>;
@group(2) @binding(4) var image_color_sampler: sampler;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let r = distance(in.uv, vec2<f32>(0.5));

    let radius = 0.5;

    if r > radius {
        discard;
    }

    /// Blend the texture with the background
    let image_color = textureSample(image_color_texture, image_color_sampler, in.uv);
    let color = mix(background_color.rgb, image_color.rgb, image_color.a);
    
    return select(outline_color, vec4<f32>(color, background_color.a), r <= radius - outline_width);
}

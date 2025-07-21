#import bevy_sprite::mesh2d_vertex_output::VertexOutput

struct DiskMaterial {
    background_color: vec4<f32>,
    outline_color: vec4<f32>,
    outline_width: f32,
    texture_scale: f32,
}

@group(2) @binding(0) var<uniform> disk_material: DiskMaterial;
@group(2) @binding(1) var image_color_texture: texture_2d<f32>;
@group(2) @binding(2) var image_color_sampler: sampler;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let texture_scale = disk_material.texture_scale;
    let outline_color = disk_material.outline_color;
    let outline_width = disk_material.outline_width;
    let background_color = disk_material.background_color;

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
    var color = mix(background_color.rgb, image_color.rgb, image_color.a);

    // Fill with the background color outside the scaled texture.
    var texture_valid = true;
    if scaled_uv.r < 0 || scaled_uv.r > 1 {
        color = background_color.rgb;
    }
    if scaled_uv.g < 0 || scaled_uv.g > 1 {
        color = background_color.rgb;
    }

    return select(outline_color, vec4<f32>(color, 1.0), r <= radius - outline_width);
}

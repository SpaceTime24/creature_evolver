// Camera uniform (group 0, binding 0).
struct Camera {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};
@group(0) @binding(0)
var<uniform> camera: Camera;

// Per-vertex attributes.
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

// Per-instance attributes: a model matrix (4 columns) plus a color.
struct InstanceInput {
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
    @location(6) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) color: vec3<f32>,
};

@vertex
fn vs_main(vert: VertexInput, inst: InstanceInput) -> VertexOutput {
    let model = mat4x4<f32>(inst.model_0, inst.model_1, inst.model_2, inst.model_3);

    var out: VertexOutput;
    let world_position = model * vec4<f32>(vert.position, 1.0);
    out.clip_position = camera.view_proj * world_position;

    out.world_normal = normalize((model * vec4<f32>(vert.normal, 0.0)).xyz);
    out.world_position = world_position.xyz;

    let crazy_color = (mat3x3(
        -1, 0.5, 0.5,
        0.5, -1, 0.5,
        0.5, 0.5, -1
    ) * vert.normal + vec3(1.0, 1.0, 1.0)) * 0.5;
    //out.color = crazy_color;
    out.color = inst.color.rgb;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);

    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let point_to_camera = normalize(camera.camera_pos.xyz - in.world_position);

    let light_reflect = reflect(-light_dir, n);

    let spec = pow(
        max(dot(light_reflect, point_to_camera), 0.0),
        12.0
    );
    let diffuse = max(dot(n, light_dir), 0.0);
    let ambient = 0.25;
    let shade = ambient + diffuse * 0.8 + spec;

    return vec4<f32>(in.color * shade, 1.0);
}

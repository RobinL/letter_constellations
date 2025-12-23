// Magical Glitter shader for drawing effects
// Creates fine, subtle glitter sparkles like light catching tiny particles

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

const MAX_TRAIL_POINTS: u32 = 64u;

struct Uniforms {
    time: f32,
    resolution: vec2<f32>,
    mouse_active: f32,
    mouse_pos: vec2<f32>,
    mouse_velocity: vec2<f32>,
    trail_count: f32,
    _padding: f32,
};

struct TrailPoint {
    pos: vec2<f32>,
    time: f32,
    _padding: f32,
};

struct TrailBuffer {
    points: array<TrailPoint, 64>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> trail: TrailBuffer;

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32(i32(vertexIndex & 1u) * 4 - 1);
    let y = f32(i32(vertexIndex & 2u) * 2 - 1);
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return output;
}

// Sharp hash for glitter positions
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

// Glitter - sharp, discrete sparkle points
fn glitter(p: vec2<f32>, time: f32) -> f32 {
    // High frequency grid for tiny glitter particles
    let gridScale = 300.0;
    let cell = floor(p * gridScale);
    let cellUV = fract(p * gridScale);

    // Random position within cell
    let rand = hash21(cell);
    let rand2 = hash21(cell + vec2<f32>(127.0, 311.0));

    // More cells have glitter (less sparse)
    if (rand > 0.82) {
        // Glitter position within cell
        let glitterPos = vec2<f32>(
            hash21(cell + vec2<f32>(1.0, 0.0)),
            hash21(cell + vec2<f32>(0.0, 1.0))
        );

        // Distance to glitter point
        let d = length(cellUV - glitterPos);

        // Sharp falloff - tiny point
        let size = 0.12 + rand2 * 0.08;
        let point = 1.0 - smoothstep(0.0, size, d);

        // Twinkle - glitter catches light momentarily
        let twinkleSpeed = 2.0 + rand * 6.0;
        let twinklePhase = rand2 * 6.28;
        let twinkle = sin(time * twinkleSpeed + twinklePhase);

        // Sharp threshold - either catching light or not
        let catchLight = smoothstep(0.3, 0.6, twinkle);

        return point * catchLight;
    }

    return 0.0;
}

// Multi-layer glitter for depth
fn glitterLayers(p: vec2<f32>, time: f32) -> f32 {
    var total = 0.0;

    // Multiple offset layers for richness
    total += glitter(p, time);
    total += glitter(p + vec2<f32>(0.1, 0.2), time * 1.1 + 1.0) * 0.8;
    total += glitter(p + vec2<f32>(0.3, 0.1), time * 0.9 + 2.0) * 0.6;

    return min(total, 1.0);
}

// Aurora-themed glitter color
fn glitterColor(seed: f32, time: f32) -> vec3<f32> {
    let cyan = vec3<f32>(0.6, 0.95, 1.0);
    let green = vec3<f32>(0.5, 1.0, 0.7);
    let purple = vec3<f32>(0.8, 0.6, 1.0);
    let white = vec3<f32>(1.0, 1.0, 1.0);

    let hue = fract(seed + time * 0.02);
    var color: vec3<f32>;

    if (hue < 0.33) {
        color = mix(cyan, green, hue * 3.0);
    } else if (hue < 0.66) {
        color = mix(green, purple, (hue - 0.33) * 3.0);
    } else {
        color = mix(purple, cyan, (hue - 0.66) * 3.0);
    }

    // Glitter is mostly white with color tint
    color = mix(white, color, 0.3);

    return color;
}

// Calculate proximity to trail
fn trailProximity(uv: vec2<f32>, aspect: f32, time: f32) -> vec2<f32> {
    var minDist = 1000.0;
    var closestAge = 1.0;

    let trailCount = u32(uniforms.trail_count);
    let step = max(1u, trailCount / 24u);

    for (var i = 0u; i < trailCount; i += step) {
        let point = trail.points[i];
        let age = time - point.time;

        if (age > 3.0) {
            continue;
        }

        let diff = uv - point.pos;
        let correctedDiff = vec2<f32>(diff.x * aspect, diff.y);
        let dist = length(correctedDiff);

        if (dist < minDist) {
            minDist = dist;
            closestAge = age / 3.0;
        }
    }

    return vec2<f32>(minDist, closestAge);
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let time = uniforms.time;
    let aspect = uniforms.resolution.x / uniforms.resolution.y;

    // Get distance to trail
    let trailInfo = trailProximity(uv, aspect, time);
    let dist = trailInfo.x;
    let age = trailInfo.y;

    // Glitter radius - area around the path
    let radius = 0.08 * (1.0 - age * 0.5);

    // Early exit if too far
    if (dist > radius * 2.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Proximity falloff - glitter density decreases with distance
    let proximity = 1.0 - smoothstep(0.0, radius, dist);
    let ageFade = 1.0 - smoothstep(0.0, 1.0, age);

    // Generate glitter
    let g = glitterLayers(uv, time);

    // Glitter only appears near the trail, fading with distance and age
    var intensity = g * proximity * ageFade;

    // Extra glitter burst at cursor
    if (uniforms.mouse_active > 0.5) {
        let cursorDiff = uv - uniforms.mouse_pos;
        let cursorCorrected = vec2<f32>(cursorDiff.x * aspect, cursorDiff.y);
        let cursorDist = length(cursorCorrected);

        let cursorProximity = 1.0 - smoothstep(0.0, 0.06, cursorDist);
        let cursorGlitter = glitterLayers(uv + uniforms.mouse_velocity * 0.02, time * 1.3);
        intensity += cursorGlitter * cursorProximity * 1.5;
    }

    // Color with subtle variation
    let colorSeed = hash21(floor(uv * 300.0));
    let color = glitterColor(colorSeed, time);

    // Output
    intensity = min(intensity, 1.0);

    return vec4<f32>(color * intensity, intensity);
}

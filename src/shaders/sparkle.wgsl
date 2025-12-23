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

// Glitter - layered sparkle points with glow
fn glitter(
    p: vec2<f32>,
    time: f32,
    gridScale: f32,
    density: f32,
    sizeMin: f32,
    sizeMax: f32,
    glowScale: f32,
    twinkleBase: f32
) -> f32 {
    let cell = floor(p * gridScale);
    let cellUV = fract(p * gridScale);
    var sparkle = 0.0;

    // Sample neighboring cells so glow doesn't clip to square tiles
    for (var oy: i32 = -1; oy <= 1; oy = oy + 1) {
        for (var ox: i32 = -1; ox <= 1; ox = ox + 1) {
            let offset = vec2<f32>(f32(ox), f32(oy));
            let neighbor = cell + offset;
            let rand = hash21(neighbor);

            if (rand < (1.0 - density)) {
                continue;
            }

            let rand2 = hash21(neighbor + vec2<f32>(127.0, 311.0));
            let glitterPos = vec2<f32>(
                hash21(neighbor + vec2<f32>(1.0, 0.0)),
                hash21(neighbor + vec2<f32>(0.0, 1.0))
            );

            let q = cellUV - (glitterPos + offset);
            let d = length(q);

            // Sharper core to kill mid haze
            let size = mix(sizeMin, sizeMax, rand2);
            var core = 1.0 - smoothstep(0.0, size, d);
            core = pow(core, 6.0);

            // Glow that doesn't dominate mids
            var glow = 1.0 - smoothstep(size, size * glowScale, d);
            glow = pow(glow, 2.0);

            // Star flare rays for sparkle character
            let ang = atan2(q.y, q.x) + rand2 * 6.28318;
            let rays = pow(abs(cos(ang * 6.0)), 18.0);
            let flare = (1.0 - smoothstep(size * 0.25, size * 2.2, d)) * rays;

            // Spiky twinkle for rare bright flashes
            let twinkleSpeed = twinkleBase + rand * 6.0;
            let tw = 0.5 + 0.5 * sin(time * twinkleSpeed + rand2 * 6.28318);
            let catchLight = pow(tw, 12.0);

            let hit = (core + glow * 0.45 + flare * 0.9) * catchLight;
            sparkle = max(sparkle, hit);
        }
    }

    return sparkle;
}

// Multi-layer glitter for depth
fn glitterLayers(p: vec2<f32>, time: f32) -> f32 {
    var total = 0.0;

    // Multiple offset layers for richness and size variation
    total += glitter(p, time * 0.7, 240.0, 0.16, 0.12, 0.22, 3.2, 1.4);
    total += glitter(p + vec2<f32>(0.17, 0.11), time * 0.75 + 0.7, 175.0, 0.13, 0.16, 0.28, 3.8, 1.1) * 0.9;
    total += glitter(p + vec2<f32>(0.31, 0.23), time * 0.6 + 1.9, 120.0, 0.1, 0.22, 0.36, 4.8, 0.9) * 0.8;
    total += glitter(p + vec2<f32>(0.41, 0.37), time * 0.5 + 2.8, 75.0, 0.08, 0.28, 0.42, 5.4, 0.7) * 0.6;

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
    color = mix(white, color, 0.65);

    return color;
}

// Calculate proximity to trail
fn trailProximity(uv: vec2<f32>, aspect: f32, time: f32) -> vec2<f32> {
    var minDist = 1000.0;
    var closestAge = 999.0;

    let trailCount = u32(uniforms.trail_count);
    let step = max(1u, trailCount / 24u);

    for (var i = 0u; i < trailCount; i += step) {
        let point = trail.points[i];
        let age = time - point.time;

        if (age > 4.0) {
            continue;
        }

        let diff = uv - point.pos;
        let correctedDiff = vec2<f32>(diff.x * aspect, diff.y);
        let dist = length(correctedDiff);

        if (dist < minDist) {
            minDist = dist;
            closestAge = age;
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
    let ageSeconds = trailInfo.y;
    let age = clamp(ageSeconds / 4.0, 0.0, 1.0);

    // Glitter radius - area around the path
    let radius = 0.13 * (1.0 - age * 0.4);

    // Early exit if too far
    if (dist > radius * 2.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Proximity falloff - glitter density decreases with distance
    let proximity = 1.0 - smoothstep(0.0, radius, dist);
    let ageFade = 1.0 - smoothstep(0.0, 1.0, age);
    let fadeIn = smoothstep(0.08, 0.45, ageSeconds);
    let cursorFade = smoothstep(0.12, 0.65, ageSeconds);

    // Generate glitter
    let g = glitterLayers(uv, time);

    let activation = smoothstep(0.0, 1.0, clamp(uniforms.mouse_active, 0.0, 1.0));
    let cursorActivation = pow(activation, 2.2);

    // Glitter only appears near the trail, fading with distance and age
    var intensity = g * proximity * ageFade * fadeIn;
    let glow = pow(g, 0.5) * proximity * ageFade * 0.35;
    intensity += glow;
    intensity = pow(intensity, 1.1);
    intensity *= activation;

    // Extra glitter burst at cursor
    if (activation > 0.01) {
        let cursorDiff = uv - uniforms.mouse_pos;
        let cursorCorrected = vec2<f32>(cursorDiff.x * aspect, cursorDiff.y);
        let cursorDist = length(cursorCorrected);
        let speed = length(uniforms.mouse_velocity);
        let speedBoost = smoothstep(0.05, 2.8, speed);

        let cursorRadius = mix(0.09, 0.18, speedBoost);
        let cursorProximity = 1.0 - smoothstep(0.0, cursorRadius, cursorDist);
        let cursorGlitter = glitterLayers(uv + uniforms.mouse_velocity * 0.05, time * (0.9 + speedBoost * 0.3));
        intensity += cursorGlitter * cursorProximity * (2.1 + speedBoost * 1.8) * cursorActivation * cursorFade;

        let a = atan2(cursorCorrected.y, cursorCorrected.x);
        let rayBurst = pow(abs(cos(a * 10.0 + time * 1.5)), 28.0);
        let burstFalloff = exp(-cursorDist * 35.0);
        intensity += rayBurst * burstFalloff * speedBoost * 0.9 * cursorActivation * cursorFade;
    }

    // Color with subtle variation
    let colorSeed = hash21(floor(uv * 180.0));
    let color = glitterColor(colorSeed, time);
    let alpha = clamp(intensity, 0.0, 1.0);

    // Output
    let finalColor = color * alpha;
    return vec4<f32>(finalColor, alpha);
}

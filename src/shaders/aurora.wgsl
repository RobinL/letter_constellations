// Aurora Borealis shader using WebGPU
// Renders animated aurora effect over a background texture

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct Uniforms {
    time: f32,
    resolution: vec2<f32>,
    _padding: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var backgroundTexture: texture_2d<f32>;
@group(0) @binding(2) var backgroundSampler: sampler;

// --- Stars (cheap, no loops) -----------------------------------------------
fn hash_u32(x: u32) -> u32 {
    var v = x;
    v ^= v >> 16u;
    v *= 0x7feb352du;
    v ^= v >> 15u;
    v *= 0x846ca68bu;
    v ^= v >> 16u;
    return v;
}

fn hash2_u32(p: vec2<u32>, salt: u32) -> u32 {
    let x = p.x ^ (p.y * 0x9e3779b9u) ^ (salt * 0x85ebca6bu);
    return hash_u32(x);
}

fn hash01(p: vec2<u32>, salt: u32) -> f32 {
    let h = hash2_u32(p, salt);
    return f32(h) / 4294967296.0;
}

fn starFromCell(
    p: vec2<f32>,
    cellI: vec2<i32>,
    time: f32,
    cellPx: f32,
    density: f32,
    sizeMinPx: f32,
    sizeMaxPx: f32,
    brightness: f32
) -> vec3<f32> {
    if (any(cellI < vec2<i32>(0))) {
        return vec3<f32>(0.0);
    }

    let cell = vec2<u32>(cellI);
    let r0 = hash01(cell, 0u);
    if (r0 > density) {
        return vec3<f32>(0.0);
    }

    let center01 = vec2<f32>(hash01(cell, 1u), hash01(cell, 2u));
    let centerPx = (vec2<f32>(cell) + center01) * cellPx;
    let dPx = distance(p, centerPx);

    let rSize = hash01(cell, 3u);
    let sizePx = mix(sizeMinPx, sizeMaxPx, pow(rSize, 2.4));

    var core = 1.0 - smoothstep(0.0, sizePx, dPx);
    core = pow(core, 6.0);

    var glow = 1.0 - smoothstep(sizePx, sizePx * 5.0, dPx);
    glow = pow(glow, 2.0) * 0.35;

    let phase = hash01(cell, 4u) * 6.2831853;
    let baseSpeed = mix(0.34, 1.20, hash01(cell, 5u)) * (10.0 / max(cellPx, 6.0));
    let t1 = 0.5 + 0.5 * sin(time * baseSpeed + phase);
    let t2 = 0.5 + 0.5 * sin(time * (baseSpeed * 1.73) + phase * 1.37);
    let sparkle = pow(max(t1, t2), 4.4);
    let slowBreath = 0.85 + 0.15 * sin(time * 0.20 + phase * 0.5);
    let tw = mix(0.40, 2.55, sparkle) * slowBreath;

    let tintMix = hash01(cell, 6u);
    let cool = vec3<f32>(0.80, 0.88, 1.00);
    let warm = vec3<f32>(1.00, 0.95, 0.86);
    let tint = mix(cool, warm, tintMix);
    let color = mix(vec3<f32>(1.0), tint, 0.25 + sparkle * 0.15);

    let brightJitter = mix(0.60, 1.60, hash01(cell, 7u));
    let intensity = (core + glow) * tw * brightness * brightJitter;
    return color * intensity;
}

fn starLayer(
    uv: vec2<f32>,
    time: f32,
    cellPx: f32,
    density: f32,
    sizeMinPx: f32,
    sizeMaxPx: f32,
    brightness: f32
) -> vec3<f32> {
    let p = uv * uniforms.resolution;
    let baseCell = vec2<i32>(floor(p / cellPx));
    var s = vec3<f32>(0.0);

    s += starFromCell(p, baseCell + vec2<i32>(-1, -1), time, cellPx, density, sizeMinPx, sizeMaxPx, brightness);
    s += starFromCell(p, baseCell + vec2<i32>(0, -1), time, cellPx, density, sizeMinPx, sizeMaxPx, brightness);
    s += starFromCell(p, baseCell + vec2<i32>(1, -1), time, cellPx, density, sizeMinPx, sizeMaxPx, brightness);
    s += starFromCell(p, baseCell + vec2<i32>(-1, 0), time, cellPx, density, sizeMinPx, sizeMaxPx, brightness);
    s += starFromCell(p, baseCell, time, cellPx, density, sizeMinPx, sizeMaxPx, brightness);
    s += starFromCell(p, baseCell + vec2<i32>(1, 0), time, cellPx, density, sizeMinPx, sizeMaxPx, brightness);
    s += starFromCell(p, baseCell + vec2<i32>(-1, 1), time, cellPx, density, sizeMinPx, sizeMaxPx, brightness);
    s += starFromCell(p, baseCell + vec2<i32>(0, 1), time, cellPx, density, sizeMinPx, sizeMaxPx, brightness);
    s += starFromCell(p, baseCell + vec2<i32>(1, 1), time, cellPx, density, sizeMinPx, sizeMaxPx, brightness);

    return s;
}

fn starField(uv: vec2<f32>, time: f32) -> vec3<f32> {
    let skyMask = 1.0 - smoothstep(0.55, 0.95, uv.y);
    if (skyMask <= 0.0) {
        return vec3<f32>(0.0);
    }

    var s = vec3<f32>(0.0);
    s += starLayer(uv, time, 7.0, 0.018, 0.9, 1.9, 0.34);
    s += starLayer(uv, time, 16.0, 0.012, 1.2, 3.0, 0.46);
    s += starLayer(uv, time, 34.0, 0.005, 2.0, 4.6, 0.62);

    return s * skyMask;
}

// Fullscreen triangle vertex shader
@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var output: VertexOutput;

    // Generate oversized fullscreen triangle (covers [-1,1] clip space when clipped)
    // vertex 0: (-1, -1), vertex 1: (3, -1), vertex 2: (-1, 3)
    let x = f32(i32(vertexIndex & 1u) * 4 - 1);
    let y = f32(i32(vertexIndex & 2u) * 2 - 1);

    output.position = vec4<f32>(x, y, 0.0, 1.0);
    // UV: map clip space to [0,1] texture coordinates
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);

    return output;
}

// Simplex-like noise functions for aurora
fn hash(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);

    let a = hash(i);
    let b = hash(i + vec2<f32>(1.0, 0.0));
    let c = hash(i + vec2<f32>(0.0, 1.0));
    let d = hash(i + vec2<f32>(1.0, 1.0));

    let u = f * f * (3.0 - 2.0 * f);

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;

    for (var i = 0; i < 3; i++) {
        value += amplitude * noise(pos * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
        pos += vec2<f32>(1.7, 9.2);
    }

    return value;
}

// Aurora color palette - greens and teals typical of northern lights
fn auroraColor(t: f32, intensity: f32) -> vec3<f32> {
    // Base green aurora color
    let green = vec3<f32>(0.2, 0.9, 0.4);
    // Teal/cyan accent
    let teal = vec3<f32>(0.1, 0.7, 0.8);
    // Purple edge color
    let purple = vec3<f32>(0.5, 0.2, 0.7);
    // Pink highlights
    let pink = vec3<f32>(0.8, 0.3, 0.5);

    // Mix colors based on position and intensity
    var color = mix(green, teal, sin(t * 3.14159) * 0.5 + 0.5);
    color = mix(color, purple, smoothstep(0.6, 1.0, intensity));
    color = mix(color, pink, smoothstep(0.8, 1.0, intensity) * 0.3);

    return color;
}

// Main aurora function
fn aurora(uv: vec2<f32>, time: f32) -> vec4<f32> {
    var totalIntensity = 0.0;
    var totalColor = vec3<f32>(0.0);
    let edgeFade = smoothstep(0.0, 0.3, uv.x) * smoothstep(1.0, 0.7, uv.x);
    let topHalfFade = smoothstep(0.5, 0.45, uv.y);

    // Create multiple aurora curtains/ribbons
    for (var i = 0; i < 2; i++) {
        let fi = f32(i);

        // Horizontal wave parameters
        let waveSpeed = 0.15 + fi * 0.05;
        let waveFreq = 1.5 + fi * 0.5;
        let waveAmp = 0.08 + fi * 0.03;

        // Vertical position of this aurora band - in top third of screen
        let baseY = 0.16 + fi * 0.05;

        // Create flowing wave motion
        let wave1 = sin(uv.x * waveFreq + time * waveSpeed) * waveAmp;
        let wave2 = sin(uv.x * waveFreq * 2.3 - time * waveSpeed * 1.3) * waveAmp * 0.5;
        let wave3 = fbm(vec2<f32>(uv.x * 3.0 + time * 0.1, time * 0.05 + fi)) * 0.15;

        let waveOffset = wave1 + wave2 + wave3;

        // Distance from aurora center line
        let auroraY = baseY + waveOffset;
        let dist = abs(uv.y - auroraY);

        // Aurora thickness varies with noise
        let thickness = 0.08 + fbm(vec2<f32>(uv.x * 2.0 + time * 0.1, fi * 10.0)) * 0.06;

        // Intensity falloff from center
        var intensity = smoothstep(thickness, 0.0, dist);

        // Add vertical streaks (curtain effect)
        let streakNoise = fbm(vec2<f32>(uv.x * 15.0 + fi * 5.0, time * 0.2));
        let streaks = pow(streakNoise, 2.0);
        intensity *= 0.5 + streaks * 0.8;

        // Fade aurora at edges of screen + keep it in the top half
        intensity *= edgeFade * topHalfFade;

        // Add shimmer effect
        let shimmer = sin(uv.x * 50.0 + time * 2.0 + fi * 20.0) * 0.1 + 0.9;
        intensity *= shimmer;

        // Get color for this layer
        let layerColor = auroraColor(uv.x + fi * 0.3 + time * 0.1, intensity);

        // Accumulate
        totalColor += layerColor * intensity * (1.0 - fi * 0.2);
        totalIntensity += intensity * (1.0 - fi * 0.2);
    }

    // Normalize and apply glow
    totalIntensity = min(totalIntensity, 1.0);

    return vec4<f32>(totalColor, totalIntensity * 0.7);
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let time = uniforms.time;

    // Sample background texture
    let background = textureSample(backgroundTexture, backgroundSampler, uv);

    // Generate aurora
    let auroraEffect = aurora(uv, time);

    // Stars behind aurora (dim where aurora is strongest)
    let stars = starField(uv, time) * (1.0 - auroraEffect.a * 0.65);

    // Blend aurora over background using additive-like blending
    // This makes the aurora glow over the scene
    let blendedColor = background.rgb + stars + auroraEffect.rgb * auroraEffect.a;

    return vec4<f32>(blendedColor, 1.0);
}

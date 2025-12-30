// Comet nib shader - renders a comet effect at the line tracing nib position
// Only enabled on high graphics settings
// Particles are emitted from the nib and drift with their own momentum

struct Uniforms {
    time: f32,
    _pad0: f32,
    resolution: vec2<f32>,
    nibActive: f32,
    numParticles: f32,      // Number of valid particles (0-32)
    _pad1: f32,
    _pad2: f32,
};

// Each particle: vec4(posX, posY, velX, velY) and vec4(spawnTime, size, seed, 0)
// 256 particles max, stored as 512 vec4s (256 pos/vel + 256 spawn/size)
struct ParticleData {
    data: array<vec4<f32>, 512>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> particles: ParticleData;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = f32(i32(vertexIndex & 1u) * 4 - 1);
    let y = f32(i32(vertexIndex & 2u) * 2 - 1);
    output.position = vec4<f32>(x, y, 0.0, 1.0);
    output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return output;
}

// Hash functions for randomization
fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    let n = vec2<f32>(
        dot(p, vec2<f32>(127.1, 311.7)),
        dot(p, vec2<f32>(269.5, 183.3))
    );
    return fract(sin(n) * 43758.5453123);
}

// Comet head (bright core at current nib position)
fn cometHead(p: vec2<f32>, headPos: vec2<f32>, size: f32) -> f32 {
    let d = length(p - headPos);
    let core = smoothstep(size, 0.0, d);
    let glow = smoothstep(size * 5.0, 0.0, d);
    return core * 2.5 + glow * 0.6;
}

// Single particle glow
fn particleGlow(p: vec2<f32>, particlePos: vec2<f32>, size: f32, brightness: f32) -> f32 {
    let d = length(p - particlePos);
    let core = smoothstep(size, size * 0.3, d);
    let glow = smoothstep(size * 2.5, 0.0, d);
    return (core * 1.5 + glow * 0.8) * brightness;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
    // Early exit if nib is not active
    if (uniforms.nibActive < 0.5) {
        return vec4<f32>(0.0);
    }

    let numParticles = i32(uniforms.numParticles);
    if (numParticles < 1) {
        return vec4<f32>(0.0);
    }

    let uv = input.uv;
    let time = uniforms.time;
    let aspect = uniforms.resolution.x / uniforms.resolution.y;

    // Convert UV to aspect-corrected space centered at origin
    var p = uv - 0.5;
    p.x *= aspect;

    // Get head position (first particle is always the current nib position)
    let headData = particles.data[0];
    var headPos = vec2<f32>(headData.x, headData.y) - 0.5;
    headPos.x *= aspect;

    var totalLight = vec3<f32>(0.0);

    // Early distance culling - check if we're near any particle
    // Sample every 4th particle for faster culling check
    var minDist = 1000.0;
    for (var check = 0; check < 256; check += 4) {
        if (check >= numParticles) { break; }
        let posVel = particles.data[check];
        let spawnData = particles.data[check + 256];

        let age = time - spawnData.x;
        if (age < 0.0 || age > 4.0) { continue; }

        // Simulate particle position based on velocity and time
        var particlePos = vec2<f32>(posVel.x, posVel.y) - 0.5;
        particlePos.x *= aspect;

        // Apply velocity over time (particles drift away symmetrically)
        let vel = vec2<f32>(posVel.z * aspect, posVel.w);
        particlePos += vel * age;

        minDist = min(minDist, length(p - particlePos));
    }
    if (minDist > 0.3) {
        return vec4<f32>(0.0);
    }

    // Render comet head - bright golden core
    let headSize = 0.007;
    let headIntensity = cometHead(p, headPos, headSize);
    let headColor = vec3<f32>(1.0, 0.9, 0.25);
    totalLight += headColor * headIntensity * 1.8; // Reduced from 2.5

    // Render each emitted particle
    for (var i = 1; i < 256; i++) {
        if (i >= numParticles) { break; }

        let posVel = particles.data[i];
        let spawnData = particles.data[i + 256];

        let spawnTime = spawnData.x;
        let size = spawnData.y;
        let seed = spawnData.z;

        let age = time - spawnTime;
        let lifetime = 2.8 + seed * 1.2; // 2.8-4.0 second lifetime

        if (age < 0.0 || age > lifetime) { continue; }

        // Normalized age (0 = just spawned, 1 = about to die)
        let t = age / lifetime;

        // Calculate particle position from spawn position + velocity * time
        var particlePos = vec2<f32>(posVel.x, posVel.y) - 0.5;
        particlePos.x *= aspect;

        // Apply velocity (particles drift symmetrically outward)
        let vel = vec2<f32>(posVel.z * aspect, posVel.w);
        particlePos += vel * age;

        // Early distance check - skip particles too far from current pixel
        let distToParticle = length(p - particlePos);
        if (distToParticle > 0.08) { continue; }

        // Minimal turbulence for slight organic feel
        let turbAmp = 0.003 * t;
        particlePos.x += sin(seed * 50.0 + age * 2.0) * turbAmp;
        particlePos.y += cos(seed * 37.0 + age * 1.7) * turbAmp;

        // Particle size grows slightly then shrinks
        let sizeMultiplier = (1.0 - t * t) * (0.5 + t * 0.5);
        let particleSize = size * sizeMultiplier;

        // Brightness fades out smoothly
        let brightness = (1.0 - t * t) * 0.55;

        let light = particleGlow(p, particlePos, particleSize, brightness);

        // Color gradient: bright golden -> orange -> red as particle ages
        let particleColor = mix(
            mix(
                vec3<f32>(1.0, 0.92, 0.35),  // Bright golden (young)
                vec3<f32>(1.0, 0.6, 0.2),    // Orange (middle)
                smoothstep(0.0, 0.5, t)
            ),
            vec3<f32>(0.85, 0.25, 0.08),     // Red-orange (old)
            smoothstep(0.4, 1.0, t)
        );

        totalLight += particleColor * light * 1.2;
    }

    // Add soft glow around head
    let glowDist = length(p - headPos);
    let softGlow = smoothstep(0.12, 0.0, glowDist) * 0.3;
    totalLight += vec3<f32>(1.0, 0.7, 0.2) * softGlow;

    let intensity = length(totalLight);
    return vec4<f32>(totalLight, min(intensity, 1.0));
}

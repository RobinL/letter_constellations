// Instanced “spark / glitter” particles.
// Stateless: position is derived from (time - spawnTime), so no compute pass is needed.

struct Uniforms {
  time: f32,
  _pad0: f32,
  resolution: vec2<f32>,
  global: f32,
  _pad1: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,   // local (dir/perp) space, includes stretch
  @location(1) seed: f32,
  @location(2) age01: f32,
  @location(3) speed: f32,
  @location(4) sizePx: f32,
};

fn hash11(n: f32) -> f32 {
  return fract(sin(n) * 43758.5453123);
}

fn sparkleColor(seed: f32, time: f32) -> vec3<f32> {
  let cyan   = vec3<f32>(0.55, 0.95, 1.0);
  let green  = vec3<f32>(0.55, 1.0, 0.72);
  let purple = vec3<f32>(0.85, 0.62, 1.0);
  let pink   = vec3<f32>(1.0, 0.55, 0.78);
  let white  = vec3<f32>(1.0, 1.0, 1.0);

  let h = fract(seed + time * 0.02);
  var c: vec3<f32>;
  if (h < 0.25) {
    c = mix(cyan, green, h / 0.25);
  } else if (h < 0.5) {
    c = mix(green, purple, (h - 0.25) / 0.25);
  } else if (h < 0.75) {
    c = mix(purple, pink, (h - 0.5) / 0.25);
  } else {
    c = mix(pink, cyan, (h - 0.75) / 0.25);
  }

  return mix(white, c, 0.7);
}

@vertex
fn sparkVertexMain(
  @location(0) quad: vec2<f32>,   // -1..+1
  @location(1) a: vec4<f32>,      // pos0.xy, vel.xy
  @location(2) b: vec4<f32>,      // spawnTime, life, sizePx, seed
) -> VertexOut {
  var out: VertexOut;

  let pos0 = a.xy;
  let vel = a.zw;
  let spawnTime = b.x;
  let life = max(b.y, 0.0001);
  let sizePx = b.z;
  let seed = b.w;

  let age = uniforms.time - spawnTime;

  // Dead particles are moved outside clip space (no fragments).
  if (age < 0.0 || age > life || sizePx <= 0.0) {
    out.position = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
    out.local = vec2<f32>(0.0, 0.0);
    out.seed = 0.0;
    out.age01 = 1.0;
    out.speed = 0.0;
    out.sizePx = 0.0;
    return out;
  }

  // Slight gravity (down is +y in UV)
  let g = vec2<f32>(0.0, 0.16);
  let pos = pos0 + vel * age + 0.5 * g * age * age;

  let speed = length(vel);
  let dir = select(vel / max(speed, 1e-5), vec2<f32>(1.0, 0.0), speed < 1e-5);
  let perp = vec2<f32>(-dir.y, dir.x);

  let speedBoost = smoothstep(0.15, 2.8, speed);
  let stretch = 1.0 + speedBoost * 2.6;

  // local space in dir/perp basis (passed to fragment)
  let local = vec2<f32>(quad.x * stretch, quad.y);
  let basis = dir * local.x + perp * local.y;

  let scaleUv = vec2<f32>(
    sizePx / uniforms.resolution.x,
    sizePx / uniforms.resolution.y
  );
  let finalUv = pos + vec2<f32>(basis.x * scaleUv.x, basis.y * scaleUv.y);

  let clip = vec2<f32>(finalUv.x * 2.0 - 1.0, 1.0 - finalUv.y * 2.0);
  out.position = vec4<f32>(clip, 0.0, 1.0);

  out.local = local;
  out.seed = seed;
  out.age01 = clamp(age / life, 0.0, 1.0);
  out.speed = speed;
  out.sizePx = sizePx;
  return out;
}

@fragment
fn sparkFragmentMain(in: VertexOut) -> @location(0) vec4<f32> {
  let r = length(in.local);

  // Soft kill outside bounds (reduces overdraw).
  if (r > 1.4) {
    return vec4<f32>(0.0);
  }

  let fadeIn  = smoothstep(0.0, 0.12, in.age01);
  let fadeOut = 1.0 - smoothstep(0.65, 1.0, in.age01);
  let lifeFade = fadeIn * fadeOut;

  let core = pow(max(0.0, 1.0 - r), 10.0);
  let glow = pow(max(0.0, 1.0 - r), 2.0) * 0.35;

  // Rays show up more for larger sprites.
  let isSpark = smoothstep(4.0, 8.0, in.sizePx);
  let ang = atan2(in.local.y, in.local.x);
  let rayCount = 6.0 + floor(hash11(in.seed * 19.17) * 6.0);
  let rays =
    pow(abs(cos(ang * rayCount + in.seed * 6.28318)), 18.0) *
    smoothstep(1.3, 0.2, r);

  // Spiky twinkle (rare bright flashes)
  let speedBoost = smoothstep(0.05, 2.8, in.speed);

  // Two oscillators -> richer flicker, plus baseline shimmer
  let twBase = 10.0 + 24.0 * hash11(in.seed * 3.91);
  let twSpeed = twBase * (1.0 + speedBoost * 0.55);

  let tw1 = 0.5 + 0.5 * sin(uniforms.time * twSpeed + in.seed * 6.28318);
  let tw2 = 0.5 + 0.5 * sin(uniforms.time * (twSpeed * 1.73) + in.seed * 2.123);

  // Visible baseline + rare bright pops
  var twinkle = mix(0.35, 1.0, pow(max(tw1, tw2), 10.0));

  // Small per-fragment shimmer so the sparkle “surface” dances
  let shimmerFreq = 28.0 + 60.0 * hash11(in.seed * 9.2);
  let shimmer = 0.85 + 0.15 * sin(uniforms.time * shimmerFreq + dot(in.local, vec2<f32>(12.3, 9.7)));
  twinkle *= shimmer;

  var intensity = (core + glow + rays * 0.9 * isSpark) * twinkle;
  intensity *= lifeFade;
  intensity *= (0.75 + speedBoost * 1.8);
  intensity *= clamp(uniforms.global, 0.0, 1.0);

  // Tiny dust needs a touch more energy.
  intensity *= mix(1.15, 1.0, isSpark);

  intensity = min(intensity, 1.0);

  let color = sparkleColor(in.seed, uniforms.time);

  // Premultiplied output
  return vec4<f32>(color * intensity, intensity);
}

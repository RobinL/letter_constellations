var e=Object.defineProperty,t=t=>{let n={};for(var r in t)e(n,r,{get:t[r],enumerable:!0});return n};(function(){let e=document.createElement(`link`).relList;if(e&&e.supports&&e.supports(`modulepreload`))return;for(let e of document.querySelectorAll(`link[rel="modulepreload"]`))n(e);new MutationObserver(e=>{for(let t of e)if(t.type===`childList`)for(let e of t.addedNodes)e.tagName===`LINK`&&e.rel===`modulepreload`&&n(e)}).observe(document,{childList:!0,subtree:!0});function t(e){let t={};return e.integrity&&(t.integrity=e.integrity),e.referrerPolicy&&(t.referrerPolicy=e.referrerPolicy),e.crossOrigin===`use-credentials`?t.credentials=`include`:e.crossOrigin===`anonymous`?t.credentials=`omit`:t.credentials=`same-origin`,t}function n(e){if(e.ep)return;e.ep=!0;let n=t(e);fetch(e.href,n)}})();var n=`// Aurora Borealis shader using WebGPU
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

    for (var i = 0; i < 5; i++) {
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

    // Create multiple aurora curtains/ribbons
    for (var i = 0; i < 3; i++) {
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

        // Fade aurora at edges of screen
        let edgeFade = smoothstep(0.0, 0.3, uv.x) * smoothstep(1.0, 0.7, uv.x);
        intensity *= edgeFade;

        // Only show aurora in top third of screen
        let topHalfFade = smoothstep(0.40, 0.32, uv.y);
        intensity *= topHalfFade;

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
`,r=`/letter_constellations/assets/background_landscape-DE4pyiv7.png`,i=`/letter_constellations/assets/background_portrait-C7A3VQQf.png`;async function a(e){let t=await fetch(e);if(!t.ok)throw Error(`Failed to load image: ${e}`);let n=await t.blob();return createImageBitmap(n)}var o=class{canvas;device;context;pipeline;uniformBuffer;bindGroup;bindGroupLayout;backgroundLandscapeView;backgroundPortraitView;sampler;currentBackground=`landscape`;startTime;constructor(e){this.canvas=e,this.startTime=performance.now()}async initialize(){if(!navigator.gpu)return console.error(`WebGPU not supported in this browser`),!1;let e=await navigator.gpu.requestAdapter();if(!e)return console.error(`Failed to get GPU adapter`),!1;this.device=await e.requestDevice(),this.context=this.canvas.getContext(`webgpu`);let t=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device:this.device,format:t,alphaMode:`premultiplied`});let[o,s]=await Promise.all([a(r),a(i)]);this.backgroundLandscapeView=this.createTextureFromImage(o).createView(),this.backgroundPortraitView=this.createTextureFromImage(s).createView(),this.sampler=this.device.createSampler({magFilter:`linear`,minFilter:`linear`}),this.uniformBuffer=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});let c=this.device.createShaderModule({code:n});this.bindGroupLayout=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:`filtering`}}]}),this.updateBackgroundBindGroup();let l=this.device.createPipelineLayout({bindGroupLayouts:[this.bindGroupLayout]});return this.pipeline=this.device.createRenderPipeline({layout:l,vertex:{module:c,entryPoint:`vertexMain`},fragment:{module:c,entryPoint:`fragmentMain`,targets:[{format:t}]},primitive:{topology:`triangle-list`}}),!0}createTextureFromImage(e){let t=this.device.createTexture({size:[e.width,e.height,1],format:`rgba8unorm`,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});return this.device.queue.copyExternalImageToTexture({source:e},{texture:t},[e.width,e.height]),t}render(){let e=(performance.now()-this.startTime)/1e3,t=new Float32Array([e,0,this.canvas.width,this.canvas.height,0,0,0,0]);this.device.queue.writeBuffer(this.uniformBuffer,0,t);let n=this.device.createCommandEncoder(),r=n.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:`clear`,storeOp:`store`}]});r.setPipeline(this.pipeline),r.setBindGroup(0,this.bindGroup),r.draw(3),r.end(),this.device.queue.submit([n.finish()])}resize(e,t){this.canvas.width=e,this.canvas.height=t,this.updateBackgroundBindGroup();let n=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device:this.device,format:n,alphaMode:`premultiplied`})}getDevice(){return this.device}updateBackgroundBindGroup(){if(!this.uniformBuffer||!this.bindGroupLayout)return;let e=this.canvas.height>this.canvas.width?`portrait`:`landscape`;if(e===this.currentBackground&&this.bindGroup)return;this.currentBackground=e;let t=e===`portrait`?this.backgroundPortraitView:this.backgroundLandscapeView;this.bindGroup=this.device.createBindGroup({layout:this.bindGroupLayout,entries:[{binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:t},{binding:2,resource:this.sampler}]})}},s=`// Instanced star dots for the dot-to-dot letters.
// Draws twinkling, state-colored stars with soft glow and rays.

struct Uniforms {
  time: f32,
  _pad0: f32,
  resolution: vec2<f32>,
  global: f32,
  mouseStrength: f32,
  mouse: vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) state: f32,
  @location(2) seed: f32,
  @location(3) posUv: vec2<f32>,
  @location(4) stateTime: f32,
};

fn hash11(n: f32) -> f32 {
  return fract(sin(n) * 43758.5453123);
}

@vertex
fn dotVertexMain(
  @location(0) quad: vec2<f32>,    // -1..+1
  @location(1) instA: vec4<f32>,   // pos.xy, sizePx, state
  @location(2) instB: vec4<f32>,   // seed, unused...
) -> VertexOut {
  var out: VertexOut;

  let pos = instA.xy;
  let sizePx = instA.z;
  let state = instA.w;
  let seed = instB.x;
  let stateTime = instB.y;

  if (sizePx <= 0.0) {
    out.position = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
    out.local = vec2<f32>(0.0, 0.0);
    out.state = 0.0;
    out.seed = 0.0;
    return out;
  }

  let scaleUv = vec2<f32>(
    sizePx / uniforms.resolution.x,
    sizePx / uniforms.resolution.y
  );
  let uv = pos + quad * scaleUv;
  let clip = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

  out.position = vec4<f32>(clip, 0.0, 1.0);
  out.local = quad;
  out.state = state;
  out.seed = seed;
  out.posUv = pos;
  out.stateTime = stateTime;
  return out;
}

fn easeOutBack(t: f32) -> f32 {
  let c1 = 1.70158;
  let c3 = c1 + 1.0;
  let x = t - 1.0;
  return 1.0 + c3 * x * x * x + c1 * x * x;
}

@fragment
fn dotFragmentMain(in: VertexOut) -> @location(0) vec4<f32> {
  let state = in.state;
  let seed = in.seed;
  let stateAge = max(0.0, uniforms.time - in.stateTime);

  var scale = 1.0;
  if (state > 0.5 && state < 1.5) {
    let a = clamp(stateAge / 0.5, 0.0, 1.0);
    scale = mix(0.6, 1.0, easeOutBack(a));
  } else if (state >= 1.5) {
    let settle = 1.0 - exp(-stateAge * 6.0);
    scale = mix(1.15, 1.0, settle);
  }

  let q = in.local / scale;
  let d = length(q);

  if (d > 1.6) {
    return vec4<f32>(0.0);
  }

  // Twinkle (state-dependent)
  let baseSpeed = 1.0 + hash11(seed * 19.1) * 2.6;
  let t1 = 0.5 + 0.5 * sin(uniforms.time * baseSpeed + seed * 6.28318);
  let t2 = 0.5 + 0.5 * sin(uniforms.time * (baseSpeed * 1.7) + seed * 2.123);
  let shimmerFreq = 6.0 + 9.0 * hash11(seed * 7.7);
  let shimmer = 0.85 + 0.15 * sin(uniforms.time * shimmerFreq + dot(q, vec2<f32>(11.3, 9.7)));

  var twinkle = mix(0.6, 1.0, pow(max(t1, t2), 8.0)) * shimmer;
  if (state < 0.5) {
    twinkle = mix(0.55, 0.95, pow(t1, 5.0)) * shimmer;
  } else if (state < 1.5) {
    let breath = 0.7 + 0.3 * sin(uniforms.time * 1.7 + seed * 6.28318);
    twinkle = mix(0.75, 1.15, pow(max(t1, t2), 6.0)) * breath;
  } else {
    twinkle = mix(0.75, 1.0, pow(t1, 3.0)) * 0.9;
  }

  let ang = atan2(q.y, q.x) + seed * 6.28318;
  let rayCount = 5.0 + floor(hash11(seed * 5.3) * 4.0);

  var core = pow(max(0.0, 1.0 - d), 7.0);
  var glow = pow(max(0.0, 1.0 - d), 2.1) * 0.85;
  var rays = pow(abs(cos(ang * rayCount)), 22.0) * smoothstep(1.3, 0.15, d);

  // Target gets extra cross rays + a pulsing ring
  var ring = 0.0;
  if (state > 0.5 && state < 1.5) {
    let cross = pow(abs(cos(ang * 2.0)), 18.0) * smoothstep(1.2, 0.12, d) * 0.7;
    rays = max(rays, cross);

    let pulse = fract((uniforms.time * 0.65) + seed);
    let ringR = mix(0.18, 1.05, pulse);
    let ringW = 0.05;
    ring = smoothstep(ringR + ringW, ringR, d) * smoothstep(ringR - ringW, ringR, d) * (1.2 - pulse);
  }

  // Done shockwave + flash
  var flash = 0.0;
  var shock = 0.0;
  if (state >= 1.5) {
    flash = exp(-stateAge * 14.0);
    let shockR = stateAge * 1.1 + 0.35;
    let shockW = 0.06;
    shock = smoothstep(shockR + shockW, shockR, d) * smoothstep(shockR - shockW, shockR, d);
  }

  var intensity = (core * 1.45 + glow * 1.2 + rays * 1.15 + ring + shock * 1.3 + flash * 1.4)
    * twinkle
    * uniforms.global;
  intensity = min(intensity, 1.0);

  // State colors: future red, target orange, done green
  let futureColor = vec3<f32>(1.0, 0.15, 0.08);
  let targetColor = vec3<f32>(1.0, 0.86, 0.18);
  let doneColor = vec3<f32>(0.15, 1.0, 0.5);

  var baseColor = futureColor;
  if (state > 0.5 && state < 1.5) {
    baseColor = targetColor;
  } else if (state >= 1.5) {
    let doneMix = clamp(stateAge / 0.35, 0.0, 1.0);
    baseColor = mix(targetColor, doneColor, doneMix);
  }

  let coreBoost = pow(max(0.0, 1.0 - d), 6.0);
  var color = mix(baseColor, vec3<f32>(1.0), 0.18 + coreBoost * 0.55);

  // Magnet glow for target dot
  if (state > 0.5 && state < 1.5) {
    let delta = (in.posUv - uniforms.mouse) * uniforms.resolution;
    let distPx = length(delta);
    let mag = (1.0 - smoothstep(160.0, 30.0, distPx)) * uniforms.mouseStrength;
    intensity *= (1.0 + mag * 0.85);
    color = mix(color, vec3<f32>(1.0, 0.95, 0.75), mag * 0.55);
  }

  return vec4<f32>(color * intensity, intensity);
}
`,c=`// Instanced “spark / glitter” particles.
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
  @location(5) life: f32,
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
  let life = max(b.y, 0.0001) * 10.0;
  let sizePx = b.z;
  let seedKind = b.w;

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

  // Gravity + drag so particles drift and settle like snow
  let g = vec2<f32>(0.0, 0.28);
  let drag = 2.2;
  let damp = exp(-drag * age);
  let velDamped = vel * damp;
  let pos = pos0 + vel * (1.0 - damp) / drag + 0.5 * g * age * age;

  let speed = length(velDamped);
  let dir = select(velDamped / max(speed, 1e-5), vec2<f32>(1.0, 0.0), speed < 1e-5);
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
  out.seed = seedKind;
  out.age01 = clamp(age / life, 0.0, 1.0);
  out.speed = speed;
  out.sizePx = sizePx;
  out.life = life;
  return out;
}

@fragment
fn sparkFragmentMain(in: VertexOut) -> @location(0) vec4<f32> {
  let kind = floor(in.seed);
  let seed = fract(in.seed);

  let speedBoost = smoothstep(0.05, 2.8, in.speed);

  // Reconstruct the same stretch the vertex used so we can “un-stretch”
  var stretch = 1.0 + speedBoost * 3.2;
  if (kind > 1.5 && kind < 2.5) { stretch = 1.0; }     // hero star: not stretched
  if (kind > 2.5) { stretch *= 3.0; }                  // comet: extra long

  // Work in un-stretched space for stable distance
  let q = vec2<f32>(in.local.x / stretch, in.local.y);
  let d = length(q);

  if (d > 1.55) {
    return vec4<f32>(0.0);
  }

  let ageSeconds = in.age01 * in.life;
  let fadeIn  = smoothstep(0.0, 0.12, ageSeconds);
  let fadeOut = 1.0 - smoothstep(0.68, 1.0, in.age01);
  let lifeFade = fadeIn * fadeOut;

  // --- Twinkle (rich shimmer + occasional pops) ---
  let base = 8.0 + 22.0 * hash11(seed * 17.3 + kind * 4.1);
  let twSpeed = base * (1.0 + speedBoost * 0.55);

  let t1 = 0.5 + 0.5 * sin(uniforms.time * twSpeed + seed * 6.28318);
  let t2 = 0.5 + 0.5 * sin(uniforms.time * (twSpeed * 1.73) + seed * 2.123);

  let shimmerFreq = 24.0 + 80.0 * hash11(seed * 9.2);
  let shimmer = 0.85 + 0.15 * sin(uniforms.time * shimmerFreq + dot(q, vec2<f32>(11.3, 9.7)));

  var twinkle = mix(0.3, 1.0, pow(max(t1, t2), 10.0)) * shimmer;

  // Kind-specific twinkle behavior
  if (kind < 0.5) { // dust: gentle sparkle
    twinkle = mix(0.55, 1.0, pow(t1, 6.0)) * shimmer;
  } else if (kind > 1.5 && kind < 2.5) { // hero star: slow “breathing” magic
    let breath = 0.65 + 0.35 * sin(uniforms.time * 1.6 + seed * 6.28318);
    twinkle = mix(0.5, 1.0, pow(max(t1, t2), 8.0)) * breath * shimmer;
  } else if (kind > 2.5) { // comet: sharp pops
    twinkle = mix(0.35, 1.0, pow(max(t1, t2), 12.0)) * shimmer;
  }

  // --- Shape ---
  var core = 0.0;
  var glow = 0.0;
  var rays = 0.0;

  let ang = atan2(q.y, q.x) + seed * 6.28318;

  if (kind < 0.5) {
    // tiny fairy dust
    core = pow(max(0.0, 1.0 - d), 12.0);
    glow = pow(max(0.0, 1.0 - d), 2.2) * 0.6;
    rays = pow(abs(cos(ang * 4.0)), 18.0) * smoothstep(1.2, 0.15, d) * 0.35;
  } else if (kind < 1.5) {
    // spark
    core = pow(max(0.0, 1.0 - d), 10.0);
    glow = pow(max(0.0, 1.0 - d), 2.0) * 0.45;
    let rc = 7.0 + floor(hash11(seed * 19.17) * 7.0);
    rays = pow(abs(cos(ang * rc)), 18.0) * smoothstep(1.3, 0.2, d);
  } else if (kind < 2.5) {
    // hero star (big 5-point star + cross sparkle)
    let spikes = pow(abs(cos(ang * 5.0)), 26.0);
    let cross  = pow(abs(cos(ang * 2.0)), 22.0) * 0.55;
    let star = max(spikes, cross);

    core = pow(max(0.0, 1.0 - d), 6.0) * (0.55 + star * 1.4);
    glow = pow(max(0.0, 1.0 - d), 1.8) * 1.15;
    rays = star * smoothstep(1.25, 0.1, d) * 1.1;
  } else {
    // comet (shooting star)
    let t = clamp(q.x * 0.5 + 0.5, 0.0, 1.0);  // 0 tail .. 1 head
    let head = pow(t, 10.0);
    let tail = pow(1.0 - t, 1.6);

    let width = exp(-q.y * q.y * 10.0);

    core = width * (head * 1.25 + tail * 0.55) * pow(max(0.0, 1.0 - abs(q.x)), 0.6);
    glow = width * pow(max(0.0, 1.0 - d), 1.6) * 0.95;
    rays = pow(abs(cos(ang * 6.0)), 18.0) * smoothstep(1.2, 0.12, d) * 0.55;
  }

  var intensity = (core + glow + rays) * twinkle;
  intensity *= lifeFade;
  intensity *= clamp(uniforms.global, 0.0, 1.0);

  // Motion energy (don’t overboost hero stars)
  if (kind < 2.5) {
    intensity *= (0.55 + speedBoost * 1.25);
  } else {
    intensity *= 1.05;
  }

  intensity = min(intensity, 1.0);

  // --- Color (aurora palette + kid-friendly gold stars) ---
  var color = sparkleColor(seed, uniforms.time);

  if (kind < 0.5) {
    color = mix(vec3<f32>(1.0, 1.0, 1.0), color, 0.35); // dust whiter
  } else if (kind < 1.5) {
    color = mix(color, vec3<f32>(1.0, 1.0, 1.0), 0.15); // sparks slightly whiter
  } else if (kind < 2.5) {
    let gold = vec3<f32>(1.0, 0.9, 0.55);
    color = mix(gold, color, 0.25); // golden hero stars
  } else {
    color = mix(vec3<f32>(1.0, 1.0, 1.0), color, 0.55); // comets bright
  }

  return vec4<f32>(color * intensity, intensity);
}
`,l=24576,u=8,d=u*4,f=1536,p=512,m=8,h=m*4,g=0,_=1,v=2,y=3;function b(e,t,n){let r=Math.max(0,Math.min(1,(n-e)/(t-e)));return r*r*(3-2*r)}var x=class{canvas;device;context;pipeline;dotPipeline;uniformBuffer;dotUniformBuffer;bindGroup;dotBindGroup;quadVertexBuffer;particleBuffer;dotBuffer;startTimeMs;uniformScratch=new Float32Array(8);dotUniformScratch=new Float32Array(8);mouseRamp=0;prevActive=!1;mousePos={x:.5,y:.5};mouseVelocity={x:0,y:0};lastSimTimeSec=0;dotCount=0;dotScratch=new Float32Array(p*m);dotStates=new Int8Array(p);dotStateTimes=new Float32Array(p);nextParticleIndex=0;pendingStart=0;pendingCount=0;pendingData=new Float32Array(f*u);sparkAccumulator=0;dustAccumulator=0;burstCooldown=0;haloPhase=0;haloAccumulator=0;stillSeconds=0;lastStampTime=-999;rngState=305419896;constructor(e){this.canvas=e,this.startTimeMs=performance.now()}async initialize(e){if(e)this.device=e;else{if(!navigator.gpu)return console.error(`WebGPU not supported`),!1;let e=await navigator.gpu.requestAdapter();if(!e)return console.error(`Failed to get GPU adapter`),!1;this.device=await e.requestDevice()}this.context=this.canvas.getContext(`webgpu`);let t=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device:this.device,format:t,alphaMode:`premultiplied`}),this.uniformBuffer=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.dotUniformBuffer=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.quadVertexBuffer=this.device.createBuffer({size:48,usage:GPUBufferUsage.VERTEX,mappedAtCreation:!0}),new Float32Array(this.quadVertexBuffer.getMappedRange()).set([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),this.quadVertexBuffer.unmap(),this.particleBuffer=this.device.createBuffer({size:l*d,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Float32Array(this.particleBuffer.getMappedRange()).fill(0),this.particleBuffer.unmap(),this.dotBuffer=this.device.createBuffer({size:p*h,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Float32Array(this.dotBuffer.getMappedRange()).fill(0),this.dotBuffer.unmap();let n=this.device.createShaderModule({code:s}),r=this.device.createShaderModule({code:c}),i=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}}]});this.bindGroup=this.device.createBindGroup({layout:i,entries:[{binding:0,resource:{buffer:this.uniformBuffer}}]}),this.dotBindGroup=this.device.createBindGroup({layout:i,entries:[{binding:0,resource:{buffer:this.dotUniformBuffer}}]});let a=this.device.createPipelineLayout({bindGroupLayouts:[i]});return this.pipeline=this.device.createRenderPipeline({layout:a,vertex:{module:r,entryPoint:`sparkVertexMain`,buffers:[{arrayStride:8,stepMode:`vertex`,attributes:[{shaderLocation:0,offset:0,format:`float32x2`}]},{arrayStride:d,stepMode:`instance`,attributes:[{shaderLocation:1,offset:0,format:`float32x4`},{shaderLocation:2,offset:16,format:`float32x4`}]}]},fragment:{module:r,entryPoint:`sparkFragmentMain`,targets:[{format:t,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one-minus-src-alpha`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),this.dotPipeline=this.device.createRenderPipeline({layout:a,vertex:{module:n,entryPoint:`dotVertexMain`,buffers:[{arrayStride:8,stepMode:`vertex`,attributes:[{shaderLocation:0,offset:0,format:`float32x2`}]},{arrayStride:h,stepMode:`instance`,attributes:[{shaderLocation:1,offset:0,format:`float32x4`},{shaderLocation:2,offset:16,format:`float32x4`}]}]},fragment:{module:n,entryPoint:`dotFragmentMain`,targets:[{format:t,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one-minus-src-alpha`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),!0}setDots(e,t,n){let r=Math.min(e.length,p);if(this.dotCount=r,r===0)return;let i=(performance.now()-this.startTimeMs)/1e3,a=Math.max(1,window.devicePixelRatio||1),o=Math.max(1,this.canvas.width/a),s=Math.max(1,this.canvas.height/a),c=Math.max(10,n*1.3);for(let n=0;n<r;n+=1){let r=e[n],l=r.x/o,u=r.y/s,d=0;n<t?d=2:n===t&&(d=1),this.dotStates[n]!==d&&(this.dotStates[n]=d,this.dotStateTimes[n]=i,d===2&&this.emitDotCompleteBurst(i,{x:l,y:u},a));let f=(d===1?c*1.45:d===2?c*1.2:c*1.05)*a,p=n*m;this.dotScratch[p+0]=l,this.dotScratch[p+1]=u,this.dotScratch[p+2]=f,this.dotScratch[p+3]=d,this.dotScratch[p+4]=this.hashSeed(n),this.dotScratch[p+5]=this.dotStateTimes[n],this.dotScratch[p+6]=0,this.dotScratch[p+7]=0}this.device.queue.writeBuffer(this.dotBuffer,0,this.dotScratch.subarray(0,r*m))}updateMouseState(e,t,n,r){let i=(performance.now()-this.startTimeMs)/1e3,a=this.lastSimTimeSec===0?0:i-this.lastSimTimeSec,o=Math.max(0,Math.min(.05,a));if(this.lastSimTimeSec=i,o>0){let t=o/.45;this.mouseRamp=e?Math.min(1,this.mouseRamp+t):Math.max(0,this.mouseRamp-t)}let s=Math.max(1,window.devicePixelRatio||1),c=Math.max(1,this.canvas.width/s),l=Math.max(1,this.canvas.height/s),u=Math.max(0,Math.min(1,t/c)),d=Math.max(0,Math.min(1,n/l)),f={...this.mousePos},p={x:u,y:d};if(this.mousePos=p,o>0){let e=(p.x-f.x)/o,t=(p.y-f.y)/o;this.mouseVelocity.x=this.mouseVelocity.x*.7+e*.3,this.mouseVelocity.y=this.mouseVelocity.y*.7+t*.3}let m=Math.hypot(this.mouseVelocity.x,this.mouseVelocity.y),h=b(.08,2.4,m);if(o>0&&this.mouseRamp>.02){this.haloPhase+=o*(1.2+h*5.5);let e=(18+h*140)*this.mouseRamp;this.haloAccumulator+=e*o;let t=Math.floor(this.haloAccumulator);this.haloAccumulator-=t,t=Math.min(t,12);let n=.014+.012*h;for(let e=0;e<t;e++){let e=this.haloPhase*Math.PI*2+this.rand01()*Math.PI*2,t=Math.cos(e)*n,r=Math.sin(e)*n,a=.18+h*.65,o=-Math.sin(e)*a+t*.8,c=Math.cos(e)*a+r*.8,l=.35+this.rand01()*.55,u=(.9+this.rand01()*2.2)*s;this.queueParticle(p.x+t,p.y+r,o,c,i,l,u,this.makeSeed(g))}}if(o>0&&(e&&m<.07?this.stillSeconds+=o:this.stillSeconds=0,e&&this.stillSeconds>.22&&i-this.lastStampTime>.35&&(this.stampHeroStar(i,p,s,10,18,1.1,1.8),this.emitDustRing(i,p,s,18),this.lastStampTime=i,this.stillSeconds=.12)),e&&!this.prevActive&&(this.emitRadialBurst(i,p,s,300),this.burstCooldown=.12),o>0&&(this.burstCooldown=Math.max(0,this.burstCooldown-o),e&&this.burstCooldown<=0&&h>.85&&this.mouseRamp>.25&&(this.emitDirectionalBurst(i,p,s,110),this.burstCooldown=.12)),o>0){let e=(120+h*1700)*this.mouseRamp;this.sparkAccumulator+=e*o;let t=Math.floor(this.sparkAccumulator);if(this.sparkAccumulator-=t,t=Math.min(t,180),t>0){let e=this.mouseVelocity.x,n=this.mouseVelocity.y,r=Math.hypot(e,n);if(r>1e-5)e/=r,n/=r;else{let t=this.rand01()*Math.PI*2;e=Math.cos(t),n=Math.sin(t)}let a=-n,o=e,c=1.25-h*.9;for(let r=0;r<t;r++){let t=this.rand01(),r=f.x+(p.x-f.x)*t,l=f.y+(p.y-f.y)*t,u=(this.rand01()-.5)*.01,d=a*u,m=o*u,g=(this.rand01()-.5)*c,v=Math.cos(g),b=Math.sin(g),x=e*v-n*b,S=e*b+n*v,C=.1+this.rand01()*.2+h*.75,w=(this.rand01()-.5)*(.36-h*.18),T=x*C+a*w*C,E=S*C+o*w*C,D=h>.75&&this.rand01()<.06,O=D?y:_,k=D?.45+this.rand01()*.55:.2+this.rand01()*.38+h*.18,A=(D?6+this.rand01()*10+h*10:1.4+this.rand01()*3.4+h*8)*s;this.queueParticle(r+d,l+m,T,E,i,k,A,this.makeSeed(O))}}}if(o>0&&r.length>0){let e=(45+160*Math.min(1,r.length/160))*(.25+.75*this.mouseRamp);this.dustAccumulator+=e*o;let t=Math.floor(this.dustAccumulator);this.dustAccumulator-=t,t=Math.min(t,60);for(let e=0;e<t;e++){let e=r[this.rand01()*r.length|0],t=Math.max(0,Math.min(1,e.x/c)),n=Math.max(0,Math.min(1,e.y/l)),a=this.rand01()*Math.PI*2,o=.014+this.rand01()*.03,u=Math.cos(a)*o,d=Math.sin(a)*o,f=.45+this.rand01()*1.2,p=(.9+this.rand01()*2.8)*s;this.queueParticle(t,n,u,d,i,f,p,this.makeSeed(g))}}e||(this.mouseVelocity.x*=.95,this.mouseVelocity.y*=.95),!e&&this.prevActive&&(this.stampHeroStar(i,p,s,14,26,1.4,2.4),this.emitDustRing(i,p,s,28)),this.prevActive=e}render(){let e=(performance.now()-this.startTimeMs)/1e3;this.flushPendingParticles(),this.uniformScratch[0]=e,this.uniformScratch[1]=0,this.uniformScratch[2]=this.canvas.width,this.uniformScratch[3]=this.canvas.height,this.uniformScratch[4]=this.mouseRamp,this.uniformScratch[5]=0,this.uniformScratch[6]=0,this.uniformScratch[7]=0,this.device.queue.writeBuffer(this.uniformBuffer,0,this.uniformScratch),this.dotUniformScratch[0]=e,this.dotUniformScratch[1]=0,this.dotUniformScratch[2]=this.canvas.width,this.dotUniformScratch[3]=this.canvas.height,this.dotUniformScratch[4]=1,this.dotUniformScratch[5]=this.mouseRamp,this.dotUniformScratch[6]=this.mousePos.x,this.dotUniformScratch[7]=this.mousePos.y,this.device.queue.writeBuffer(this.dotUniformBuffer,0,this.dotUniformScratch);let t=this.device.createCommandEncoder(),n=t.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:`clear`,storeOp:`store`}]});this.dotCount>0&&(n.setPipeline(this.dotPipeline),n.setBindGroup(0,this.dotBindGroup),n.setVertexBuffer(0,this.quadVertexBuffer),n.setVertexBuffer(1,this.dotBuffer),n.draw(6,this.dotCount,0,0)),n.setPipeline(this.pipeline),n.setBindGroup(0,this.bindGroup),n.setVertexBuffer(0,this.quadVertexBuffer),n.setVertexBuffer(1,this.particleBuffer),n.draw(6,l,0,0),n.end(),this.device.queue.submit([t.finish()])}resize(e,t){this.canvas.width=e,this.canvas.height=t;let n=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device:this.device,format:n,alphaMode:`premultiplied`})}rand01(){let e=this.rngState|0;return e^=e<<13,e^=e>>>17,e^=e<<5,this.rngState=e,(e>>>0)/4294967296}hashSeed(e){let t=(e+1)*2654435761;return t^=t<<13,t^=t>>>17,t^=t<<5,(t>>>0)/4294967296}makeSeed(e){return e+this.rand01()}queueParticle(e,t,n,r,i,a,o,s){if(this.pendingCount>=f)return;this.pendingCount===0&&(this.pendingStart=this.nextParticleIndex),this.nextParticleIndex=(this.nextParticleIndex+1)%l;let c=this.pendingCount*u,d=this.pendingData;d[c+0]=e,d[c+1]=t,d[c+2]=n,d[c+3]=r,d[c+4]=i,d[c+5]=a,d[c+6]=o,d[c+7]=s,this.pendingCount++}flushPendingParticles(){if(this.pendingCount===0)return;let e=this.pendingCount,t=this.pendingStart,n=Math.min(e,l-t),r=n*u;if(this.device.queue.writeBuffer(this.particleBuffer,t*d,this.pendingData.subarray(0,r)),e>n){let t=(e-n)*u;this.device.queue.writeBuffer(this.particleBuffer,0,this.pendingData.subarray(r,r+t))}this.pendingCount=0}emitRadialBurst(e,t,n,r){let i=Math.min(r,f);for(let r=0;r<i;r++){let r=this.rand01()*Math.PI*2,i=.14+this.rand01()*.8,a=Math.cos(r)*i,o=Math.sin(r)*i,s=.32+this.rand01()*.75,c=(2.2+this.rand01()*7)*n;this.queueParticle(t.x,t.y,a,o,e,s,c,this.makeSeed(_))}}emitDirectionalBurst(e,t,n,r){let i=this.mouseVelocity.x,a=this.mouseVelocity.y,o=Math.hypot(i,a);o>1e-5?(i/=o,a/=o):(i=1,a=0);let s=-a,c=i,l=Math.min(r,f);for(let r=0;r<l;r++){let r=(this.rand01()-.5)*.35,o=Math.cos(r),l=Math.sin(r),u=i*o-a*l,d=i*l+a*o,f=.17+this.rand01()*1,p=(this.rand01()-.5)*.25,m=u*f+s*p,h=d*f+c*p,g=.28+this.rand01()*.6,v=(2+this.rand01()*8.5)*n;this.queueParticle(t.x,t.y,m,h,e,g,v,this.makeSeed(_))}}stampHeroStar(e,t,n,r,i,a,o){let s=(r+this.rand01()*(i-r))*n,c=a+this.rand01()*(o-a),l=(this.rand01()-.5)*.05,u=-.05-this.rand01()*.07;this.queueParticle(t.x,t.y,l,u,e,c,s,this.makeSeed(v))}emitDustRing(e,t,n,r){let i=Math.min(r,40),a=.018;for(let r=0;r<i;r++){let r=this.rand01()*Math.PI*2,i=Math.cos(r)*a*(.6+this.rand01()*.6),o=Math.sin(r)*a*(.6+this.rand01()*.6),s=.03+this.rand01()*.08,c=i*s*30,l=o*s*30,u=.5+this.rand01()*.7,d=(.8+this.rand01()*2)*n;this.queueParticle(t.x+i,t.y+o,c,l,e,u,d,this.makeSeed(g))}}emitDotCompleteBurst(e,t,n){this.emitRadialBurst(e,t,n,320),this.emitDustRing(e,t,n,80),this.stampHeroStar(e,t,n,18,28,1.2,2)}},S=class{size;constructor(e,t,n){this.auroraCanvas=e,this.gameCanvas=t,this.sparkleCanvas=n,this.size={width:0,height:0,dpr:1}}resize(){let e=window.innerWidth,t=window.innerHeight,n=Math.max(1,window.devicePixelRatio||1);return this.applySize(this.auroraCanvas,e,t,n),this.applySize(this.gameCanvas,e,t,n),this.sparkleCanvas&&this.applySize(this.sparkleCanvas,e,t,n),this.size={width:e,height:t,dpr:n},this.size}configureGameContext(e){let{dpr:t}=this.size;e.setTransform(t,0,0,t,0,0),e.imageSmoothingEnabled=!0}getPointFromEvent(e,t){let n=e.getBoundingClientRect();return{x:t.clientX-n.left,y:t.clientY-n.top}}getSize(){return this.size}applySize(e,t,n,r){e.width=Math.max(1,Math.floor(t*r)),e.height=Math.max(1,Math.floor(n*r)),e.style.width=`${t}px`,e.style.height=`${n}px`}},C=class{callbacks={};activePointerId=null;constructor(e,t){this.canvas=e,this.getPoint=t,this.onPointerDown=this.onPointerDown.bind(this),this.onPointerMove=this.onPointerMove.bind(this),this.onPointerUp=this.onPointerUp.bind(this),this.canvas.addEventListener(`pointerdown`,this.onPointerDown),this.canvas.addEventListener(`pointermove`,this.onPointerMove),this.canvas.addEventListener(`pointerup`,this.onPointerUp),this.canvas.addEventListener(`pointercancel`,this.onPointerUp),this.canvas.addEventListener(`pointerleave`,this.onPointerUp)}setCallbacks(e){this.callbacks=e}dispose(){this.canvas.removeEventListener(`pointerdown`,this.onPointerDown),this.canvas.removeEventListener(`pointermove`,this.onPointerMove),this.canvas.removeEventListener(`pointerup`,this.onPointerUp),this.canvas.removeEventListener(`pointercancel`,this.onPointerUp),this.canvas.removeEventListener(`pointerleave`,this.onPointerUp)}onPointerDown(e){if(this.activePointerId!==null)return;this.activePointerId=e.pointerId,this.canvas.setPointerCapture(e.pointerId),e.preventDefault();let t=this.buildPoint(e);this.callbacks.onStart?.(t)}onPointerMove(e){if(this.activePointerId!==e.pointerId)return;e.preventDefault();let t=this.buildPoint(e);this.callbacks.onMove?.(t)}onPointerUp(e){if(this.activePointerId!==e.pointerId)return;e.preventDefault();let t=this.buildPoint(e);this.callbacks.onEnd?.(t),this.canvas.releasePointerCapture(e.pointerId),this.activePointerId=null}buildPoint(e){let{x:t,y:n}=this.getPoint(e);return{x:t,y:n,id:e.pointerId,time:performance.now()/1e3}}},w=t({default:()=>T}),T=[{order:1,x:725,y:320,display:`keystone`,note:`Start at Black Dot`},{order:2,x:660,y:255,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:580,y:215,display:`intermediate`,note:`Curve Up-Left`},{order:4,x:500,y:200,display:`keystone`,note:`Top Arch Center`},{order:5,x:420,y:220,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:350,y:265,display:`intermediate`,note:`Curve Down-Left`},{order:7,x:290,y:340,display:`keystone`,note:`Top Left Shoulder`},{order:8,x:250,y:435,display:`intermediate`,note:`Left Vertical`},{order:9,x:240,y:535,display:`intermediate`,note:`Left Vertical`},{order:10,x:260,y:635,display:`keystone`,note:`Bottom Left Turn`},{order:11,x:320,y:720,display:`intermediate`,note:`Bottom Curve`},{order:12,x:410,y:775,display:`intermediate`,note:`Bottom Curve`},{order:13,x:510,y:780,display:`keystone`,note:`Bottom Center`},{order:14,x:610,y:750,display:`intermediate`,note:`Curve Up-Right`},{order:15,x:680,y:690,display:`intermediate`,note:`Curve Up-Right`},{order:16,x:720,y:610,display:`keystone`,note:`Closing the Loop`},{order:17,x:740,y:530,display:`intermediate`,note:`Loop End`},{order:18,x:810,y:180,display:`intermediate`,newStroke:!0,note:`Lift Pen -> Top of Stem`},{order:19,x:810,y:265,display:`keystone`,note:`Stem Down`},{order:20,x:810,y:350,display:`intermediate`,note:`Stem Down`},{order:21,x:810,y:435,display:`intermediate`,note:`Stem Down`},{order:22,x:810,y:520,display:`keystone`,note:`Stem Center`},{order:23,x:810,y:605,display:`intermediate`,note:`Stem Down`},{order:24,x:810,y:690,display:`intermediate`,note:`Stem Down`},{order:25,x:810,y:775,display:`keystone`,note:`Stem Bottom`},{order:26,x:810,y:860,display:`intermediate`,note:`Stem Finish`}],E=t({default:()=>D}),D=[{order:1,x:240,y:150,display:`keystone`,note:`Start Dot (Top of Stem)`},{order:2,x:240,y:230,display:`intermediate`,note:`Stem Down`},{order:3,x:240,y:310,display:`intermediate`,note:`Stem Down`},{order:4,x:240,y:390,display:`keystone`,note:`Stem Middle`},{order:5,x:240,y:470,display:`intermediate`,note:`Stem Down`},{order:6,x:240,y:550,display:`intermediate`,note:`Stem Down`},{order:7,x:240,y:630,display:`keystone`,note:`Stem Lower`},{order:8,x:240,y:710,display:`intermediate`,note:`Stem Down`},{order:9,x:240,y:790,display:`intermediate`,note:`Stem Down`},{order:10,x:240,y:870,display:`keystone`,note:`Stem Bottom Arrow`},{order:11,x:240,y:450,display:`intermediate`,newStroke:!0,note:`Lift Pen -> Start Loop (Mid-Stem)`},{order:12,x:290,y:390,display:`intermediate`,note:`Loop Curve Up-Right`},{order:13,x:380,y:350,display:`keystone`,note:`Loop Top Left`},{order:14,x:490,y:335,display:`intermediate`,note:`Loop Top Center`},{order:15,x:610,y:350,display:`intermediate`,note:`Loop Top Right`},{order:16,x:710,y:410,display:`keystone`,note:`Loop Top Curve`},{order:17,x:780,y:500,display:`intermediate`,note:`Loop Far Right`},{order:18,x:800,y:600,display:`intermediate`,note:`Loop Far Right Mid`},{order:19,x:780,y:700,display:`keystone`,note:`Loop Bottom Right Turn`},{order:20,x:710,y:790,display:`intermediate`,note:`Loop Bottom Curve`},{order:21,x:610,y:845,display:`intermediate`,note:`Loop Bottom Right`},{order:22,x:500,y:870,display:`keystone`,note:`Loop Bottom Center`},{order:23,x:390,y:860,display:`intermediate`,note:`Loop Bottom Left`},{order:24,x:310,y:820,display:`intermediate`,note:`Loop Closing In`},{order:25,x:260,y:850,display:`keystone`,note:`Loop Connect to Stem`},{order:26,x:240,y:870,display:`intermediate`,note:`Finish`}],O=t({default:()=>k}),k=[{order:1,x:760,y:230,display:`keystone`,note:`Start Black Dot`},{order:2,x:700,y:190,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:635,y:165,display:`intermediate`,note:`Curve Up-Left`},{order:4,x:565,y:150,display:`keystone`,note:`Top Arch Right`},{order:5,x:495,y:150,display:`intermediate`,note:`Top Arch Center`},{order:6,x:425,y:165,display:`intermediate`,note:`Top Arch Left`},{order:7,x:360,y:195,display:`keystone`,note:`Curve Down-Left`},{order:8,x:300,y:240,display:`intermediate`,note:`Curve Down-Left`},{order:9,x:250,y:300,display:`intermediate`,note:`Upper Left`},{order:10,x:210,y:370,display:`keystone`,note:`Left Curve High`},{order:11,x:180,y:450,display:`intermediate`,note:`Left Curve Mid`},{order:12,x:170,y:530,display:`intermediate`,note:`Left Vertical Center`},{order:13,x:180,y:610,display:`keystone`,note:`Left Curve Low`},{order:14,x:210,y:690,display:`intermediate`,note:`Bottom Left Turn`},{order:15,x:260,y:760,display:`intermediate`,note:`Bottom Left Curve`},{order:16,x:320,y:810,display:`keystone`,note:`Bottom Arch Left`},{order:17,x:390,y:845,display:`intermediate`,note:`Bottom Arch Mid-Left`},{order:18,x:470,y:860,display:`intermediate`,note:`Bottom Arch Center`},{order:19,x:550,y:855,display:`keystone`,note:`Bottom Arch Right`},{order:20,x:620,y:830,display:`intermediate`,note:`Curve Up-Right`},{order:21,x:680,y:790,display:`intermediate`,note:`Curve Up-Right`},{order:22,x:730,y:740,display:`keystone`,note:`Lower Right Turn`},{order:26,x:780,y:660,display:`intermediate`,note:`Finish Arrow Tip`}],A=t({default:()=>j}),j=[{order:1,x:640,y:460,display:`keystone`,note:`Start Black Dot`},{order:2,x:580,y:380,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:500,y:330,display:`intermediate`,note:`Curve Up-Left`},{order:4,x:400,y:310,display:`keystone`,note:`Top of Loop`},{order:5,x:300,y:340,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:220,y:420,display:`intermediate`,note:`Curve Down-Left`},{order:7,x:160,y:550,display:`keystone`,note:`Leftmost Curve`},{order:8,x:200,y:680,display:`intermediate`,note:`Curve Down-Right`},{order:9,x:280,y:760,display:`intermediate`,note:`Curve Down-Right`},{order:10,x:400,y:800,display:`keystone`,note:`Bottom of Loop`},{order:11,x:520,y:780,display:`intermediate`,note:`Curve Up-Right`},{order:12,x:650,y:720,display:`intermediate`,note:`Approach Stem`},{order:13,x:780,y:620,display:`keystone`,note:`Connect to Stem (Go Up)`},{order:14,x:780,y:520,display:`intermediate`,note:`Stem Up`},{order:15,x:780,y:420,display:`intermediate`,note:`Stem Up`},{order:16,x:780,y:320,display:`keystone`,note:`Stem Up High`},{order:17,x:780,y:200,display:`intermediate`,note:`Stem Up`},{order:18,x:780,y:100,display:`intermediate`,note:`Reach Top`},{order:19,x:780,y:90,display:`keystone`,note:`Top Turnaround`},{order:20,x:780,y:250,display:`intermediate`,note:`Stem Down (Retrace)`},{order:21,x:780,y:400,display:`intermediate`,note:`Stem Down`},{order:22,x:780,y:550,display:`keystone`,note:`Stem Down Mid`},{order:23,x:780,y:700,display:`intermediate`,note:`Stem Down`},{order:24,x:780,y:800,display:`intermediate`,note:`Stem Down`},{order:25,x:780,y:900,display:`keystone`,note:`Stem Bottom`},{order:26,x:780,y:930,display:`intermediate`,note:`Finish Arrow`}],M=t({default:()=>N}),N=[{order:1,x:220,y:440,display:`keystone`,note:`Start Black Dot`},{order:2,x:330,y:440,display:`intermediate`,note:`Horizontal Bar`},{order:3,x:440,y:440,display:`intermediate`,note:`Horizontal Bar`},{order:4,x:550,y:440,display:`keystone`,note:`Center of Bar`},{order:5,x:660,y:440,display:`intermediate`,note:`Horizontal Bar`},{order:6,x:770,y:440,display:`intermediate`,note:`End of Bar (Turn Up)`},{order:7,x:760,y:360,display:`keystone`,note:`Curve Up-Left`},{order:8,x:700,y:280,display:`intermediate`,note:`Curve Up-Left`},{order:9,x:620,y:220,display:`intermediate`,note:`Top Right Curve`},{order:10,x:500,y:190,display:`keystone`,note:`Top Center`},{order:11,x:380,y:200,display:`intermediate`,note:`Top Left Curve`},{order:12,x:280,y:250,display:`intermediate`,note:`Curve Down-Left`},{order:13,x:200,y:330,display:`keystone`,note:`Top Left Shoulder`},{order:14,x:150,y:430,display:`intermediate`,note:`Left Spine High`},{order:15,x:140,y:530,display:`intermediate`,note:`Left Spine Mid`},{order:16,x:160,y:630,display:`keystone`,note:`Bottom Left Turn`},{order:17,x:210,y:720,display:`intermediate`,note:`Curve Down-Right`},{order:18,x:290,y:790,display:`intermediate`,note:`Curve Down-Right`},{order:19,x:390,y:840,display:`keystone`,note:`Bottom Center`},{order:20,x:500,y:850,display:`intermediate`,note:`Bottom Right`},{order:21,x:600,y:820,display:`intermediate`,note:`Curve Up-Right`},{order:22,x:680,y:770,display:`keystone`,note:`Tail Curve`},{order:23,x:730,y:720,display:`intermediate`,note:`Approach Arrow`},{order:24,x:750,y:690,display:`intermediate`,note:`Arrow Tail`},{order:25,x:760,y:670,display:`keystone`,note:`Arrow Head`},{order:26,x:770,y:650,display:`intermediate`,note:`Finish Tip`}],P=t({default:()=>F}),F=[{order:1,x:620,y:140,display:`keystone`,note:`Start Black Dot (Under Arch)`},{order:2,x:560,y:110,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:500,y:90,display:`intermediate`,note:`Top Arch Right`},{order:4,x:440,y:90,display:`keystone`,note:`Top Arch Center`},{order:5,x:390,y:130,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:380,y:190,display:`intermediate`,note:`Join Vertical Stem`},{order:7,x:380,y:270,display:`keystone`,note:`Stem High`},{order:8,x:380,y:360,display:`intermediate`,note:`Stem (Crossing Point)`},{order:9,x:380,y:450,display:`intermediate`,note:`Stem Middle`},{order:10,x:380,y:540,display:`keystone`,note:`Stem Mid-Low`},{order:11,x:380,y:630,display:`intermediate`,note:`Stem Lower`},{order:12,x:380,y:720,display:`intermediate`,note:`Stem Lower`},{order:13,x:380,y:810,display:`keystone`,note:`Stem Bottom Approach`},{order:14,x:380,y:860,display:`intermediate`,note:`Stem Arrow Start`},{order:15,x:380,y:900,display:`intermediate`,note:`Stem Arrow Body`},{order:16,x:380,y:930,display:`keystone`,note:`Stem Arrow Tip`},{order:17,x:380,y:940,display:`intermediate`,note:`Finish Stroke 1`},{order:18,x:380,y:940,display:`intermediate`,note:`Pen Lift Point`},{order:19,x:180,y:360,display:`keystone`,newStroke:!0,note:`Start Stroke 2 (The 'X')`},{order:20,x:250,y:360,display:`intermediate`,note:`Crossbar Left`},{order:21,x:320,y:360,display:`intermediate`,note:`Crossbar Approach`},{order:22,x:390,y:360,display:`keystone`,note:`Crossbar Center (After Stem)`},{order:23,x:460,y:360,display:`intermediate`,note:`Crossbar Right`},{order:24,x:530,y:360,display:`intermediate`,note:`Crossbar Right`},{order:25,x:600,y:360,display:`keystone`,note:`Crossbar Arrow Start`},{order:26,x:660,y:360,display:`intermediate`,note:`Finish Crossbar Arrow`}],I={"./assets/letters_json/a.json":w,"./assets/letters_json/b.json":E,"./assets/letters_json/c.json":O,"./assets/letters_json/d.json":A,"./assets/letters_json/e.json":M,"./assets/letters_json/f.json":P},L=Object.values(I).map(e=>e.default).filter(e=>Array.isArray(e)),R=()=>L.length===0?[]:L[Math.floor(Math.random()*L.length)].map(e=>({order:e.order,x:e.x,y:e.y})).sort((e,t)=>e.order-t.order),z=e=>{if(e.length===0)return{minX:0,maxX:0,minY:0,maxY:0};let t=e[0].x,n=e[0].x,r=e[0].y,i=e[0].y;for(let a of e)a.x<t&&(t=a.x),a.x>n&&(n=a.x),a.y<r&&(r=a.y),a.y>i&&(i=a.y);return{minX:t,maxX:n,minY:r,maxY:i}},B=class{plotPoints=[];paths=[];currentPath=null;fadeSeconds=9;maxPoints=600;isDrawing=!1;currentMousePos={x:0,y:0};plotBounds={minX:0,maxX:0,minY:0,maxY:0};scaledPlotPoints=[];lastPlotSize={width:0,height:0};needsRescale=!0;currentTargetIndex=0;dotRadius=20;hitRadiusScale=1.5;lineSegmentIndex=0;lineSegmentT=0;linePauseRemaining=0;lineSegmentSeconds=.37;lineLoopPauseSeconds=.4;completionMessageUntil=0;pendingLetterReset=!1;completionMessageSeconds=2;callbacks;constructor(e,t={}){this.plotPoints=R(),this.plotBounds=z(this.plotPoints),this.callbacks=t,e.setCallbacks({onStart:e=>this.startPath(e),onMove:e=>this.extendPath(e),onEnd:e=>this.endPath(e)})}getDrawingState(){let e=this.paths.flatMap(e=>e);return{isDrawing:this.isDrawing,mouseX:this.currentMousePos.x,mouseY:this.currentMousePos.y,trailPoints:e}}getDotState(){return{dots:this.scaledPlotPoints,currentIndex:this.currentTargetIndex,radius:this.dotRadius}}clearUserPaths(){this.paths=[],this.currentPath=null,this.isDrawing=!1}resetCurrentLetter(){this.clearUserPaths(),this.currentTargetIndex=0,this.lineSegmentIndex=0,this.lineSegmentT=0,this.linePauseRemaining=0,this.pendingLetterReset=!1,this.completionMessageUntil=0}setViewportSize(e,t){if(this.plotPoints.length===0)return;let n=Math.max(1,e),r=Math.max(1,t);if(n===this.lastPlotSize.width&&r===this.lastPlotSize.height&&!this.needsRescale)return;this.lastPlotSize={width:n,height:r},this.needsRescale=!1;let i=this.plotBounds.maxX-this.plotBounds.minX,a=this.plotBounds.maxY-this.plotBounds.minY;if(i<=0||a<=0)return;let o=n*.5,s=r*.5,c=Math.min(o/i,s/a),l=(n-i*c)/2-this.plotBounds.minX*c,u=(r-a*c)/2-this.plotBounds.minY*c;this.scaledPlotPoints=this.plotPoints.map(e=>({order:e.order,x:e.x*c+l,y:e.y*c+u}))}update(e){this.advancePlotAnimation(e);let t=performance.now()/1e3;if(this.paths.length>0){let e=t-this.fadeSeconds;for(let t of this.paths){let n=0;for(;n<t.length&&t[n].time<e;)n+=1;n>0&&t.splice(0,n),t.length>this.maxPoints&&t.splice(0,t.length-this.maxPoints)}this.paths=this.paths.filter(e=>e.length>0)}this.pendingLetterReset&&t>=this.completionMessageUntil&&(this.resetForNewLetter(),this.pendingLetterReset=!1,this.completionMessageUntil=0)}render(e){if(e.save(),e.setTransform(1,0,0,1,0,0),e.clearRect(0,0,e.canvas.width,e.canvas.height),e.restore(),this.renderPlotLines(e),this.paths.length>0){e.strokeStyle=`rgba(200, 230, 255, 0.9)`,e.lineWidth=15,e.lineJoin=`round`,e.lineCap=`round`;for(let t of this.paths)t.length!==0&&(e.beginPath(),t.forEach((t,n)=>{n===0?e.moveTo(t.x,t.y):e.lineTo(t.x,t.y)}),e.stroke())}this.renderCompletionMessage(e)}startPath(e){this.pendingLetterReset||(this.isDrawing=!0,this.currentMousePos={x:e.x,y:e.y},this.currentPath=[{x:e.x,y:e.y,time:e.time}],this.paths.push(this.currentPath),this.tryAdvanceTarget(e))}extendPath(e){this.pendingLetterReset||(this.currentMousePos={x:e.x,y:e.y},this.currentPath&&(this.currentPath.push({x:e.x,y:e.y,time:e.time}),this.tryAdvanceTarget(e)))}endPath(e){this.pendingLetterReset||(this.currentMousePos={x:e.x,y:e.y},this.currentPath&&this.currentPath.push({x:e.x,y:e.y,time:e.time}),this.tryAdvanceTarget(e),this.isDrawing=!1,this.currentPath=null)}renderPlotLines(e){if(this.scaledPlotPoints.length<2)return;e.strokeStyle=`rgba(255, 220, 0, 0.5)`,e.lineWidth=9,e.lineJoin=`round`,e.lineCap=`round`,e.beginPath();let t=Math.min(Math.max(0,this.currentTargetIndex-1),this.scaledPlotPoints.length-1),n=this.scaledPlotPoints.slice(t);if(!(n.length<2)){if(e.moveTo(n[0].x,n[0].y),this.linePauseRemaining>0)for(let t=1;t<n.length;t+=1)e.lineTo(n[t].x,n[t].y);else{let t=Math.min(this.lineSegmentIndex,n.length-1);for(let r=1;r<=t;r+=1)e.lineTo(n[r].x,n[r].y);let r=Math.min(this.lineSegmentIndex+1,n.length-1);if(r>this.lineSegmentIndex){let t=n[this.lineSegmentIndex],i=n[r],a=t.x+(i.x-t.x)*this.lineSegmentT,o=t.y+(i.y-t.y)*this.lineSegmentT;e.lineTo(a,o)}}e.stroke()}}tryAdvanceTarget(e){if(!this.isDrawing||this.scaledPlotPoints.length===0||this.currentTargetIndex>=this.scaledPlotPoints.length)return;let t=this.scaledPlotPoints[this.currentTargetIndex],n=e.x-t.x,r=e.y-t.y,i=this.getTargetDotRadius()*this.hitRadiusScale;n*n+r*r<=i*i&&(this.currentTargetIndex=Math.min(this.currentTargetIndex+1,this.scaledPlotPoints.length),this.callbacks.onPointHit?.(),this.lineSegmentIndex=0,this.lineSegmentT=0,this.linePauseRemaining=0,this.currentTargetIndex>=this.scaledPlotPoints.length&&this.handleLetterComplete())}getTargetDotRadius(){return Math.max(10,this.dotRadius*1.3)*1.45}advancePlotAnimation(e){if(e<=0||this.scaledPlotPoints.length<2)return;let t=Math.max(0,this.scaledPlotPoints.length-Math.max(0,this.currentTargetIndex-1));if(t<2)return;let n=t-1;if(this.lineSegmentIndex=Math.min(this.lineSegmentIndex,n-1),this.linePauseRemaining>0){if(this.linePauseRemaining=Math.max(0,this.linePauseRemaining-e),this.linePauseRemaining>0)return;this.lineSegmentIndex=0,this.lineSegmentT=0}for(this.lineSegmentT+=e/this.lineSegmentSeconds;this.lineSegmentT>=1;)if(--this.lineSegmentT,this.lineSegmentIndex+=1,this.lineSegmentIndex>=n){this.linePauseRemaining=this.lineLoopPauseSeconds,this.lineSegmentIndex=0,this.lineSegmentT=0;break}}handleLetterComplete(){this.pendingLetterReset||(this.completionMessageUntil=performance.now()/1e3+this.completionMessageSeconds,this.pendingLetterReset=!0,this.isDrawing=!1,this.currentPath=null)}resetForNewLetter(){this.plotPoints=R(),this.plotBounds=z(this.plotPoints),this.scaledPlotPoints=[],this.currentTargetIndex=0,this.lineSegmentIndex=0,this.lineSegmentT=0,this.linePauseRemaining=0,this.paths=[],this.currentPath=null,this.isDrawing=!1,this.needsRescale=!0,this.lastPlotSize.width>0&&this.lastPlotSize.height>0&&this.setViewportSize(this.lastPlotSize.width,this.lastPlotSize.height)}renderCompletionMessage(e){if(performance.now()/1e3>=this.completionMessageUntil)return;let t=e.canvas.width/(window.devicePixelRatio||1),n=e.canvas.height/(window.devicePixelRatio||1);e.save(),e.textAlign=`center`,e.textBaseline=`middle`,e.fillStyle=`rgba(255, 244, 210, 0.95)`,e.strokeStyle=`rgba(0, 0, 0, 0.45)`,e.lineWidth=6,e.font=`700 48px system-ui, -apple-system, sans-serif`;let r=`Well done!`;e.strokeText(r,t/2,n*.45),e.fillText(r,t/2,n*.45),e.restore()}};async function V(){let e=document.querySelector(`#app`),t=new URL(`/letter_constellations/assets/aurora-vQgsr6MW.mp3`,``+import.meta.url).toString(),n=new Audio(t);n.loop=!0,n.preload=`auto`,n.volume=1;let r=new URL(`/letter_constellations/assets/chime-DLBpLcOd.mp3`,``+import.meta.url).toString(),i=new Audio(r);i.preload=`auto`;let a=null,s=null,c=()=>{if(a)return;a=new AudioContext;let e=a.createMediaElementSource(n);s=a.createGain(),s.gain.value=.33,e.connect(s).connect(a.destination)},l=document.createElement(`canvas`);l.id=`aurora-canvas`,e.appendChild(l);let u=document.createElement(`canvas`);u.id=`sparkle-canvas`,e.appendChild(u);let d=document.createElement(`canvas`);d.id=`game-canvas`,e.appendChild(d);let f=document.createElement(`div`);f.className=`audio-attribution`,f.innerHTML=`
    <span>Music: </span>
    <a href="https://freemusicarchive.org/music/Kevin_Hartnell/Umbra_1955/Kevin_Hartnell_-_Umbra_-_10_Aurora/" target="_blank" rel="noopener noreferrer">Aurora</a>
    <span> by Kevin Hartnell — </span>
    <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener noreferrer">CC BY 4.0</a>
    <span> (no changes)</span>
  `,e.appendChild(f);let p=document.createElement(`button`);p.className=`mute-button`,p.type=`button`,p.textContent=`Mute`,p.setAttribute(`aria-pressed`,`false`),e.appendChild(p);let m=document.createElement(`div`);m.className=`top-controls`;let h=document.createElement(`button`);h.className=`control-button`,h.type=`button`,h.textContent=`Clear`,m.appendChild(h);let g=document.createElement(`button`);g.className=`control-button`,g.type=`button`,g.textContent=`Reset`,m.appendChild(g),e.appendChild(m);let _=document.createElement(`div`);_.className=`volume-control`,_.innerHTML=`
    <label for="volume-slider">Volume</label>
    <input id="volume-slider" type="range" min="0" max="2" step="0.05" value="0.33" />
  `,e.appendChild(_);let v=_.querySelector(`#volume-slider`);i.volume=Math.min(1,Number(v.value));let y=()=>{let e=n.muted;p.textContent=e?`Unmute`:`Mute`,p.setAttribute(`aria-pressed`,String(e)),i.muted=e};y();let b=!1,w=()=>{b&&(c(),a&&a.state===`suspended`&&a.resume().catch(()=>{}),n.play().catch(()=>{}))},T=()=>{b=!0,w()},E={once:!0,capture:!0};document.addEventListener(`pointerdown`,T,E),document.addEventListener(`touchstart`,T,E),document.addEventListener(`click`,T,E),document.addEventListener(`keydown`,T,E),p.addEventListener(`click`,e=>{e.stopPropagation(),n.muted=!n.muted,n.muted||w(),y()}),v.addEventListener(`input`,()=>{c(),s&&(s.gain.value=Number(v.value)),i.volume=Math.min(1,Number(v.value))});let D=null,O=null,k=null,A=new S(l,d,u),j=d.getContext(`2d`);if(!j)throw Error(`Failed to get 2D context for game canvas.`);let M=()=>{let{width:e,height:t,dpr:n}=A.resize();A.configureGameContext(j),k&&k.setViewportSize(e,t),D&&D.resize(e*n,t*n),O&&O.resize(e*n,t*n)};M(),window.addEventListener(`resize`,M),D=new o(l);let N=await D.initialize();O=new x(u);let P=await O.initialize(D.getDevice());if(M(),!N||!P){let e=document.createElement(`div`);e.className=`error-message`,e.innerHTML=`
            <h1>WebGPU Not Supported</h1>
            <p>Your browser doesn't support WebGPU yet.</p>
            <p>Try using Chrome 113+ or Edge 113+ with WebGPU enabled.</p>
        `,document.body.appendChild(e);return}k=new B(new C(d,e=>A.getPointFromEvent(d,e)),{onPointHit:()=>{i.currentTime=0,i.play().catch(()=>{})}}),h.addEventListener(`click`,e=>{e.stopPropagation(),k?.clearUserPaths()}),g.addEventListener(`click`,e=>{e.stopPropagation(),k?.resetCurrentLetter()});let F=A.getSize();k.setViewportSize(F.width,F.height);let I=performance.now();function L(e){let t=(e-I)/1e3;if(I=e,!k||!D||!O)return;k.update(t);let n=k.getDrawingState();O.updateMouseState(n.isDrawing,n.mouseX,n.mouseY,n.trailPoints);let r=k.getDotState();O.setDots(r.dots,r.currentIndex,r.radius),D.render(),O.render(),j&&k.render(j),requestAnimationFrame(L)}L(performance.now())}V();
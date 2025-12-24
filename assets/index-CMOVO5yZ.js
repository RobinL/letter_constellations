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

fn pow8(x: f32) -> f32 {
    // x^8 using 3 squarings (cheap, predictable)
    var v = x * x;
    v = v * v;
    v = v * v;
    return v;
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
    let baseSpeed = mix(0.20, 0.70, hash01(cell, 5u)) * (10.0 / max(cellPx, 6.0));
    let t1 = 0.5 + 0.5 * sin(time * baseSpeed + phase);
    let t2 = 0.5 + 0.5 * sin(time * (baseSpeed * 1.73) + phase * 1.37);
    let pmax = max(t1, t2);

    let p2 = pmax * pmax;
    let p4 = p2 * p2;
    let sparkle = p4 * pmax;

    let p8 = p4 * p4;
    let glint = p8 * p8;

    let slowBreath = 0.85 + 0.15 * sin(time * 0.12 + phase * 0.5);
    let tw = mix(0.82, 1.22, sparkle) * slowBreath;

    let big = smoothstep(1.3, 3.0, sizePx);
    let q = (p - centerPx) / max(1.0, sizePx * 3.0);
    let qd = vec2<f32>(q.x + q.y, q.x - q.y) * 0.70710678;

    let lx = max(0.0, 1.0 - abs(q.x));
    let ly = max(0.0, 1.0 - abs(q.y));
    let ldx = max(0.0, 1.0 - abs(qd.x));
    let ldy = max(0.0, 1.0 - abs(qd.y));

    let cross = pow8(lx) + pow8(ly);
    let diag = pow8(ldx) + pow8(ldy);

    let rayEnv = smoothstep(1.15, 0.0, length(q));
    let rays = (cross * 0.75 + diag * 0.35) * rayEnv;

    let glintJitter = mix(0.65, 1.35, hash01(cell, 8u));
    let sparkleBurst = rays * glint * big * glintJitter * 1.2;

    let tintMix = hash01(cell, 6u);
    let cool = vec3<f32>(0.80, 0.88, 1.00);
    let warm = vec3<f32>(1.00, 0.95, 0.86);
    let tint = mix(cool, warm, tintMix);
    var color = mix(vec3<f32>(1.0), tint, 0.22 + sparkle * 0.14);
    color = mix(color, vec3<f32>(1.0), glint * big * 0.45);

    let brightJitter = mix(0.60, 1.60, hash01(cell, 7u));
    let intensity = ((core + glow) * tw + sparkleBurst) * brightness * brightJitter;
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
`,r=`/letter_constellations/assets/background_landscape-CSs9pKZF.webp`,i=`/letter_constellations/assets/background_portrait-CoUykP07.webp`;async function a(e){let t=await fetch(e);if(!t.ok)throw Error(`Failed to load image: ${e}`);let n=await t.blob();return createImageBitmap(n)}var o=class{canvas;device;context;pipeline;uniformBuffer;bindGroup;bindGroupLayout;backgroundLandscapeView;backgroundPortraitView;sampler;currentBackground=`landscape`;startTime;constructor(e){this.canvas=e,this.startTime=performance.now()}async initialize(){if(!navigator.gpu)return console.error(`WebGPU not supported in this browser`),!1;let e=await navigator.gpu.requestAdapter();if(!e)return console.error(`Failed to get GPU adapter`),!1;this.device=await e.requestDevice(),this.context=this.canvas.getContext(`webgpu`);let t=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device:this.device,format:t,alphaMode:`premultiplied`});let[o,s]=await Promise.all([a(r),a(i)]);this.backgroundLandscapeView=this.createTextureFromImage(o).createView(),this.backgroundPortraitView=this.createTextureFromImage(s).createView(),this.sampler=this.device.createSampler({magFilter:`linear`,minFilter:`linear`}),this.uniformBuffer=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});let c=this.device.createShaderModule({code:n});this.bindGroupLayout=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:`filtering`}}]}),this.updateBackgroundBindGroup();let l=this.device.createPipelineLayout({bindGroupLayouts:[this.bindGroupLayout]});return this.pipeline=this.device.createRenderPipeline({layout:l,vertex:{module:c,entryPoint:`vertexMain`},fragment:{module:c,entryPoint:`fragmentMain`,targets:[{format:t}]},primitive:{topology:`triangle-list`}}),!0}createTextureFromImage(e){let t=this.device.createTexture({size:[e.width,e.height,1],format:`rgba8unorm`,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});return this.device.queue.copyExternalImageToTexture({source:e},{texture:t},[e.width,e.height]),t}render(){let e=(performance.now()-this.startTime)/1e3,t=new Float32Array([e,0,this.canvas.width,this.canvas.height,0,0,0,0]);this.device.queue.writeBuffer(this.uniformBuffer,0,t);let n=this.device.createCommandEncoder(),r=n.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:`clear`,storeOp:`store`}]});r.setPipeline(this.pipeline),r.setBindGroup(0,this.bindGroup),r.draw(3),r.end(),this.device.queue.submit([n.finish()])}resize(e,t){this.canvas.width=e,this.canvas.height=t,this.updateBackgroundBindGroup();let n=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device:this.device,format:n,alphaMode:`premultiplied`})}getDevice(){return this.device}updateBackgroundBindGroup(){if(!this.uniformBuffer||!this.bindGroupLayout)return;let e=this.canvas.height>this.canvas.width?`portrait`:`landscape`;if(e===this.currentBackground&&this.bindGroup)return;this.currentBackground=e;let t=e===`portrait`?this.backgroundPortraitView:this.backgroundLandscapeView;this.bindGroup=this.device.createBindGroup({layout:this.bindGroupLayout,entries:[{binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:t},{binding:2,resource:this.sampler}]})}},s=`// Instanced star dots for the dot-to-dot letters.
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

  var emphasis = 1.0;
  if (state > 0.5 && state < 1.5) {
    emphasis = 1.25;
  } else if (state >= 1.5) {
    emphasis = 0.9;
  } else {
    emphasis = 0.8;
  }

  var intensity = (core * 1.45 + glow * 1.2 + rays * 1.15 + ring + shock * 1.3 + flash * 1.4)
    * twinkle
    * uniforms.global
    * emphasis;
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
`,l=16384,u=8,d=u*4,f=1024,p=512,m=8,h=m*4,g=0,_=1,v=2,y=3,b=.7,x=.8,S=12,C=.03;function w(e,t,n){let r=Math.max(0,Math.min(1,(n-e)/(t-e)));return r*r*(3-2*r)}var ee=class{canvas;device;context;pipeline;dotPipeline;uniformBuffer;dotUniformBuffer;bindGroup;dotBindGroup;quadVertexBuffer;particleBuffer;dotBuffer;startTimeMs;uniformScratch=new Float32Array(8);dotUniformScratch=new Float32Array(8);mouseRamp=0;prevActive=!1;mousePos={x:.5,y:.5};mouseVelocity={x:0,y:0};lastSimTimeSec=0;dotCount=0;dotScratch=new Float32Array(p*m);dotStates=new Int8Array(p);dotStateTimes=new Float32Array(p);nextParticleIndex=0;pendingStart=0;pendingCount=0;pendingData=new Float32Array(f*u);sparkAccumulator=0;dustAccumulator=0;burstCooldown=0;haloPhase=0;haloAccumulator=0;stillSeconds=0;lastStampTime=-999;rngState=305419896;constructor(e){this.canvas=e,this.startTimeMs=performance.now()}async initialize(e){if(e)this.device=e;else{if(!navigator.gpu)return console.error(`WebGPU not supported`),!1;let e=await navigator.gpu.requestAdapter();if(!e)return console.error(`Failed to get GPU adapter`),!1;this.device=await e.requestDevice()}this.context=this.canvas.getContext(`webgpu`);let t=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device:this.device,format:t,alphaMode:`premultiplied`}),this.uniformBuffer=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.dotUniformBuffer=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.quadVertexBuffer=this.device.createBuffer({size:48,usage:GPUBufferUsage.VERTEX,mappedAtCreation:!0}),new Float32Array(this.quadVertexBuffer.getMappedRange()).set([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),this.quadVertexBuffer.unmap(),this.particleBuffer=this.device.createBuffer({size:l*d,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Float32Array(this.particleBuffer.getMappedRange()).fill(0),this.particleBuffer.unmap(),this.dotBuffer=this.device.createBuffer({size:p*h,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Float32Array(this.dotBuffer.getMappedRange()).fill(0),this.dotBuffer.unmap();let n=this.device.createShaderModule({code:s}),r=this.device.createShaderModule({code:c}),i=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}}]});this.bindGroup=this.device.createBindGroup({layout:i,entries:[{binding:0,resource:{buffer:this.uniformBuffer}}]}),this.dotBindGroup=this.device.createBindGroup({layout:i,entries:[{binding:0,resource:{buffer:this.dotUniformBuffer}}]});let a=this.device.createPipelineLayout({bindGroupLayouts:[i]});return this.pipeline=this.device.createRenderPipeline({layout:a,vertex:{module:r,entryPoint:`sparkVertexMain`,buffers:[{arrayStride:8,stepMode:`vertex`,attributes:[{shaderLocation:0,offset:0,format:`float32x2`}]},{arrayStride:d,stepMode:`instance`,attributes:[{shaderLocation:1,offset:0,format:`float32x4`},{shaderLocation:2,offset:16,format:`float32x4`}]}]},fragment:{module:r,entryPoint:`sparkFragmentMain`,targets:[{format:t,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one-minus-src-alpha`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),this.dotPipeline=this.device.createRenderPipeline({layout:a,vertex:{module:n,entryPoint:`dotVertexMain`,buffers:[{arrayStride:8,stepMode:`vertex`,attributes:[{shaderLocation:0,offset:0,format:`float32x2`}]},{arrayStride:h,stepMode:`instance`,attributes:[{shaderLocation:1,offset:0,format:`float32x4`},{shaderLocation:2,offset:16,format:`float32x4`}]}]},fragment:{module:n,entryPoint:`dotFragmentMain`,targets:[{format:t,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one-minus-src-alpha`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),!0}setDots(e,t,n){let r=Math.min(e.length,p);if(this.dotCount=r,r===0)return;let i=(performance.now()-this.startTimeMs)/1e3,{dpr:a,cssW:o,cssH:s}=this.getCanvasMetrics(),c=Math.max(10,n*1.3);for(let n=0;n<r;n+=1){let r=e[n],l=r.x/o,u=r.y/s,d=0;n<t?d=2:n===t&&(d=1),this.dotStates[n]!==d&&(this.dotStates[n]=d,this.dotStateTimes[n]=i,d===2&&this.emitDotCompleteBurst(i,{x:l,y:u},a));let f=c*(d===1?1.7:d===2?.95:.85)*a,p=n*m;this.dotScratch[p+0]=l,this.dotScratch[p+1]=u,this.dotScratch[p+2]=f,this.dotScratch[p+3]=d,this.dotScratch[p+4]=this.hashSeed(n),this.dotScratch[p+5]=this.dotStateTimes[n],this.dotScratch[p+6]=0,this.dotScratch[p+7]=0}this.device.queue.writeBuffer(this.dotBuffer,0,this.dotScratch.subarray(0,r*m))}updateMouseState(e,t,n,r){let i=(performance.now()-this.startTimeMs)/1e3,a=this.lastSimTimeSec===0?0:i-this.lastSimTimeSec,o=Math.max(0,Math.min(.05,a));if(this.lastSimTimeSec=i,o>0){let t=o/.45;this.mouseRamp=e?Math.min(1,this.mouseRamp+t):Math.max(0,this.mouseRamp-t)}let{dpr:s,cssW:c,cssH:l}=this.getCanvasMetrics(),u=Math.max(0,Math.min(1,t/c)),d=Math.max(0,Math.min(1,n/l)),f={...this.mousePos},p={x:u,y:d};if(this.mousePos=p,o>0){let e=(p.x-f.x)/o,t=(p.y-f.y)/o;this.mouseVelocity.x=this.mouseVelocity.x*.7+e*.3,this.mouseVelocity.y=this.mouseVelocity.y*.7+t*.3}let m=Math.hypot(this.mouseVelocity.x,this.mouseVelocity.y),h=w(.08,2.4,m);if(o>0&&this.mouseRamp>.02){this.haloPhase+=o*(1.2+h*5.5);let e=(18+h*140)*this.mouseRamp*b;this.haloAccumulator+=e*o;let t=Math.floor(this.haloAccumulator);this.haloAccumulator-=t,t=Math.min(t,10);let n=.014+.012*h;for(let e=0;e<t;e++){let e=this.haloPhase*Math.PI*2+this.rand01()*Math.PI*2,t=Math.cos(e)*n,r=Math.sin(e)*n,a=.18+h*.65,o=-Math.sin(e)*a+t*.8,c=Math.cos(e)*a+r*.8,l=.35+this.rand01()*.55,u=(.9+this.rand01()*2.2)*x,d=Math.min(u,S)*s;this.queueParticle(p.x+t,p.y+r,o,c,i,l,d,this.makeSeed(g))}}if(o>0&&(e&&m<.07?this.stillSeconds+=o:this.stillSeconds=0,e&&this.stillSeconds>.22&&i-this.lastStampTime>.35&&(this.stampHeroStar(i,p,s,10,18,1.1,1.8),this.emitDustRing(i,p,s,18),this.lastStampTime=i,this.stillSeconds=.12)),e&&!this.prevActive&&(this.emitRadialBurst(i,p,s,300),this.burstCooldown=.12),o>0&&(this.burstCooldown=Math.max(0,this.burstCooldown-o),e&&this.burstCooldown<=0&&h>.85&&this.mouseRamp>.25&&(this.emitDirectionalBurst(i,p,s,110),this.burstCooldown=.12)),o>0){let e=(120+h*1700)*this.mouseRamp*b;this.sparkAccumulator+=e*o;let t=Math.floor(this.sparkAccumulator);if(this.sparkAccumulator-=t,t=Math.min(t,120),t>0){let e=this.mouseVelocity.x,n=this.mouseVelocity.y,r=Math.hypot(e,n);if(r>1e-5)e/=r,n/=r;else{let t=this.rand01()*Math.PI*2;e=Math.cos(t),n=Math.sin(t)}let a=-n,o=e,c=1.25-h*.9;for(let r=0;r<t;r++){let t=this.rand01(),r=f.x+(p.x-f.x)*t,l=f.y+(p.y-f.y)*t,u=(this.rand01()-.5)*.01,d=a*u,m=o*u,g=(this.rand01()-.5)*c,v=Math.cos(g),b=Math.sin(g),w=e*v-n*b,ee=e*b+n*v,T=.1+this.rand01()*.2+h*.75,E=(this.rand01()-.5)*(.36-h*.18),D=w*T+a*E*T,O=ee*T+o*E*T,k=h>.75&&this.rand01()<C,A=k?y:_,j=k?.45+this.rand01()*.55:.2+this.rand01()*.38+h*.18,M=(k?6+this.rand01()*10+h*10:1.4+this.rand01()*3.4+h*8)*x,N=Math.min(M,S)*s;this.queueParticle(r+d,l+m,D,O,i,j,N,this.makeSeed(A))}}}if(o>0&&r.length>0){let e=(45+160*Math.min(1,r.length/160))*(.25+.75*this.mouseRamp)*b;this.dustAccumulator+=e*o;let t=Math.floor(this.dustAccumulator);this.dustAccumulator-=t,t=Math.min(t,45);for(let e=0;e<t;e++){let e=r[this.rand01()*r.length|0],t=Math.max(0,Math.min(1,e.x/c)),n=Math.max(0,Math.min(1,e.y/l)),a=this.rand01()*Math.PI*2,o=.014+this.rand01()*.03,u=Math.cos(a)*o,d=Math.sin(a)*o,f=.45+this.rand01()*1.2,p=(.9+this.rand01()*2.8)*x,m=Math.min(p,S)*s;this.queueParticle(t,n,u,d,i,f,m,this.makeSeed(g))}}e||(this.mouseVelocity.x*=.95,this.mouseVelocity.y*=.95),!e&&this.prevActive&&(this.stampHeroStar(i,p,s,14,26,1.4,2.4),this.emitDustRing(i,p,s,28)),this.prevActive=e}render(){let e=(performance.now()-this.startTimeMs)/1e3;this.flushPendingParticles(),this.uniformScratch[0]=e,this.uniformScratch[1]=0,this.uniformScratch[2]=this.canvas.width,this.uniformScratch[3]=this.canvas.height,this.uniformScratch[4]=this.mouseRamp,this.uniformScratch[5]=0,this.uniformScratch[6]=0,this.uniformScratch[7]=0,this.device.queue.writeBuffer(this.uniformBuffer,0,this.uniformScratch),this.dotUniformScratch[0]=e,this.dotUniformScratch[1]=0,this.dotUniformScratch[2]=this.canvas.width,this.dotUniformScratch[3]=this.canvas.height,this.dotUniformScratch[4]=1,this.dotUniformScratch[5]=this.mouseRamp,this.dotUniformScratch[6]=this.mousePos.x,this.dotUniformScratch[7]=this.mousePos.y,this.device.queue.writeBuffer(this.dotUniformBuffer,0,this.dotUniformScratch);let t=this.device.createCommandEncoder(),n=t.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:`clear`,storeOp:`store`}]});this.dotCount>0&&(n.setPipeline(this.dotPipeline),n.setBindGroup(0,this.dotBindGroup),n.setVertexBuffer(0,this.quadVertexBuffer),n.setVertexBuffer(1,this.dotBuffer),n.draw(6,this.dotCount,0,0)),n.setPipeline(this.pipeline),n.setBindGroup(0,this.bindGroup),n.setVertexBuffer(0,this.quadVertexBuffer),n.setVertexBuffer(1,this.particleBuffer),n.draw(6,l,0,0),n.end(),this.device.queue.submit([t.finish()])}resize(e,t){this.canvas.width=e,this.canvas.height=t;let n=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device:this.device,format:n,alphaMode:`premultiplied`})}getCanvasMetrics(){let e=Math.max(1,this.canvas.clientWidth||this.canvas.width),t=Math.max(1,this.canvas.clientHeight||this.canvas.height),n=this.canvas.width/e;return{dpr:Math.max(.5,n),cssW:e,cssH:t}}rand01(){let e=this.rngState|0;return e^=e<<13,e^=e>>>17,e^=e<<5,this.rngState=e,(e>>>0)/4294967296}hashSeed(e){let t=(e+1)*2654435761;return t^=t<<13,t^=t>>>17,t^=t<<5,(t>>>0)/4294967296}makeSeed(e){return e+this.rand01()}queueParticle(e,t,n,r,i,a,o,s){if(this.pendingCount>=f)return;this.pendingCount===0&&(this.pendingStart=this.nextParticleIndex),this.nextParticleIndex=(this.nextParticleIndex+1)%l;let c=this.pendingCount*u,d=this.pendingData;d[c+0]=e,d[c+1]=t,d[c+2]=n,d[c+3]=r,d[c+4]=i,d[c+5]=a,d[c+6]=o,d[c+7]=s,this.pendingCount++}flushPendingParticles(){if(this.pendingCount===0)return;let e=this.pendingCount,t=this.pendingStart,n=Math.min(e,l-t),r=n*u;if(this.device.queue.writeBuffer(this.particleBuffer,t*d,this.pendingData.subarray(0,r)),e>n){let t=(e-n)*u;this.device.queue.writeBuffer(this.particleBuffer,0,this.pendingData.subarray(r,r+t))}this.pendingCount=0}emitRadialBurst(e,t,n,r){let i=Math.min(Math.floor(r*b),f);for(let r=0;r<i;r++){let r=this.rand01()*Math.PI*2,i=.14+this.rand01()*.8,a=Math.cos(r)*i,o=Math.sin(r)*i,s=.32+this.rand01()*.75,c=(2.2+this.rand01()*7)*x,l=Math.min(c,S)*n;this.queueParticle(t.x,t.y,a,o,e,s,l,this.makeSeed(_))}}emitDirectionalBurst(e,t,n,r){let i=this.mouseVelocity.x,a=this.mouseVelocity.y,o=Math.hypot(i,a);o>1e-5?(i/=o,a/=o):(i=1,a=0);let s=-a,c=i,l=Math.min(Math.floor(r*b),f);for(let r=0;r<l;r++){let r=(this.rand01()-.5)*.35,o=Math.cos(r),l=Math.sin(r),u=i*o-a*l,d=i*l+a*o,f=.17+this.rand01()*1,p=(this.rand01()-.5)*.25,m=u*f+s*p,h=d*f+c*p,g=.28+this.rand01()*.6,v=(2+this.rand01()*8.5)*x,y=Math.min(v,S)*n;this.queueParticle(t.x,t.y,m,h,e,g,y,this.makeSeed(_))}}stampHeroStar(e,t,n,r,i,a,o){let s=(r+this.rand01()*(i-r))*x,c=Math.min(s,S*1.5)*n,l=a+this.rand01()*(o-a),u=(this.rand01()-.5)*.05,d=-.05-this.rand01()*.07;this.queueParticle(t.x,t.y,u,d,e,l,c,this.makeSeed(v))}emitDustRing(e,t,n,r){let i=Math.min(Math.floor(r*b),30),a=.018;for(let r=0;r<i;r++){let r=this.rand01()*Math.PI*2,i=Math.cos(r)*a*(.6+this.rand01()*.6),o=Math.sin(r)*a*(.6+this.rand01()*.6),s=.03+this.rand01()*.08,c=i*s*30,l=o*s*30,u=.5+this.rand01()*.7,d=(.8+this.rand01()*2)*x,f=Math.min(d,S)*n;this.queueParticle(t.x+i,t.y+o,c,l,e,u,f,this.makeSeed(g))}}emitDotCompleteBurst(e,t,n){this.emitRadialBurst(e,t,n,320),this.emitDustRing(e,t,n,80),this.stampHeroStar(e,t,n,18,28,1.2,2)}},T=class{size;auroraScale=.66;sparkleScale=1;constructor(e,t,n){this.auroraCanvas=e,this.gameCanvas=t,this.sparkleCanvas=n,this.size={width:0,height:0,dpr:1}}resize(){let e=window.innerWidth,t=window.innerHeight,n=Math.max(1,window.devicePixelRatio||1),r=Math.min(1,n),i=Math.max(.5,r*this.auroraScale),a=Math.max(.5,r*this.sparkleScale);return this.applySize(this.auroraCanvas,e,t,i),this.applySize(this.gameCanvas,e,t,n),this.sparkleCanvas&&this.applySize(this.sparkleCanvas,e,t,a),this.size={width:e,height:t,dpr:n},this.size}configureGameContext(e){let{dpr:t}=this.size;e.setTransform(t,0,0,t,0,0),e.imageSmoothingEnabled=!0}getPointFromEvent(e,t){let n=e.getBoundingClientRect();return{x:t.clientX-n.left,y:t.clientY-n.top}}getSize(){return this.size}applySize(e,t,n,r){e.width=Math.max(1,Math.floor(t*r)),e.height=Math.max(1,Math.floor(n*r)),e.style.width=`${t}px`,e.style.height=`${n}px`}},E=class{callbacks={};activePointerId=null;constructor(e,t){this.canvas=e,this.getPoint=t,this.onPointerDown=this.onPointerDown.bind(this),this.onPointerMove=this.onPointerMove.bind(this),this.onPointerUp=this.onPointerUp.bind(this),this.canvas.addEventListener(`pointerdown`,this.onPointerDown),this.canvas.addEventListener(`pointermove`,this.onPointerMove),this.canvas.addEventListener(`pointerup`,this.onPointerUp),this.canvas.addEventListener(`pointercancel`,this.onPointerUp),this.canvas.addEventListener(`pointerleave`,this.onPointerUp)}setCallbacks(e){this.callbacks=e}dispose(){this.canvas.removeEventListener(`pointerdown`,this.onPointerDown),this.canvas.removeEventListener(`pointermove`,this.onPointerMove),this.canvas.removeEventListener(`pointerup`,this.onPointerUp),this.canvas.removeEventListener(`pointercancel`,this.onPointerUp),this.canvas.removeEventListener(`pointerleave`,this.onPointerUp)}onPointerDown(e){if(this.activePointerId!==null)return;this.activePointerId=e.pointerId,this.canvas.setPointerCapture(e.pointerId),e.preventDefault();let t=this.buildPoint(e);this.callbacks.onStart?.(t)}onPointerMove(e){if(this.activePointerId!==e.pointerId)return;e.preventDefault();let t=this.buildPoint(e);this.callbacks.onMove?.(t)}onPointerUp(e){if(this.activePointerId!==e.pointerId)return;e.preventDefault();let t=this.buildPoint(e);this.callbacks.onEnd?.(t),this.canvas.releasePointerCapture(e.pointerId),this.activePointerId=null}buildPoint(e){let{x:t,y:n}=this.getPoint(e);return{x:t,y:n,id:e.pointerId,time:performance.now()/1e3}}},D=t({default:()=>O}),O=[{order:1,x:725,y:320,display:`keystone`,note:`Start at Black Dot`},{order:2,x:660,y:255,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:580,y:215,display:`intermediate`,note:`Curve Up-Left`},{order:4,x:500,y:200,display:`keystone`,note:`Top Arch Center`},{order:5,x:420,y:220,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:350,y:265,display:`intermediate`,note:`Curve Down-Left`},{order:7,x:290,y:340,display:`keystone`,note:`Top Left Shoulder`},{order:8,x:250,y:435,display:`intermediate`,note:`Left Vertical`},{order:9,x:240,y:535,display:`intermediate`,note:`Left Vertical`},{order:10,x:260,y:635,display:`keystone`,note:`Bottom Left Turn`},{order:11,x:320,y:720,display:`intermediate`,note:`Bottom Curve`},{order:12,x:410,y:775,display:`intermediate`,note:`Bottom Curve`},{order:13,x:510,y:780,display:`keystone`,note:`Bottom Center`},{order:14,x:610,y:750,display:`intermediate`,note:`Curve Up-Right`},{order:15,x:680,y:690,display:`intermediate`,note:`Curve Up-Right`},{order:16,x:720,y:610,display:`keystone`,note:`Closing the Loop`},{order:17,x:740,y:530,display:`intermediate`,note:`Loop End`},{order:18,x:810,y:180,display:`intermediate`,newStroke:!0,note:`Lift Pen -> Top of Stem`},{order:19,x:810,y:265,display:`keystone`,note:`Stem Down`},{order:20,x:810,y:350,display:`intermediate`,note:`Stem Down`},{order:21,x:810,y:435,display:`intermediate`,note:`Stem Down`},{order:22,x:810,y:520,display:`keystone`,note:`Stem Center`},{order:23,x:810,y:605,display:`intermediate`,note:`Stem Down`},{order:24,x:810,y:690,display:`intermediate`,note:`Stem Down`},{order:25,x:810,y:775,display:`keystone`,note:`Stem Bottom`},{order:26,x:810,y:860,display:`intermediate`,note:`Stem Finish`}],k=t({default:()=>A}),A=[{order:1,x:240,y:150,display:`keystone`,note:`Start Dot (Top of Stem)`},{order:2,x:240,y:230,display:`intermediate`,note:`Stem Down`},{order:3,x:240,y:310,display:`intermediate`,note:`Stem Down`},{order:4,x:240,y:390,display:`keystone`,note:`Stem Middle`},{order:5,x:240,y:470,display:`intermediate`,note:`Stem Down`},{order:6,x:240,y:550,display:`intermediate`,note:`Stem Down`},{order:7,x:240,y:630,display:`keystone`,note:`Stem Lower`},{order:8,x:240,y:710,display:`intermediate`,note:`Stem Down`},{order:9,x:240,y:790,display:`intermediate`,note:`Stem Down`},{order:10,x:240,y:870,display:`keystone`,note:`Stem Bottom Arrow`},{order:11,x:240,y:450,display:`intermediate`,newStroke:!0,note:`Lift Pen -> Start Loop (Mid-Stem)`},{order:12,x:290,y:390,display:`intermediate`,note:`Loop Curve Up-Right`},{order:13,x:380,y:350,display:`keystone`,note:`Loop Top Left`},{order:14,x:490,y:335,display:`intermediate`,note:`Loop Top Center`},{order:15,x:610,y:350,display:`intermediate`,note:`Loop Top Right`},{order:16,x:710,y:410,display:`keystone`,note:`Loop Top Curve`},{order:17,x:780,y:500,display:`intermediate`,note:`Loop Far Right`},{order:18,x:800,y:600,display:`intermediate`,note:`Loop Far Right Mid`},{order:19,x:780,y:700,display:`keystone`,note:`Loop Bottom Right Turn`},{order:20,x:710,y:790,display:`intermediate`,note:`Loop Bottom Curve`},{order:21,x:610,y:845,display:`intermediate`,note:`Loop Bottom Right`},{order:22,x:500,y:870,display:`keystone`,note:`Loop Bottom Center`},{order:23,x:390,y:860,display:`intermediate`,note:`Loop Bottom Left`},{order:24,x:310,y:820,display:`intermediate`,note:`Loop Closing In`},{order:25,x:260,y:850,display:`keystone`,note:`Loop Connect to Stem`},{order:26,x:240,y:870,display:`intermediate`,note:`Finish`}],j=t({default:()=>M}),M=[{order:1,x:760,y:230,display:`keystone`,note:`Start Black Dot`},{order:2,x:700,y:190,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:635,y:165,display:`intermediate`,note:`Curve Up-Left`},{order:4,x:565,y:150,display:`keystone`,note:`Top Arch Right`},{order:5,x:495,y:150,display:`intermediate`,note:`Top Arch Center`},{order:6,x:425,y:165,display:`intermediate`,note:`Top Arch Left`},{order:7,x:360,y:195,display:`keystone`,note:`Curve Down-Left`},{order:8,x:300,y:240,display:`intermediate`,note:`Curve Down-Left`},{order:9,x:250,y:300,display:`intermediate`,note:`Upper Left`},{order:10,x:210,y:370,display:`keystone`,note:`Left Curve High`},{order:11,x:180,y:450,display:`intermediate`,note:`Left Curve Mid`},{order:12,x:170,y:530,display:`intermediate`,note:`Left Vertical Center`},{order:13,x:180,y:610,display:`keystone`,note:`Left Curve Low`},{order:14,x:210,y:690,display:`intermediate`,note:`Bottom Left Turn`},{order:15,x:260,y:760,display:`intermediate`,note:`Bottom Left Curve`},{order:16,x:320,y:810,display:`keystone`,note:`Bottom Arch Left`},{order:17,x:390,y:845,display:`intermediate`,note:`Bottom Arch Mid-Left`},{order:18,x:470,y:860,display:`intermediate`,note:`Bottom Arch Center`},{order:19,x:550,y:855,display:`keystone`,note:`Bottom Arch Right`},{order:20,x:620,y:830,display:`intermediate`,note:`Curve Up-Right`},{order:21,x:680,y:790,display:`intermediate`,note:`Curve Up-Right`},{order:22,x:730,y:740,display:`keystone`,note:`Lower Right Turn`},{order:26,x:780,y:660,display:`intermediate`,note:`Finish Arrow Tip`}],N=t({default:()=>P}),P=[{order:1,x:640,y:460,display:`keystone`,note:`Start Black Dot`},{order:2,x:580,y:380,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:500,y:330,display:`intermediate`,note:`Curve Up-Left`},{order:4,x:400,y:310,display:`keystone`,note:`Top of Loop`},{order:5,x:300,y:340,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:220,y:420,display:`intermediate`,note:`Curve Down-Left`},{order:7,x:160,y:550,display:`keystone`,note:`Leftmost Curve`},{order:8,x:200,y:680,display:`intermediate`,note:`Curve Down-Right`},{order:9,x:280,y:760,display:`intermediate`,note:`Curve Down-Right`},{order:10,x:400,y:800,display:`keystone`,note:`Bottom of Loop`},{order:11,x:520,y:780,display:`intermediate`,note:`Curve Up-Right`},{order:12,x:650,y:720,display:`intermediate`,note:`Approach Stem`},{order:13,x:780,y:620,display:`keystone`,note:`Connect to Stem (Go Up)`},{order:14,x:780,y:520,display:`intermediate`,note:`Stem Up`},{order:15,x:780,y:420,display:`intermediate`,note:`Stem Up`},{order:16,x:780,y:320,display:`keystone`,note:`Stem Up High`},{order:17,x:780,y:200,display:`intermediate`,note:`Stem Up`},{order:18,x:780,y:100,display:`intermediate`,note:`Reach Top`},{order:19,x:780,y:90,display:`keystone`,note:`Top Turnaround`},{order:20,x:780,y:250,display:`intermediate`,note:`Stem Down (Retrace)`},{order:21,x:780,y:400,display:`intermediate`,note:`Stem Down`},{order:22,x:780,y:550,display:`keystone`,note:`Stem Down Mid`},{order:23,x:780,y:700,display:`intermediate`,note:`Stem Down`},{order:24,x:780,y:800,display:`intermediate`,note:`Stem Down`},{order:25,x:780,y:900,display:`keystone`,note:`Stem Bottom`},{order:26,x:780,y:930,display:`intermediate`,note:`Finish Arrow`}],F=t({default:()=>te}),te=[{order:1,x:220,y:440,display:`keystone`,note:`Start Black Dot`},{order:2,x:330,y:440,display:`intermediate`,note:`Horizontal Bar`},{order:3,x:440,y:440,display:`intermediate`,note:`Horizontal Bar`},{order:4,x:550,y:440,display:`keystone`,note:`Center of Bar`},{order:5,x:660,y:440,display:`intermediate`,note:`Horizontal Bar`},{order:6,x:770,y:440,display:`intermediate`,note:`End of Bar (Turn Up)`},{order:7,x:760,y:360,display:`keystone`,note:`Curve Up-Left`},{order:8,x:700,y:280,display:`intermediate`,note:`Curve Up-Left`},{order:9,x:620,y:220,display:`intermediate`,note:`Top Right Curve`},{order:10,x:500,y:190,display:`keystone`,note:`Top Center`},{order:11,x:380,y:200,display:`intermediate`,note:`Top Left Curve`},{order:12,x:280,y:250,display:`intermediate`,note:`Curve Down-Left`},{order:13,x:200,y:330,display:`keystone`,note:`Top Left Shoulder`},{order:14,x:150,y:430,display:`intermediate`,note:`Left Spine High`},{order:15,x:140,y:530,display:`intermediate`,note:`Left Spine Mid`},{order:16,x:160,y:630,display:`keystone`,note:`Bottom Left Turn`},{order:17,x:210,y:720,display:`intermediate`,note:`Curve Down-Right`},{order:18,x:290,y:790,display:`intermediate`,note:`Curve Down-Right`},{order:19,x:390,y:840,display:`keystone`,note:`Bottom Center`},{order:20,x:500,y:850,display:`intermediate`,note:`Bottom Right`},{order:21,x:600,y:820,display:`intermediate`,note:`Curve Up-Right`},{order:22,x:680,y:770,display:`keystone`,note:`Tail Curve`},{order:23,x:730,y:720,display:`intermediate`,note:`Approach Arrow`},{order:24,x:750,y:690,display:`intermediate`,note:`Arrow Tail`},{order:25,x:760,y:670,display:`keystone`,note:`Arrow Head`},{order:26,x:770,y:650,display:`intermediate`,note:`Finish Tip`}],I=t({default:()=>L}),L=[{order:1,x:620,y:140,display:`keystone`,note:`Start Black Dot (Under Arch)`},{order:2,x:560,y:110,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:500,y:90,display:`intermediate`,note:`Top Arch Right`},{order:4,x:440,y:90,display:`keystone`,note:`Top Arch Center`},{order:5,x:390,y:130,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:380,y:190,display:`intermediate`,note:`Join Vertical Stem`},{order:7,x:380,y:270,display:`keystone`,note:`Stem High`},{order:8,x:380,y:360,display:`intermediate`,note:`Stem (Crossing Point)`},{order:9,x:380,y:450,display:`intermediate`,note:`Stem Middle`},{order:10,x:380,y:540,display:`keystone`,note:`Stem Mid-Low`},{order:11,x:380,y:630,display:`intermediate`,note:`Stem Lower`},{order:12,x:380,y:720,display:`intermediate`,note:`Stem Lower`},{order:13,x:380,y:810,display:`keystone`,note:`Stem Bottom Approach`},{order:14,x:380,y:860,display:`intermediate`,note:`Stem Arrow Start`},{order:15,x:380,y:900,display:`intermediate`,note:`Stem Arrow Body`},{order:16,x:380,y:930,display:`keystone`,note:`Stem Arrow Tip`},{order:17,x:380,y:940,display:`intermediate`,note:`Finish Stroke 1`},{order:18,x:380,y:940,display:`intermediate`,note:`Pen Lift Point`},{order:19,x:180,y:360,display:`keystone`,newStroke:!0,note:`Start Stroke 2 (The 'X')`},{order:20,x:250,y:360,display:`intermediate`,note:`Crossbar Left`},{order:21,x:320,y:360,display:`intermediate`,note:`Crossbar Approach`},{order:22,x:390,y:360,display:`keystone`,note:`Crossbar Center (After Stem)`},{order:23,x:460,y:360,display:`intermediate`,note:`Crossbar Right`},{order:24,x:530,y:360,display:`intermediate`,note:`Crossbar Right`},{order:25,x:600,y:360,display:`keystone`,note:`Crossbar Arrow Start`},{order:26,x:660,y:360,display:`intermediate`,note:`Finish Crossbar Arrow`}],R=t({default:()=>z}),z=[{order:1,x:710,y:240,display:`keystone`,note:`Start Black Dot`},{order:2,x:630,y:180,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:520,y:150,display:`intermediate`,note:`Top Curve`},{order:4,x:410,y:160,display:`keystone`,note:`Top Center`},{order:5,x:300,y:220,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:210,y:320,display:`intermediate`,note:`Left Curve High`},{order:7,x:180,y:440,display:`keystone`,note:`Left Side Center`},{order:8,x:210,y:560,display:`intermediate`,note:`Left Curve Low`},{order:9,x:300,y:640,display:`intermediate`,note:`Bottom Curve Left`},{order:10,x:430,y:660,display:`keystone`,note:`Bottom of Loop`},{order:11,x:580,y:620,display:`intermediate`,note:`Curve Up-Right`},{order:12,x:700,y:520,display:`intermediate`,note:`Approach Stem`},{order:13,x:780,y:400,display:`keystone`,note:`Connect to Stem`},{order:14,x:780,y:300,display:`intermediate`,note:`Stem Up`},{order:15,x:780,y:200,display:`intermediate`,note:`Stem Up`},{order:16,x:780,y:120,display:`keystone`,note:`Top of Stem`},{order:17,x:780,y:300,display:`intermediate`,note:`Stem Down (Retrace)`},{order:18,x:780,y:500,display:`intermediate`,note:`Stem Down Middle`},{order:19,x:780,y:700,display:`keystone`,note:`Descender Start`},{order:20,x:780,y:820,display:`intermediate`,note:`Descender Down`},{order:21,x:720,y:900,display:`intermediate`,note:`Tail Curve Start`},{order:22,x:600,y:940,display:`keystone`,note:`Tail Bottom Center`},{order:23,x:480,y:940,display:`intermediate`,note:`Tail Curve Left`},{order:24,x:360,y:900,display:`intermediate`,note:`Tail Curve Up`},{order:25,x:260,y:840,display:`keystone`,note:`Tail Arrow Head`},{order:26,x:200,y:800,display:`intermediate`,note:`Finish Tip`}],B=t({default:()=>V}),V=[{order:1,x:220,y:120,display:`keystone`,note:`Start Black Dot`},{order:2,x:220,y:230,display:`intermediate`,note:`Stem Down`},{order:3,x:220,y:310,display:`intermediate`,note:`Stem Down`},{order:4,x:220,y:390,display:`keystone`,note:`Stem Upper Middle`},{order:5,x:220,y:470,display:`intermediate`,note:`Stem Middle`},{order:6,x:220,y:550,display:`intermediate`,note:`Stem Lower Middle`},{order:7,x:220,y:630,display:`keystone`,note:`Stem Low`},{order:8,x:220,y:710,display:`intermediate`,note:`Stem Lower`},{order:9,x:220,y:790,display:`intermediate`,note:`Stem Approach Bottom`},{order:10,x:220,y:870,display:`keystone`,note:`Stem Left Arrow`},{order:11,x:220,y:500,display:`intermediate`,newStroke:!0,note:`Branch Point (Mid-Stem)`},{order:12,x:280,y:420,display:`intermediate`,note:`Arch Up-Right`},{order:13,x:360,y:360,display:`keystone`,note:`Arch Left Shoulder`},{order:14,x:460,y:330,display:`intermediate`,note:`Arch Top Center`},{order:15,x:560,y:330,display:`intermediate`,note:`Arch Top Right`},{order:16,x:660,y:360,display:`keystone`,note:`Arch Right Shoulder`},{order:17,x:740,y:420,display:`intermediate`,note:`Arch Curve Down`},{order:18,x:780,y:500,display:`intermediate`,note:`Right Leg Start`},{order:19,x:780,y:580,display:`keystone`,note:`Right Leg Mid-High`},{order:20,x:780,y:650,display:`intermediate`,note:`Right Leg Middle`},{order:21,x:780,y:720,display:`intermediate`,note:`Right Leg Low`},{order:22,x:780,y:790,display:`keystone`,note:`Right Leg Lower`},{order:23,x:780,y:830,display:`intermediate`,note:`Approach Arrow`},{order:24,x:780,y:850,display:`intermediate`,note:`Arrow Tail`},{order:25,x:780,y:870,display:`keystone`,note:`Right Arrow Tip`},{order:26,x:780,y:890,display:`intermediate`,note:`Finish`}],ne=t({default:()=>re}),re=[{order:1,x:500,y:350,display:`keystone`,note:`Start Black Dot (Stem Top)`},{order:2,x:500,y:420,display:`intermediate`,note:`Stem Down`},{order:3,x:500,y:490,display:`intermediate`,note:`Stem Down`},{order:4,x:500,y:560,display:`keystone`,note:`Stem Middle`},{order:5,x:500,y:630,display:`intermediate`,note:`Stem Down`},{order:6,x:500,y:700,display:`intermediate`,note:`Stem Lower`},{order:7,x:500,y:770,display:`keystone`,note:`Begin Turn`},{order:8,x:510,y:830,display:`intermediate`,note:`Curve Right`},{order:9,x:535,y:875,display:`intermediate`,note:`Curve Bottom`},{order:10,x:570,y:900,display:`keystone`,note:`Hook Upwards`},{order:11,x:620,y:910,display:`intermediate`,note:`Finish Hook`},{order:12,x:500,y:150,display:`keystone`,newStroke:!0,note:`The Dot (Top X)`}],H=t({default:()=>U}),U=[{order:1,x:550,y:280,display:`keystone`,note:`Start Black Dot`},{order:2,x:550,y:380,display:`intermediate`,note:`Stem Down`},{order:3,x:550,y:480,display:`intermediate`,note:`Stem Down`},{order:4,x:550,y:580,display:`keystone`,note:`Stem Middle`},{order:5,x:550,y:680,display:`intermediate`,note:`Stem Down`},{order:6,x:550,y:760,display:`intermediate`,note:`Start Turn`},{order:7,x:530,y:840,display:`keystone`,note:`Curve Left`},{order:8,x:490,y:900,display:`intermediate`,note:`Curve Bottom Right`},{order:9,x:430,y:930,display:`intermediate`,note:`Curve Bottom Center`},{order:10,x:360,y:920,display:`keystone`,note:`Curve Bottom Left`},{order:11,x:310,y:890,display:`intermediate`,note:`Arrow Approach`},{order:12,x:280,y:870,display:`intermediate`,note:`Finish Arrow Tip`},{order:13,x:550,y:140,display:`keystone`,newStroke:!0,note:`The Dot (Top X)`}],ie=t({default:()=>W}),W=[{order:1,x:220,y:180,display:`keystone`,note:`Start Black Dot (Top of Stem)`},{order:2,x:220,y:360,display:`intermediate`,note:`Stem Down`},{order:3,x:220,y:540,display:`intermediate`,note:`Stem Middle`},{order:4,x:220,y:720,display:`keystone`,note:`Stem Lower`},{order:5,x:220,y:900,display:`intermediate`,note:`Stem Finish Arrow`},{order:6,x:680,y:380,display:`keystone`,newStroke:!0,note:`Start 'X' (Top Right)`},{order:7,x:550,y:460,display:`intermediate`,note:`Diagonal In`},{order:8,x:420,y:540,display:`intermediate`,note:`Diagonal In`},{order:9,x:300,y:620,display:`keystone`,note:`The Turn (Touch Stem)`},{order:10,x:410,y:690,display:`intermediate`,note:`Diagonal Out`},{order:11,x:520,y:760,display:`intermediate`,note:`Diagonal Out`},{order:12,x:630,y:830,display:`intermediate`,note:`Diagonal Out`},{order:13,x:740,y:900,display:`keystone`,note:`Leg Finish Arrow`}],ae=t({default:()=>oe}),oe=[{order:1,x:180,y:200,display:`keystone`,note:`Start Black Dot (Top Left)`},{order:2,x:180,y:340,display:`intermediate`,note:`Left Stem Down`},{order:3,x:180,y:480,display:`intermediate`,note:`Left Stem Middle`},{order:4,x:180,y:620,display:`keystone`,note:`Left Stem Lower`},{order:5,x:180,y:760,display:`intermediate`,note:`Left Stem Down`},{order:6,x:180,y:900,display:`intermediate`,note:`Left Stem Finish Arrow`},{order:7,x:180,y:380,display:`keystone`,newStroke:!0,note:`Branch Point 1 (Start Middle Arch)`},{order:8,x:240,y:260,display:`intermediate`,note:`Curve Up-Right`},{order:9,x:340,y:210,display:`intermediate`,note:`Arch 1 Top`},{order:10,x:440,y:260,display:`keystone`,note:`Arch 1 Down-Right`},{order:11,x:500,y:380,display:`intermediate`,note:`Middle Stem Start`},{order:12,x:500,y:550,display:`intermediate`,note:`Middle Stem Down`},{order:13,x:500,y:720,display:`keystone`,note:`Middle Stem Lower`},{order:14,x:500,y:850,display:`intermediate`,note:`Middle Stem Approach`},{order:15,x:500,y:900,display:`intermediate`,note:`Middle Stem Finish Arrow`},{order:16,x:500,y:380,display:`keystone`,newStroke:!0,note:`Branch Point 2 (Start Right Arch)`},{order:17,x:560,y:260,display:`intermediate`,note:`Curve Up-Right`},{order:18,x:660,y:210,display:`intermediate`,note:`Arch 2 Top`},{order:19,x:760,y:260,display:`keystone`,note:`Arch 2 Down-Right`},{order:20,x:820,y:380,display:`intermediate`,note:`Right Stem Start`},{order:21,x:820,y:510,display:`intermediate`,note:`Right Stem Down`},{order:22,x:820,y:640,display:`keystone`,note:`Right Stem Middle`},{order:23,x:820,y:770,display:`intermediate`,note:`Right Stem Lower`},{order:24,x:820,y:850,display:`intermediate`,note:`Right Stem Approach`},{order:25,x:820,y:890,display:`keystone`,note:`Right Stem Arrow Body`},{order:26,x:820,y:910,display:`intermediate`,note:`Right Stem Finish`}],se=t({default:()=>ce}),ce=[{order:1,x:250,y:200,display:`keystone`,note:`Start Black Dot (Top Left)`},{order:2,x:250,y:340,display:`intermediate`,note:`Left Stem Down`},{order:3,x:250,y:480,display:`intermediate`,note:`Left Stem Middle`},{order:4,x:250,y:620,display:`keystone`,note:`Left Stem Lower`},{order:5,x:250,y:760,display:`intermediate`,note:`Left Stem Down`},{order:6,x:250,y:900,display:`intermediate`,note:`Left Stem Finish Arrow`},{order:7,x:250,y:450,display:`keystone`,newStroke:!0,note:`Branch Point (Start Arch)`},{order:8,x:300,y:320,display:`intermediate`,note:`Curve Up-Right`},{order:9,x:400,y:240,display:`intermediate`,note:`Arch Top Left`},{order:10,x:500,y:210,display:`keystone`,note:`Arch Top Center`},{order:11,x:600,y:240,display:`intermediate`,note:`Arch Top Right`},{order:12,x:700,y:320,display:`intermediate`,note:`Curve Down-Right`},{order:13,x:750,y:450,display:`keystone`,note:`Right Stem Start`},{order:14,x:750,y:600,display:`intermediate`,note:`Right Stem Down`},{order:15,x:750,y:750,display:`intermediate`,note:`Right Stem Approach`},{order:16,x:750,y:900,display:`keystone`,note:`Right Stem Finish Arrow`}],le=t({default:()=>ue}),ue=[{order:1,x:740,y:260,display:`keystone`,note:`Start Black Dot`},{order:2,x:670,y:200,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:590,y:160,display:`intermediate`,note:`Curve Up-Left`},{order:4,x:500,y:150,display:`keystone`,note:`Top Center`},{order:5,x:410,y:160,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:330,y:200,display:`intermediate`,note:`Curve Down-Left`},{order:7,x:260,y:260,display:`keystone`,note:`Top Left Curve`},{order:8,x:200,y:330,display:`intermediate`,note:`Left Side High`},{order:9,x:160,y:410,display:`intermediate`,note:`Left Side Mid`},{order:10,x:150,y:500,display:`keystone`,note:`Left Center`},{order:11,x:160,y:590,display:`intermediate`,note:`Left Side Low`},{order:12,x:200,y:670,display:`intermediate`,note:`Bottom Left Turn`},{order:13,x:260,y:740,display:`keystone`,note:`Bottom Left Curve`},{order:14,x:330,y:800,display:`intermediate`,note:`Curve Down-Right`},{order:15,x:410,y:840,display:`intermediate`,note:`Bottom Curve`},{order:16,x:500,y:850,display:`keystone`,note:`Bottom Center`},{order:17,x:590,y:840,display:`intermediate`,note:`Bottom Curve Right`},{order:18,x:670,y:800,display:`intermediate`,note:`Curve Up-Right`},{order:19,x:740,y:740,display:`keystone`,note:`Bottom Right Curve`},{order:20,x:800,y:670,display:`intermediate`,note:`Right Side Low`},{order:21,x:840,y:590,display:`intermediate`,note:`Right Side Mid`},{order:22,x:850,y:500,display:`keystone`,note:`Right Center`},{order:23,x:840,y:410,display:`intermediate`,note:`Right Side High`},{order:24,x:800,y:330,display:`intermediate`,note:`Approach Top`},{order:25,x:760,y:290,display:`keystone`,note:`Close Loop`},{order:26,x:730,y:280,display:`intermediate`,note:`Finish Arrow Tip`}],de=t({default:()=>fe}),fe=[{order:1,x:230,y:160,display:`keystone`,note:`Start Black Dot (Top Left)`},{order:2,x:230,y:260,display:`intermediate`,note:`Stem Down`},{order:3,x:230,y:360,display:`intermediate`,note:`Stem Down`},{order:4,x:230,y:460,display:`keystone`,note:`Stem Middle`},{order:5,x:230,y:560,display:`intermediate`,note:`Stem Down`},{order:6,x:230,y:660,display:`intermediate`,note:`Stem Down`},{order:7,x:230,y:760,display:`keystone`,note:`Stem Lower`},{order:8,x:230,y:880,display:`intermediate`,note:`Stem Finish Arrow`},{order:9,x:230,y:340,display:`keystone`,newStroke:!0,note:`Branch Point (Start Lobe)`},{order:10,x:280,y:240,display:`intermediate`,note:`Curve Up-Right`},{order:11,x:380,y:180,display:`intermediate`,note:`Top Curve Left`},{order:12,x:500,y:150,display:`keystone`,note:`Lobe Top Center`},{order:13,x:620,y:160,display:`intermediate`,note:`Top Curve Right`},{order:14,x:720,y:220,display:`intermediate`,note:`Curve Down-Right`},{order:15,x:800,y:320,display:`keystone`,note:`Right Side High`},{order:16,x:840,y:420,display:`intermediate`,note:`Right Side Middle`},{order:17,x:840,y:520,display:`intermediate`,note:`Right Side Low`},{order:18,x:800,y:620,display:`keystone`,note:`Bottom Right Curve`},{order:19,x:720,y:700,display:`intermediate`,note:`Curve Left-Down`},{order:20,x:620,y:740,display:`intermediate`,note:`Bottom Curve Right`},{order:21,x:500,y:750,display:`keystone`,note:`Lobe Bottom Center`},{order:22,x:380,y:740,display:`intermediate`,note:`Bottom Curve Left`},{order:23,x:320,y:710,display:`intermediate`,note:`Approach Stem`},{order:24,x:280,y:680,display:`keystone`,note:`Close Lobe Arrow`},{order:25,x:250,y:660,display:`intermediate`,note:`Connect`},{order:26,x:230,y:640,display:`intermediate`,note:`Join Stem`}],pe={"./assets/letters_json/a.json":D,"./assets/letters_json/b.json":k,"./assets/letters_json/c.json":j,"./assets/letters_json/d.json":N,"./assets/letters_json/e.json":F,"./assets/letters_json/f.json":I,"./assets/letters_json/g.json":R,"./assets/letters_json/h.json":B,"./assets/letters_json/i.json":ne,"./assets/letters_json/j.json":H,"./assets/letters_json/k.json":ie,"./assets/letters_json/m.json":ae,"./assets/letters_json/n.json":se,"./assets/letters_json/o.json":le,"./assets/letters_json/p.json":de},G=Object.entries(pe).map(([e,t])=>{let n=t.default;return Array.isArray(n)?{name:(e.split(`/`).pop()??`unknown`).replace(/\.json$/i,``),points:n}:null}).filter(e=>e!==null),me=()=>{if(G.length===0)return{name:`unknown`,points:[]};let e=G[Math.floor(Math.random()*G.length)];return console.info(`Selected letter:`,e.name),{name:e.name,points:e.points.map(e=>({order:e.order,x:e.x,y:e.y})).sort((e,t)=>e.order-t.order)}},he=e=>{if(e.length===0)return{minX:0,maxX:0,minY:0,maxY:0};let t=e[0].x,n=e[0].x,r=e[0].y,i=e[0].y;for(let a of e)a.x<t&&(t=a.x),a.x>n&&(n=a.x),a.y<r&&(r=a.y),a.y>i&&(i=a.y);return{minX:t,maxX:n,minY:r,maxY:i}},ge=3.5,_e=e=>{let t=[...e].sort((e,t)=>e-t),n=Math.floor(t.length/2);return t.length%2==0?(t[n-1]+t[n])/2:t[n]},ve=e=>{if(e.length<2)return 1/0;let t=[];for(let n=0;n<e.length-1;n+=1){let r=e[n+1].x-e[n].x,i=e[n+1].y-e[n].y;t.push(Math.hypot(r,i))}let n=t.filter(e=>e>0),r=n.length>0?n:t;return r.length===0?1/0:_e(r)*ge},K=(e,t)=>{let n=[];if(e.length<2)return n;for(let r=0;r<e.length-1;r+=1){let i=e[r+1].x-e[r].x,a=e[r+1].y-e[r].y;Math.hypot(i,a)>t||n.push({from:r,to:r+1})}return n},ye=class{plotPoints=[];paths=[];currentPath=null;fadeSeconds=9;maxPoints=600;isDrawing=!1;currentMousePos={x:0,y:0};plotBounds={minX:0,maxX:0,minY:0,maxY:0};scaledPlotPoints=[];lastPlotSize={width:0,height:0};needsRescale=!0;currentTargetIndex=0;dotRadius=20;hitRadiusScale=1.5;penUpDistanceThreshold=1/0;lineSegmentIndex=0;lineSegmentT=0;linePauseRemaining=0;lineSegmentSeconds=.123;lineLoopPauseSeconds=.4;completionMessageUntil=0;pendingLetterReset=!1;completionMessageSeconds=2;callbacks;currentLetterName=`unknown`;constructor(e,t={}){let n=me();this.plotPoints=n.points,this.plotBounds=he(this.plotPoints),this.callbacks=t,this.currentLetterName=n.name,this.callbacks.onLetterChange?.(this.currentLetterName),e.setCallbacks({onStart:e=>this.startPath(e),onMove:e=>this.extendPath(e),onEnd:e=>this.endPath(e)})}getDrawingState(){let e=this.paths.flatMap(e=>e);return{isDrawing:this.isDrawing,mouseX:this.currentMousePos.x,mouseY:this.currentMousePos.y,trailPoints:e}}getDotState(){return{dots:this.scaledPlotPoints,currentIndex:this.currentTargetIndex,radius:this.dotRadius}}clearUserPaths(){this.paths=[],this.currentPath=null,this.isDrawing=!1}resetCurrentLetter(){this.clearUserPaths(),this.currentTargetIndex=0,this.lineSegmentIndex=0,this.lineSegmentT=0,this.linePauseRemaining=0,this.pendingLetterReset=!1,this.completionMessageUntil=0}setViewportSize(e,t){if(this.plotPoints.length===0)return;let n=Math.max(1,e),r=Math.max(1,t);if(n===this.lastPlotSize.width&&r===this.lastPlotSize.height&&!this.needsRescale)return;this.lastPlotSize={width:n,height:r},this.needsRescale=!1;let i=this.plotBounds.maxX-this.plotBounds.minX,a=this.plotBounds.maxY-this.plotBounds.minY;if(i<=0||a<=0)return;let o=n*.5,s=r*.5,c=Math.min(o/i,s/a),l=(n-i*c)/2-this.plotBounds.minX*c,u=(r-a*c)/2-this.plotBounds.minY*c;this.scaledPlotPoints=this.plotPoints.map(e=>({order:e.order,x:e.x*c+l,y:e.y*c+u})),this.penUpDistanceThreshold=ve(this.scaledPlotPoints)}update(e){this.advancePlotAnimation(e);let t=performance.now()/1e3;if(this.paths.length>0){let e=t-this.fadeSeconds;for(let t of this.paths){let n=0;for(;n<t.length&&t[n].time<e;)n+=1;n>0&&t.splice(0,n),t.length>this.maxPoints&&t.splice(0,t.length-this.maxPoints)}this.paths=this.paths.filter(e=>e.length>0)}this.pendingLetterReset&&t>=this.completionMessageUntil&&(this.resetForNewLetter(),this.pendingLetterReset=!1,this.completionMessageUntil=0)}render(e){if(e.save(),e.setTransform(1,0,0,1,0,0),e.clearRect(0,0,e.canvas.width,e.canvas.height),e.restore(),this.renderPlotLines(e),this.paths.length>0){e.strokeStyle=`rgba(200, 230, 255, 0.9)`,e.lineWidth=15,e.lineJoin=`round`,e.lineCap=`round`;for(let t of this.paths)t.length!==0&&(e.beginPath(),t.forEach((t,n)=>{n===0?e.moveTo(t.x,t.y):e.lineTo(t.x,t.y)}),e.stroke())}this.renderCompletionMessage(e)}startPath(e){this.pendingLetterReset||(this.isDrawing=!0,this.currentMousePos={x:e.x,y:e.y},this.currentPath=[{x:e.x,y:e.y,time:e.time}],this.paths.push(this.currentPath),this.tryAdvanceTarget(e))}extendPath(e){this.pendingLetterReset||(this.currentMousePos={x:e.x,y:e.y},this.currentPath&&(this.currentPath.push({x:e.x,y:e.y,time:e.time}),this.tryAdvanceTarget(e)))}endPath(e){this.pendingLetterReset||(this.currentMousePos={x:e.x,y:e.y},this.currentPath&&this.currentPath.push({x:e.x,y:e.y,time:e.time}),this.tryAdvanceTarget(e),this.isDrawing=!1,this.currentPath=null)}renderPlotLines(e){if(this.scaledPlotPoints.length<2)return;e.strokeStyle=`rgba(255, 220, 0, 0.5)`,e.lineWidth=9,e.lineJoin=`round`,e.lineCap=`round`,e.beginPath();let t=Math.min(Math.max(0,this.currentTargetIndex-1),this.scaledPlotPoints.length-1),n=this.scaledPlotPoints.slice(t),r=K(n,this.penUpDistanceThreshold);if(r.length!==0){if(this.linePauseRemaining>0)this.drawLineSegments(e,n,r,r.length-1);else{let t=Math.min(this.lineSegmentIndex,r.length-1);this.drawLineSegments(e,n,r,t-1),this.drawPartialLineSegment(e,n,r,t,this.lineSegmentT)}e.stroke()}}drawLineSegments(e,t,n,r){if(r<0)return;let i=Math.min(r,n.length-1),a=null;for(let r=0;r<=i;r+=1){let i=n[r],o=t[i.from],s=t[i.to];(!a||a.to!==i.from)&&e.moveTo(o.x,o.y),e.lineTo(s.x,s.y),a=i}}drawPartialLineSegment(e,t,n,r,i){if(r<0||r>=n.length)return;let a=n[r],o=t[a.from],s=t[a.to],c=r>0?n[r-1]:null;(!c||c.to!==a.from)&&e.moveTo(o.x,o.y);let l=o.x+(s.x-o.x)*i,u=o.y+(s.y-o.y)*i;e.lineTo(l,u)}tryAdvanceTarget(e){if(!this.isDrawing||this.scaledPlotPoints.length===0||this.currentTargetIndex>=this.scaledPlotPoints.length)return;let t=this.scaledPlotPoints[this.currentTargetIndex],n=e.x-t.x,r=e.y-t.y,i=this.getTargetDotRadius()*this.hitRadiusScale;n*n+r*r<=i*i&&(this.currentTargetIndex=Math.min(this.currentTargetIndex+1,this.scaledPlotPoints.length),this.callbacks.onPointHit?.(),this.lineSegmentIndex=0,this.lineSegmentT=0,this.linePauseRemaining=0,this.currentTargetIndex>=this.scaledPlotPoints.length&&this.handleLetterComplete())}getTargetDotRadius(){return Math.max(10,this.dotRadius*1.3)*1.45}advancePlotAnimation(e){if(e<=0||this.scaledPlotPoints.length<2)return;let t=Math.min(Math.max(0,this.currentTargetIndex-1),this.scaledPlotPoints.length-1),n=K(this.scaledPlotPoints.slice(t),this.penUpDistanceThreshold).length;if(n!==0){if(this.lineSegmentIndex=Math.min(this.lineSegmentIndex,n-1),this.linePauseRemaining>0){if(this.linePauseRemaining=Math.max(0,this.linePauseRemaining-e),this.linePauseRemaining>0)return;this.lineSegmentIndex=0,this.lineSegmentT=0}for(this.lineSegmentT+=e/this.lineSegmentSeconds;this.lineSegmentT>=1;)if(--this.lineSegmentT,this.lineSegmentIndex+=1,this.lineSegmentIndex>=n){this.linePauseRemaining=this.lineLoopPauseSeconds,this.lineSegmentIndex=0,this.lineSegmentT=0;break}}}handleLetterComplete(){this.pendingLetterReset||(this.completionMessageUntil=performance.now()/1e3+this.completionMessageSeconds,this.pendingLetterReset=!0,this.isDrawing=!1,this.currentPath=null)}resetForNewLetter(){let e=me();this.plotPoints=e.points,this.plotBounds=he(this.plotPoints),this.currentLetterName=e.name,this.callbacks.onLetterChange?.(this.currentLetterName),this.scaledPlotPoints=[],this.currentTargetIndex=0,this.lineSegmentIndex=0,this.lineSegmentT=0,this.linePauseRemaining=0,this.paths=[],this.currentPath=null,this.isDrawing=!1,this.needsRescale=!0,this.lastPlotSize.width>0&&this.lastPlotSize.height>0&&this.setViewportSize(this.lastPlotSize.width,this.lastPlotSize.height)}renderCompletionMessage(e){if(performance.now()/1e3>=this.completionMessageUntil)return;let t=e.canvas.width/(window.devicePixelRatio||1),n=e.canvas.height/(window.devicePixelRatio||1);e.save(),e.textAlign=`center`,e.textBaseline=`middle`,e.fillStyle=`rgba(255, 244, 210, 0.95)`,e.strokeStyle=`rgba(0, 0, 0, 0.45)`,e.lineWidth=6,e.font=`700 48px system-ui, -apple-system, sans-serif`;let r=`Well done!`;e.strokeText(r,t/2,n*.45),e.fillText(r,t/2,n*.45),e.restore()}},be=`/letter_constellations/assets/accordion-CyoQk0r2.webp`,xe=`/letter_constellations/assets/acorn-B-P3IZra.webp`,Se=`/letter_constellations/assets/airplane-DZQqSBvK.webp`,Ce=`/letter_constellations/assets/alligator-Ce5bwrBv.webp`,we=`/letter_constellations/assets/anchor-BV-M7FTg.webp`,Te=`/letter_constellations/assets/ant-CvU0-Xki.webp`,Ee=`/letter_constellations/assets/apple-DE3ezPUv.webp`,De=`/letter_constellations/assets/arrow-C4cWjJLo.webp`,Oe=`/letter_constellations/assets/astronaut-uRVMZixm.webp`,ke=`/letter_constellations/assets/bananas-VvvO5iFW.webp`,Ae=`/letter_constellations/assets/beachball-BbGYNBZh.webp`,je=`/letter_constellations/assets/bear-CvctF6Z1.webp`,Me=`/letter_constellations/assets/bee-BCPxoZBd.webp`,Ne=`/letter_constellations/assets/bird-CPHJbX7S.webp`,Pe=`/letter_constellations/assets/boat-5S1Aura4.webp`,Fe=`/letter_constellations/assets/books-DzB7841Q.webp`,Ie=`/letter_constellations/assets/bus-Cr08LBdp.webp`,Le=`/letter_constellations/assets/butterfly-DJ37Qz2P.webp`,Re=`/letter_constellations/assets/cake-C6PvIUA0.webp`,ze=`/letter_constellations/assets/car-BLjHiQLL.webp`,Be=`/letter_constellations/assets/castle-kQcpwewP.webp`,Ve=`/letter_constellations/assets/cat-_X2VtWwN.webp`,He=`/letter_constellations/assets/clock-BqkzNyNk.webp`,Ue=`/letter_constellations/assets/coat-OQ0hqgvS.webp`,We=`/letter_constellations/assets/cow-BU0iB5gA.webp`,Ge=`/letter_constellations/assets/crayon-DT3SkfbL.webp`,Ke=`/letter_constellations/assets/cup-B6QWjkyx.webp`,qe=`/letter_constellations/assets/dog-DC0ZhRtT.webp`,Je=`/letter_constellations/assets/doll-DAHtNuVs.webp`,Ye=`/letter_constellations/assets/dolphin-C2ILYpRK.webp`,Xe=`/letter_constellations/assets/donkey-BMyke4TY.webp`,Ze=`/letter_constellations/assets/door-CVCIUMCt.webp`,Qe=`/letter_constellations/assets/dragon-BAjlAFRA.webp`,$e=`/letter_constellations/assets/dress-DvrNR5gf.webp`,et=`/letter_constellations/assets/drum-CwxRV3Cf.webp`,tt=`/letter_constellations/assets/duck-U27dc_u8.webp`,nt=`/letter_constellations/assets/eagle-DiKLg3yz.webp`,rt=`/letter_constellations/assets/ear-BgYxZnRS.webp`,it=`/letter_constellations/assets/earth-r6cd7hOH.webp`,at=`/letter_constellations/assets/easel-EvkDAxpG.webp`,ot=`/letter_constellations/assets/egg-Dm6XQc4n.webp`,st=`/letter_constellations/assets/elephant-CIvbMIFa.webp`,ct=`/letter_constellations/assets/envelope-DU_Wnco4.webp`,lt=`/letter_constellations/assets/eraser-MC2wEqI8.webp`,ut=`/letter_constellations/assets/feather-DOx_OLgY.webp`,dt=`/letter_constellations/assets/fence-CwF42dkY.webp`,ft=`/letter_constellations/assets/fish-DyOqSklD.webp`,pt=`/letter_constellations/assets/flower-B__p0fXU.webp`,mt=`/letter_constellations/assets/foot-BXSJM_nF.webp`,ht=`/letter_constellations/assets/fork-yucdu-VG.webp`,gt=`/letter_constellations/assets/fox-BE5prvLj.webp`,_t=`/letter_constellations/assets/frog-Dh1keBjF.webp`,vt=`/letter_constellations/assets/garden-BXEz5faC.webp`,yt=`/letter_constellations/assets/ghost-C3-5NTro.webp`,bt=`/letter_constellations/assets/giraffe-BN8FpAKY.webp`,xt=`/letter_constellations/assets/gloves-BJzKgdkI.webp`,St=`/letter_constellations/assets/goat-Cp1Bn2Pw.webp`,Ct=`/letter_constellations/assets/goldfish-C02QaEXu.webp`,wt=`/letter_constellations/assets/grapes-DhMMMjx_.webp`,Tt=`/letter_constellations/assets/guitar-BhL2ZgvO.webp`,Et=`/letter_constellations/assets/hammer-CfkG7qG8.webp`,Dt=`/letter_constellations/assets/hand-C1darVYo.webp`,Ot=`/letter_constellations/assets/hat-5HvhUkFa.webp`,kt=`/letter_constellations/assets/heart-qdzpJsno.webp`,At=`/letter_constellations/assets/hedgehog-BaJaSoQw.webp`,jt=`/letter_constellations/assets/helicopter-DC0W3QSo.webp`,Mt=`/letter_constellations/assets/horse-DfgajmWZ.webp`,Nt=`/letter_constellations/assets/hotdog-DTQ4uHOu.webp`,Pt=`/letter_constellations/assets/house-BZe7-xWI.webp`,Ft=`/letter_constellations/assets/ice_cream-DT_mc0_P.webp`,It=`/letter_constellations/assets/igloo-CC6ArNRt.webp`,Lt=`/letter_constellations/assets/iguana-BXYH3WVB.webp`,Rt=`/letter_constellations/assets/inkpot-DnxrzNHx.webp`,zt=`/letter_constellations/assets/instrument-1utLT-6T.webp`,Bt=`/letter_constellations/assets/invitation-z-PWuchC.webp`,Vt=`/letter_constellations/assets/iron-CtDYRgb4.webp`,Ht=`/letter_constellations/assets/island-B8Bp5Tsi.webp`,Ut=`/letter_constellations/assets/jaguar-DHXPvMuk.webp`,Wt=`/letter_constellations/assets/jar-Dwv58Vzr.webp`,Gt=`/letter_constellations/assets/jean_jacket-BULMDhzs.webp`,Kt=`/letter_constellations/assets/jelly-CVVY10k9.webp`,qt=`/letter_constellations/assets/jellyfish-BC5AuiGF.webp`,Jt=`/letter_constellations/assets/jigsaw-BOvQFAyk.webp`,Yt=`/letter_constellations/assets/juggler-GzS6Gfiw.webp`,Xt=`/letter_constellations/assets/juice-BqR4izBV.webp`,Zt=`/letter_constellations/assets/kangaroo-BxEtKT1L.webp`,Qt=`/letter_constellations/assets/key-C62Kti4m.webp`,$t=`/letter_constellations/assets/kid-CjHrY2fb.webp`,en=`/letter_constellations/assets/kite-BB0XKGyC.webp`,tn=`/letter_constellations/assets/kiwi-B0KGm69A.webp`,nn=`/letter_constellations/assets/koala-CjVCRMJZ.webp`,rn=`/letter_constellations/assets/ladybug-CpSwAR2D.webp`,an=`/letter_constellations/assets/lamp-DUz1kO0o.webp`,on=`/letter_constellations/assets/lemon-Dd__w2Pq.webp`,sn=`/letter_constellations/assets/lemur-DHKaat7q.webp`,cn=`/letter_constellations/assets/lettuce-D8mc2q-U.webp`,ln=`/letter_constellations/assets/library-DKV5tmFO.webp`,un=`/letter_constellations/assets/llama-Bwh9dRAG.webp`,dn=`/letter_constellations/assets/map-vQmueQII.webp`,fn=`/letter_constellations/assets/milk_carton-5cuoXE3T.webp`,pn=`/letter_constellations/assets/mitten-BeKduZUD.webp`,mn=`/letter_constellations/assets/monkey-Cx-TqYqO.webp`,hn=`/letter_constellations/assets/moon-jSXzUMnz.webp`,gn=`/letter_constellations/assets/mountain-CZsGBOtO.webp`,_n=`/letter_constellations/assets/mouse-CmzFJelu.webp`,vn=`/letter_constellations/assets/muffin-BhJF_GvY.webp`,yn=`/letter_constellations/assets/mushroom-DHqipsMi.webp`,bn=`/letter_constellations/assets/nail-2NdCu6nX.webp`,xn=`/letter_constellations/assets/needle-WCtT707e.webp`,Sn=`/letter_constellations/assets/nest-Bs_bSd3C.webp`,Cn=`/letter_constellations/assets/net-CcQz3K1r.webp`,wn=`/letter_constellations/assets/newt-DMKiaP-a.webp`,Tn=`/letter_constellations/assets/night-loJiPRO9.webp`,En=`/letter_constellations/assets/nose-BGs_8Ldz.webp`,Dn=`/letter_constellations/assets/nurse-DnB4a9sK.webp`,On=`/letter_constellations/assets/nuts-dWD5IlYg.webp`,kn=`/letter_constellations/assets/oar-BJP2E8X6.webp`,An=`/letter_constellations/assets/octopus-Bjrhg0Kz.webp`,jn=`/letter_constellations/assets/onion-D-sGtoXB.webp`,Mn=`/letter_constellations/assets/orange-DUVPpQC1.webp`,Nn=`/letter_constellations/assets/orangutan-GXcPQi2e.webp`,Pn=`/letter_constellations/assets/ostrich-CbDM_d-O.webp`,Fn=`/letter_constellations/assets/owl-Du9V2IZL.webp`,In=`/letter_constellations/assets/panda-Ce4wJnci.webp`,Ln=`/letter_constellations/assets/pen-XTsE5Tp9.webp`,Rn=`/letter_constellations/assets/pencil-CiXeWLpi.webp`,zn=`/letter_constellations/assets/penguin-DQXXzk60.webp`,Bn=`/letter_constellations/assets/pig-f6PyeKkO.webp`,Vn=`/letter_constellations/assets/pineapple-DEyw_76-.webp`,Hn=`/letter_constellations/assets/pirate-CR8dih_s.webp`,Un=`/letter_constellations/assets/pizza-BHofEroO.webp`,Wn=`/letter_constellations/assets/pumpkin-BK91jTFE.webp`,Gn=`/letter_constellations/assets/quail-BPC2-Z_g.webp`,Kn=`/letter_constellations/assets/queen-fiZYgCAh.webp`,qn=`/letter_constellations/assets/question-WkEi71Wr.webp`,Jn=`/letter_constellations/assets/quill-KEF9k-Kr.webp`,Yn=`/letter_constellations/assets/quilt-B-SSp3IP.webp`,Xn=`/letter_constellations/assets/quince-CGOlv5XI.webp`,Zn=`/letter_constellations/assets/quiver-ClNEjwQB.webp`,Qn=`/letter_constellations/assets/rabbit-DZWfw0Tv.webp`,$n=`/letter_constellations/assets/radishes-B0KgqPIP.webp`,er=`/letter_constellations/assets/rainbow-CsvdimZC.webp`,tr=`/letter_constellations/assets/ring-xmCTdpMO.webp`,nr=`/letter_constellations/assets/robin-X5gA9bFr.webp`,rr=`/letter_constellations/assets/robot-ozHBRcKM.webp`,ir=`/letter_constellations/assets/rocket-CitKkJev.webp`,ar=`/letter_constellations/assets/rose-DL-Oy0Ek.webp`,or=`/letter_constellations/assets/ruler-C1DPZ1Nf.webp`,sr=`/letter_constellations/assets/sailboat-BlcB2Z3i.webp`,cr=`/letter_constellations/assets/sheep-C-5w7NTE.webp`,lr=`/letter_constellations/assets/snake-CbOrYwEs.webp`,ur=`/letter_constellations/assets/snowman-DjpZB2fz.webp`,dr=`/letter_constellations/assets/socks-Dp64Nj0U.webp`,fr=`/letter_constellations/assets/star-CjyC3Ct8.webp`,pr=`/letter_constellations/assets/strawberry-ftyABBXF.webp`,mr=`/letter_constellations/assets/sun-BSCyCAh2.webp`,hr=`/letter_constellations/assets/table-BNMmdjGK.webp`,gr=`/letter_constellations/assets/teapot-D56nDidw.webp`,_r=`/letter_constellations/assets/tiger-afEfjDX9.webp`,vr=`/letter_constellations/assets/tomato-BFAkscRK.webp`,yr=`/letter_constellations/assets/tractor-tug0JOHL.webp`,br=`/letter_constellations/assets/train-NYNpzrwW.webp`,xr=`/letter_constellations/assets/tree-BlYlAreg.webp`,Sr=`/letter_constellations/assets/turtle-Cj9D3d81.webp`,Cr=`/letter_constellations/assets/ukulele-DCuOtxfK.webp`,wr=`/letter_constellations/assets/umbrella-Dz1MlrtA.webp`,Tr=`/letter_constellations/assets/unicorn-CriH96qk.webp`,Er=`/letter_constellations/assets/unicycle-gQror61r.webp`,Dr=`/letter_constellations/assets/unidentified_object-CUDu-AOC.webp`,Or=`/letter_constellations/assets/urchin-_JmHjkoD.webp`,kr=`/letter_constellations/assets/utensils-DFAZ2efj.webp`,Ar=`/letter_constellations/assets/vacuum-BLJAk_jE.webp`,jr=`/letter_constellations/assets/van-Dpdv5-Hp.webp`,Mr=`/letter_constellations/assets/vase-CpSUtdT6.webp`,Nr=`/letter_constellations/assets/vegetables-ChjTiEM-.webp`,Pr=`/letter_constellations/assets/vest-D3gDumUU.webp`,Fr=`/letter_constellations/assets/village-NwxI-P5i.webp`,Ir=`/letter_constellations/assets/violin-Bwy5HY6e.webp`,Lr=`/letter_constellations/assets/volcano-DCRqkKRz.webp`,Rr=`/letter_constellations/assets/vulture-C-ErfnZu.webp`,zr=`/letter_constellations/assets/wagon-DW5tu-i6.webp`,Br=`/letter_constellations/assets/watch-ahI5puCQ.webp`,Vr=`/letter_constellations/assets/watermelon-DOsensRc.webp`,Hr=`/letter_constellations/assets/web-Ce6CPeZ_.webp`,Ur=`/letter_constellations/assets/whale-DAEVlAKS.webp`,Wr=`/letter_constellations/assets/wind-B6DREdK4.webp`,Gr=`/letter_constellations/assets/wizard-C8Dnr-1B.webp`,Kr=`/letter_constellations/assets/wolf-aZD8PyPP.webp`,qr=`/letter_constellations/assets/worm-Ds6LH9wX.webp`,Jr=`/letter_constellations/assets/accordion-DqQtriLm.webm`,Yr=`/letter_constellations/assets/acorn-BIezBOPs.webm`,Xr=`/letter_constellations/assets/airplane-C_Wfm5g-.webm`,Zr=`/letter_constellations/assets/alligator-DYepK-g8.webm`,Qr=`/letter_constellations/assets/anchor-D_H3HrJm.webm`,$r=`/letter_constellations/assets/ant-D6QhcluX.webm`,ei=`/letter_constellations/assets/apple-CZDvf2t3.webm`,ti=`/letter_constellations/assets/arrow-Cs627Rik.webm`,ni=`/letter_constellations/assets/astronaut-B9QACnUz.webm`,q=`/letter_constellations/assets/a-CMkzIMFv.webm`,J=`/letter_constellations/assets/b-C2kyMVsg.webm`,ri=`/letter_constellations/assets/c-IjVD-Vre.webm`,ii=`/letter_constellations/assets/d-9vk5XyCw.webm`,ai=`/letter_constellations/assets/e-VRhdBt2g.webm`,oi=`/letter_constellations/assets/f-BVhXOumA.webm`,si=`/letter_constellations/assets/g-DIYEe5HW.webm`,ci=`/letter_constellations/assets/h-DyPaSF0e.webm`,li=`/letter_constellations/assets/i-C5tqm2EV.webm`,ui=`/letter_constellations/assets/j-Be2zckp_.webm`,di=`/letter_constellations/assets/k-BakCspA1.webm`,fi=`/letter_constellations/assets/l-NUuunizq.webm`,pi=`/letter_constellations/assets/m-BB98DVPZ.webm`,mi=`/letter_constellations/assets/n-CnIX4TYm.webm`,hi=`/letter_constellations/assets/o-BiLMB4YT.webm`,gi=`/letter_constellations/assets/p-C4ciUhaN.webm`,_i=`/letter_constellations/assets/q-DF2D67v_.webm`,vi=`/letter_constellations/assets/r-CS0DW1cp.webm`,yi=`/letter_constellations/assets/s-An_rASvU.webm`,bi=`/letter_constellations/assets/t-CVflKspQ.webm`,xi=`/letter_constellations/assets/u-Dudg3DnT.webm`,Y=`/letter_constellations/assets/v-WZ8s5vPh.webm`,Si=`/letter_constellations/assets/w-BElagC9L.webm`,Ci=`/letter_constellations/assets/x-BCyjQ6tg.webm`,wi=`/letter_constellations/assets/y-CxUp37AX.webm`,Ti=`/letter_constellations/assets/z-CmvHDWPa.webm`,Ei=`/letter_constellations/assets/bananas-h3WNqzRq.webm`,Di=`/letter_constellations/assets/beachball-CHdPyc90.webm`,Oi=`/letter_constellations/assets/bear-5zJMp45m.webm`,ki=`/letter_constellations/assets/bee-CIa4b6b3.webm`,Ai=`/letter_constellations/assets/bird-C-mCHkrZ.webm`,ji=`/letter_constellations/assets/boat-Egh2j8Fn.webm`,Mi=`/letter_constellations/assets/books-CDA8xLKG.webm`,Ni=`/letter_constellations/assets/bus-BcHciuah.webm`,Pi=`/letter_constellations/assets/butterfly-Cc832W2U.webm`,Fi=`/letter_constellations/assets/cake-CE8GeVD6.webm`,Ii=`/letter_constellations/assets/car-D5SmH24e.webm`,Li=`/letter_constellations/assets/castle-BoljANwC.webm`,Ri=`/letter_constellations/assets/cat-BDJFtO1J.webm`,zi=`/letter_constellations/assets/clock-BdR7qum2.webm`,Bi=`/letter_constellations/assets/coat-16JLlN-R.webm`,Vi=`/letter_constellations/assets/cow-_gfKTNxD.webm`,Hi=`/letter_constellations/assets/crayon-DcBF0plc.webm`,Ui=`/letter_constellations/assets/cup-U1NYHdiC.webm`,Wi=`/letter_constellations/assets/dog-L0-qQmmf.webm`,Gi=`/letter_constellations/assets/doll-DHzmlk22.webm`,Ki=`/letter_constellations/assets/dolphin-DHN3M5dU.webm`,qi=`/letter_constellations/assets/donkey-DZGy0jN1.webm`,Ji=`/letter_constellations/assets/door-D03rYjG6.webm`,Yi=`/letter_constellations/assets/dragon-Bv-oOD4o.webm`,Xi=`/letter_constellations/assets/dress-DioYKYZv.webm`,Zi=`/letter_constellations/assets/drum-Cf0je1UD.webm`,Qi=`/letter_constellations/assets/duck-Bgo1sGLJ.webm`,$i=`/letter_constellations/assets/eagle-BgbhiRSg.webm`,ea=`/letter_constellations/assets/ear-Dl7REk6c.webm`,ta=`/letter_constellations/assets/earth-B5sQyPAK.webm`,na=`/letter_constellations/assets/easel-DrjyCnNQ.webm`,ra=`/letter_constellations/assets/egg-Cp_mjYOa.webm`,ia=`/letter_constellations/assets/elephant-DXDMMvRo.webm`,aa=`/letter_constellations/assets/engine-CGoW8nln.webm`,oa=`/letter_constellations/assets/envelope-CZkB1Qdz.webm`,sa=`/letter_constellations/assets/eraser-CL0SN2d1.webm`,ca=`/letter_constellations/assets/feather-BTNKnrGo.webm`,la=`/letter_constellations/assets/fence-BgF_34t7.webm`,ua=`/letter_constellations/assets/firefly-B7E-EXaq.webm`,da=`/letter_constellations/assets/fish-DsqXYDML.webm`,fa=`/letter_constellations/assets/flower-DzzwNf_U.webm`,pa=`/letter_constellations/assets/foot-CqvpFi2I.webm`,ma=`/letter_constellations/assets/fork-BbXVs51Y.webm`,ha=`/letter_constellations/assets/fox-xbmCA8tV.webm`,ga=`/letter_constellations/assets/frog-ylp_1Suh.webm`,_a=`/letter_constellations/assets/garden-B_iCSqP1.webm`,va=`/letter_constellations/assets/ghost-DCvyj5gL.webm`,ya=`/letter_constellations/assets/gift-Cy86EoFO.webm`,ba=`/letter_constellations/assets/giraffe-CADFBhnh.webm`,xa=`/letter_constellations/assets/gloves-BLFuRMLi.webm`,Sa=`/letter_constellations/assets/goat-BTaP2ceE.webm`,Ca=`/letter_constellations/assets/goldfish-CfKPbugz.webm`,wa=`/letter_constellations/assets/grapes-CvsbRxCI.webm`,Ta=`/letter_constellations/assets/guitar-DdMBQt2m.webm`,Ea=`/letter_constellations/assets/hammer-Ceh7AMku.webm`,Da=`/letter_constellations/assets/hand-KB-VfcaO.webm`,Oa=`/letter_constellations/assets/hat-BiSruZqk.webm`,ka=`/letter_constellations/assets/heart-CgLg_Xw1.webm`,Aa=`/letter_constellations/assets/hedgehog-DM6G4aTi.webm`,ja=`/letter_constellations/assets/helicopter-BDdKO2XC.webm`,Ma=`/letter_constellations/assets/horse-BQ3oQppo.webm`,Na=`/letter_constellations/assets/hotdog-q3lfgc-H.webm`,Pa=`/letter_constellations/assets/house-FiY5MHVH.webm`,Fa=`/letter_constellations/assets/ice_cream-uwet6mas.webm`,Ia=`/letter_constellations/assets/igloo-Bm1n6RqU.webm`,La=`/letter_constellations/assets/iguana-Cj9OFN_N.webm`,Ra=`/letter_constellations/assets/inkpot-CedBlT6t.webm`,za=`/letter_constellations/assets/insect-BVXEJtav.webm`,Ba=`/letter_constellations/assets/instrument-DEj5EZot.webm`,Va=`/letter_constellations/assets/invitation-yDuZq8gj.webm`,Ha=`/letter_constellations/assets/iron-Cx2pzUpR.webm`,Ua=`/letter_constellations/assets/island-fs6PoIj0.webm`,Wa=`/letter_constellations/assets/jaguar-BBmfc63Y.webm`,Ga=`/letter_constellations/assets/jar-DHu15TBb.webm`,Ka=`/letter_constellations/assets/jean_jacket-6ocVOvCD.webm`,qa=`/letter_constellations/assets/jelly-EK6xdlfq.webm`,Ja=`/letter_constellations/assets/jellyfish-BRbFjms6.webm`,Ya=`/letter_constellations/assets/jet-ByhMGSXW.webm`,Xa=`/letter_constellations/assets/jigsaw-DZYjOBmp.webm`,Za=`/letter_constellations/assets/juggler-DBOlauIy.webm`,Qa=`/letter_constellations/assets/juice-D-68Bu_u.webm`,$a=`/letter_constellations/assets/kangaroo-CcIkE6dA.webm`,eo=`/letter_constellations/assets/kettle_drum-D-Fw6JsA.webm`,to=`/letter_constellations/assets/key-KD60Ug_d.webm`,no=`/letter_constellations/assets/kid-tJFEsvg6.webm`,ro=`/letter_constellations/assets/kite-BvkQpXQ4.webm`,io=`/letter_constellations/assets/kiwi-DSByUtRu.webm`,ao=`/letter_constellations/assets/knight-BHnTxxNg.webm`,oo=`/letter_constellations/assets/knot-BFtAc-vX.webm`,so=`/letter_constellations/assets/koala-BOrbjEQC.webm`,co=`/letter_constellations/assets/labrador-C2atZ3eI.webm`,lo=`/letter_constellations/assets/ladybug-D_CzUcBl.webm`,uo=`/letter_constellations/assets/lamp-CA6Xv_tn.webm`,fo=`/letter_constellations/assets/lemon-CPuugyKa.webm`,po=`/letter_constellations/assets/lemur-CPypRhBI.webm`,mo=`/letter_constellations/assets/lettuce-9ltxxL4a.webm`,ho=`/letter_constellations/assets/library-Bt30cqOU.webm`,go=`/letter_constellations/assets/llama-CnavT3OA.webm`,_o=`/letter_constellations/assets/lungs-CtTA6hzs.webm`,vo=`/letter_constellations/assets/map-DpXTnY61.webm`,yo=`/letter_constellations/assets/milk_carton-CYrSpFIF.webm`,bo=`/letter_constellations/assets/mitten-CjMBbbHc.webm`,xo=`/letter_constellations/assets/monkey-Da7Co2b4.webm`,So=`/letter_constellations/assets/moon-BQlWDz1v.webm`,Co=`/letter_constellations/assets/mountain-BXOdjgsP.webm`,wo=`/letter_constellations/assets/mouse-CX-udEkX.webm`,To=`/letter_constellations/assets/muffin-ZyoQalKv.webm`,Eo=`/letter_constellations/assets/mushroom-ma49-9Es.webm`,Do=`/letter_constellations/assets/nail-BDgqn4Oi.webm`,Oo=`/letter_constellations/assets/needle-DfSM5BBx.webm`,ko=`/letter_constellations/assets/nest-D8oW8sHs.webm`,Ao=`/letter_constellations/assets/net-BOWyW7Pd.webm`,jo=`/letter_constellations/assets/newt-Di4rr-kM.webm`,Mo=`/letter_constellations/assets/night-DYJXIJ6m.webm`,No=`/letter_constellations/assets/nose-wU6bmTQo.webm`,Po=`/letter_constellations/assets/nurse-Dvz6wvdQ.webm`,Fo=`/letter_constellations/assets/nuts-DOYhPI3p.webm`,Io=`/letter_constellations/assets/oar-NPG07m8x.webm`,Lo=`/letter_constellations/assets/octopus-BlAhrR5x.webm`,Ro=`/letter_constellations/assets/onion-C3UnQO2e.webm`,zo=`/letter_constellations/assets/orange-CAjV4sVP.webm`,Bo=`/letter_constellations/assets/orangutan-jdEjO_N5.webm`,Vo=`/letter_constellations/assets/orbit-Cd0ALAeU.webm`,Ho=`/letter_constellations/assets/ostrich-Dku7vhly.webm`,Uo=`/letter_constellations/assets/overalls-CI6JhbtH.webm`,Wo=`/letter_constellations/assets/owl-CQWsqBJC.webm`,Go=`/letter_constellations/assets/panda-BmfrrnU6.webm`,Ko=`/letter_constellations/assets/pen-ZTmVznwu.webm`,qo=`/letter_constellations/assets/pencil-BVQ8rqGT.webm`,Jo=`/letter_constellations/assets/penguin-DeD256Gp.webm`,Yo=`/letter_constellations/assets/pig-DzgF3-tq.webm`,Xo=`/letter_constellations/assets/pineapple-DxiQhkpX.webm`,Zo=`/letter_constellations/assets/pirate-D-3TP_NB.webm`,Qo=`/letter_constellations/assets/pizza-FIglZQ7d.webm`,$o=`/letter_constellations/assets/pumpkin-BindYuWd.webm`,es=`/letter_constellations/assets/quaff-C0HVHSvx.webm`,ts=`/letter_constellations/assets/quail-B8-wr5Ky.webm`,ns=`/letter_constellations/assets/quarter-DYSIAC14.webm`,rs=`/letter_constellations/assets/queen-DJtNeLa0.webm`,is=`/letter_constellations/assets/question-V1fIR47w.webm`,as=`/letter_constellations/assets/quill-Bh53ye5V.webm`,os=`/letter_constellations/assets/quilt-Ju7WSa2i.webm`,ss=`/letter_constellations/assets/quince-DQk-aRBI.webm`,cs=`/letter_constellations/assets/quiver-KF2M2SC_.webm`,ls=`/letter_constellations/assets/rabbit-Bn1L-ab9.webm`,us=`/letter_constellations/assets/radishes-Caj0bGTw.webm`,ds=`/letter_constellations/assets/rainbow-D_0DDL10.webm`,fs=`/letter_constellations/assets/ring-CvaEKfC1.webm`,ps=`/letter_constellations/assets/robin-B4Wyw6Kt.webm`,ms=`/letter_constellations/assets/robot-O0G0NIz8.webm`,hs=`/letter_constellations/assets/rocket-DyCWYIqK.webm`,gs=`/letter_constellations/assets/rose-DNpS-rzs.webm`,_s=`/letter_constellations/assets/ruler-3jiqdeeH.webm`,vs=`/letter_constellations/assets/sailboat-C9PCPNTV.webm`,ys=`/letter_constellations/assets/sheep-BD1Qex-a.webm`,bs=`/letter_constellations/assets/snake-3VL3pI0b.webm`,xs=`/letter_constellations/assets/snowman-BVngaE17.webm`,Ss=`/letter_constellations/assets/socks-BlZGzMoX.webm`,Cs=`/letter_constellations/assets/spoon-Bpezv9ur.webm`,ws=`/letter_constellations/assets/star-ByhnbkyF.webm`,Ts=`/letter_constellations/assets/strawberry-WDKIdltR.webm`,Es=`/letter_constellations/assets/sun-OXh-YcDj.webm`,Ds=`/letter_constellations/assets/table-DdVbos9W.webm`,Os=`/letter_constellations/assets/teapot-C8QdlDfa.webm`,ks=`/letter_constellations/assets/teddy_bear-DbuTfBst.webm`,As=`/letter_constellations/assets/tiger-DDYpj3Yp.webm`,js=`/letter_constellations/assets/tomato-WtQkNxZo.webm`,Ms=`/letter_constellations/assets/tractor-CCgqU7qU.webm`,Ns=`/letter_constellations/assets/train-C5Zj-m0t.webm`,Ps=`/letter_constellations/assets/tree-nuB9A3vs.webm`,Fs=`/letter_constellations/assets/turtle-BDkcySK2.webm`,Is=`/letter_constellations/assets/ukulele-CLqXYXhe.webm`,Ls=`/letter_constellations/assets/umbrella-r_963ubY.webm`,Rs=`/letter_constellations/assets/ungulate-CTO96oPD.webm`,zs=`/letter_constellations/assets/unicorn-CY2TK8f5.webm`,Bs=`/letter_constellations/assets/unicycle-DGKk_v6u.webm`,Vs=`/letter_constellations/assets/unidentified_object--Ro5ICPR.webm`,Hs=`/letter_constellations/assets/uniform-DqfMcPDd.webm`,Us=`/letter_constellations/assets/urchin-iCRAGDls.webm`,Ws=`/letter_constellations/assets/utensils-BzdOuCsZ.webm`,Gs=`/letter_constellations/assets/vacuum-DoCcb9z-.webm`,Ks=`/letter_constellations/assets/van-CQ1lJBiN.webm`,qs=`/letter_constellations/assets/vase-CccRv94X.webm`,Js=`/letter_constellations/assets/vegetables-DLtroWc1.webm`,Ys=`/letter_constellations/assets/vest-DBXjbVnZ.webm`,Xs=`/letter_constellations/assets/village-CVG8tels.webm`,Zs=`/letter_constellations/assets/violin-BE21TgO1.webm`,Qs=`/letter_constellations/assets/volcano-gCdpROrp.webm`,$s=`/letter_constellations/assets/vulture-TiasgFRu.webm`,ec=`/letter_constellations/assets/wagon-BpMibSu1.webm`,tc=`/letter_constellations/assets/watch-DR-zilf0.webm`,nc=`/letter_constellations/assets/watermelon-NcZQdRkY.webm`,rc=`/letter_constellations/assets/web-CjQIo_1Q.webm`,ic=`/letter_constellations/assets/whale-cDbKOtue.webm`,ac=`/letter_constellations/assets/wind-DK91ZQgX.webm`,oc=`/letter_constellations/assets/wizard-DhJfciPy.webm`,sc=`/letter_constellations/assets/wolf-DHbkNJpk.webm`,cc=`/letter_constellations/assets/worm-BWnc7YI5.webm`,lc={"./assets/items/a/accordion.png":be,"./assets/items/a/acorn.png":xe,"./assets/items/a/airplane.png":Se,"./assets/items/a/alligator.png":Ce,"./assets/items/a/anchor.png":we,"./assets/items/a/ant.png":Te,"./assets/items/a/apple.png":Ee,"./assets/items/a/arrow.png":De,"./assets/items/a/astronaut.png":Oe,"./assets/items/b/bananas.png":ke,"./assets/items/b/beachball.png":Ae,"./assets/items/b/bear.png":je,"./assets/items/b/bee.png":Me,"./assets/items/b/bird.png":Ne,"./assets/items/b/boat.png":Pe,"./assets/items/b/books.png":Fe,"./assets/items/b/bus.png":Ie,"./assets/items/b/butterfly.png":Le,"./assets/items/c/cake.png":Re,"./assets/items/c/car.png":ze,"./assets/items/c/castle.png":Be,"./assets/items/c/cat.png":Ve,"./assets/items/c/clock.png":He,"./assets/items/c/coat.png":Ue,"./assets/items/c/cow.png":We,"./assets/items/c/crayon.png":Ge,"./assets/items/c/cup.png":Ke,"./assets/items/d/dog.png":qe,"./assets/items/d/doll.png":Je,"./assets/items/d/dolphin.png":Ye,"./assets/items/d/donkey.png":Xe,"./assets/items/d/door.png":Ze,"./assets/items/d/dragon.png":Qe,"./assets/items/d/dress.png":$e,"./assets/items/d/drum.png":et,"./assets/items/d/duck.png":tt,"./assets/items/e/eagle.png":nt,"./assets/items/e/ear.png":rt,"./assets/items/e/earth.png":it,"./assets/items/e/easel.png":at,"./assets/items/e/egg.png":ot,"./assets/items/e/elephant.png":st,"./assets/items/e/envelope.png":ct,"./assets/items/e/eraser.png":lt,"./assets/items/f/feather.png":ut,"./assets/items/f/fence.png":dt,"./assets/items/f/fish.png":ft,"./assets/items/f/flower.png":pt,"./assets/items/f/foot.png":mt,"./assets/items/f/fork.png":ht,"./assets/items/f/fox.png":gt,"./assets/items/f/frog.png":_t,"./assets/items/g/garden.png":vt,"./assets/items/g/ghost.png":yt,"./assets/items/g/giraffe.png":bt,"./assets/items/g/gloves.png":xt,"./assets/items/g/goat.png":St,"./assets/items/g/goldfish.png":Ct,"./assets/items/g/grapes.png":wt,"./assets/items/g/guitar.png":Tt,"./assets/items/h/hammer.png":Et,"./assets/items/h/hand.png":Dt,"./assets/items/h/hat.png":Ot,"./assets/items/h/heart.png":kt,"./assets/items/h/hedgehog.png":At,"./assets/items/h/helicopter.png":jt,"./assets/items/h/horse.png":Mt,"./assets/items/h/hotdog.png":Nt,"./assets/items/h/house.png":Pt,"./assets/items/i/ice_cream.png":Ft,"./assets/items/i/igloo.png":It,"./assets/items/i/iguana.png":Lt,"./assets/items/i/inkpot.png":Rt,"./assets/items/i/instrument.png":zt,"./assets/items/i/invitation.png":Bt,"./assets/items/i/iron.png":Vt,"./assets/items/i/island.png":Ht,"./assets/items/j/jaguar.png":Ut,"./assets/items/j/jar.png":Wt,"./assets/items/j/jean_jacket.png":Gt,"./assets/items/j/jelly.png":Kt,"./assets/items/j/jellyfish.png":qt,"./assets/items/j/jigsaw.png":Jt,"./assets/items/j/juggler.png":Yt,"./assets/items/j/juice.png":Xt,"./assets/items/k/kangaroo.png":Zt,"./assets/items/k/key.png":Qt,"./assets/items/k/kid.png":$t,"./assets/items/k/kite.png":en,"./assets/items/k/kiwi.png":tn,"./assets/items/k/koala.png":nn,"./assets/items/l/ladybug.png":rn,"./assets/items/l/lamp.png":an,"./assets/items/l/lemon.png":on,"./assets/items/l/lemur.png":sn,"./assets/items/l/lettuce.png":cn,"./assets/items/l/library.png":ln,"./assets/items/l/llama.png":un,"./assets/items/m/map.png":dn,"./assets/items/m/milk_carton.png":fn,"./assets/items/m/mitten.png":pn,"./assets/items/m/monkey.png":mn,"./assets/items/m/moon.png":hn,"./assets/items/m/mountain.png":gn,"./assets/items/m/mouse.png":_n,"./assets/items/m/muffin.png":vn,"./assets/items/m/mushroom.png":yn,"./assets/items/n/nail.png":bn,"./assets/items/n/needle.png":xn,"./assets/items/n/nest.png":Sn,"./assets/items/n/net.png":Cn,"./assets/items/n/newt.png":wn,"./assets/items/n/night.png":Tn,"./assets/items/n/nose.png":En,"./assets/items/n/nurse.png":Dn,"./assets/items/n/nuts.png":On,"./assets/items/o/oar.png":kn,"./assets/items/o/octopus.png":An,"./assets/items/o/onion.png":jn,"./assets/items/o/orange.png":Mn,"./assets/items/o/orangutan.png":Nn,"./assets/items/o/ostrich.png":Pn,"./assets/items/o/owl.png":Fn,"./assets/items/p/panda.png":In,"./assets/items/p/pen.png":Ln,"./assets/items/p/pencil.png":Rn,"./assets/items/p/penguin.png":zn,"./assets/items/p/pig.png":Bn,"./assets/items/p/pineapple.png":Vn,"./assets/items/p/pirate.png":Hn,"./assets/items/p/pizza.png":Un,"./assets/items/p/pumpkin.png":Wn,"./assets/items/q/quail.png":Gn,"./assets/items/q/queen.png":Kn,"./assets/items/q/question.png":qn,"./assets/items/q/quill.png":Jn,"./assets/items/q/quilt.png":Yn,"./assets/items/q/quince.png":Xn,"./assets/items/q/quiver.png":Zn,"./assets/items/r/rabbit.png":Qn,"./assets/items/r/radishes.png":$n,"./assets/items/r/rainbow.png":er,"./assets/items/r/ring.png":tr,"./assets/items/r/robin.png":nr,"./assets/items/r/robot.png":rr,"./assets/items/r/rocket.png":ir,"./assets/items/r/rose.png":ar,"./assets/items/r/ruler.png":or,"./assets/items/s/sailboat.png":sr,"./assets/items/s/sheep.png":cr,"./assets/items/s/snake.png":lr,"./assets/items/s/snowman.png":ur,"./assets/items/s/socks.png":dr,"./assets/items/s/star.png":fr,"./assets/items/s/strawberry.png":pr,"./assets/items/s/sun.png":mr,"./assets/items/t/table.png":hr,"./assets/items/t/teapot.png":gr,"./assets/items/t/tiger.png":_r,"./assets/items/t/tomato.png":vr,"./assets/items/t/tractor.png":yr,"./assets/items/t/train.png":br,"./assets/items/t/tree.png":xr,"./assets/items/t/turtle.png":Sr,"./assets/items/u/ukulele.png":Cr,"./assets/items/u/umbrella.png":wr,"./assets/items/u/unicorn.png":Tr,"./assets/items/u/unicycle.png":Er,"./assets/items/u/unidentified_object.png":Dr,"./assets/items/u/urchin.png":Or,"./assets/items/u/utensils.png":kr,"./assets/items/v/vacuum.png":Ar,"./assets/items/v/van.png":jr,"./assets/items/v/vase.png":Mr,"./assets/items/v/vegetables.png":Nr,"./assets/items/v/vest.png":Pr,"./assets/items/v/village.png":Fr,"./assets/items/v/violin.png":Ir,"./assets/items/v/volcano.png":Lr,"./assets/items/v/vulture.png":Rr,"./assets/items/w/wagon.png":zr,"./assets/items/w/watch.png":Br,"./assets/items/w/watermelon.png":Vr,"./assets/items/w/web.png":Hr,"./assets/items/w/whale.png":Ur,"./assets/items/w/wind.png":Wr,"./assets/items/w/wizard.png":Gr,"./assets/items/w/wolf.png":Kr,"./assets/items/w/worm.png":qr},uc={"./assets/voice/a/accordion.webm":Jr,"./assets/voice/a/acorn.webm":Yr,"./assets/voice/a/airplane.webm":Xr,"./assets/voice/a/alligator.webm":Zr,"./assets/voice/a/anchor.webm":Qr,"./assets/voice/a/ant.webm":$r,"./assets/voice/a/apple.webm":ei,"./assets/voice/a/arrow.webm":ti,"./assets/voice/a/astronaut.webm":ni,"./assets/voice/alphabet/a.webm":q,"./assets/voice/alphabet/b.webm":J,"./assets/voice/alphabet/c.webm":ri,"./assets/voice/alphabet/d.webm":ii,"./assets/voice/alphabet/e.webm":ai,"./assets/voice/alphabet/f.webm":oi,"./assets/voice/alphabet/g.webm":si,"./assets/voice/alphabet/h.webm":ci,"./assets/voice/alphabet/i.webm":li,"./assets/voice/alphabet/j.webm":ui,"./assets/voice/alphabet/k.webm":di,"./assets/voice/alphabet/l.webm":fi,"./assets/voice/alphabet/m.webm":pi,"./assets/voice/alphabet/n.webm":mi,"./assets/voice/alphabet/o.webm":hi,"./assets/voice/alphabet/p.webm":gi,"./assets/voice/alphabet/q.webm":_i,"./assets/voice/alphabet/r.webm":vi,"./assets/voice/alphabet/s.webm":yi,"./assets/voice/alphabet/t.webm":bi,"./assets/voice/alphabet/u.webm":xi,"./assets/voice/alphabet/v.webm":Y,"./assets/voice/alphabet/w.webm":Si,"./assets/voice/alphabet/x.webm":Ci,"./assets/voice/alphabet/y.webm":wi,"./assets/voice/alphabet/z.webm":Ti,"./assets/voice/b/bananas.webm":Ei,"./assets/voice/b/beachball.webm":Di,"./assets/voice/b/bear.webm":Oi,"./assets/voice/b/bee.webm":ki,"./assets/voice/b/bird.webm":Ai,"./assets/voice/b/boat.webm":ji,"./assets/voice/b/books.webm":Mi,"./assets/voice/b/bus.webm":Ni,"./assets/voice/b/butterfly.webm":Pi,"./assets/voice/c/cake.webm":Fi,"./assets/voice/c/car.webm":Ii,"./assets/voice/c/castle.webm":Li,"./assets/voice/c/cat.webm":Ri,"./assets/voice/c/clock.webm":zi,"./assets/voice/c/coat.webm":Bi,"./assets/voice/c/cow.webm":Vi,"./assets/voice/c/crayon.webm":Hi,"./assets/voice/c/cup.webm":Ui,"./assets/voice/d/dog.webm":Wi,"./assets/voice/d/doll.webm":Gi,"./assets/voice/d/dolphin.webm":Ki,"./assets/voice/d/donkey.webm":qi,"./assets/voice/d/door.webm":Ji,"./assets/voice/d/dragon.webm":Yi,"./assets/voice/d/dress.webm":Xi,"./assets/voice/d/drum.webm":Zi,"./assets/voice/d/duck.webm":Qi,"./assets/voice/e/eagle.webm":$i,"./assets/voice/e/ear.webm":ea,"./assets/voice/e/earth.webm":ta,"./assets/voice/e/easel.webm":na,"./assets/voice/e/egg.webm":ra,"./assets/voice/e/elephant.webm":ia,"./assets/voice/e/engine.webm":aa,"./assets/voice/e/envelope.webm":oa,"./assets/voice/e/eraser.webm":sa,"./assets/voice/f/feather.webm":ca,"./assets/voice/f/fence.webm":la,"./assets/voice/f/firefly.webm":ua,"./assets/voice/f/fish.webm":da,"./assets/voice/f/flower.webm":fa,"./assets/voice/f/foot.webm":pa,"./assets/voice/f/fork.webm":ma,"./assets/voice/f/fox.webm":ha,"./assets/voice/f/frog.webm":ga,"./assets/voice/g/garden.webm":_a,"./assets/voice/g/ghost.webm":va,"./assets/voice/g/gift.webm":ya,"./assets/voice/g/giraffe.webm":ba,"./assets/voice/g/gloves.webm":xa,"./assets/voice/g/goat.webm":Sa,"./assets/voice/g/goldfish.webm":Ca,"./assets/voice/g/grapes.webm":wa,"./assets/voice/g/guitar.webm":Ta,"./assets/voice/h/hammer.webm":Ea,"./assets/voice/h/hand.webm":Da,"./assets/voice/h/hat.webm":Oa,"./assets/voice/h/heart.webm":ka,"./assets/voice/h/hedgehog.webm":Aa,"./assets/voice/h/helicopter.webm":ja,"./assets/voice/h/horse.webm":Ma,"./assets/voice/h/hotdog.webm":Na,"./assets/voice/h/house.webm":Pa,"./assets/voice/i/ice_cream.webm":Fa,"./assets/voice/i/igloo.webm":Ia,"./assets/voice/i/iguana.webm":La,"./assets/voice/i/inkpot.webm":Ra,"./assets/voice/i/insect.webm":za,"./assets/voice/i/instrument.webm":Ba,"./assets/voice/i/invitation.webm":Va,"./assets/voice/i/iron.webm":Ha,"./assets/voice/i/island.webm":Ua,"./assets/voice/j/jaguar.webm":Wa,"./assets/voice/j/jar.webm":Ga,"./assets/voice/j/jean_jacket.webm":Ka,"./assets/voice/j/jelly.webm":qa,"./assets/voice/j/jellyfish.webm":Ja,"./assets/voice/j/jet.webm":Ya,"./assets/voice/j/jigsaw.webm":Xa,"./assets/voice/j/juggler.webm":Za,"./assets/voice/j/juice.webm":Qa,"./assets/voice/k/kangaroo.webm":$a,"./assets/voice/k/kettle_drum.webm":eo,"./assets/voice/k/key.webm":to,"./assets/voice/k/kid.webm":no,"./assets/voice/k/kite.webm":ro,"./assets/voice/k/kiwi.webm":io,"./assets/voice/k/knight.webm":ao,"./assets/voice/k/knot.webm":oo,"./assets/voice/k/koala.webm":so,"./assets/voice/l/labrador.webm":co,"./assets/voice/l/ladybug.webm":lo,"./assets/voice/l/lamp.webm":uo,"./assets/voice/l/lemon.webm":fo,"./assets/voice/l/lemur.webm":po,"./assets/voice/l/lettuce.webm":mo,"./assets/voice/l/library.webm":ho,"./assets/voice/l/llama.webm":go,"./assets/voice/l/lungs.webm":_o,"./assets/voice/m/map.webm":vo,"./assets/voice/m/milk_carton.webm":yo,"./assets/voice/m/mitten.webm":bo,"./assets/voice/m/monkey.webm":xo,"./assets/voice/m/moon.webm":So,"./assets/voice/m/mountain.webm":Co,"./assets/voice/m/mouse.webm":wo,"./assets/voice/m/muffin.webm":To,"./assets/voice/m/mushroom.webm":Eo,"./assets/voice/n/nail.webm":Do,"./assets/voice/n/needle.webm":Oo,"./assets/voice/n/nest.webm":ko,"./assets/voice/n/net.webm":Ao,"./assets/voice/n/newt.webm":jo,"./assets/voice/n/night.webm":Mo,"./assets/voice/n/nose.webm":No,"./assets/voice/n/nurse.webm":Po,"./assets/voice/n/nuts.webm":Fo,"./assets/voice/o/oar.webm":Io,"./assets/voice/o/octopus.webm":Lo,"./assets/voice/o/onion.webm":Ro,"./assets/voice/o/orange.webm":zo,"./assets/voice/o/orangutan.webm":Bo,"./assets/voice/o/orbit.webm":Vo,"./assets/voice/o/ostrich.webm":Ho,"./assets/voice/o/overalls.webm":Uo,"./assets/voice/o/owl.webm":Wo,"./assets/voice/p/panda.webm":Go,"./assets/voice/p/pen.webm":Ko,"./assets/voice/p/pencil.webm":qo,"./assets/voice/p/penguin.webm":Jo,"./assets/voice/p/pig.webm":Yo,"./assets/voice/p/pineapple.webm":Xo,"./assets/voice/p/pirate.webm":Zo,"./assets/voice/p/pizza.webm":Qo,"./assets/voice/p/pumpkin.webm":$o,"./assets/voice/q/quaff.webm":es,"./assets/voice/q/quail.webm":ts,"./assets/voice/q/quarter.webm":ns,"./assets/voice/q/queen.webm":rs,"./assets/voice/q/question.webm":is,"./assets/voice/q/quill.webm":as,"./assets/voice/q/quilt.webm":os,"./assets/voice/q/quince.webm":ss,"./assets/voice/q/quiver.webm":cs,"./assets/voice/r/rabbit.webm":ls,"./assets/voice/r/radishes.webm":us,"./assets/voice/r/rainbow.webm":ds,"./assets/voice/r/ring.webm":fs,"./assets/voice/r/robin.webm":ps,"./assets/voice/r/robot.webm":ms,"./assets/voice/r/rocket.webm":hs,"./assets/voice/r/rose.webm":gs,"./assets/voice/r/ruler.webm":_s,"./assets/voice/s/sailboat.webm":vs,"./assets/voice/s/sheep.webm":ys,"./assets/voice/s/snake.webm":bs,"./assets/voice/s/snowman.webm":xs,"./assets/voice/s/socks.webm":Ss,"./assets/voice/s/spoon.webm":Cs,"./assets/voice/s/star.webm":ws,"./assets/voice/s/strawberry.webm":Ts,"./assets/voice/s/sun.webm":Es,"./assets/voice/t/table.webm":Ds,"./assets/voice/t/teapot.webm":Os,"./assets/voice/t/teddy_bear.webm":ks,"./assets/voice/t/tiger.webm":As,"./assets/voice/t/tomato.webm":js,"./assets/voice/t/tractor.webm":Ms,"./assets/voice/t/train.webm":Ns,"./assets/voice/t/tree.webm":Ps,"./assets/voice/t/turtle.webm":Fs,"./assets/voice/u/ukulele.webm":Is,"./assets/voice/u/umbrella.webm":Ls,"./assets/voice/u/ungulate.webm":Rs,"./assets/voice/u/unicorn.webm":zs,"./assets/voice/u/unicycle.webm":Bs,"./assets/voice/u/unidentified_object.webm":Vs,"./assets/voice/u/uniform.webm":Hs,"./assets/voice/u/urchin.webm":Us,"./assets/voice/u/utensils.webm":Ws,"./assets/voice/v/vacuum.webm":Gs,"./assets/voice/v/van.webm":Ks,"./assets/voice/v/vase.webm":qs,"./assets/voice/v/vegetables.webm":Js,"./assets/voice/v/vest.webm":Ys,"./assets/voice/v/village.webm":Xs,"./assets/voice/v/violin.webm":Zs,"./assets/voice/v/volcano.webm":Qs,"./assets/voice/v/vulture.webm":$s,"./assets/voice/w/wagon.webm":ec,"./assets/voice/w/watch.webm":tc,"./assets/voice/w/watermelon.webm":nc,"./assets/voice/w/web.webm":rc,"./assets/voice/w/whale.webm":ic,"./assets/voice/w/wind.webm":ac,"./assets/voice/w/wizard.webm":oc,"./assets/voice/w/wolf.webm":sc,"./assets/voice/w/worm.webm":cc},dc={"./assets/voice/alphabet/a.webm":q,"./assets/voice/alphabet/b.webm":J,"./assets/voice/alphabet/c.webm":ri,"./assets/voice/alphabet/d.webm":ii,"./assets/voice/alphabet/e.webm":ai,"./assets/voice/alphabet/f.webm":oi,"./assets/voice/alphabet/g.webm":si,"./assets/voice/alphabet/h.webm":ci,"./assets/voice/alphabet/i.webm":li,"./assets/voice/alphabet/j.webm":ui,"./assets/voice/alphabet/k.webm":di,"./assets/voice/alphabet/l.webm":fi,"./assets/voice/alphabet/m.webm":pi,"./assets/voice/alphabet/n.webm":mi,"./assets/voice/alphabet/o.webm":hi,"./assets/voice/alphabet/p.webm":gi,"./assets/voice/alphabet/q.webm":_i,"./assets/voice/alphabet/r.webm":vi,"./assets/voice/alphabet/s.webm":yi,"./assets/voice/alphabet/t.webm":bi,"./assets/voice/alphabet/u.webm":xi,"./assets/voice/alphabet/v.webm":Y,"./assets/voice/alphabet/w.webm":Si,"./assets/voice/alphabet/x.webm":Ci,"./assets/voice/alphabet/y.webm":wi,"./assets/voice/alphabet/z.webm":Ti},X=e=>{let t=e.split(`/`),n=t[t.length-1]??``;return{folder:t[t.length-2]??``,name:n.replace(/\.[^.]+$/,``)}},Z=new Map;for(let[e,t]of Object.entries(uc)){let{folder:n,name:r}=X(e);if(n===`alphabet`)continue;let i=t;Z.has(n)||Z.set(n,new Map),Z.get(n).set(r,i)}var Q=new Map;for(let[e,t]of Object.entries(lc)){let{folder:n,name:r}=X(e),i=r.replace(/_/g,` `),a=Z.get(n)?.get(r);Q.has(n)||Q.set(n,[]),Q.get(n).push({name:r,label:i,imageUrl:t,audioUrl:a})}var $=new Map;for(let[e,t]of Object.entries(dc)){let{name:n}=X(e);$.set(n,t)}var fc=(e,t)=>{let n=[...e];for(let e=n.length-1;e>0;--e){let t=Math.floor(Math.random()*(e+1));[n[e],n[t]]=[n[t],n[e]]}return n.slice(0,Math.min(t,n.length))};async function pc(){let e=document.querySelector(`#app`),t=new URL(`/letter_constellations/assets/aurora-BhVtVCtR.mp3`,``+import.meta.url).toString(),n=new Audio(t);n.loop=!0,n.preload=`auto`,n.volume=1;let r=new URL(`/letter_constellations/assets/chime-DLBpLcOd.mp3`,``+import.meta.url).toString(),i=new Audio(r);i.preload=`auto`;let a=null,s=null,c=.33,l=()=>{if(a)return;a=new AudioContext;let e=a.createMediaElementSource(n);s=a.createGain(),s.gain.value=.25*c,e.connect(s).connect(a.destination)},u=document.createElement(`canvas`);u.id=`aurora-canvas`,e.appendChild(u);let d=document.createElement(`canvas`);d.id=`sparkle-canvas`,e.appendChild(d);let f=document.createElement(`canvas`);f.id=`game-canvas`,e.appendChild(f);let p=document.createElement(`button`);p.className=`mute-button`,p.type=`button`,p.textContent=`Mute`,p.setAttribute(`aria-pressed`,`false`),e.appendChild(p);let m=document.createElement(`div`);m.className=`top-controls`;let h=document.createElement(`button`);h.className=`control-button`,h.type=`button`,h.textContent=`Clear`,m.appendChild(h);let g=document.createElement(`button`);g.className=`control-button`,g.type=`button`,g.textContent=`Reset`,m.appendChild(g),e.appendChild(m);let _=document.createElement(`div`);_.className=`volume-control`,_.innerHTML=`
    <label for="volume-slider">Volume</label>
    <input id="volume-slider" type="range" min="0" max="2" step="0.05" value="0.33" />
  `,e.appendChild(_);let v=_.querySelector(`#volume-slider`);c=Number(v.value),i.volume=Math.min(1,c);let y=new Map,b=new Set,x=null,S=0,C=null,w=e=>{let t=y.get(e);return t||(t=new Audio(e),t.preload=`auto`,y.set(e,t),b.add(t)),t.volume=Math.min(1,Number(v.value)),t.muted=n.muted,t},D=()=>{x&&(x.pause(),x.currentTime=0)},O=async e=>{if(e.length===0)return;let t=++S;D();for(let n of e){if(t!==S)return;let e=w(n);x=e,e.currentTime=0;try{await e.play()}catch{return}await new Promise(t=>{let n=()=>{e.removeEventListener(`ended`,n),e.removeEventListener(`error`,n),t()};e.addEventListener(`ended`,n),e.addEventListener(`error`,n)})}},k=e=>{let t=$.get(e);if(t){if(!j){C=t;return}O([t])}},A=()=>{let e=n.muted;p.textContent=e?`Unmute`:`Mute`,p.setAttribute(`aria-pressed`,String(e)),i.muted=e;for(let t of b)t.muted=e};A();let j=!1,M=()=>{j&&(l(),a&&a.state===`suspended`&&a.resume().catch(()=>{}),n.play().catch(()=>{}))},N=()=>{if(j=!0,M(),C){let e=C;C=null,O([e])}},P={once:!0,capture:!0};document.addEventListener(`pointerdown`,N,P),document.addEventListener(`touchstart`,N,P),document.addEventListener(`click`,N,P),document.addEventListener(`keydown`,N,P),p.addEventListener(`click`,e=>{e.stopPropagation(),n.muted=!n.muted,n.muted||M(),A()}),v.addEventListener(`input`,()=>{l();let e=Number(v.value);c=e,s&&(s.gain.value=.25*e),i.volume=Math.min(1,e);let t=Math.min(1,e);for(let e of b)e.volume=t});let F=document.createElement(`div`);F.className=`item-tray`,e.appendChild(F);let te=e=>{F.replaceChildren();let t=Q.get(e)??[];if(t.length===0){let t=document.createElement(`div`);t.className=`item-empty`,t.textContent=`No ${e.toUpperCase()} items yet.`,F.appendChild(t);return}let n=fc(t,3);for(let t of n){let n=document.createElement(`button`);n.type=`button`,n.className=`item-card`,n.setAttribute(`aria-label`,t.label),n.addEventListener(`click`,n=>{n.stopPropagation();let r=[],i=$.get(e);i&&r.push(i),t.audioUrl&&r.push(t.audioUrl),O(r)});let r=document.createElement(`img`);r.src=t.imageUrl,r.alt=t.label,r.loading=`lazy`,r.decoding=`async`,n.appendChild(r),F.appendChild(n)}},I=null,L=null,R=null,z=new T(u,f,d),B=f.getContext(`2d`);if(!B)throw Error(`Failed to get 2D context for game canvas.`);let V=()=>{let{width:e,height:t}=z.resize();z.configureGameContext(B),R&&R.setViewportSize(e,t),I&&I.resize(u.width,u.height),L&&L.resize(d.width,d.height)};V(),window.addEventListener(`resize`,V),I=new o(u);let ne=await I.initialize();L=new ee(d);let re=await L.initialize(I.getDevice());if(V(),!ne||!re){let e=document.createElement(`div`);e.className=`error-message`,e.innerHTML=`
            <h1>WebGPU Not Supported</h1>
            <p>Your browser doesn't support WebGPU yet.</p>
            <p>Try using Chrome 113+ or Edge 113+ with WebGPU enabled.</p>
        `,document.body.appendChild(e);return}R=new ye(new E(f,e=>z.getPointFromEvent(f,e)),{onPointHit:()=>{i.currentTime=0,i.play().catch(()=>{})},onLetterChange:e=>{te(e),k(e)}}),h.addEventListener(`click`,e=>{e.stopPropagation(),R?.clearUserPaths()}),g.addEventListener(`click`,e=>{e.stopPropagation(),R?.resetCurrentLetter()});let H=z.getSize();R.setViewportSize(H.width,H.height);let U=performance.now(),ie=0;function W(e){let t=(e-U)/1e3;if(U=e,!R||!I||!L)return;R.update(t);let n=R.getDrawingState();L.updateMouseState(n.isDrawing,n.mouseX,n.mouseY,n.trailPoints);let r=R.getDotState();L.setDots(r.dots,r.currentIndex,r.radius),ie++&1||I.render(),L.render(),B&&R.render(B),requestAnimationFrame(W)}W(performance.now())}pc();
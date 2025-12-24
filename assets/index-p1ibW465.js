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
`,l=16384,u=8,d=u*4,f=1024,p=512,m=8,h=m*4,g=0,_=1,v=2,y=3,b=.7,x=.8,S=12,C=.03;function w(e,t,n){let r=Math.max(0,Math.min(1,(n-e)/(t-e)));return r*r*(3-2*r)}var ee=class{canvas;device;context;pipeline;dotPipeline;uniformBuffer;dotUniformBuffer;bindGroup;dotBindGroup;quadVertexBuffer;particleBuffer;dotBuffer;startTimeMs;uniformScratch=new Float32Array(8);dotUniformScratch=new Float32Array(8);mouseRamp=0;prevActive=!1;mousePos={x:.5,y:.5};mouseVelocity={x:0,y:0};lastSimTimeSec=0;dotCount=0;dotScratch=new Float32Array(p*m);dotStates=new Int8Array(p);dotStateTimes=new Float32Array(p);nextParticleIndex=0;pendingStart=0;pendingCount=0;pendingData=new Float32Array(f*u);sparkAccumulator=0;dustAccumulator=0;burstCooldown=0;haloPhase=0;haloAccumulator=0;stillSeconds=0;lastStampTime=-999;rngState=305419896;constructor(e){this.canvas=e,this.startTimeMs=performance.now()}async initialize(e){if(e)this.device=e;else{if(!navigator.gpu)return console.error(`WebGPU not supported`),!1;let e=await navigator.gpu.requestAdapter();if(!e)return console.error(`Failed to get GPU adapter`),!1;this.device=await e.requestDevice()}this.context=this.canvas.getContext(`webgpu`);let t=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device:this.device,format:t,alphaMode:`premultiplied`}),this.uniformBuffer=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.dotUniformBuffer=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.quadVertexBuffer=this.device.createBuffer({size:48,usage:GPUBufferUsage.VERTEX,mappedAtCreation:!0}),new Float32Array(this.quadVertexBuffer.getMappedRange()).set([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),this.quadVertexBuffer.unmap(),this.particleBuffer=this.device.createBuffer({size:l*d,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Float32Array(this.particleBuffer.getMappedRange()).fill(0),this.particleBuffer.unmap(),this.dotBuffer=this.device.createBuffer({size:p*h,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Float32Array(this.dotBuffer.getMappedRange()).fill(0),this.dotBuffer.unmap();let n=this.device.createShaderModule({code:s}),r=this.device.createShaderModule({code:c}),i=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}}]});this.bindGroup=this.device.createBindGroup({layout:i,entries:[{binding:0,resource:{buffer:this.uniformBuffer}}]}),this.dotBindGroup=this.device.createBindGroup({layout:i,entries:[{binding:0,resource:{buffer:this.dotUniformBuffer}}]});let a=this.device.createPipelineLayout({bindGroupLayouts:[i]});return this.pipeline=this.device.createRenderPipeline({layout:a,vertex:{module:r,entryPoint:`sparkVertexMain`,buffers:[{arrayStride:8,stepMode:`vertex`,attributes:[{shaderLocation:0,offset:0,format:`float32x2`}]},{arrayStride:d,stepMode:`instance`,attributes:[{shaderLocation:1,offset:0,format:`float32x4`},{shaderLocation:2,offset:16,format:`float32x4`}]}]},fragment:{module:r,entryPoint:`sparkFragmentMain`,targets:[{format:t,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one-minus-src-alpha`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),this.dotPipeline=this.device.createRenderPipeline({layout:a,vertex:{module:n,entryPoint:`dotVertexMain`,buffers:[{arrayStride:8,stepMode:`vertex`,attributes:[{shaderLocation:0,offset:0,format:`float32x2`}]},{arrayStride:h,stepMode:`instance`,attributes:[{shaderLocation:1,offset:0,format:`float32x4`},{shaderLocation:2,offset:16,format:`float32x4`}]}]},fragment:{module:n,entryPoint:`dotFragmentMain`,targets:[{format:t,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one-minus-src-alpha`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),!0}setDots(e,t,n){let r=Math.min(e.length,p);if(this.dotCount=r,r===0)return;let i=(performance.now()-this.startTimeMs)/1e3,{dpr:a,cssW:o,cssH:s}=this.getCanvasMetrics(),c=Math.max(10,n*1.3);for(let n=0;n<r;n+=1){let r=e[n],l=r.x/o,u=r.y/s,d=0;n<t?d=2:n===t&&(d=1),this.dotStates[n]!==d&&(this.dotStates[n]=d,this.dotStateTimes[n]=i,d===2&&this.emitDotCompleteBurst(i,{x:l,y:u},a));let f=c*(d===1?1.7:d===2?.95:.85)*a,p=n*m;this.dotScratch[p+0]=l,this.dotScratch[p+1]=u,this.dotScratch[p+2]=f,this.dotScratch[p+3]=d,this.dotScratch[p+4]=this.hashSeed(n),this.dotScratch[p+5]=this.dotStateTimes[n],this.dotScratch[p+6]=0,this.dotScratch[p+7]=0}this.device.queue.writeBuffer(this.dotBuffer,0,this.dotScratch.subarray(0,r*m))}updateMouseState(e,t,n,r){let i=(performance.now()-this.startTimeMs)/1e3,a=this.lastSimTimeSec===0?0:i-this.lastSimTimeSec,o=Math.max(0,Math.min(.05,a));if(this.lastSimTimeSec=i,o>0){let t=o/.45;this.mouseRamp=e?Math.min(1,this.mouseRamp+t):Math.max(0,this.mouseRamp-t)}let{dpr:s,cssW:c,cssH:l}=this.getCanvasMetrics(),u=Math.max(0,Math.min(1,t/c)),d=Math.max(0,Math.min(1,n/l)),f={...this.mousePos},p={x:u,y:d};if(this.mousePos=p,o>0){let e=(p.x-f.x)/o,t=(p.y-f.y)/o;this.mouseVelocity.x=this.mouseVelocity.x*.7+e*.3,this.mouseVelocity.y=this.mouseVelocity.y*.7+t*.3}let m=Math.hypot(this.mouseVelocity.x,this.mouseVelocity.y),h=w(.08,2.4,m);if(o>0&&this.mouseRamp>.02){this.haloPhase+=o*(1.2+h*5.5);let e=(18+h*140)*this.mouseRamp*b;this.haloAccumulator+=e*o;let t=Math.floor(this.haloAccumulator);this.haloAccumulator-=t,t=Math.min(t,10);let n=.014+.012*h;for(let e=0;e<t;e++){let e=this.haloPhase*Math.PI*2+this.rand01()*Math.PI*2,t=Math.cos(e)*n,r=Math.sin(e)*n,a=.18+h*.65,o=-Math.sin(e)*a+t*.8,c=Math.cos(e)*a+r*.8,l=.35+this.rand01()*.55,u=(.9+this.rand01()*2.2)*x,d=Math.min(u,S)*s;this.queueParticle(p.x+t,p.y+r,o,c,i,l,d,this.makeSeed(g))}}if(o>0&&(e&&m<.07?this.stillSeconds+=o:this.stillSeconds=0,e&&this.stillSeconds>.22&&i-this.lastStampTime>.35&&(this.stampHeroStar(i,p,s,10,18,1.1,1.8),this.emitDustRing(i,p,s,18),this.lastStampTime=i,this.stillSeconds=.12)),e&&!this.prevActive&&(this.emitRadialBurst(i,p,s,300),this.burstCooldown=.12),o>0&&(this.burstCooldown=Math.max(0,this.burstCooldown-o),e&&this.burstCooldown<=0&&h>.85&&this.mouseRamp>.25&&(this.emitDirectionalBurst(i,p,s,110),this.burstCooldown=.12)),o>0){let e=(120+h*1700)*this.mouseRamp*b;this.sparkAccumulator+=e*o;let t=Math.floor(this.sparkAccumulator);if(this.sparkAccumulator-=t,t=Math.min(t,120),t>0){let e=this.mouseVelocity.x,n=this.mouseVelocity.y,r=Math.hypot(e,n);if(r>1e-5)e/=r,n/=r;else{let t=this.rand01()*Math.PI*2;e=Math.cos(t),n=Math.sin(t)}let a=-n,o=e,c=1.25-h*.9;for(let r=0;r<t;r++){let t=this.rand01(),r=f.x+(p.x-f.x)*t,l=f.y+(p.y-f.y)*t,u=(this.rand01()-.5)*.01,d=a*u,m=o*u,g=(this.rand01()-.5)*c,v=Math.cos(g),b=Math.sin(g),w=e*v-n*b,ee=e*b+n*v,T=.1+this.rand01()*.2+h*.75,E=(this.rand01()-.5)*(.36-h*.18),D=w*T+a*E*T,O=ee*T+o*E*T,k=h>.75&&this.rand01()<C,A=k?y:_,j=k?.45+this.rand01()*.55:.2+this.rand01()*.38+h*.18,M=(k?6+this.rand01()*10+h*10:1.4+this.rand01()*3.4+h*8)*x,N=Math.min(M,S)*s;this.queueParticle(r+d,l+m,D,O,i,j,N,this.makeSeed(A))}}}if(o>0&&r.length>0){let e=(45+160*Math.min(1,r.length/160))*(.25+.75*this.mouseRamp)*b;this.dustAccumulator+=e*o;let t=Math.floor(this.dustAccumulator);this.dustAccumulator-=t,t=Math.min(t,45);for(let e=0;e<t;e++){let e=r[this.rand01()*r.length|0],t=Math.max(0,Math.min(1,e.x/c)),n=Math.max(0,Math.min(1,e.y/l)),a=this.rand01()*Math.PI*2,o=.014+this.rand01()*.03,u=Math.cos(a)*o,d=Math.sin(a)*o,f=.45+this.rand01()*1.2,p=(.9+this.rand01()*2.8)*x,m=Math.min(p,S)*s;this.queueParticle(t,n,u,d,i,f,m,this.makeSeed(g))}}e||(this.mouseVelocity.x*=.95,this.mouseVelocity.y*=.95),!e&&this.prevActive&&(this.stampHeroStar(i,p,s,14,26,1.4,2.4),this.emitDustRing(i,p,s,28)),this.prevActive=e}render(){let e=(performance.now()-this.startTimeMs)/1e3;this.flushPendingParticles(),this.uniformScratch[0]=e,this.uniformScratch[1]=0,this.uniformScratch[2]=this.canvas.width,this.uniformScratch[3]=this.canvas.height,this.uniformScratch[4]=this.mouseRamp,this.uniformScratch[5]=0,this.uniformScratch[6]=0,this.uniformScratch[7]=0,this.device.queue.writeBuffer(this.uniformBuffer,0,this.uniformScratch),this.dotUniformScratch[0]=e,this.dotUniformScratch[1]=0,this.dotUniformScratch[2]=this.canvas.width,this.dotUniformScratch[3]=this.canvas.height,this.dotUniformScratch[4]=1,this.dotUniformScratch[5]=this.mouseRamp,this.dotUniformScratch[6]=this.mousePos.x,this.dotUniformScratch[7]=this.mousePos.y,this.device.queue.writeBuffer(this.dotUniformBuffer,0,this.dotUniformScratch);let t=this.device.createCommandEncoder(),n=t.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:`clear`,storeOp:`store`}]});this.dotCount>0&&(n.setPipeline(this.dotPipeline),n.setBindGroup(0,this.dotBindGroup),n.setVertexBuffer(0,this.quadVertexBuffer),n.setVertexBuffer(1,this.dotBuffer),n.draw(6,this.dotCount,0,0)),n.setPipeline(this.pipeline),n.setBindGroup(0,this.bindGroup),n.setVertexBuffer(0,this.quadVertexBuffer),n.setVertexBuffer(1,this.particleBuffer),n.draw(6,l,0,0),n.end(),this.device.queue.submit([t.finish()])}resize(e,t){this.canvas.width=e,this.canvas.height=t;let n=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device:this.device,format:n,alphaMode:`premultiplied`})}getCanvasMetrics(){let e=Math.max(1,this.canvas.clientWidth||this.canvas.width),t=Math.max(1,this.canvas.clientHeight||this.canvas.height),n=this.canvas.width/e;return{dpr:Math.max(.5,n),cssW:e,cssH:t}}rand01(){let e=this.rngState|0;return e^=e<<13,e^=e>>>17,e^=e<<5,this.rngState=e,(e>>>0)/4294967296}hashSeed(e){let t=(e+1)*2654435761;return t^=t<<13,t^=t>>>17,t^=t<<5,(t>>>0)/4294967296}makeSeed(e){return e+this.rand01()}queueParticle(e,t,n,r,i,a,o,s){if(this.pendingCount>=f)return;this.pendingCount===0&&(this.pendingStart=this.nextParticleIndex),this.nextParticleIndex=(this.nextParticleIndex+1)%l;let c=this.pendingCount*u,d=this.pendingData;d[c+0]=e,d[c+1]=t,d[c+2]=n,d[c+3]=r,d[c+4]=i,d[c+5]=a,d[c+6]=o,d[c+7]=s,this.pendingCount++}flushPendingParticles(){if(this.pendingCount===0)return;let e=this.pendingCount,t=this.pendingStart,n=Math.min(e,l-t),r=n*u;if(this.device.queue.writeBuffer(this.particleBuffer,t*d,this.pendingData.subarray(0,r)),e>n){let t=(e-n)*u;this.device.queue.writeBuffer(this.particleBuffer,0,this.pendingData.subarray(r,r+t))}this.pendingCount=0}emitRadialBurst(e,t,n,r){let i=Math.min(Math.floor(r*b),f);for(let r=0;r<i;r++){let r=this.rand01()*Math.PI*2,i=.14+this.rand01()*.8,a=Math.cos(r)*i,o=Math.sin(r)*i,s=.32+this.rand01()*.75,c=(2.2+this.rand01()*7)*x,l=Math.min(c,S)*n;this.queueParticle(t.x,t.y,a,o,e,s,l,this.makeSeed(_))}}emitDirectionalBurst(e,t,n,r){let i=this.mouseVelocity.x,a=this.mouseVelocity.y,o=Math.hypot(i,a);o>1e-5?(i/=o,a/=o):(i=1,a=0);let s=-a,c=i,l=Math.min(Math.floor(r*b),f);for(let r=0;r<l;r++){let r=(this.rand01()-.5)*.35,o=Math.cos(r),l=Math.sin(r),u=i*o-a*l,d=i*l+a*o,f=.17+this.rand01()*1,p=(this.rand01()-.5)*.25,m=u*f+s*p,h=d*f+c*p,g=.28+this.rand01()*.6,v=(2+this.rand01()*8.5)*x,y=Math.min(v,S)*n;this.queueParticle(t.x,t.y,m,h,e,g,y,this.makeSeed(_))}}stampHeroStar(e,t,n,r,i,a,o){let s=(r+this.rand01()*(i-r))*x,c=Math.min(s,S*1.5)*n,l=a+this.rand01()*(o-a),u=(this.rand01()-.5)*.05,d=-.05-this.rand01()*.07;this.queueParticle(t.x,t.y,u,d,e,l,c,this.makeSeed(v))}emitDustRing(e,t,n,r){let i=Math.min(Math.floor(r*b),30),a=.018;for(let r=0;r<i;r++){let r=this.rand01()*Math.PI*2,i=Math.cos(r)*a*(.6+this.rand01()*.6),o=Math.sin(r)*a*(.6+this.rand01()*.6),s=.03+this.rand01()*.08,c=i*s*30,l=o*s*30,u=.5+this.rand01()*.7,d=(.8+this.rand01()*2)*x,f=Math.min(d,S)*n;this.queueParticle(t.x+i,t.y+o,c,l,e,u,f,this.makeSeed(g))}}emitDotCompleteBurst(e,t,n){this.emitRadialBurst(e,t,n,320),this.emitDustRing(e,t,n,80),this.stampHeroStar(e,t,n,18,28,1.2,2)}},T=class{size;auroraScale=.66;sparkleScale=1;constructor(e,t,n){this.auroraCanvas=e,this.gameCanvas=t,this.sparkleCanvas=n,this.size={width:0,height:0,dpr:1}}resize(){let e=window.innerWidth,t=window.innerHeight,n=Math.max(1,window.devicePixelRatio||1),r=Math.min(1,n),i=Math.max(.5,r*this.auroraScale),a=Math.max(.5,r*this.sparkleScale);return this.applySize(this.auroraCanvas,e,t,i),this.applySize(this.gameCanvas,e,t,n),this.sparkleCanvas&&this.applySize(this.sparkleCanvas,e,t,a),this.size={width:e,height:t,dpr:n},this.size}configureGameContext(e){let{dpr:t}=this.size;e.setTransform(t,0,0,t,0,0),e.imageSmoothingEnabled=!0}getPointFromEvent(e,t){let n=e.getBoundingClientRect();return{x:t.clientX-n.left,y:t.clientY-n.top}}getSize(){return this.size}applySize(e,t,n,r){e.width=Math.max(1,Math.floor(t*r)),e.height=Math.max(1,Math.floor(n*r)),e.style.width=`${t}px`,e.style.height=`${n}px`}},E=class{callbacks={};activePointerId=null;constructor(e,t){this.canvas=e,this.getPoint=t,this.onPointerDown=this.onPointerDown.bind(this),this.onPointerMove=this.onPointerMove.bind(this),this.onPointerUp=this.onPointerUp.bind(this),this.canvas.addEventListener(`pointerdown`,this.onPointerDown),this.canvas.addEventListener(`pointermove`,this.onPointerMove),this.canvas.addEventListener(`pointerup`,this.onPointerUp),this.canvas.addEventListener(`pointercancel`,this.onPointerUp),this.canvas.addEventListener(`pointerleave`,this.onPointerUp)}setCallbacks(e){this.callbacks=e}dispose(){this.canvas.removeEventListener(`pointerdown`,this.onPointerDown),this.canvas.removeEventListener(`pointermove`,this.onPointerMove),this.canvas.removeEventListener(`pointerup`,this.onPointerUp),this.canvas.removeEventListener(`pointercancel`,this.onPointerUp),this.canvas.removeEventListener(`pointerleave`,this.onPointerUp)}onPointerDown(e){if(this.activePointerId!==null)return;this.activePointerId=e.pointerId,this.canvas.setPointerCapture(e.pointerId),e.preventDefault();let t=this.buildPoint(e);this.callbacks.onStart?.(t)}onPointerMove(e){if(this.activePointerId!==e.pointerId)return;e.preventDefault();let t=this.buildPoint(e);this.callbacks.onMove?.(t)}onPointerUp(e){if(this.activePointerId!==e.pointerId)return;e.preventDefault();let t=this.buildPoint(e);this.callbacks.onEnd?.(t),this.canvas.releasePointerCapture(e.pointerId),this.activePointerId=null}buildPoint(e){let{x:t,y:n}=this.getPoint(e);return{x:t,y:n,id:e.pointerId,time:performance.now()/1e3}}},D=t({default:()=>O}),O=[{order:1,x:725,y:320,display:`keystone`,note:`Start at Black Dot`},{order:2,x:660,y:255,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:580,y:215,display:`intermediate`,note:`Curve Up-Left`},{order:4,x:500,y:200,display:`keystone`,note:`Top Arch Center`},{order:5,x:420,y:220,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:350,y:265,display:`intermediate`,note:`Curve Down-Left`},{order:7,x:290,y:340,display:`keystone`,note:`Top Left Shoulder`},{order:8,x:250,y:435,display:`intermediate`,note:`Left Vertical`},{order:9,x:240,y:535,display:`intermediate`,note:`Left Vertical`},{order:10,x:260,y:635,display:`keystone`,note:`Bottom Left Turn`},{order:11,x:320,y:720,display:`intermediate`,note:`Bottom Curve`},{order:12,x:410,y:775,display:`intermediate`,note:`Bottom Curve`},{order:13,x:510,y:780,display:`keystone`,note:`Bottom Center`},{order:14,x:610,y:750,display:`intermediate`,note:`Curve Up-Right`},{order:15,x:680,y:690,display:`intermediate`,note:`Curve Up-Right`},{order:16,x:720,y:610,display:`keystone`,note:`Closing the Loop`},{order:17,x:740,y:530,display:`intermediate`,note:`Loop End`},{order:18,x:810,y:180,display:`intermediate`,newStroke:!0,note:`Lift Pen -> Top of Stem`},{order:19,x:810,y:265,display:`keystone`,note:`Stem Down`},{order:20,x:810,y:350,display:`intermediate`,note:`Stem Down`},{order:21,x:810,y:435,display:`intermediate`,note:`Stem Down`},{order:22,x:810,y:520,display:`keystone`,note:`Stem Center`},{order:23,x:810,y:605,display:`intermediate`,note:`Stem Down`},{order:24,x:810,y:690,display:`intermediate`,note:`Stem Down`},{order:25,x:810,y:775,display:`keystone`,note:`Stem Bottom`},{order:26,x:810,y:860,display:`intermediate`,note:`Stem Finish`}],k=t({default:()=>A}),A=[{order:1,x:240,y:150,display:`keystone`,note:`Start Dot (Top of Stem)`},{order:2,x:240,y:230,display:`intermediate`,note:`Stem Down`},{order:3,x:240,y:310,display:`intermediate`,note:`Stem Down`},{order:4,x:240,y:390,display:`keystone`,note:`Stem Middle`},{order:5,x:240,y:470,display:`intermediate`,note:`Stem Down`},{order:6,x:240,y:550,display:`intermediate`,note:`Stem Down`},{order:7,x:240,y:630,display:`keystone`,note:`Stem Lower`},{order:8,x:240,y:710,display:`intermediate`,note:`Stem Down`},{order:9,x:240,y:790,display:`intermediate`,note:`Stem Down`},{order:10,x:240,y:870,display:`keystone`,note:`Stem Bottom Arrow`},{order:11,x:240,y:450,display:`intermediate`,newStroke:!0,note:`Lift Pen -> Start Loop (Mid-Stem)`},{order:12,x:290,y:390,display:`intermediate`,note:`Loop Curve Up-Right`},{order:13,x:380,y:350,display:`keystone`,note:`Loop Top Left`},{order:14,x:490,y:335,display:`intermediate`,note:`Loop Top Center`},{order:15,x:610,y:350,display:`intermediate`,note:`Loop Top Right`},{order:16,x:710,y:410,display:`keystone`,note:`Loop Top Curve`},{order:17,x:780,y:500,display:`intermediate`,note:`Loop Far Right`},{order:18,x:800,y:600,display:`intermediate`,note:`Loop Far Right Mid`},{order:19,x:780,y:700,display:`keystone`,note:`Loop Bottom Right Turn`},{order:20,x:710,y:790,display:`intermediate`,note:`Loop Bottom Curve`},{order:21,x:610,y:845,display:`intermediate`,note:`Loop Bottom Right`},{order:22,x:500,y:870,display:`keystone`,note:`Loop Bottom Center`},{order:23,x:390,y:860,display:`intermediate`,note:`Loop Bottom Left`},{order:24,x:310,y:820,display:`intermediate`,note:`Loop Closing In`},{order:25,x:260,y:850,display:`keystone`,note:`Loop Connect to Stem`},{order:26,x:240,y:870,display:`intermediate`,note:`Finish`}],j=t({default:()=>M}),M=[{order:1,x:760,y:230,display:`keystone`,note:`Start Black Dot`},{order:2,x:700,y:190,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:635,y:165,display:`intermediate`,note:`Curve Up-Left`},{order:4,x:565,y:150,display:`keystone`,note:`Top Arch Right`},{order:5,x:495,y:150,display:`intermediate`,note:`Top Arch Center`},{order:6,x:425,y:165,display:`intermediate`,note:`Top Arch Left`},{order:7,x:360,y:195,display:`keystone`,note:`Curve Down-Left`},{order:8,x:300,y:240,display:`intermediate`,note:`Curve Down-Left`},{order:9,x:250,y:300,display:`intermediate`,note:`Upper Left`},{order:10,x:210,y:370,display:`keystone`,note:`Left Curve High`},{order:11,x:180,y:450,display:`intermediate`,note:`Left Curve Mid`},{order:12,x:170,y:530,display:`intermediate`,note:`Left Vertical Center`},{order:13,x:180,y:610,display:`keystone`,note:`Left Curve Low`},{order:14,x:210,y:690,display:`intermediate`,note:`Bottom Left Turn`},{order:15,x:260,y:760,display:`intermediate`,note:`Bottom Left Curve`},{order:16,x:320,y:810,display:`keystone`,note:`Bottom Arch Left`},{order:17,x:390,y:845,display:`intermediate`,note:`Bottom Arch Mid-Left`},{order:18,x:470,y:860,display:`intermediate`,note:`Bottom Arch Center`},{order:19,x:550,y:855,display:`keystone`,note:`Bottom Arch Right`},{order:20,x:620,y:830,display:`intermediate`,note:`Curve Up-Right`},{order:21,x:680,y:790,display:`intermediate`,note:`Curve Up-Right`},{order:22,x:730,y:740,display:`keystone`,note:`Lower Right Turn`},{order:26,x:780,y:660,display:`intermediate`,note:`Finish Arrow Tip`}],N=t({default:()=>P}),P=[{order:1,x:640,y:460,display:`keystone`,note:`Start Black Dot`},{order:2,x:580,y:380,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:500,y:330,display:`intermediate`,note:`Curve Up-Left`},{order:4,x:400,y:310,display:`keystone`,note:`Top of Loop`},{order:5,x:300,y:340,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:220,y:420,display:`intermediate`,note:`Curve Down-Left`},{order:7,x:160,y:550,display:`keystone`,note:`Leftmost Curve`},{order:8,x:200,y:680,display:`intermediate`,note:`Curve Down-Right`},{order:9,x:280,y:760,display:`intermediate`,note:`Curve Down-Right`},{order:10,x:400,y:800,display:`keystone`,note:`Bottom of Loop`},{order:11,x:520,y:780,display:`intermediate`,note:`Curve Up-Right`},{order:12,x:650,y:720,display:`intermediate`,note:`Approach Stem`},{order:13,x:780,y:620,display:`keystone`,note:`Connect to Stem (Go Up)`},{order:14,x:780,y:520,display:`intermediate`,note:`Stem Up`},{order:15,x:780,y:420,display:`intermediate`,note:`Stem Up`},{order:16,x:780,y:320,display:`keystone`,note:`Stem Up High`},{order:17,x:780,y:200,display:`intermediate`,note:`Stem Up`},{order:18,x:780,y:100,display:`intermediate`,note:`Reach Top`},{order:19,x:780,y:90,display:`keystone`,note:`Top Turnaround`},{order:20,x:780,y:250,display:`intermediate`,note:`Stem Down (Retrace)`},{order:21,x:780,y:400,display:`intermediate`,note:`Stem Down`},{order:22,x:780,y:550,display:`keystone`,note:`Stem Down Mid`},{order:23,x:780,y:700,display:`intermediate`,note:`Stem Down`},{order:24,x:780,y:800,display:`intermediate`,note:`Stem Down`},{order:25,x:780,y:900,display:`keystone`,note:`Stem Bottom`},{order:26,x:780,y:930,display:`intermediate`,note:`Finish Arrow`}],F=t({default:()=>te}),te=[{order:1,x:220,y:440,display:`keystone`,note:`Start Black Dot`},{order:2,x:330,y:440,display:`intermediate`,note:`Horizontal Bar`},{order:3,x:440,y:440,display:`intermediate`,note:`Horizontal Bar`},{order:4,x:550,y:440,display:`keystone`,note:`Center of Bar`},{order:5,x:660,y:440,display:`intermediate`,note:`Horizontal Bar`},{order:6,x:770,y:440,display:`intermediate`,note:`End of Bar (Turn Up)`},{order:7,x:760,y:360,display:`keystone`,note:`Curve Up-Left`},{order:8,x:700,y:280,display:`intermediate`,note:`Curve Up-Left`},{order:9,x:620,y:220,display:`intermediate`,note:`Top Right Curve`},{order:10,x:500,y:190,display:`keystone`,note:`Top Center`},{order:11,x:380,y:200,display:`intermediate`,note:`Top Left Curve`},{order:12,x:280,y:250,display:`intermediate`,note:`Curve Down-Left`},{order:13,x:200,y:330,display:`keystone`,note:`Top Left Shoulder`},{order:14,x:150,y:430,display:`intermediate`,note:`Left Spine High`},{order:15,x:140,y:530,display:`intermediate`,note:`Left Spine Mid`},{order:16,x:160,y:630,display:`keystone`,note:`Bottom Left Turn`},{order:17,x:210,y:720,display:`intermediate`,note:`Curve Down-Right`},{order:18,x:290,y:790,display:`intermediate`,note:`Curve Down-Right`},{order:19,x:390,y:840,display:`keystone`,note:`Bottom Center`},{order:20,x:500,y:850,display:`intermediate`,note:`Bottom Right`},{order:21,x:600,y:820,display:`intermediate`,note:`Curve Up-Right`},{order:22,x:680,y:770,display:`keystone`,note:`Tail Curve`},{order:23,x:730,y:720,display:`intermediate`,note:`Approach Arrow`},{order:24,x:750,y:690,display:`intermediate`,note:`Arrow Tail`},{order:25,x:760,y:670,display:`keystone`,note:`Arrow Head`},{order:26,x:770,y:650,display:`intermediate`,note:`Finish Tip`}],I=t({default:()=>L}),L=[{order:1,x:620,y:140,display:`keystone`,note:`Start Black Dot (Under Arch)`},{order:2,x:560,y:110,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:500,y:90,display:`intermediate`,note:`Top Arch Right`},{order:4,x:440,y:90,display:`keystone`,note:`Top Arch Center`},{order:5,x:390,y:130,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:380,y:190,display:`intermediate`,note:`Join Vertical Stem`},{order:7,x:380,y:270,display:`keystone`,note:`Stem High`},{order:8,x:380,y:360,display:`intermediate`,note:`Stem (Crossing Point)`},{order:9,x:380,y:450,display:`intermediate`,note:`Stem Middle`},{order:10,x:380,y:540,display:`keystone`,note:`Stem Mid-Low`},{order:11,x:380,y:630,display:`intermediate`,note:`Stem Lower`},{order:12,x:380,y:720,display:`intermediate`,note:`Stem Lower`},{order:13,x:380,y:810,display:`keystone`,note:`Stem Bottom Approach`},{order:14,x:380,y:860,display:`intermediate`,note:`Stem Arrow Start`},{order:15,x:380,y:900,display:`intermediate`,note:`Stem Arrow Body`},{order:16,x:380,y:930,display:`keystone`,note:`Stem Arrow Tip`},{order:17,x:380,y:940,display:`intermediate`,note:`Finish Stroke 1`},{order:18,x:380,y:940,display:`intermediate`,note:`Pen Lift Point`},{order:19,x:180,y:360,display:`keystone`,newStroke:!0,note:`Start Stroke 2 (The 'X')`},{order:20,x:250,y:360,display:`intermediate`,note:`Crossbar Left`},{order:21,x:320,y:360,display:`intermediate`,note:`Crossbar Approach`},{order:22,x:390,y:360,display:`keystone`,note:`Crossbar Center (After Stem)`},{order:23,x:460,y:360,display:`intermediate`,note:`Crossbar Right`},{order:24,x:530,y:360,display:`intermediate`,note:`Crossbar Right`},{order:25,x:600,y:360,display:`keystone`,note:`Crossbar Arrow Start`},{order:26,x:660,y:360,display:`intermediate`,note:`Finish Crossbar Arrow`}],R=t({default:()=>z}),z=[{order:1,x:710,y:240,display:`keystone`,note:`Start Black Dot`},{order:2,x:630,y:180,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:520,y:150,display:`intermediate`,note:`Top Curve`},{order:4,x:410,y:160,display:`keystone`,note:`Top Center`},{order:5,x:300,y:220,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:210,y:320,display:`intermediate`,note:`Left Curve High`},{order:7,x:180,y:440,display:`keystone`,note:`Left Side Center`},{order:8,x:210,y:560,display:`intermediate`,note:`Left Curve Low`},{order:9,x:300,y:640,display:`intermediate`,note:`Bottom Curve Left`},{order:10,x:430,y:660,display:`keystone`,note:`Bottom of Loop`},{order:11,x:580,y:620,display:`intermediate`,note:`Curve Up-Right`},{order:12,x:700,y:520,display:`intermediate`,note:`Approach Stem`},{order:13,x:780,y:400,display:`keystone`,note:`Connect to Stem`},{order:14,x:780,y:300,display:`intermediate`,note:`Stem Up`},{order:15,x:780,y:200,display:`intermediate`,note:`Stem Up`},{order:16,x:780,y:120,display:`keystone`,note:`Top of Stem`},{order:17,x:780,y:300,display:`intermediate`,note:`Stem Down (Retrace)`},{order:18,x:780,y:500,display:`intermediate`,note:`Stem Down Middle`},{order:19,x:780,y:700,display:`keystone`,note:`Descender Start`},{order:20,x:780,y:820,display:`intermediate`,note:`Descender Down`},{order:21,x:720,y:900,display:`intermediate`,note:`Tail Curve Start`},{order:22,x:600,y:940,display:`keystone`,note:`Tail Bottom Center`},{order:23,x:480,y:940,display:`intermediate`,note:`Tail Curve Left`},{order:24,x:360,y:900,display:`intermediate`,note:`Tail Curve Up`},{order:25,x:260,y:840,display:`keystone`,note:`Tail Arrow Head`},{order:26,x:200,y:800,display:`intermediate`,note:`Finish Tip`}],B=t({default:()=>V}),V=[{order:1,x:220,y:120,display:`keystone`,note:`Start Black Dot`},{order:2,x:220,y:230,display:`intermediate`,note:`Stem Down`},{order:3,x:220,y:310,display:`intermediate`,note:`Stem Down`},{order:4,x:220,y:390,display:`keystone`,note:`Stem Upper Middle`},{order:5,x:220,y:470,display:`intermediate`,note:`Stem Middle`},{order:6,x:220,y:550,display:`intermediate`,note:`Stem Lower Middle`},{order:7,x:220,y:630,display:`keystone`,note:`Stem Low`},{order:8,x:220,y:710,display:`intermediate`,note:`Stem Lower`},{order:9,x:220,y:790,display:`intermediate`,note:`Stem Approach Bottom`},{order:10,x:220,y:870,display:`keystone`,note:`Stem Left Arrow`},{order:11,x:220,y:500,display:`intermediate`,newStroke:!0,note:`Branch Point (Mid-Stem)`},{order:12,x:280,y:420,display:`intermediate`,note:`Arch Up-Right`},{order:13,x:360,y:360,display:`keystone`,note:`Arch Left Shoulder`},{order:14,x:460,y:330,display:`intermediate`,note:`Arch Top Center`},{order:15,x:560,y:330,display:`intermediate`,note:`Arch Top Right`},{order:16,x:660,y:360,display:`keystone`,note:`Arch Right Shoulder`},{order:17,x:740,y:420,display:`intermediate`,note:`Arch Curve Down`},{order:18,x:780,y:500,display:`intermediate`,note:`Right Leg Start`},{order:19,x:780,y:580,display:`keystone`,note:`Right Leg Mid-High`},{order:20,x:780,y:650,display:`intermediate`,note:`Right Leg Middle`},{order:21,x:780,y:720,display:`intermediate`,note:`Right Leg Low`},{order:22,x:780,y:790,display:`keystone`,note:`Right Leg Lower`},{order:23,x:780,y:830,display:`intermediate`,note:`Approach Arrow`},{order:24,x:780,y:850,display:`intermediate`,note:`Arrow Tail`},{order:25,x:780,y:870,display:`keystone`,note:`Right Arrow Tip`},{order:26,x:780,y:890,display:`intermediate`,note:`Finish`}],ne=t({default:()=>re}),re=[{order:1,x:500,y:350,display:`keystone`,note:`Start Black Dot (Stem Top)`},{order:2,x:500,y:420,display:`intermediate`,note:`Stem Down`},{order:3,x:500,y:490,display:`intermediate`,note:`Stem Down`},{order:4,x:500,y:560,display:`keystone`,note:`Stem Middle`},{order:5,x:500,y:630,display:`intermediate`,note:`Stem Down`},{order:6,x:500,y:700,display:`intermediate`,note:`Stem Lower`},{order:7,x:500,y:770,display:`keystone`,note:`Begin Turn`},{order:8,x:510,y:830,display:`intermediate`,note:`Curve Right`},{order:9,x:535,y:875,display:`intermediate`,note:`Curve Bottom`},{order:10,x:570,y:900,display:`keystone`,note:`Hook Upwards`},{order:11,x:620,y:910,display:`intermediate`,note:`Finish Hook`},{order:12,x:500,y:150,display:`keystone`,newStroke:!0,note:`The Dot (Top X)`}],H=t({default:()=>U}),U=[{order:1,x:550,y:280,display:`keystone`,note:`Start Black Dot`},{order:2,x:550,y:380,display:`intermediate`,note:`Stem Down`},{order:3,x:550,y:480,display:`intermediate`,note:`Stem Down`},{order:4,x:550,y:580,display:`keystone`,note:`Stem Middle`},{order:5,x:550,y:680,display:`intermediate`,note:`Stem Down`},{order:6,x:550,y:760,display:`intermediate`,note:`Start Turn`},{order:7,x:530,y:840,display:`keystone`,note:`Curve Left`},{order:8,x:490,y:900,display:`intermediate`,note:`Curve Bottom Right`},{order:9,x:430,y:930,display:`intermediate`,note:`Curve Bottom Center`},{order:10,x:360,y:920,display:`keystone`,note:`Curve Bottom Left`},{order:11,x:310,y:890,display:`intermediate`,note:`Arrow Approach`},{order:12,x:280,y:870,display:`intermediate`,note:`Finish Arrow Tip`},{order:13,x:550,y:140,display:`keystone`,newStroke:!0,note:`The Dot (Top X)`}],ie=t({default:()=>W}),W=[{order:1,x:220,y:180,display:`keystone`,note:`Start Black Dot (Top of Stem)`},{order:2,x:220,y:360,display:`intermediate`,note:`Stem Down`},{order:3,x:220,y:540,display:`intermediate`,note:`Stem Middle`},{order:4,x:220,y:720,display:`keystone`,note:`Stem Lower`},{order:5,x:220,y:900,display:`intermediate`,note:`Stem Finish Arrow`},{order:6,x:680,y:380,display:`keystone`,newStroke:!0,note:`Start 'X' (Top Right)`},{order:7,x:550,y:460,display:`intermediate`,note:`Diagonal In`},{order:8,x:420,y:540,display:`intermediate`,note:`Diagonal In`},{order:9,x:300,y:620,display:`keystone`,note:`The Turn (Touch Stem)`},{order:10,x:410,y:690,display:`intermediate`,note:`Diagonal Out`},{order:11,x:520,y:760,display:`intermediate`,note:`Diagonal Out`},{order:12,x:630,y:830,display:`intermediate`,note:`Diagonal Out`},{order:13,x:740,y:900,display:`keystone`,note:`Leg Finish Arrow`}],ae=t({default:()=>oe}),oe=[{order:1,x:180,y:200,display:`keystone`,note:`Start Black Dot (Top Left)`},{order:2,x:180,y:340,display:`intermediate`,note:`Left Stem Down`},{order:3,x:180,y:480,display:`intermediate`,note:`Left Stem Middle`},{order:4,x:180,y:620,display:`keystone`,note:`Left Stem Lower`},{order:5,x:180,y:760,display:`intermediate`,note:`Left Stem Down`},{order:6,x:180,y:900,display:`intermediate`,note:`Left Stem Finish Arrow`},{order:7,x:180,y:380,display:`keystone`,newStroke:!0,note:`Branch Point 1 (Start Middle Arch)`},{order:8,x:240,y:260,display:`intermediate`,note:`Curve Up-Right`},{order:9,x:340,y:210,display:`intermediate`,note:`Arch 1 Top`},{order:10,x:440,y:260,display:`keystone`,note:`Arch 1 Down-Right`},{order:11,x:500,y:380,display:`intermediate`,note:`Middle Stem Start`},{order:12,x:500,y:550,display:`intermediate`,note:`Middle Stem Down`},{order:13,x:500,y:720,display:`keystone`,note:`Middle Stem Lower`},{order:14,x:500,y:850,display:`intermediate`,note:`Middle Stem Approach`},{order:15,x:500,y:900,display:`intermediate`,note:`Middle Stem Finish Arrow`},{order:16,x:500,y:380,display:`keystone`,newStroke:!0,note:`Branch Point 2 (Start Right Arch)`},{order:17,x:560,y:260,display:`intermediate`,note:`Curve Up-Right`},{order:18,x:660,y:210,display:`intermediate`,note:`Arch 2 Top`},{order:19,x:760,y:260,display:`keystone`,note:`Arch 2 Down-Right`},{order:20,x:820,y:380,display:`intermediate`,note:`Right Stem Start`},{order:21,x:820,y:510,display:`intermediate`,note:`Right Stem Down`},{order:22,x:820,y:640,display:`keystone`,note:`Right Stem Middle`},{order:23,x:820,y:770,display:`intermediate`,note:`Right Stem Lower`},{order:24,x:820,y:850,display:`intermediate`,note:`Right Stem Approach`},{order:25,x:820,y:890,display:`keystone`,note:`Right Stem Arrow Body`},{order:26,x:820,y:910,display:`intermediate`,note:`Right Stem Finish`}],se=t({default:()=>ce}),ce=[{order:1,x:250,y:200,display:`keystone`,note:`Start Black Dot (Top Left)`},{order:2,x:250,y:340,display:`intermediate`,note:`Left Stem Down`},{order:3,x:250,y:480,display:`intermediate`,note:`Left Stem Middle`},{order:4,x:250,y:620,display:`keystone`,note:`Left Stem Lower`},{order:5,x:250,y:760,display:`intermediate`,note:`Left Stem Down`},{order:6,x:250,y:900,display:`intermediate`,note:`Left Stem Finish Arrow`},{order:7,x:250,y:450,display:`keystone`,newStroke:!0,note:`Branch Point (Start Arch)`},{order:8,x:300,y:320,display:`intermediate`,note:`Curve Up-Right`},{order:9,x:400,y:240,display:`intermediate`,note:`Arch Top Left`},{order:10,x:500,y:210,display:`keystone`,note:`Arch Top Center`},{order:11,x:600,y:240,display:`intermediate`,note:`Arch Top Right`},{order:12,x:700,y:320,display:`intermediate`,note:`Curve Down-Right`},{order:13,x:750,y:450,display:`keystone`,note:`Right Stem Start`},{order:14,x:750,y:600,display:`intermediate`,note:`Right Stem Down`},{order:15,x:750,y:750,display:`intermediate`,note:`Right Stem Approach`},{order:16,x:750,y:900,display:`keystone`,note:`Right Stem Finish Arrow`}],le=t({default:()=>ue}),ue=[{order:1,x:740,y:260,display:`keystone`,note:`Start Black Dot`},{order:2,x:670,y:200,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:590,y:160,display:`intermediate`,note:`Curve Up-Left`},{order:4,x:500,y:150,display:`keystone`,note:`Top Center`},{order:5,x:410,y:160,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:330,y:200,display:`intermediate`,note:`Curve Down-Left`},{order:7,x:260,y:260,display:`keystone`,note:`Top Left Curve`},{order:8,x:200,y:330,display:`intermediate`,note:`Left Side High`},{order:9,x:160,y:410,display:`intermediate`,note:`Left Side Mid`},{order:10,x:150,y:500,display:`keystone`,note:`Left Center`},{order:11,x:160,y:590,display:`intermediate`,note:`Left Side Low`},{order:12,x:200,y:670,display:`intermediate`,note:`Bottom Left Turn`},{order:13,x:260,y:740,display:`keystone`,note:`Bottom Left Curve`},{order:14,x:330,y:800,display:`intermediate`,note:`Curve Down-Right`},{order:15,x:410,y:840,display:`intermediate`,note:`Bottom Curve`},{order:16,x:500,y:850,display:`keystone`,note:`Bottom Center`},{order:17,x:590,y:840,display:`intermediate`,note:`Bottom Curve Right`},{order:18,x:670,y:800,display:`intermediate`,note:`Curve Up-Right`},{order:19,x:740,y:740,display:`keystone`,note:`Bottom Right Curve`},{order:20,x:800,y:670,display:`intermediate`,note:`Right Side Low`},{order:21,x:840,y:590,display:`intermediate`,note:`Right Side Mid`},{order:22,x:850,y:500,display:`keystone`,note:`Right Center`},{order:23,x:840,y:410,display:`intermediate`,note:`Right Side High`},{order:24,x:800,y:330,display:`intermediate`,note:`Approach Top`},{order:25,x:760,y:290,display:`keystone`,note:`Close Loop`},{order:26,x:730,y:280,display:`intermediate`,note:`Finish Arrow Tip`}],de=t({default:()=>fe}),fe=[{order:1,x:230,y:160,display:`keystone`,note:`Start Black Dot (Top Left)`},{order:2,x:230,y:260,display:`intermediate`,note:`Stem Down`},{order:3,x:230,y:360,display:`intermediate`,note:`Stem Down`},{order:4,x:230,y:460,display:`keystone`,note:`Stem Middle`},{order:5,x:230,y:560,display:`intermediate`,note:`Stem Down`},{order:6,x:230,y:660,display:`intermediate`,note:`Stem Down`},{order:7,x:230,y:760,display:`keystone`,note:`Stem Lower`},{order:8,x:230,y:880,display:`intermediate`,note:`Stem Finish Arrow`},{order:9,x:230,y:340,display:`keystone`,newStroke:!0,note:`Branch Point (Start Lobe)`},{order:10,x:280,y:240,display:`intermediate`,note:`Curve Up-Right`},{order:11,x:380,y:180,display:`intermediate`,note:`Top Curve Left`},{order:12,x:500,y:150,display:`keystone`,note:`Lobe Top Center`},{order:13,x:620,y:160,display:`intermediate`,note:`Top Curve Right`},{order:14,x:720,y:220,display:`intermediate`,note:`Curve Down-Right`},{order:15,x:800,y:320,display:`keystone`,note:`Right Side High`},{order:16,x:840,y:420,display:`intermediate`,note:`Right Side Middle`},{order:17,x:840,y:520,display:`intermediate`,note:`Right Side Low`},{order:18,x:800,y:620,display:`keystone`,note:`Bottom Right Curve`},{order:19,x:720,y:700,display:`intermediate`,note:`Curve Left-Down`},{order:20,x:620,y:740,display:`intermediate`,note:`Bottom Curve Right`},{order:21,x:500,y:750,display:`keystone`,note:`Lobe Bottom Center`},{order:22,x:380,y:740,display:`intermediate`,note:`Bottom Curve Left`},{order:23,x:320,y:710,display:`intermediate`,note:`Approach Stem`},{order:24,x:280,y:680,display:`keystone`,note:`Close Lobe Arrow`},{order:25,x:250,y:660,display:`intermediate`,note:`Connect`},{order:26,x:230,y:640,display:`intermediate`,note:`Join Stem`}],pe={"./assets/letters_json/a.json":D,"./assets/letters_json/b.json":k,"./assets/letters_json/c.json":j,"./assets/letters_json/d.json":N,"./assets/letters_json/e.json":F,"./assets/letters_json/f.json":I,"./assets/letters_json/g.json":R,"./assets/letters_json/h.json":B,"./assets/letters_json/i.json":ne,"./assets/letters_json/j.json":H,"./assets/letters_json/k.json":ie,"./assets/letters_json/m.json":ae,"./assets/letters_json/n.json":se,"./assets/letters_json/o.json":le,"./assets/letters_json/p.json":de},G=Object.entries(pe).map(([e,t])=>{let n=t.default;return Array.isArray(n)?{name:(e.split(`/`).pop()??`unknown`).replace(/\.json$/i,``),points:n}:null}).filter(e=>e!==null),me=()=>{if(G.length===0)return{name:`unknown`,points:[]};let e=G[Math.floor(Math.random()*G.length)];return console.info(`Selected letter:`,e.name),{name:e.name,points:e.points.map(e=>({order:e.order,x:e.x,y:e.y})).sort((e,t)=>e.order-t.order)}},he=e=>{if(e.length===0)return{minX:0,maxX:0,minY:0,maxY:0};let t=e[0].x,n=e[0].x,r=e[0].y,i=e[0].y;for(let a of e)a.x<t&&(t=a.x),a.x>n&&(n=a.x),a.y<r&&(r=a.y),a.y>i&&(i=a.y);return{minX:t,maxX:n,minY:r,maxY:i}},ge=3.5,_e=e=>{let t=[...e].sort((e,t)=>e-t),n=Math.floor(t.length/2);return t.length%2==0?(t[n-1]+t[n])/2:t[n]},ve=e=>{if(e.length<2)return 1/0;let t=[];for(let n=0;n<e.length-1;n+=1){let r=e[n+1].x-e[n].x,i=e[n+1].y-e[n].y;t.push(Math.hypot(r,i))}let n=t.filter(e=>e>0),r=n.length>0?n:t;return r.length===0?1/0:_e(r)*ge},K=(e,t)=>{let n=[];if(e.length<2)return n;for(let r=0;r<e.length-1;r+=1){let i=e[r+1].x-e[r].x,a=e[r+1].y-e[r].y;Math.hypot(i,a)>t||n.push({from:r,to:r+1})}return n},ye=class{plotPoints=[];paths=[];currentPath=null;fadeSeconds=9;maxPoints=600;isDrawing=!1;currentMousePos={x:0,y:0};plotBounds={minX:0,maxX:0,minY:0,maxY:0};scaledPlotPoints=[];lastPlotSize={width:0,height:0};needsRescale=!0;currentTargetIndex=0;dotRadius=20;hitRadiusScale=1.5;penUpDistanceThreshold=1/0;lineSegmentIndex=0;lineSegmentT=0;linePauseRemaining=0;lineSegmentSeconds=.123;lineLoopPauseSeconds=.4;completionMessageUntil=0;pendingLetterReset=!1;completionMessageSeconds=2;callbacks;currentLetterName=`unknown`;constructor(e,t={}){let n=me();this.plotPoints=n.points,this.plotBounds=he(this.plotPoints),this.callbacks=t,this.currentLetterName=n.name,this.callbacks.onLetterChange?.(this.currentLetterName),e.setCallbacks({onStart:e=>this.startPath(e),onMove:e=>this.extendPath(e),onEnd:e=>this.endPath(e)})}getDrawingState(){let e=this.paths.flatMap(e=>e);return{isDrawing:this.isDrawing,mouseX:this.currentMousePos.x,mouseY:this.currentMousePos.y,trailPoints:e}}getDotState(){return{dots:this.scaledPlotPoints,currentIndex:this.currentTargetIndex,radius:this.dotRadius}}clearUserPaths(){this.paths=[],this.currentPath=null,this.isDrawing=!1}resetCurrentLetter(){this.clearUserPaths(),this.currentTargetIndex=0,this.lineSegmentIndex=0,this.lineSegmentT=0,this.linePauseRemaining=0,this.pendingLetterReset=!1,this.completionMessageUntil=0}setViewportSize(e,t){if(this.plotPoints.length===0)return;let n=Math.max(1,e),r=Math.max(1,t);if(n===this.lastPlotSize.width&&r===this.lastPlotSize.height&&!this.needsRescale)return;this.lastPlotSize={width:n,height:r},this.needsRescale=!1;let i=this.plotBounds.maxX-this.plotBounds.minX,a=this.plotBounds.maxY-this.plotBounds.minY;if(i<=0||a<=0)return;let o=n*.5,s=r*.5,c=Math.min(o/i,s/a),l=(n-i*c)/2-this.plotBounds.minX*c,u=(r-a*c)/2-this.plotBounds.minY*c;this.scaledPlotPoints=this.plotPoints.map(e=>({order:e.order,x:e.x*c+l,y:e.y*c+u})),this.penUpDistanceThreshold=ve(this.scaledPlotPoints)}update(e){this.advancePlotAnimation(e);let t=performance.now()/1e3;if(this.paths.length>0){let e=t-this.fadeSeconds;for(let t of this.paths){let n=0;for(;n<t.length&&t[n].time<e;)n+=1;n>0&&t.splice(0,n),t.length>this.maxPoints&&t.splice(0,t.length-this.maxPoints)}this.paths=this.paths.filter(e=>e.length>0)}this.pendingLetterReset&&t>=this.completionMessageUntil&&(this.resetForNewLetter(),this.pendingLetterReset=!1,this.completionMessageUntil=0)}render(e){if(e.save(),e.setTransform(1,0,0,1,0,0),e.clearRect(0,0,e.canvas.width,e.canvas.height),e.restore(),this.renderPlotLines(e),this.paths.length>0){e.strokeStyle=`rgba(200, 230, 255, 0.9)`,e.lineWidth=15,e.lineJoin=`round`,e.lineCap=`round`;for(let t of this.paths)t.length!==0&&(e.beginPath(),t.forEach((t,n)=>{n===0?e.moveTo(t.x,t.y):e.lineTo(t.x,t.y)}),e.stroke())}this.renderCompletionMessage(e)}startPath(e){this.pendingLetterReset||(this.isDrawing=!0,this.currentMousePos={x:e.x,y:e.y},this.currentPath=[{x:e.x,y:e.y,time:e.time}],this.paths.push(this.currentPath),this.tryAdvanceTarget(e))}extendPath(e){this.pendingLetterReset||(this.currentMousePos={x:e.x,y:e.y},this.currentPath&&(this.currentPath.push({x:e.x,y:e.y,time:e.time}),this.tryAdvanceTarget(e)))}endPath(e){this.pendingLetterReset||(this.currentMousePos={x:e.x,y:e.y},this.currentPath&&this.currentPath.push({x:e.x,y:e.y,time:e.time}),this.tryAdvanceTarget(e),this.isDrawing=!1,this.currentPath=null)}renderPlotLines(e){if(this.scaledPlotPoints.length<2)return;e.strokeStyle=`rgba(255, 220, 0, 0.5)`,e.lineWidth=9,e.lineJoin=`round`,e.lineCap=`round`,e.beginPath();let t=Math.min(Math.max(0,this.currentTargetIndex-1),this.scaledPlotPoints.length-1),n=this.scaledPlotPoints.slice(t),r=K(n,this.penUpDistanceThreshold);if(r.length!==0){if(this.linePauseRemaining>0)this.drawLineSegments(e,n,r,r.length-1);else{let t=Math.min(this.lineSegmentIndex,r.length-1);this.drawLineSegments(e,n,r,t-1),this.drawPartialLineSegment(e,n,r,t,this.lineSegmentT)}e.stroke()}}drawLineSegments(e,t,n,r){if(r<0)return;let i=Math.min(r,n.length-1),a=null;for(let r=0;r<=i;r+=1){let i=n[r],o=t[i.from],s=t[i.to];(!a||a.to!==i.from)&&e.moveTo(o.x,o.y),e.lineTo(s.x,s.y),a=i}}drawPartialLineSegment(e,t,n,r,i){if(r<0||r>=n.length)return;let a=n[r],o=t[a.from],s=t[a.to],c=r>0?n[r-1]:null;(!c||c.to!==a.from)&&e.moveTo(o.x,o.y);let l=o.x+(s.x-o.x)*i,u=o.y+(s.y-o.y)*i;e.lineTo(l,u)}tryAdvanceTarget(e){if(!this.isDrawing||this.scaledPlotPoints.length===0||this.currentTargetIndex>=this.scaledPlotPoints.length)return;let t=this.scaledPlotPoints[this.currentTargetIndex],n=e.x-t.x,r=e.y-t.y,i=this.getTargetDotRadius()*this.hitRadiusScale;n*n+r*r<=i*i&&(this.currentTargetIndex=Math.min(this.currentTargetIndex+1,this.scaledPlotPoints.length),this.callbacks.onPointHit?.(),this.lineSegmentIndex=0,this.lineSegmentT=0,this.linePauseRemaining=0,this.currentTargetIndex>=this.scaledPlotPoints.length&&this.handleLetterComplete())}getTargetDotRadius(){return Math.max(10,this.dotRadius*1.3)*1.45}advancePlotAnimation(e){if(e<=0||this.scaledPlotPoints.length<2)return;let t=Math.min(Math.max(0,this.currentTargetIndex-1),this.scaledPlotPoints.length-1),n=K(this.scaledPlotPoints.slice(t),this.penUpDistanceThreshold).length;if(n!==0){if(this.lineSegmentIndex=Math.min(this.lineSegmentIndex,n-1),this.linePauseRemaining>0){if(this.linePauseRemaining=Math.max(0,this.linePauseRemaining-e),this.linePauseRemaining>0)return;this.lineSegmentIndex=0,this.lineSegmentT=0}for(this.lineSegmentT+=e/this.lineSegmentSeconds;this.lineSegmentT>=1;)if(--this.lineSegmentT,this.lineSegmentIndex+=1,this.lineSegmentIndex>=n){this.linePauseRemaining=this.lineLoopPauseSeconds,this.lineSegmentIndex=0,this.lineSegmentT=0;break}}}handleLetterComplete(){this.pendingLetterReset||(this.completionMessageUntil=performance.now()/1e3+this.completionMessageSeconds,this.pendingLetterReset=!0,this.isDrawing=!1,this.currentPath=null)}resetForNewLetter(){let e=me();this.plotPoints=e.points,this.plotBounds=he(this.plotPoints),this.currentLetterName=e.name,this.callbacks.onLetterChange?.(this.currentLetterName),this.scaledPlotPoints=[],this.currentTargetIndex=0,this.lineSegmentIndex=0,this.lineSegmentT=0,this.linePauseRemaining=0,this.paths=[],this.currentPath=null,this.isDrawing=!1,this.needsRescale=!0,this.lastPlotSize.width>0&&this.lastPlotSize.height>0&&this.setViewportSize(this.lastPlotSize.width,this.lastPlotSize.height)}renderCompletionMessage(e){if(performance.now()/1e3>=this.completionMessageUntil)return;let t=e.canvas.width/(window.devicePixelRatio||1),n=e.canvas.height/(window.devicePixelRatio||1);e.save(),e.textAlign=`center`,e.textBaseline=`middle`,e.fillStyle=`rgba(255, 244, 210, 0.95)`,e.strokeStyle=`rgba(0, 0, 0, 0.45)`,e.lineWidth=6,e.font=`700 48px system-ui, -apple-system, sans-serif`;let r=`Well done!`;e.strokeText(r,t/2,n*.45),e.fillText(r,t/2,n*.45),e.restore()}},be=t({default:()=>xe}),xe=`/letter_constellations/assets/accordion-CyoQk0r2.webp`,Se=t({default:()=>Ce}),Ce=`/letter_constellations/assets/acorn-B-P3IZra.webp`,we=t({default:()=>Te}),Te=`/letter_constellations/assets/airplane-DZQqSBvK.webp`,Ee=t({default:()=>De}),De=`/letter_constellations/assets/alligator-Ce5bwrBv.webp`,Oe=t({default:()=>ke}),ke=`/letter_constellations/assets/anchor-BV-M7FTg.webp`,Ae=t({default:()=>je}),je=`/letter_constellations/assets/ant-CvU0-Xki.webp`,Me=t({default:()=>Ne}),Ne=`/letter_constellations/assets/apple-DE3ezPUv.webp`,Pe=t({default:()=>Fe}),Fe=`/letter_constellations/assets/arrow-C4cWjJLo.webp`,Ie=t({default:()=>Le}),Le=`/letter_constellations/assets/astronaut-uRVMZixm.webp`,Re=t({default:()=>ze}),ze=`/letter_constellations/assets/bananas-VvvO5iFW.webp`,Be=t({default:()=>Ve}),Ve=`/letter_constellations/assets/beachball-BbGYNBZh.webp`,He=t({default:()=>Ue}),Ue=`/letter_constellations/assets/bear-CvctF6Z1.webp`,We=t({default:()=>Ge}),Ge=`/letter_constellations/assets/bee-BCPxoZBd.webp`,Ke=t({default:()=>qe}),qe=`/letter_constellations/assets/bird-CPHJbX7S.webp`,Je=t({default:()=>Ye}),Ye=`/letter_constellations/assets/boat-5S1Aura4.webp`,Xe=t({default:()=>Ze}),Ze=`/letter_constellations/assets/books-DzB7841Q.webp`,Qe=t({default:()=>$e}),$e=`/letter_constellations/assets/bus-Cr08LBdp.webp`,et=t({default:()=>tt}),tt=`/letter_constellations/assets/butterfly-DJ37Qz2P.webp`,nt=t({default:()=>rt}),rt=`/letter_constellations/assets/cake-C6PvIUA0.webp`,it=t({default:()=>at}),at=`/letter_constellations/assets/car-BLjHiQLL.webp`,ot=t({default:()=>st}),st=`/letter_constellations/assets/castle-kQcpwewP.webp`,ct=t({default:()=>lt}),lt=`/letter_constellations/assets/cat-_X2VtWwN.webp`,ut=t({default:()=>dt}),dt=`/letter_constellations/assets/clock-BqkzNyNk.webp`,ft=t({default:()=>pt}),pt=`/letter_constellations/assets/coat-OQ0hqgvS.webp`,mt=t({default:()=>ht}),ht=`/letter_constellations/assets/cow-BU0iB5gA.webp`,gt=t({default:()=>_t}),_t=`/letter_constellations/assets/crayon-DT3SkfbL.webp`,vt=t({default:()=>yt}),yt=`/letter_constellations/assets/cup-B6QWjkyx.webp`,bt=t({default:()=>xt}),xt=`/letter_constellations/assets/dog-DC0ZhRtT.webp`,St=t({default:()=>Ct}),Ct=`/letter_constellations/assets/doll-DAHtNuVs.webp`,wt=t({default:()=>Tt}),Tt=`/letter_constellations/assets/dolphin-C2ILYpRK.webp`,Et=t({default:()=>Dt}),Dt=`/letter_constellations/assets/donkey-BMyke4TY.webp`,Ot=t({default:()=>kt}),kt=`/letter_constellations/assets/door-CVCIUMCt.webp`,At=t({default:()=>jt}),jt=`/letter_constellations/assets/dragon-BAjlAFRA.webp`,Mt=t({default:()=>Nt}),Nt=`/letter_constellations/assets/dress-DvrNR5gf.webp`,Pt=t({default:()=>Ft}),Ft=`/letter_constellations/assets/drum-CwxRV3Cf.webp`,It=t({default:()=>Lt}),Lt=`/letter_constellations/assets/duck-U27dc_u8.webp`,Rt=t({default:()=>zt}),zt=`/letter_constellations/assets/eagle-DiKLg3yz.webp`,Bt=t({default:()=>Vt}),Vt=`/letter_constellations/assets/ear-BgYxZnRS.webp`,Ht=t({default:()=>Ut}),Ut=`/letter_constellations/assets/earth-r6cd7hOH.webp`,Wt=t({default:()=>Gt}),Gt=`/letter_constellations/assets/easel-EvkDAxpG.webp`,Kt=t({default:()=>qt}),qt=`/letter_constellations/assets/egg-Dm6XQc4n.webp`,Jt=t({default:()=>Yt}),Yt=`/letter_constellations/assets/elephant-CIvbMIFa.webp`,Xt=t({default:()=>Zt}),Zt=`/letter_constellations/assets/envelope-DU_Wnco4.webp`,Qt=t({default:()=>$t}),$t=`/letter_constellations/assets/eraser-MC2wEqI8.webp`,en=t({default:()=>tn}),tn=`/letter_constellations/assets/feather-DOx_OLgY.webp`,nn=t({default:()=>rn}),rn=`/letter_constellations/assets/fence-CwF42dkY.webp`,an=t({default:()=>on}),on=`/letter_constellations/assets/fish-DyOqSklD.webp`,sn=t({default:()=>cn}),cn=`/letter_constellations/assets/flower-B__p0fXU.webp`,ln=t({default:()=>un}),un=`/letter_constellations/assets/foot-BXSJM_nF.webp`,dn=t({default:()=>fn}),fn=`/letter_constellations/assets/fork-yucdu-VG.webp`,pn=t({default:()=>mn}),mn=`/letter_constellations/assets/fox-BE5prvLj.webp`,hn=t({default:()=>gn}),gn=`/letter_constellations/assets/frog-Dh1keBjF.webp`,_n=t({default:()=>vn}),vn=`/letter_constellations/assets/garden-BXEz5faC.webp`,yn=t({default:()=>bn}),bn=`/letter_constellations/assets/ghost-C3-5NTro.webp`,xn=t({default:()=>Sn}),Sn=`/letter_constellations/assets/giraffe-BN8FpAKY.webp`,Cn=t({default:()=>wn}),wn=`/letter_constellations/assets/gloves-BJzKgdkI.webp`,Tn=t({default:()=>En}),En=`/letter_constellations/assets/goat-Cp1Bn2Pw.webp`,Dn=t({default:()=>On}),On=`/letter_constellations/assets/goldfish-C02QaEXu.webp`,kn=t({default:()=>An}),An=`/letter_constellations/assets/grapes-DhMMMjx_.webp`,jn=t({default:()=>Mn}),Mn=`/letter_constellations/assets/guitar-BhL2ZgvO.webp`,Nn=t({default:()=>Pn}),Pn=`/letter_constellations/assets/hammer-CfkG7qG8.webp`,Fn=t({default:()=>In}),In=`/letter_constellations/assets/hand-C1darVYo.webp`,Ln=t({default:()=>Rn}),Rn=`/letter_constellations/assets/hat-5HvhUkFa.webp`,zn=t({default:()=>Bn}),Bn=`/letter_constellations/assets/heart-qdzpJsno.webp`,Vn=t({default:()=>Hn}),Hn=`/letter_constellations/assets/hedgehog-BaJaSoQw.webp`,Un=t({default:()=>Wn}),Wn=`/letter_constellations/assets/helicopter-DC0W3QSo.webp`,Gn=t({default:()=>Kn}),Kn=`/letter_constellations/assets/horse-DfgajmWZ.webp`,qn=t({default:()=>Jn}),Jn=`/letter_constellations/assets/hotdog-DTQ4uHOu.webp`,Yn=t({default:()=>Xn}),Xn=`/letter_constellations/assets/house-BZe7-xWI.webp`,Zn=t({default:()=>Qn}),Qn=`/letter_constellations/assets/ice_cream-DT_mc0_P.webp`,$n=t({default:()=>er}),er=`/letter_constellations/assets/igloo-CC6ArNRt.webp`,tr=t({default:()=>nr}),nr=`/letter_constellations/assets/iguana-BXYH3WVB.webp`,rr=t({default:()=>ir}),ir=`/letter_constellations/assets/inkpot-DnxrzNHx.webp`,ar=t({default:()=>or}),or=`/letter_constellations/assets/instrument-1utLT-6T.webp`,sr=t({default:()=>cr}),cr=`/letter_constellations/assets/invitation-z-PWuchC.webp`,lr=t({default:()=>ur}),ur=`/letter_constellations/assets/iron-CtDYRgb4.webp`,dr=t({default:()=>fr}),fr=`/letter_constellations/assets/island-B8Bp5Tsi.webp`,pr=t({default:()=>mr}),mr=`/letter_constellations/assets/jaguar-DHXPvMuk.webp`,hr=t({default:()=>gr}),gr=`/letter_constellations/assets/jar-Dwv58Vzr.webp`,_r=t({default:()=>vr}),vr=`/letter_constellations/assets/jean_jacket-BULMDhzs.webp`,yr=t({default:()=>br}),br=`/letter_constellations/assets/jelly-CVVY10k9.webp`,xr=t({default:()=>Sr}),Sr=`/letter_constellations/assets/jellyfish-BC5AuiGF.webp`,Cr=t({default:()=>wr}),wr=`/letter_constellations/assets/jigsaw-BOvQFAyk.webp`,Tr=t({default:()=>Er}),Er=`/letter_constellations/assets/juggler-GzS6Gfiw.webp`,Dr=t({default:()=>Or}),Or=`/letter_constellations/assets/juice-BqR4izBV.webp`,kr=t({default:()=>Ar}),Ar=`/letter_constellations/assets/kangaroo-BxEtKT1L.webp`,jr=t({default:()=>Mr}),Mr=`/letter_constellations/assets/key-C62Kti4m.webp`,Nr=t({default:()=>Pr}),Pr=`/letter_constellations/assets/kid-CjHrY2fb.webp`,Fr=t({default:()=>Ir}),Ir=`/letter_constellations/assets/kite-BB0XKGyC.webp`,Lr=t({default:()=>Rr}),Rr=`/letter_constellations/assets/kiwi-B0KGm69A.webp`,zr=t({default:()=>Br}),Br=`/letter_constellations/assets/koala-CjVCRMJZ.webp`,Vr=t({default:()=>Hr}),Hr=`/letter_constellations/assets/ladybug-CpSwAR2D.webp`,Ur=t({default:()=>Wr}),Wr=`/letter_constellations/assets/lamp-DUz1kO0o.webp`,Gr=t({default:()=>Kr}),Kr=`/letter_constellations/assets/lemon-Dd__w2Pq.webp`,qr=t({default:()=>Jr}),Jr=`/letter_constellations/assets/lemur-DHKaat7q.webp`,Yr=t({default:()=>Xr}),Xr=`/letter_constellations/assets/lettuce-D8mc2q-U.webp`,Zr=t({default:()=>Qr}),Qr=`/letter_constellations/assets/library-DKV5tmFO.webp`,$r=t({default:()=>ei}),ei=`/letter_constellations/assets/llama-Bwh9dRAG.webp`,ti=t({default:()=>ni}),ni=`/letter_constellations/assets/map-vQmueQII.webp`,ri=t({default:()=>ii}),ii=`/letter_constellations/assets/milk_carton-5cuoXE3T.webp`,ai=t({default:()=>oi}),oi=`/letter_constellations/assets/mitten-BeKduZUD.webp`,si=t({default:()=>ci}),ci=`/letter_constellations/assets/monkey-Cx-TqYqO.webp`,li=t({default:()=>ui}),ui=`/letter_constellations/assets/moon-jSXzUMnz.webp`,di=t({default:()=>fi}),fi=`/letter_constellations/assets/mountain-CZsGBOtO.webp`,pi=t({default:()=>mi}),mi=`/letter_constellations/assets/mouse-CmzFJelu.webp`,hi=t({default:()=>gi}),gi=`/letter_constellations/assets/muffin-BhJF_GvY.webp`,_i=t({default:()=>vi}),vi=`/letter_constellations/assets/mushroom-DHqipsMi.webp`,yi=t({default:()=>bi}),bi=`/letter_constellations/assets/nail-2NdCu6nX.webp`,xi=t({default:()=>Si}),Si=`/letter_constellations/assets/needle-WCtT707e.webp`,Ci=t({default:()=>wi}),wi=`/letter_constellations/assets/nest-Bs_bSd3C.webp`,Ti=t({default:()=>Ei}),Ei=`/letter_constellations/assets/net-CcQz3K1r.webp`,Di=t({default:()=>Oi}),Oi=`/letter_constellations/assets/newt-DMKiaP-a.webp`,ki=t({default:()=>Ai}),Ai=`/letter_constellations/assets/night-loJiPRO9.webp`,ji=t({default:()=>Mi}),Mi=`/letter_constellations/assets/nose-BGs_8Ldz.webp`,Ni=t({default:()=>Pi}),Pi=`/letter_constellations/assets/nurse-DnB4a9sK.webp`,Fi=t({default:()=>Ii}),Ii=`/letter_constellations/assets/nuts-dWD5IlYg.webp`,Li=t({default:()=>Ri}),Ri=`/letter_constellations/assets/oar-BJP2E8X6.webp`,zi=t({default:()=>Bi}),Bi=`/letter_constellations/assets/octopus-Bjrhg0Kz.webp`,Vi=t({default:()=>Hi}),Hi=`/letter_constellations/assets/onion-D-sGtoXB.webp`,Ui=t({default:()=>Wi}),Wi=`/letter_constellations/assets/orange-DUVPpQC1.webp`,Gi=t({default:()=>Ki}),Ki=`/letter_constellations/assets/orangutan-GXcPQi2e.webp`,qi=t({default:()=>Ji}),Ji=`/letter_constellations/assets/ostrich-CbDM_d-O.webp`,Yi=t({default:()=>Xi}),Xi=`/letter_constellations/assets/owl-Du9V2IZL.webp`,Zi=t({default:()=>Qi}),Qi=`/letter_constellations/assets/panda-Ce4wJnci.webp`,$i=t({default:()=>ea}),ea=`/letter_constellations/assets/pen-XTsE5Tp9.webp`,ta=t({default:()=>na}),na=`/letter_constellations/assets/pencil-CiXeWLpi.webp`,ra=t({default:()=>ia}),ia=`/letter_constellations/assets/penguin-DQXXzk60.webp`,aa=t({default:()=>oa}),oa=`/letter_constellations/assets/pig-f6PyeKkO.webp`,sa=t({default:()=>ca}),ca=`/letter_constellations/assets/pineapple-DEyw_76-.webp`,la=t({default:()=>ua}),ua=`/letter_constellations/assets/pirate-CR8dih_s.webp`,da=t({default:()=>fa}),fa=`/letter_constellations/assets/pizza-BHofEroO.webp`,pa=t({default:()=>ma}),ma=`/letter_constellations/assets/pumpkin-BK91jTFE.webp`,ha=t({default:()=>ga}),ga=`/letter_constellations/assets/quail-BPC2-Z_g.webp`,_a=t({default:()=>va}),va=`/letter_constellations/assets/queen-fiZYgCAh.webp`,ya=t({default:()=>ba}),ba=`/letter_constellations/assets/question-WkEi71Wr.webp`,xa=t({default:()=>Sa}),Sa=`/letter_constellations/assets/quill-KEF9k-Kr.webp`,Ca=t({default:()=>wa}),wa=`/letter_constellations/assets/quilt-B-SSp3IP.webp`,Ta=t({default:()=>Ea}),Ea=`/letter_constellations/assets/quince-CGOlv5XI.webp`,Da=t({default:()=>Oa}),Oa=`/letter_constellations/assets/quiver-ClNEjwQB.webp`,ka=t({default:()=>Aa}),Aa=`/letter_constellations/assets/rabbit-DZWfw0Tv.webp`,ja=t({default:()=>Ma}),Ma=`/letter_constellations/assets/radishes-B0KgqPIP.webp`,Na=t({default:()=>Pa}),Pa=`/letter_constellations/assets/rainbow-CsvdimZC.webp`,Fa=t({default:()=>Ia}),Ia=`/letter_constellations/assets/ring-xmCTdpMO.webp`,La=t({default:()=>Ra}),Ra=`/letter_constellations/assets/robin-X5gA9bFr.webp`,za=t({default:()=>Ba}),Ba=`/letter_constellations/assets/robot-ozHBRcKM.webp`,Va=t({default:()=>Ha}),Ha=`/letter_constellations/assets/rocket-CitKkJev.webp`,Ua=t({default:()=>Wa}),Wa=`/letter_constellations/assets/rose-DL-Oy0Ek.webp`,Ga=t({default:()=>Ka}),Ka=`/letter_constellations/assets/ruler-C1DPZ1Nf.webp`,qa=t({default:()=>Ja}),Ja=`/letter_constellations/assets/sailboat-BlcB2Z3i.webp`,Ya=t({default:()=>Xa}),Xa=`/letter_constellations/assets/sheep-C-5w7NTE.webp`,Za=t({default:()=>Qa}),Qa=`/letter_constellations/assets/snake-CbOrYwEs.webp`,$a=t({default:()=>eo}),eo=`/letter_constellations/assets/snowman-DjpZB2fz.webp`,to=t({default:()=>no}),no=`/letter_constellations/assets/socks-Dp64Nj0U.webp`,ro=t({default:()=>io}),io=`/letter_constellations/assets/star-CjyC3Ct8.webp`,ao=t({default:()=>oo}),oo=`/letter_constellations/assets/strawberry-ftyABBXF.webp`,so=t({default:()=>co}),co=`/letter_constellations/assets/sun-BSCyCAh2.webp`,lo=t({default:()=>uo}),uo=`/letter_constellations/assets/table-BNMmdjGK.webp`,fo=t({default:()=>po}),po=`/letter_constellations/assets/teapot-D56nDidw.webp`,mo=t({default:()=>ho}),ho=`/letter_constellations/assets/tiger-afEfjDX9.webp`,go=t({default:()=>_o}),_o=`/letter_constellations/assets/tomato-BFAkscRK.webp`,vo=t({default:()=>yo}),yo=`/letter_constellations/assets/tractor-tug0JOHL.webp`,bo=t({default:()=>xo}),xo=`/letter_constellations/assets/train-NYNpzrwW.webp`,So=t({default:()=>Co}),Co=`/letter_constellations/assets/tree-BlYlAreg.webp`,wo=t({default:()=>To}),To=`/letter_constellations/assets/turtle-Cj9D3d81.webp`,Eo=t({default:()=>Do}),Do=`/letter_constellations/assets/ukulele-DCuOtxfK.webp`,Oo=t({default:()=>ko}),ko=`/letter_constellations/assets/umbrella-Dz1MlrtA.webp`,Ao=t({default:()=>jo}),jo=`/letter_constellations/assets/unicorn-CriH96qk.webp`,Mo=t({default:()=>No}),No=`/letter_constellations/assets/unicycle-gQror61r.webp`,Po=t({default:()=>Fo}),Fo=`/letter_constellations/assets/unidentified_object-CUDu-AOC.webp`,Io=t({default:()=>Lo}),Lo=`/letter_constellations/assets/urchin-_JmHjkoD.webp`,Ro=t({default:()=>zo}),zo=`/letter_constellations/assets/utensils-DFAZ2efj.webp`,Bo=t({default:()=>Vo}),Vo=`/letter_constellations/assets/vacuum-BLJAk_jE.webp`,Ho=t({default:()=>Uo}),Uo=`/letter_constellations/assets/van-Dpdv5-Hp.webp`,Wo=t({default:()=>Go}),Go=`/letter_constellations/assets/vase-CpSUtdT6.webp`,Ko=t({default:()=>qo}),qo=`/letter_constellations/assets/vegetables-ChjTiEM-.webp`,Jo=t({default:()=>Yo}),Yo=`/letter_constellations/assets/vest-D3gDumUU.webp`,Xo=t({default:()=>Zo}),Zo=`/letter_constellations/assets/village-NwxI-P5i.webp`,Qo=t({default:()=>$o}),$o=`/letter_constellations/assets/violin-Bwy5HY6e.webp`,es=t({default:()=>ts}),ts=`/letter_constellations/assets/volcano-DCRqkKRz.webp`,ns=t({default:()=>rs}),rs=`/letter_constellations/assets/vulture-C-ErfnZu.webp`,is=t({default:()=>as}),as=`/letter_constellations/assets/wagon-DW5tu-i6.webp`,os=t({default:()=>ss}),ss=`/letter_constellations/assets/watch-ahI5puCQ.webp`,cs=t({default:()=>ls}),ls=`/letter_constellations/assets/watermelon-DOsensRc.webp`,us=t({default:()=>ds}),ds=`/letter_constellations/assets/web-Ce6CPeZ_.webp`,fs=t({default:()=>ps}),ps=`/letter_constellations/assets/whale-DAEVlAKS.webp`,ms=t({default:()=>hs}),hs=`/letter_constellations/assets/wind-B6DREdK4.webp`,gs=t({default:()=>_s}),_s=`/letter_constellations/assets/wizard-C8Dnr-1B.webp`,vs=t({default:()=>ys}),ys=`/letter_constellations/assets/wolf-aZD8PyPP.webp`,bs=t({default:()=>xs}),xs=`/letter_constellations/assets/worm-Ds6LH9wX.webp`,Ss=t({default:()=>Cs}),Cs=`/letter_constellations/assets/accordion-DqQtriLm.webm`,ws=t({default:()=>Ts}),Ts=`/letter_constellations/assets/acorn-BIezBOPs.webm`,Es=t({default:()=>Ds}),Ds=`/letter_constellations/assets/airplane-C_Wfm5g-.webm`,Os=t({default:()=>ks}),ks=`/letter_constellations/assets/alligator-DYepK-g8.webm`,As=t({default:()=>js}),js=`/letter_constellations/assets/anchor-D_H3HrJm.webm`,Ms=t({default:()=>Ns}),Ns=`/letter_constellations/assets/ant-D6QhcluX.webm`,Ps=t({default:()=>Fs}),Fs=`/letter_constellations/assets/apple-CZDvf2t3.webm`,Is=t({default:()=>Ls}),Ls=`/letter_constellations/assets/arrow-Cs627Rik.webm`,Rs=t({default:()=>zs}),zs=`/letter_constellations/assets/astronaut-B9QACnUz.webm`,q=t({default:()=>Bs}),Bs=`/letter_constellations/assets/a-CMkzIMFv.webm`,J=t({default:()=>Vs}),Vs=`/letter_constellations/assets/b-C2kyMVsg.webm`,Hs=t({default:()=>Us}),Us=`/letter_constellations/assets/c-IjVD-Vre.webm`,Ws=t({default:()=>Gs}),Gs=`/letter_constellations/assets/d-9vk5XyCw.webm`,Ks=t({default:()=>qs}),qs=`/letter_constellations/assets/e-VRhdBt2g.webm`,Js=t({default:()=>Ys}),Ys=`/letter_constellations/assets/f-BVhXOumA.webm`,Xs=t({default:()=>Zs}),Zs=`/letter_constellations/assets/g-DIYEe5HW.webm`,Qs=t({default:()=>$s}),$s=`/letter_constellations/assets/h-DyPaSF0e.webm`,ec=t({default:()=>tc}),tc=`/letter_constellations/assets/i-C5tqm2EV.webm`,nc=t({default:()=>rc}),rc=`/letter_constellations/assets/j-Be2zckp_.webm`,ic=t({default:()=>ac}),ac=`/letter_constellations/assets/k-BakCspA1.webm`,oc=t({default:()=>sc}),sc=`/letter_constellations/assets/l-NUuunizq.webm`,cc=t({default:()=>lc}),lc=`/letter_constellations/assets/m-BB98DVPZ.webm`,uc=t({default:()=>dc}),dc=`/letter_constellations/assets/n-CnIX4TYm.webm`,fc=t({default:()=>pc}),pc=`/letter_constellations/assets/o-BiLMB4YT.webm`,mc=t({default:()=>hc}),hc=`/letter_constellations/assets/p-C4ciUhaN.webm`,gc=t({default:()=>_c}),_c=`/letter_constellations/assets/q-DF2D67v_.webm`,vc=t({default:()=>yc}),yc=`/letter_constellations/assets/r-CS0DW1cp.webm`,bc=t({default:()=>xc}),xc=`/letter_constellations/assets/s-An_rASvU.webm`,Sc=t({default:()=>Cc}),Cc=`/letter_constellations/assets/t-CVflKspQ.webm`,wc=t({default:()=>Tc}),Tc=`/letter_constellations/assets/u-Dudg3DnT.webm`,Y=t({default:()=>Ec}),Ec=`/letter_constellations/assets/v-WZ8s5vPh.webm`,Dc=t({default:()=>Oc}),Oc=`/letter_constellations/assets/w-BElagC9L.webm`,kc=t({default:()=>Ac}),Ac=`/letter_constellations/assets/x-BCyjQ6tg.webm`,jc=t({default:()=>Mc}),Mc=`/letter_constellations/assets/y-CxUp37AX.webm`,Nc=t({default:()=>Pc}),Pc=`/letter_constellations/assets/z-CmvHDWPa.webm`,Fc=t({default:()=>Ic}),Ic=`/letter_constellations/assets/bananas-h3WNqzRq.webm`,Lc=t({default:()=>Rc}),Rc=`/letter_constellations/assets/beachball-CHdPyc90.webm`,zc=t({default:()=>Bc}),Bc=`/letter_constellations/assets/bear-5zJMp45m.webm`,Vc=t({default:()=>Hc}),Hc=`/letter_constellations/assets/bee-CIa4b6b3.webm`,Uc=t({default:()=>Wc}),Wc=`/letter_constellations/assets/bird-C-mCHkrZ.webm`,Gc=t({default:()=>Kc}),Kc=`/letter_constellations/assets/boat-Egh2j8Fn.webm`,qc=t({default:()=>Jc}),Jc=`/letter_constellations/assets/books-CDA8xLKG.webm`,Yc=t({default:()=>Xc}),Xc=`/letter_constellations/assets/bus-BcHciuah.webm`,Zc=t({default:()=>Qc}),Qc=`/letter_constellations/assets/butterfly-Cc832W2U.webm`,$c=t({default:()=>el}),el=`/letter_constellations/assets/cake-CE8GeVD6.webm`,tl=t({default:()=>nl}),nl=`/letter_constellations/assets/car-D5SmH24e.webm`,rl=t({default:()=>il}),il=`/letter_constellations/assets/castle-BoljANwC.webm`,al=t({default:()=>ol}),ol=`/letter_constellations/assets/cat-BDJFtO1J.webm`,sl=t({default:()=>cl}),cl=`/letter_constellations/assets/clock-BdR7qum2.webm`,ll=t({default:()=>ul}),ul=`/letter_constellations/assets/coat-16JLlN-R.webm`,dl=t({default:()=>fl}),fl=`/letter_constellations/assets/cow-_gfKTNxD.webm`,pl=t({default:()=>ml}),ml=`/letter_constellations/assets/crayon-DcBF0plc.webm`,hl=t({default:()=>gl}),gl=`/letter_constellations/assets/cup-U1NYHdiC.webm`,_l=t({default:()=>vl}),vl=`/letter_constellations/assets/dog-L0-qQmmf.webm`,yl=t({default:()=>bl}),bl=`/letter_constellations/assets/doll-DHzmlk22.webm`,xl=t({default:()=>Sl}),Sl=`/letter_constellations/assets/dolphin-DHN3M5dU.webm`,Cl=t({default:()=>wl}),wl=`/letter_constellations/assets/donkey-DZGy0jN1.webm`,Tl=t({default:()=>El}),El=`/letter_constellations/assets/door-D03rYjG6.webm`,Dl=t({default:()=>Ol}),Ol=`/letter_constellations/assets/dragon-Bv-oOD4o.webm`,kl=t({default:()=>Al}),Al=`/letter_constellations/assets/dress-DioYKYZv.webm`,jl=t({default:()=>Ml}),Ml=`/letter_constellations/assets/drum-Cf0je1UD.webm`,Nl=t({default:()=>Pl}),Pl=`/letter_constellations/assets/duck-Bgo1sGLJ.webm`,Fl=t({default:()=>Il}),Il=`/letter_constellations/assets/eagle-BgbhiRSg.webm`,Ll=t({default:()=>Rl}),Rl=`/letter_constellations/assets/ear-Dl7REk6c.webm`,zl=t({default:()=>Bl}),Bl=`/letter_constellations/assets/earth-B5sQyPAK.webm`,Vl=t({default:()=>Hl}),Hl=`/letter_constellations/assets/easel-DrjyCnNQ.webm`,Ul=t({default:()=>Wl}),Wl=`/letter_constellations/assets/egg-Cp_mjYOa.webm`,Gl=t({default:()=>Kl}),Kl=`/letter_constellations/assets/elephant-DXDMMvRo.webm`,ql=t({default:()=>Jl}),Jl=`/letter_constellations/assets/engine-CGoW8nln.webm`,Yl=t({default:()=>Xl}),Xl=`/letter_constellations/assets/envelope-CZkB1Qdz.webm`,Zl=t({default:()=>Ql}),Ql=`/letter_constellations/assets/eraser-CL0SN2d1.webm`,$l=t({default:()=>eu}),eu=`/letter_constellations/assets/feather-BTNKnrGo.webm`,tu=t({default:()=>nu}),nu=`/letter_constellations/assets/fence-BgF_34t7.webm`,ru=t({default:()=>iu}),iu=`/letter_constellations/assets/firefly-B7E-EXaq.webm`,au=t({default:()=>ou}),ou=`/letter_constellations/assets/fish-DsqXYDML.webm`,su=t({default:()=>cu}),cu=`/letter_constellations/assets/flower-DzzwNf_U.webm`,lu=t({default:()=>uu}),uu=`/letter_constellations/assets/foot-CqvpFi2I.webm`,du=t({default:()=>fu}),fu=`/letter_constellations/assets/fork-BbXVs51Y.webm`,pu=t({default:()=>mu}),mu=`/letter_constellations/assets/fox-xbmCA8tV.webm`,hu=t({default:()=>gu}),gu=`/letter_constellations/assets/frog-ylp_1Suh.webm`,_u=t({default:()=>vu}),vu=`/letter_constellations/assets/garden-B_iCSqP1.webm`,yu=t({default:()=>bu}),bu=`/letter_constellations/assets/ghost-DCvyj5gL.webm`,xu=t({default:()=>Su}),Su=`/letter_constellations/assets/gift-Cy86EoFO.webm`,Cu=t({default:()=>wu}),wu=`/letter_constellations/assets/giraffe-CADFBhnh.webm`,Tu=t({default:()=>Eu}),Eu=`/letter_constellations/assets/gloves-BLFuRMLi.webm`,Du=t({default:()=>Ou}),Ou=`/letter_constellations/assets/goat-BTaP2ceE.webm`,ku=t({default:()=>Au}),Au=`/letter_constellations/assets/goldfish-CfKPbugz.webm`,ju=t({default:()=>Mu}),Mu=`/letter_constellations/assets/grapes-CvsbRxCI.webm`,Nu=t({default:()=>Pu}),Pu=`/letter_constellations/assets/guitar-DdMBQt2m.webm`,Fu=t({default:()=>Iu}),Iu=`/letter_constellations/assets/hammer-Ceh7AMku.webm`,Lu=t({default:()=>Ru}),Ru=`/letter_constellations/assets/hand-KB-VfcaO.webm`,zu=t({default:()=>Bu}),Bu=`/letter_constellations/assets/hat-BiSruZqk.webm`,Vu=t({default:()=>Hu}),Hu=`/letter_constellations/assets/heart-CgLg_Xw1.webm`,Uu=t({default:()=>Wu}),Wu=`/letter_constellations/assets/hedgehog-DM6G4aTi.webm`,Gu=t({default:()=>Ku}),Ku=`/letter_constellations/assets/helicopter-BDdKO2XC.webm`,qu=t({default:()=>Ju}),Ju=`/letter_constellations/assets/horse-BQ3oQppo.webm`,Yu=t({default:()=>Xu}),Xu=`/letter_constellations/assets/hotdog-q3lfgc-H.webm`,Zu=t({default:()=>Qu}),Qu=`/letter_constellations/assets/house-FiY5MHVH.webm`,$u=t({default:()=>ed}),ed=`/letter_constellations/assets/ice_cream-uwet6mas.webm`,td=t({default:()=>nd}),nd=`/letter_constellations/assets/igloo-Bm1n6RqU.webm`,rd=t({default:()=>id}),id=`/letter_constellations/assets/iguana-Cj9OFN_N.webm`,ad=t({default:()=>od}),od=`/letter_constellations/assets/inkpot-CedBlT6t.webm`,sd=t({default:()=>cd}),cd=`/letter_constellations/assets/insect-BVXEJtav.webm`,ld=t({default:()=>ud}),ud=`/letter_constellations/assets/instrument-DEj5EZot.webm`,dd=t({default:()=>fd}),fd=`/letter_constellations/assets/invitation-yDuZq8gj.webm`,pd=t({default:()=>md}),md=`/letter_constellations/assets/iron-Cx2pzUpR.webm`,hd=t({default:()=>gd}),gd=`/letter_constellations/assets/island-fs6PoIj0.webm`,_d=t({default:()=>vd}),vd=`/letter_constellations/assets/jaguar-BBmfc63Y.webm`,yd=t({default:()=>bd}),bd=`/letter_constellations/assets/jar-DHu15TBb.webm`,xd=t({default:()=>Sd}),Sd=`/letter_constellations/assets/jean_jacket-6ocVOvCD.webm`,Cd=t({default:()=>wd}),wd=`/letter_constellations/assets/jelly-EK6xdlfq.webm`,Td=t({default:()=>Ed}),Ed=`/letter_constellations/assets/jellyfish-BRbFjms6.webm`,Dd=t({default:()=>Od}),Od=`/letter_constellations/assets/jet-ByhMGSXW.webm`,kd=t({default:()=>Ad}),Ad=`/letter_constellations/assets/jigsaw-DZYjOBmp.webm`,jd=t({default:()=>Md}),Md=`/letter_constellations/assets/juggler-DBOlauIy.webm`,Nd=t({default:()=>Pd}),Pd=`/letter_constellations/assets/juice-D-68Bu_u.webm`,Fd=t({default:()=>Id}),Id=`/letter_constellations/assets/kangaroo-CcIkE6dA.webm`,Ld=t({default:()=>Rd}),Rd=`/letter_constellations/assets/kettle_drum-D-Fw6JsA.webm`,zd=t({default:()=>Bd}),Bd=`/letter_constellations/assets/key-KD60Ug_d.webm`,Vd=t({default:()=>Hd}),Hd=`/letter_constellations/assets/kid-tJFEsvg6.webm`,Ud=t({default:()=>Wd}),Wd=`/letter_constellations/assets/kite-BvkQpXQ4.webm`,Gd=t({default:()=>Kd}),Kd=`/letter_constellations/assets/kiwi-DSByUtRu.webm`,qd=t({default:()=>Jd}),Jd=`/letter_constellations/assets/knight-BHnTxxNg.webm`,Yd=t({default:()=>Xd}),Xd=`/letter_constellations/assets/knot-BFtAc-vX.webm`,Zd=t({default:()=>Qd}),Qd=`/letter_constellations/assets/koala-BOrbjEQC.webm`,$d=t({default:()=>ef}),ef=`/letter_constellations/assets/labrador-C2atZ3eI.webm`,tf=t({default:()=>nf}),nf=`/letter_constellations/assets/ladybug-D_CzUcBl.webm`,rf=t({default:()=>af}),af=`/letter_constellations/assets/lamp-CA6Xv_tn.webm`,of=t({default:()=>sf}),sf=`/letter_constellations/assets/lemon-CPuugyKa.webm`,cf=t({default:()=>lf}),lf=`/letter_constellations/assets/lemur-CPypRhBI.webm`,uf=t({default:()=>df}),df=`/letter_constellations/assets/lettuce-9ltxxL4a.webm`,ff=t({default:()=>pf}),pf=`/letter_constellations/assets/library-Bt30cqOU.webm`,mf=t({default:()=>hf}),hf=`/letter_constellations/assets/llama-CnavT3OA.webm`,gf=t({default:()=>_f}),_f=`/letter_constellations/assets/lungs-CtTA6hzs.webm`,vf=t({default:()=>yf}),yf=`/letter_constellations/assets/map-DpXTnY61.webm`,bf=t({default:()=>xf}),xf=`/letter_constellations/assets/milk_carton-CYrSpFIF.webm`,Sf=t({default:()=>Cf}),Cf=`/letter_constellations/assets/mitten-CjMBbbHc.webm`,wf=t({default:()=>Tf}),Tf=`/letter_constellations/assets/monkey-Da7Co2b4.webm`,Ef=t({default:()=>Df}),Df=`/letter_constellations/assets/moon-BQlWDz1v.webm`,Of=t({default:()=>kf}),kf=`/letter_constellations/assets/mountain-BXOdjgsP.webm`,Af=t({default:()=>jf}),jf=`/letter_constellations/assets/mouse-CX-udEkX.webm`,Mf=t({default:()=>Nf}),Nf=`/letter_constellations/assets/muffin-ZyoQalKv.webm`,Pf=t({default:()=>Ff}),Ff=`/letter_constellations/assets/mushroom-ma49-9Es.webm`,If=t({default:()=>Lf}),Lf=`/letter_constellations/assets/nail-BDgqn4Oi.webm`,Rf=t({default:()=>zf}),zf=`/letter_constellations/assets/needle-DfSM5BBx.webm`,Bf=t({default:()=>Vf}),Vf=`/letter_constellations/assets/nest-D8oW8sHs.webm`,Hf=t({default:()=>Uf}),Uf=`/letter_constellations/assets/net-BOWyW7Pd.webm`,Wf=t({default:()=>Gf}),Gf=`/letter_constellations/assets/newt-Di4rr-kM.webm`,Kf=t({default:()=>qf}),qf=`/letter_constellations/assets/night-DYJXIJ6m.webm`,Jf=t({default:()=>Yf}),Yf=`/letter_constellations/assets/nose-wU6bmTQo.webm`,Xf=t({default:()=>Zf}),Zf=`/letter_constellations/assets/nurse-Dvz6wvdQ.webm`,Qf=t({default:()=>$f}),$f=`/letter_constellations/assets/nuts-DOYhPI3p.webm`,ep=t({default:()=>tp}),tp=`/letter_constellations/assets/oar-NPG07m8x.webm`,np=t({default:()=>rp}),rp=`/letter_constellations/assets/octopus-BlAhrR5x.webm`,ip=t({default:()=>ap}),ap=`/letter_constellations/assets/onion-C3UnQO2e.webm`,op=t({default:()=>sp}),sp=`/letter_constellations/assets/orange-CAjV4sVP.webm`,cp=t({default:()=>lp}),lp=`/letter_constellations/assets/orangutan-jdEjO_N5.webm`,up=t({default:()=>dp}),dp=`/letter_constellations/assets/orbit-Cd0ALAeU.webm`,fp=t({default:()=>pp}),pp=`/letter_constellations/assets/ostrich-Dku7vhly.webm`,mp=t({default:()=>hp}),hp=`/letter_constellations/assets/overalls-CI6JhbtH.webm`,gp=t({default:()=>_p}),_p=`/letter_constellations/assets/owl-CQWsqBJC.webm`,vp=t({default:()=>yp}),yp=`/letter_constellations/assets/panda-BmfrrnU6.webm`,bp=t({default:()=>xp}),xp=`/letter_constellations/assets/pen-ZTmVznwu.webm`,Sp=t({default:()=>Cp}),Cp=`/letter_constellations/assets/pencil-BVQ8rqGT.webm`,wp=t({default:()=>Tp}),Tp=`/letter_constellations/assets/penguin-DeD256Gp.webm`,Ep=t({default:()=>Dp}),Dp=`/letter_constellations/assets/pig-DzgF3-tq.webm`,Op=t({default:()=>kp}),kp=`/letter_constellations/assets/pineapple-DxiQhkpX.webm`,Ap=t({default:()=>jp}),jp=`/letter_constellations/assets/pirate-D-3TP_NB.webm`,Mp=t({default:()=>Np}),Np=`/letter_constellations/assets/pizza-FIglZQ7d.webm`,Pp=t({default:()=>Fp}),Fp=`/letter_constellations/assets/pumpkin-BindYuWd.webm`,Ip=t({default:()=>Lp}),Lp=`/letter_constellations/assets/quaff-C0HVHSvx.webm`,Rp=t({default:()=>zp}),zp=`/letter_constellations/assets/quail-B8-wr5Ky.webm`,Bp=t({default:()=>Vp}),Vp=`/letter_constellations/assets/quarter-DYSIAC14.webm`,Hp=t({default:()=>Up}),Up=`/letter_constellations/assets/queen-DJtNeLa0.webm`,Wp=t({default:()=>Gp}),Gp=`/letter_constellations/assets/question-V1fIR47w.webm`,Kp=t({default:()=>qp}),qp=`/letter_constellations/assets/quill-Bh53ye5V.webm`,Jp=t({default:()=>Yp}),Yp=`/letter_constellations/assets/quilt-Ju7WSa2i.webm`,Xp=t({default:()=>Zp}),Zp=`/letter_constellations/assets/quince-DQk-aRBI.webm`,Qp=t({default:()=>$p}),$p=`/letter_constellations/assets/quiver-KF2M2SC_.webm`,em=t({default:()=>tm}),tm=`/letter_constellations/assets/rabbit-Bn1L-ab9.webm`,nm=t({default:()=>rm}),rm=`/letter_constellations/assets/radishes-Caj0bGTw.webm`,im=t({default:()=>am}),am=`/letter_constellations/assets/rainbow-D_0DDL10.webm`,om=t({default:()=>sm}),sm=`/letter_constellations/assets/ring-CvaEKfC1.webm`,cm=t({default:()=>lm}),lm=`/letter_constellations/assets/robin-B4Wyw6Kt.webm`,um=t({default:()=>dm}),dm=`/letter_constellations/assets/robot-O0G0NIz8.webm`,fm=t({default:()=>pm}),pm=`/letter_constellations/assets/rocket-DyCWYIqK.webm`,mm=t({default:()=>hm}),hm=`/letter_constellations/assets/rose-DNpS-rzs.webm`,gm=t({default:()=>_m}),_m=`/letter_constellations/assets/ruler-3jiqdeeH.webm`,vm=t({default:()=>ym}),ym=`/letter_constellations/assets/sailboat-C9PCPNTV.webm`,bm=t({default:()=>xm}),xm=`/letter_constellations/assets/sheep-BD1Qex-a.webm`,Sm=t({default:()=>Cm}),Cm=`/letter_constellations/assets/snake-3VL3pI0b.webm`,wm=t({default:()=>Tm}),Tm=`/letter_constellations/assets/snowman-BVngaE17.webm`,Em=t({default:()=>Dm}),Dm=`/letter_constellations/assets/socks-BlZGzMoX.webm`,Om=t({default:()=>km}),km=`/letter_constellations/assets/spoon-Bpezv9ur.webm`,Am=t({default:()=>jm}),jm=`/letter_constellations/assets/star-ByhnbkyF.webm`,Mm=t({default:()=>Nm}),Nm=`/letter_constellations/assets/strawberry-WDKIdltR.webm`,Pm=t({default:()=>Fm}),Fm=`/letter_constellations/assets/sun-OXh-YcDj.webm`,Im=t({default:()=>Lm}),Lm=`/letter_constellations/assets/table-DdVbos9W.webm`,Rm=t({default:()=>zm}),zm=`/letter_constellations/assets/teapot-C8QdlDfa.webm`,Bm=t({default:()=>Vm}),Vm=`/letter_constellations/assets/teddy_bear-DbuTfBst.webm`,Hm=t({default:()=>Um}),Um=`/letter_constellations/assets/tiger-DDYpj3Yp.webm`,Wm=t({default:()=>Gm}),Gm=`/letter_constellations/assets/tomato-WtQkNxZo.webm`,Km=t({default:()=>qm}),qm=`/letter_constellations/assets/tractor-CCgqU7qU.webm`,Jm=t({default:()=>Ym}),Ym=`/letter_constellations/assets/train-C5Zj-m0t.webm`,Xm=t({default:()=>Zm}),Zm=`/letter_constellations/assets/tree-nuB9A3vs.webm`,Qm=t({default:()=>$m}),$m=`/letter_constellations/assets/turtle-BDkcySK2.webm`,eh=t({default:()=>th}),th=`/letter_constellations/assets/ukulele-CLqXYXhe.webm`,nh=t({default:()=>rh}),rh=`/letter_constellations/assets/umbrella-r_963ubY.webm`,ih=t({default:()=>ah}),ah=`/letter_constellations/assets/ungulate-CTO96oPD.webm`,oh=t({default:()=>sh}),sh=`/letter_constellations/assets/unicorn-CY2TK8f5.webm`,ch=t({default:()=>lh}),lh=`/letter_constellations/assets/unicycle-DGKk_v6u.webm`,uh=t({default:()=>dh}),dh=`/letter_constellations/assets/unidentified_object--Ro5ICPR.webm`,fh=t({default:()=>ph}),ph=`/letter_constellations/assets/uniform-DqfMcPDd.webm`,mh=t({default:()=>hh}),hh=`/letter_constellations/assets/urchin-iCRAGDls.webm`,gh=t({default:()=>_h}),_h=`/letter_constellations/assets/utensils-BzdOuCsZ.webm`,vh=t({default:()=>yh}),yh=`/letter_constellations/assets/vacuum-DoCcb9z-.webm`,bh=t({default:()=>xh}),xh=`/letter_constellations/assets/van-CQ1lJBiN.webm`,Sh=t({default:()=>Ch}),Ch=`/letter_constellations/assets/vase-CccRv94X.webm`,wh=t({default:()=>Th}),Th=`/letter_constellations/assets/vegetables-DLtroWc1.webm`,Eh=t({default:()=>Dh}),Dh=`/letter_constellations/assets/vest-DBXjbVnZ.webm`,Oh=t({default:()=>kh}),kh=`/letter_constellations/assets/village-CVG8tels.webm`,Ah=t({default:()=>jh}),jh=`/letter_constellations/assets/violin-BE21TgO1.webm`,Mh=t({default:()=>Nh}),Nh=`/letter_constellations/assets/volcano-gCdpROrp.webm`,Ph=t({default:()=>Fh}),Fh=`/letter_constellations/assets/vulture-TiasgFRu.webm`,Ih=t({default:()=>Lh}),Lh=`/letter_constellations/assets/wagon-BpMibSu1.webm`,Rh=t({default:()=>zh}),zh=`/letter_constellations/assets/watch-DR-zilf0.webm`,Bh=t({default:()=>Vh}),Vh=`/letter_constellations/assets/watermelon-NcZQdRkY.webm`,Hh=t({default:()=>Uh}),Uh=`/letter_constellations/assets/web-CjQIo_1Q.webm`,Wh=t({default:()=>Gh}),Gh=`/letter_constellations/assets/whale-cDbKOtue.webm`,Kh=t({default:()=>qh}),qh=`/letter_constellations/assets/wind-DK91ZQgX.webm`,Jh=t({default:()=>Yh}),Yh=`/letter_constellations/assets/wizard-DhJfciPy.webm`,Xh=t({default:()=>Zh}),Zh=`/letter_constellations/assets/wolf-DHbkNJpk.webm`,Qh=t({default:()=>$h}),$h=`/letter_constellations/assets/worm-BWnc7YI5.webm`,eg={"./assets/items/a/accordion.png":be,"./assets/items/a/acorn.png":Se,"./assets/items/a/airplane.png":we,"./assets/items/a/alligator.png":Ee,"./assets/items/a/anchor.png":Oe,"./assets/items/a/ant.png":Ae,"./assets/items/a/apple.png":Me,"./assets/items/a/arrow.png":Pe,"./assets/items/a/astronaut.png":Ie,"./assets/items/b/bananas.png":Re,"./assets/items/b/beachball.png":Be,"./assets/items/b/bear.png":He,"./assets/items/b/bee.png":We,"./assets/items/b/bird.png":Ke,"./assets/items/b/boat.png":Je,"./assets/items/b/books.png":Xe,"./assets/items/b/bus.png":Qe,"./assets/items/b/butterfly.png":et,"./assets/items/c/cake.png":nt,"./assets/items/c/car.png":it,"./assets/items/c/castle.png":ot,"./assets/items/c/cat.png":ct,"./assets/items/c/clock.png":ut,"./assets/items/c/coat.png":ft,"./assets/items/c/cow.png":mt,"./assets/items/c/crayon.png":gt,"./assets/items/c/cup.png":vt,"./assets/items/d/dog.png":bt,"./assets/items/d/doll.png":St,"./assets/items/d/dolphin.png":wt,"./assets/items/d/donkey.png":Et,"./assets/items/d/door.png":Ot,"./assets/items/d/dragon.png":At,"./assets/items/d/dress.png":Mt,"./assets/items/d/drum.png":Pt,"./assets/items/d/duck.png":It,"./assets/items/e/eagle.png":Rt,"./assets/items/e/ear.png":Bt,"./assets/items/e/earth.png":Ht,"./assets/items/e/easel.png":Wt,"./assets/items/e/egg.png":Kt,"./assets/items/e/elephant.png":Jt,"./assets/items/e/envelope.png":Xt,"./assets/items/e/eraser.png":Qt,"./assets/items/f/feather.png":en,"./assets/items/f/fence.png":nn,"./assets/items/f/fish.png":an,"./assets/items/f/flower.png":sn,"./assets/items/f/foot.png":ln,"./assets/items/f/fork.png":dn,"./assets/items/f/fox.png":pn,"./assets/items/f/frog.png":hn,"./assets/items/g/garden.png":_n,"./assets/items/g/ghost.png":yn,"./assets/items/g/giraffe.png":xn,"./assets/items/g/gloves.png":Cn,"./assets/items/g/goat.png":Tn,"./assets/items/g/goldfish.png":Dn,"./assets/items/g/grapes.png":kn,"./assets/items/g/guitar.png":jn,"./assets/items/h/hammer.png":Nn,"./assets/items/h/hand.png":Fn,"./assets/items/h/hat.png":Ln,"./assets/items/h/heart.png":zn,"./assets/items/h/hedgehog.png":Vn,"./assets/items/h/helicopter.png":Un,"./assets/items/h/horse.png":Gn,"./assets/items/h/hotdog.png":qn,"./assets/items/h/house.png":Yn,"./assets/items/i/ice_cream.png":Zn,"./assets/items/i/igloo.png":$n,"./assets/items/i/iguana.png":tr,"./assets/items/i/inkpot.png":rr,"./assets/items/i/instrument.png":ar,"./assets/items/i/invitation.png":sr,"./assets/items/i/iron.png":lr,"./assets/items/i/island.png":dr,"./assets/items/j/jaguar.png":pr,"./assets/items/j/jar.png":hr,"./assets/items/j/jean_jacket.png":_r,"./assets/items/j/jelly.png":yr,"./assets/items/j/jellyfish.png":xr,"./assets/items/j/jigsaw.png":Cr,"./assets/items/j/juggler.png":Tr,"./assets/items/j/juice.png":Dr,"./assets/items/k/kangaroo.png":kr,"./assets/items/k/key.png":jr,"./assets/items/k/kid.png":Nr,"./assets/items/k/kite.png":Fr,"./assets/items/k/kiwi.png":Lr,"./assets/items/k/koala.png":zr,"./assets/items/l/ladybug.png":Vr,"./assets/items/l/lamp.png":Ur,"./assets/items/l/lemon.png":Gr,"./assets/items/l/lemur.png":qr,"./assets/items/l/lettuce.png":Yr,"./assets/items/l/library.png":Zr,"./assets/items/l/llama.png":$r,"./assets/items/m/map.png":ti,"./assets/items/m/milk_carton.png":ri,"./assets/items/m/mitten.png":ai,"./assets/items/m/monkey.png":si,"./assets/items/m/moon.png":li,"./assets/items/m/mountain.png":di,"./assets/items/m/mouse.png":pi,"./assets/items/m/muffin.png":hi,"./assets/items/m/mushroom.png":_i,"./assets/items/n/nail.png":yi,"./assets/items/n/needle.png":xi,"./assets/items/n/nest.png":Ci,"./assets/items/n/net.png":Ti,"./assets/items/n/newt.png":Di,"./assets/items/n/night.png":ki,"./assets/items/n/nose.png":ji,"./assets/items/n/nurse.png":Ni,"./assets/items/n/nuts.png":Fi,"./assets/items/o/oar.png":Li,"./assets/items/o/octopus.png":zi,"./assets/items/o/onion.png":Vi,"./assets/items/o/orange.png":Ui,"./assets/items/o/orangutan.png":Gi,"./assets/items/o/ostrich.png":qi,"./assets/items/o/owl.png":Yi,"./assets/items/p/panda.png":Zi,"./assets/items/p/pen.png":$i,"./assets/items/p/pencil.png":ta,"./assets/items/p/penguin.png":ra,"./assets/items/p/pig.png":aa,"./assets/items/p/pineapple.png":sa,"./assets/items/p/pirate.png":la,"./assets/items/p/pizza.png":da,"./assets/items/p/pumpkin.png":pa,"./assets/items/q/quail.png":ha,"./assets/items/q/queen.png":_a,"./assets/items/q/question.png":ya,"./assets/items/q/quill.png":xa,"./assets/items/q/quilt.png":Ca,"./assets/items/q/quince.png":Ta,"./assets/items/q/quiver.png":Da,"./assets/items/r/rabbit.png":ka,"./assets/items/r/radishes.png":ja,"./assets/items/r/rainbow.png":Na,"./assets/items/r/ring.png":Fa,"./assets/items/r/robin.png":La,"./assets/items/r/robot.png":za,"./assets/items/r/rocket.png":Va,"./assets/items/r/rose.png":Ua,"./assets/items/r/ruler.png":Ga,"./assets/items/s/sailboat.png":qa,"./assets/items/s/sheep.png":Ya,"./assets/items/s/snake.png":Za,"./assets/items/s/snowman.png":$a,"./assets/items/s/socks.png":to,"./assets/items/s/star.png":ro,"./assets/items/s/strawberry.png":ao,"./assets/items/s/sun.png":so,"./assets/items/t/table.png":lo,"./assets/items/t/teapot.png":fo,"./assets/items/t/tiger.png":mo,"./assets/items/t/tomato.png":go,"./assets/items/t/tractor.png":vo,"./assets/items/t/train.png":bo,"./assets/items/t/tree.png":So,"./assets/items/t/turtle.png":wo,"./assets/items/u/ukulele.png":Eo,"./assets/items/u/umbrella.png":Oo,"./assets/items/u/unicorn.png":Ao,"./assets/items/u/unicycle.png":Mo,"./assets/items/u/unidentified_object.png":Po,"./assets/items/u/urchin.png":Io,"./assets/items/u/utensils.png":Ro,"./assets/items/v/vacuum.png":Bo,"./assets/items/v/van.png":Ho,"./assets/items/v/vase.png":Wo,"./assets/items/v/vegetables.png":Ko,"./assets/items/v/vest.png":Jo,"./assets/items/v/village.png":Xo,"./assets/items/v/violin.png":Qo,"./assets/items/v/volcano.png":es,"./assets/items/v/vulture.png":ns,"./assets/items/w/wagon.png":is,"./assets/items/w/watch.png":os,"./assets/items/w/watermelon.png":cs,"./assets/items/w/web.png":us,"./assets/items/w/whale.png":fs,"./assets/items/w/wind.png":ms,"./assets/items/w/wizard.png":gs,"./assets/items/w/wolf.png":vs,"./assets/items/w/worm.png":bs},tg={"./assets/voice/a/accordion.webm":Ss,"./assets/voice/a/acorn.webm":ws,"./assets/voice/a/airplane.webm":Es,"./assets/voice/a/alligator.webm":Os,"./assets/voice/a/anchor.webm":As,"./assets/voice/a/ant.webm":Ms,"./assets/voice/a/apple.webm":Ps,"./assets/voice/a/arrow.webm":Is,"./assets/voice/a/astronaut.webm":Rs,"./assets/voice/alphabet/a.webm":q,"./assets/voice/alphabet/b.webm":J,"./assets/voice/alphabet/c.webm":Hs,"./assets/voice/alphabet/d.webm":Ws,"./assets/voice/alphabet/e.webm":Ks,"./assets/voice/alphabet/f.webm":Js,"./assets/voice/alphabet/g.webm":Xs,"./assets/voice/alphabet/h.webm":Qs,"./assets/voice/alphabet/i.webm":ec,"./assets/voice/alphabet/j.webm":nc,"./assets/voice/alphabet/k.webm":ic,"./assets/voice/alphabet/l.webm":oc,"./assets/voice/alphabet/m.webm":cc,"./assets/voice/alphabet/n.webm":uc,"./assets/voice/alphabet/o.webm":fc,"./assets/voice/alphabet/p.webm":mc,"./assets/voice/alphabet/q.webm":gc,"./assets/voice/alphabet/r.webm":vc,"./assets/voice/alphabet/s.webm":bc,"./assets/voice/alphabet/t.webm":Sc,"./assets/voice/alphabet/u.webm":wc,"./assets/voice/alphabet/v.webm":Y,"./assets/voice/alphabet/w.webm":Dc,"./assets/voice/alphabet/x.webm":kc,"./assets/voice/alphabet/y.webm":jc,"./assets/voice/alphabet/z.webm":Nc,"./assets/voice/b/bananas.webm":Fc,"./assets/voice/b/beachball.webm":Lc,"./assets/voice/b/bear.webm":zc,"./assets/voice/b/bee.webm":Vc,"./assets/voice/b/bird.webm":Uc,"./assets/voice/b/boat.webm":Gc,"./assets/voice/b/books.webm":qc,"./assets/voice/b/bus.webm":Yc,"./assets/voice/b/butterfly.webm":Zc,"./assets/voice/c/cake.webm":$c,"./assets/voice/c/car.webm":tl,"./assets/voice/c/castle.webm":rl,"./assets/voice/c/cat.webm":al,"./assets/voice/c/clock.webm":sl,"./assets/voice/c/coat.webm":ll,"./assets/voice/c/cow.webm":dl,"./assets/voice/c/crayon.webm":pl,"./assets/voice/c/cup.webm":hl,"./assets/voice/d/dog.webm":_l,"./assets/voice/d/doll.webm":yl,"./assets/voice/d/dolphin.webm":xl,"./assets/voice/d/donkey.webm":Cl,"./assets/voice/d/door.webm":Tl,"./assets/voice/d/dragon.webm":Dl,"./assets/voice/d/dress.webm":kl,"./assets/voice/d/drum.webm":jl,"./assets/voice/d/duck.webm":Nl,"./assets/voice/e/eagle.webm":Fl,"./assets/voice/e/ear.webm":Ll,"./assets/voice/e/earth.webm":zl,"./assets/voice/e/easel.webm":Vl,"./assets/voice/e/egg.webm":Ul,"./assets/voice/e/elephant.webm":Gl,"./assets/voice/e/engine.webm":ql,"./assets/voice/e/envelope.webm":Yl,"./assets/voice/e/eraser.webm":Zl,"./assets/voice/f/feather.webm":$l,"./assets/voice/f/fence.webm":tu,"./assets/voice/f/firefly.webm":ru,"./assets/voice/f/fish.webm":au,"./assets/voice/f/flower.webm":su,"./assets/voice/f/foot.webm":lu,"./assets/voice/f/fork.webm":du,"./assets/voice/f/fox.webm":pu,"./assets/voice/f/frog.webm":hu,"./assets/voice/g/garden.webm":_u,"./assets/voice/g/ghost.webm":yu,"./assets/voice/g/gift.webm":xu,"./assets/voice/g/giraffe.webm":Cu,"./assets/voice/g/gloves.webm":Tu,"./assets/voice/g/goat.webm":Du,"./assets/voice/g/goldfish.webm":ku,"./assets/voice/g/grapes.webm":ju,"./assets/voice/g/guitar.webm":Nu,"./assets/voice/h/hammer.webm":Fu,"./assets/voice/h/hand.webm":Lu,"./assets/voice/h/hat.webm":zu,"./assets/voice/h/heart.webm":Vu,"./assets/voice/h/hedgehog.webm":Uu,"./assets/voice/h/helicopter.webm":Gu,"./assets/voice/h/horse.webm":qu,"./assets/voice/h/hotdog.webm":Yu,"./assets/voice/h/house.webm":Zu,"./assets/voice/i/ice_cream.webm":$u,"./assets/voice/i/igloo.webm":td,"./assets/voice/i/iguana.webm":rd,"./assets/voice/i/inkpot.webm":ad,"./assets/voice/i/insect.webm":sd,"./assets/voice/i/instrument.webm":ld,"./assets/voice/i/invitation.webm":dd,"./assets/voice/i/iron.webm":pd,"./assets/voice/i/island.webm":hd,"./assets/voice/j/jaguar.webm":_d,"./assets/voice/j/jar.webm":yd,"./assets/voice/j/jean_jacket.webm":xd,"./assets/voice/j/jelly.webm":Cd,"./assets/voice/j/jellyfish.webm":Td,"./assets/voice/j/jet.webm":Dd,"./assets/voice/j/jigsaw.webm":kd,"./assets/voice/j/juggler.webm":jd,"./assets/voice/j/juice.webm":Nd,"./assets/voice/k/kangaroo.webm":Fd,"./assets/voice/k/kettle_drum.webm":Ld,"./assets/voice/k/key.webm":zd,"./assets/voice/k/kid.webm":Vd,"./assets/voice/k/kite.webm":Ud,"./assets/voice/k/kiwi.webm":Gd,"./assets/voice/k/knight.webm":qd,"./assets/voice/k/knot.webm":Yd,"./assets/voice/k/koala.webm":Zd,"./assets/voice/l/labrador.webm":$d,"./assets/voice/l/ladybug.webm":tf,"./assets/voice/l/lamp.webm":rf,"./assets/voice/l/lemon.webm":of,"./assets/voice/l/lemur.webm":cf,"./assets/voice/l/lettuce.webm":uf,"./assets/voice/l/library.webm":ff,"./assets/voice/l/llama.webm":mf,"./assets/voice/l/lungs.webm":gf,"./assets/voice/m/map.webm":vf,"./assets/voice/m/milk_carton.webm":bf,"./assets/voice/m/mitten.webm":Sf,"./assets/voice/m/monkey.webm":wf,"./assets/voice/m/moon.webm":Ef,"./assets/voice/m/mountain.webm":Of,"./assets/voice/m/mouse.webm":Af,"./assets/voice/m/muffin.webm":Mf,"./assets/voice/m/mushroom.webm":Pf,"./assets/voice/n/nail.webm":If,"./assets/voice/n/needle.webm":Rf,"./assets/voice/n/nest.webm":Bf,"./assets/voice/n/net.webm":Hf,"./assets/voice/n/newt.webm":Wf,"./assets/voice/n/night.webm":Kf,"./assets/voice/n/nose.webm":Jf,"./assets/voice/n/nurse.webm":Xf,"./assets/voice/n/nuts.webm":Qf,"./assets/voice/o/oar.webm":ep,"./assets/voice/o/octopus.webm":np,"./assets/voice/o/onion.webm":ip,"./assets/voice/o/orange.webm":op,"./assets/voice/o/orangutan.webm":cp,"./assets/voice/o/orbit.webm":up,"./assets/voice/o/ostrich.webm":fp,"./assets/voice/o/overalls.webm":mp,"./assets/voice/o/owl.webm":gp,"./assets/voice/p/panda.webm":vp,"./assets/voice/p/pen.webm":bp,"./assets/voice/p/pencil.webm":Sp,"./assets/voice/p/penguin.webm":wp,"./assets/voice/p/pig.webm":Ep,"./assets/voice/p/pineapple.webm":Op,"./assets/voice/p/pirate.webm":Ap,"./assets/voice/p/pizza.webm":Mp,"./assets/voice/p/pumpkin.webm":Pp,"./assets/voice/q/quaff.webm":Ip,"./assets/voice/q/quail.webm":Rp,"./assets/voice/q/quarter.webm":Bp,"./assets/voice/q/queen.webm":Hp,"./assets/voice/q/question.webm":Wp,"./assets/voice/q/quill.webm":Kp,"./assets/voice/q/quilt.webm":Jp,"./assets/voice/q/quince.webm":Xp,"./assets/voice/q/quiver.webm":Qp,"./assets/voice/r/rabbit.webm":em,"./assets/voice/r/radishes.webm":nm,"./assets/voice/r/rainbow.webm":im,"./assets/voice/r/ring.webm":om,"./assets/voice/r/robin.webm":cm,"./assets/voice/r/robot.webm":um,"./assets/voice/r/rocket.webm":fm,"./assets/voice/r/rose.webm":mm,"./assets/voice/r/ruler.webm":gm,"./assets/voice/s/sailboat.webm":vm,"./assets/voice/s/sheep.webm":bm,"./assets/voice/s/snake.webm":Sm,"./assets/voice/s/snowman.webm":wm,"./assets/voice/s/socks.webm":Em,"./assets/voice/s/spoon.webm":Om,"./assets/voice/s/star.webm":Am,"./assets/voice/s/strawberry.webm":Mm,"./assets/voice/s/sun.webm":Pm,"./assets/voice/t/table.webm":Im,"./assets/voice/t/teapot.webm":Rm,"./assets/voice/t/teddy_bear.webm":Bm,"./assets/voice/t/tiger.webm":Hm,"./assets/voice/t/tomato.webm":Wm,"./assets/voice/t/tractor.webm":Km,"./assets/voice/t/train.webm":Jm,"./assets/voice/t/tree.webm":Xm,"./assets/voice/t/turtle.webm":Qm,"./assets/voice/u/ukulele.webm":eh,"./assets/voice/u/umbrella.webm":nh,"./assets/voice/u/ungulate.webm":ih,"./assets/voice/u/unicorn.webm":oh,"./assets/voice/u/unicycle.webm":ch,"./assets/voice/u/unidentified_object.webm":uh,"./assets/voice/u/uniform.webm":fh,"./assets/voice/u/urchin.webm":mh,"./assets/voice/u/utensils.webm":gh,"./assets/voice/v/vacuum.webm":vh,"./assets/voice/v/van.webm":bh,"./assets/voice/v/vase.webm":Sh,"./assets/voice/v/vegetables.webm":wh,"./assets/voice/v/vest.webm":Eh,"./assets/voice/v/village.webm":Oh,"./assets/voice/v/violin.webm":Ah,"./assets/voice/v/volcano.webm":Mh,"./assets/voice/v/vulture.webm":Ph,"./assets/voice/w/wagon.webm":Ih,"./assets/voice/w/watch.webm":Rh,"./assets/voice/w/watermelon.webm":Bh,"./assets/voice/w/web.webm":Hh,"./assets/voice/w/whale.webm":Wh,"./assets/voice/w/wind.webm":Kh,"./assets/voice/w/wizard.webm":Jh,"./assets/voice/w/wolf.webm":Xh,"./assets/voice/w/worm.webm":Qh},ng={"./assets/voice/alphabet/a.webm":q,"./assets/voice/alphabet/b.webm":J,"./assets/voice/alphabet/c.webm":Hs,"./assets/voice/alphabet/d.webm":Ws,"./assets/voice/alphabet/e.webm":Ks,"./assets/voice/alphabet/f.webm":Js,"./assets/voice/alphabet/g.webm":Xs,"./assets/voice/alphabet/h.webm":Qs,"./assets/voice/alphabet/i.webm":ec,"./assets/voice/alphabet/j.webm":nc,"./assets/voice/alphabet/k.webm":ic,"./assets/voice/alphabet/l.webm":oc,"./assets/voice/alphabet/m.webm":cc,"./assets/voice/alphabet/n.webm":uc,"./assets/voice/alphabet/o.webm":fc,"./assets/voice/alphabet/p.webm":mc,"./assets/voice/alphabet/q.webm":gc,"./assets/voice/alphabet/r.webm":vc,"./assets/voice/alphabet/s.webm":bc,"./assets/voice/alphabet/t.webm":Sc,"./assets/voice/alphabet/u.webm":wc,"./assets/voice/alphabet/v.webm":Y,"./assets/voice/alphabet/w.webm":Dc,"./assets/voice/alphabet/x.webm":kc,"./assets/voice/alphabet/y.webm":jc,"./assets/voice/alphabet/z.webm":Nc},X=e=>{let t=e.split(`/`),n=t[t.length-1]??``;return{folder:t[t.length-2]??``,name:n.replace(/\.[^.]+$/,``)}},Z=new Map;for(let[e,t]of Object.entries(tg)){let{folder:n,name:r}=X(e);if(n===`alphabet`)continue;let i=t;Z.has(n)||Z.set(n,new Map),Z.get(n).set(r,i)}var Q=new Map;for(let[e,t]of Object.entries(eg)){let{folder:n,name:r}=X(e),i=r.replace(/_/g,` `),a=Z.get(n)?.get(r);Q.has(n)||Q.set(n,[]),Q.get(n).push({name:r,label:i,imageUrl:t,audioUrl:a})}var $=new Map;for(let[e,t]of Object.entries(ng)){let{name:n}=X(e);$.set(n,t)}var rg=(e,t)=>{let n=[...e];for(let e=n.length-1;e>0;--e){let t=Math.floor(Math.random()*(e+1));[n[e],n[t]]=[n[t],n[e]]}return n.slice(0,Math.min(t,n.length))};async function ig(){let e=document.querySelector(`#app`),t=new URL(`/letter_constellations/assets/aurora-BhVtVCtR.mp3`,``+import.meta.url).toString(),n=new Audio(t);n.loop=!0,n.preload=`auto`,n.volume=1;let r=new URL(`/letter_constellations/assets/chime-DLBpLcOd.mp3`,``+import.meta.url).toString(),i=new Audio(r);i.preload=`auto`;let a=null,s=null,c=.33,l=()=>{if(a)return;a=new AudioContext;let e=a.createMediaElementSource(n);s=a.createGain(),s.gain.value=.25*c,e.connect(s).connect(a.destination)},u=document.createElement(`canvas`);u.id=`aurora-canvas`,e.appendChild(u);let d=document.createElement(`canvas`);d.id=`sparkle-canvas`,e.appendChild(d);let f=document.createElement(`canvas`);f.id=`game-canvas`,e.appendChild(f);let p=document.createElement(`button`);p.className=`mute-button`,p.type=`button`,p.textContent=`Mute`,p.setAttribute(`aria-pressed`,`false`),e.appendChild(p);let m=document.createElement(`div`);m.className=`top-controls`;let h=document.createElement(`button`);h.className=`control-button`,h.type=`button`,h.textContent=`Clear`,m.appendChild(h);let g=document.createElement(`button`);g.className=`control-button`,g.type=`button`,g.textContent=`Reset`,m.appendChild(g),e.appendChild(m);let _=document.createElement(`div`);_.className=`volume-control`,_.innerHTML=`
    <label for="volume-slider">Volume</label>
    <input id="volume-slider" type="range" min="0" max="2" step="0.05" value="0.33" />
  `,e.appendChild(_);let v=_.querySelector(`#volume-slider`);c=Number(v.value),i.volume=Math.min(1,c);let y=new Map,b=new Set,x=null,S=0,C=null,w=e=>{let t=y.get(e);return t||(t=new Audio(e),t.preload=`auto`,y.set(e,t),b.add(t)),t.volume=Math.min(1,Number(v.value)),t.muted=n.muted,t},D=()=>{x&&(x.pause(),x.currentTime=0)},O=async e=>{if(e.length===0)return;let t=++S;D();for(let n of e){if(t!==S)return;let e=w(n);x=e,e.currentTime=0;try{await e.play()}catch{return}await new Promise(t=>{let n=()=>{e.removeEventListener(`ended`,n),e.removeEventListener(`error`,n),t()};e.addEventListener(`ended`,n),e.addEventListener(`error`,n)})}},k=e=>{let t=$.get(e);if(t){if(!j){C=t;return}O([t])}},A=()=>{let e=n.muted;p.textContent=e?`Unmute`:`Mute`,p.setAttribute(`aria-pressed`,String(e)),i.muted=e;for(let t of b)t.muted=e};A();let j=!1,M=()=>{j&&(l(),a&&a.state===`suspended`&&a.resume().catch(()=>{}),n.play().catch(()=>{}))},N=()=>{if(j=!0,M(),C){let e=C;C=null,O([e])}},P={once:!0,capture:!0};document.addEventListener(`pointerdown`,N,P),document.addEventListener(`touchstart`,N,P),document.addEventListener(`click`,N,P),document.addEventListener(`keydown`,N,P),p.addEventListener(`click`,e=>{e.stopPropagation(),n.muted=!n.muted,n.muted||M(),A()}),v.addEventListener(`input`,()=>{l();let e=Number(v.value);c=e,s&&(s.gain.value=.25*e),i.volume=Math.min(1,e);let t=Math.min(1,e);for(let e of b)e.volume=t});let F=document.createElement(`div`);F.className=`item-tray`,e.appendChild(F);let te=e=>{F.replaceChildren();let t=Q.get(e)??[];if(t.length===0){let t=document.createElement(`div`);t.className=`item-empty`,t.textContent=`No ${e.toUpperCase()} items yet.`,F.appendChild(t);return}let n=rg(t,3);for(let t of n){let n=document.createElement(`button`);n.type=`button`,n.className=`item-card`,n.setAttribute(`aria-label`,t.label),n.addEventListener(`click`,n=>{n.stopPropagation();let r=[],i=$.get(e);i&&r.push(i),t.audioUrl&&r.push(t.audioUrl),O(r)});let r=document.createElement(`img`);r.src=t.imageUrl,r.alt=t.label,r.loading=`lazy`,r.decoding=`async`,n.appendChild(r),F.appendChild(n)}},I=null,L=null,R=null,z=new T(u,f,d),B=f.getContext(`2d`);if(!B)throw Error(`Failed to get 2D context for game canvas.`);let V=()=>{let{width:e,height:t}=z.resize();z.configureGameContext(B),R&&R.setViewportSize(e,t),I&&I.resize(u.width,u.height),L&&L.resize(d.width,d.height)};V(),window.addEventListener(`resize`,V),I=new o(u);let ne=await I.initialize();L=new ee(d);let re=await L.initialize(I.getDevice());if(V(),!ne||!re){let e=document.createElement(`div`);e.className=`error-message`,e.innerHTML=`
            <h1>WebGPU Not Supported</h1>
            <p>Your browser doesn't support WebGPU yet.</p>
            <p>Try using Chrome 113+ or Edge 113+ with WebGPU enabled.</p>
        `,document.body.appendChild(e);return}R=new ye(new E(f,e=>z.getPointFromEvent(f,e)),{onPointHit:()=>{i.currentTime=0,i.play().catch(()=>{})},onLetterChange:e=>{te(e),k(e)}}),h.addEventListener(`click`,e=>{e.stopPropagation(),R?.clearUserPaths()}),g.addEventListener(`click`,e=>{e.stopPropagation(),R?.resetCurrentLetter()});let H=z.getSize();R.setViewportSize(H.width,H.height);let U=performance.now(),ie=0;function W(e){let t=(e-U)/1e3;if(U=e,!R||!I||!L)return;R.update(t);let n=R.getDrawingState();L.updateMouseState(n.isDrawing,n.mouseX,n.mouseY,n.trailPoints);let r=R.getDotState();L.setDots(r.dots,r.currentIndex,r.radius),ie++&1||I.render(),L.render(),B&&R.render(B),requestAnimationFrame(W)}W(performance.now())}ig();
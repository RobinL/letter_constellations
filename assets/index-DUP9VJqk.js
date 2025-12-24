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
`,r=`/letter_constellations/assets/background_landscape-C0UnJOSH.webp`,i=`/letter_constellations/assets/background_portrait-CWz74FnL.webp`;async function a(e){let t=await fetch(e);if(!t.ok)throw Error(`Failed to load image: ${e}`);let n=await t.blob();return createImageBitmap(n)}var o=class{canvas;device;context;pipeline;uniformBuffer;bindGroup;bindGroupLayout;backgroundLandscapeImage;backgroundPortraitImage;backgroundLandscapeTexture;backgroundPortraitTexture;backgroundLandscapeView;backgroundPortraitView;sampler;currentBackground=`landscape`;backgroundTextureWidth=0;backgroundTextureHeight=0;startTime;constructor(e){this.canvas=e,this.startTime=performance.now()}async initialize(){if(!navigator.gpu)return console.error(`WebGPU not supported in this browser`),!1;let e=await navigator.gpu.requestAdapter();if(!e)return console.error(`Failed to get GPU adapter`),!1;this.device=await e.requestDevice(),this.context=this.canvas.getContext(`webgpu`);let t=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device:this.device,format:t,alphaMode:`premultiplied`});let[o,s]=await Promise.all([a(r),a(i)]);this.backgroundLandscapeImage=o,this.backgroundPortraitImage=s,this.sampler=this.device.createSampler({magFilter:`linear`,minFilter:`linear`}),this.uniformBuffer=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});let c=this.device.createShaderModule({code:n});this.bindGroupLayout=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:`float`}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:`filtering`}}]}),this.updateBackgroundBindGroup();let l=this.device.createPipelineLayout({bindGroupLayouts:[this.bindGroupLayout]});return this.pipeline=this.device.createRenderPipeline({layout:l,vertex:{module:c,entryPoint:`vertexMain`},fragment:{module:c,entryPoint:`fragmentMain`,targets:[{format:t}]},primitive:{topology:`triangle-list`}}),!0}createCoveredTexture(e,t,n){let r=document.createElement(`canvas`);r.width=t,r.height=n;let i=r.getContext(`2d`);if(!i)throw Error(`Failed to get 2D context for background`);i.fillStyle=`#000`,i.fillRect(0,0,t,n);let a=e.width/e.height,o=t/n,s=t,c=n,l=0,u=0;o>a?(s=t,c=Math.round(t/a),u=Math.floor((n-c)/2)):(c=n,s=Math.round(n*a),l=Math.floor((t-s)/2)),i.drawImage(e,l,u,s,c);let d=this.device.createTexture({size:[t,n,1],format:`rgba8unorm`,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});return this.device.queue.copyExternalImageToTexture({source:r},{texture:d},[t,n]),d}refreshBackgroundTextures(e,t){e===0||t===0||(this.backgroundLandscapeTexture&&this.backgroundLandscapeTexture.destroy(),this.backgroundPortraitTexture&&this.backgroundPortraitTexture.destroy(),this.backgroundLandscapeTexture=this.createCoveredTexture(this.backgroundLandscapeImage,e,t),this.backgroundPortraitTexture=this.createCoveredTexture(this.backgroundPortraitImage,e,t),this.backgroundLandscapeView=this.backgroundLandscapeTexture.createView(),this.backgroundPortraitView=this.backgroundPortraitTexture.createView(),this.backgroundTextureWidth=e,this.backgroundTextureHeight=t)}render(){let e=(performance.now()-this.startTime)/1e3,t=new Float32Array([e,0,this.canvas.width,this.canvas.height,0,0,0,0]);this.device.queue.writeBuffer(this.uniformBuffer,0,t);let n=this.device.createCommandEncoder(),r=n.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:`clear`,storeOp:`store`}]});r.setPipeline(this.pipeline),r.setBindGroup(0,this.bindGroup),r.draw(3),r.end(),this.device.queue.submit([n.finish()])}resize(e,t){this.canvas.width=e,this.canvas.height=t,this.updateBackgroundBindGroup();let n=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device:this.device,format:n,alphaMode:`premultiplied`})}getDevice(){return this.device}updateBackgroundBindGroup(){if(!this.uniformBuffer||!this.bindGroupLayout)return;let e=!1;(this.canvas.width!==this.backgroundTextureWidth||this.canvas.height!==this.backgroundTextureHeight)&&(this.refreshBackgroundTextures(this.canvas.width,this.canvas.height),e=!0);let t=this.canvas.height>this.canvas.width?`portrait`:`landscape`;if(t===this.currentBackground&&this.bindGroup&&!e)return;this.currentBackground=t;let n=t===`portrait`?this.backgroundPortraitView:this.backgroundLandscapeView;n&&(this.bindGroup=this.device.createBindGroup({layout:this.bindGroupLayout,entries:[{binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:n},{binding:2,resource:this.sampler}]}))}},s=`// Instanced star dots for the dot-to-dot letters.
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
`,l=16384,u=8,d=u*4,f=1024,p=512,m=8,h=m*4,g=0,_=1,v=2,y=3,b=.7,x=.8,S=12,C=.03;function w(e,t,n){let r=Math.max(0,Math.min(1,(n-e)/(t-e)));return r*r*(3-2*r)}var ee=class{canvas;device;context;pipeline;dotPipeline;uniformBuffer;dotUniformBuffer;bindGroup;dotBindGroup;quadVertexBuffer;particleBuffer;dotBuffer;startTimeMs;uniformScratch=new Float32Array(8);dotUniformScratch=new Float32Array(8);mouseRamp=0;prevActive=!1;mousePos={x:.5,y:.5};mouseVelocity={x:0,y:0};lastSimTimeSec=0;dotCount=0;dotScratch=new Float32Array(p*m);dotStates=new Int8Array(p);dotStateTimes=new Float32Array(p);nextParticleIndex=0;pendingStart=0;pendingCount=0;pendingData=new Float32Array(f*u);sparkAccumulator=0;dustAccumulator=0;burstCooldown=0;haloPhase=0;haloAccumulator=0;stillSeconds=0;lastStampTime=-999;rngState=305419896;constructor(e){this.canvas=e,this.startTimeMs=performance.now()}async initialize(e){if(e)this.device=e;else{if(!navigator.gpu)return console.error(`WebGPU not supported`),!1;let e=await navigator.gpu.requestAdapter();if(!e)return console.error(`Failed to get GPU adapter`),!1;this.device=await e.requestDevice()}this.context=this.canvas.getContext(`webgpu`);let t=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device:this.device,format:t,alphaMode:`premultiplied`}),this.uniformBuffer=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.dotUniformBuffer=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.quadVertexBuffer=this.device.createBuffer({size:48,usage:GPUBufferUsage.VERTEX,mappedAtCreation:!0}),new Float32Array(this.quadVertexBuffer.getMappedRange()).set([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),this.quadVertexBuffer.unmap(),this.particleBuffer=this.device.createBuffer({size:l*d,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Float32Array(this.particleBuffer.getMappedRange()).fill(0),this.particleBuffer.unmap(),this.dotBuffer=this.device.createBuffer({size:p*h,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Float32Array(this.dotBuffer.getMappedRange()).fill(0),this.dotBuffer.unmap();let n=this.device.createShaderModule({code:s}),r=this.device.createShaderModule({code:c}),i=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:`uniform`}}]});this.bindGroup=this.device.createBindGroup({layout:i,entries:[{binding:0,resource:{buffer:this.uniformBuffer}}]}),this.dotBindGroup=this.device.createBindGroup({layout:i,entries:[{binding:0,resource:{buffer:this.dotUniformBuffer}}]});let a=this.device.createPipelineLayout({bindGroupLayouts:[i]});return this.pipeline=this.device.createRenderPipeline({layout:a,vertex:{module:r,entryPoint:`sparkVertexMain`,buffers:[{arrayStride:8,stepMode:`vertex`,attributes:[{shaderLocation:0,offset:0,format:`float32x2`}]},{arrayStride:d,stepMode:`instance`,attributes:[{shaderLocation:1,offset:0,format:`float32x4`},{shaderLocation:2,offset:16,format:`float32x4`}]}]},fragment:{module:r,entryPoint:`sparkFragmentMain`,targets:[{format:t,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one-minus-src-alpha`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),this.dotPipeline=this.device.createRenderPipeline({layout:a,vertex:{module:n,entryPoint:`dotVertexMain`,buffers:[{arrayStride:8,stepMode:`vertex`,attributes:[{shaderLocation:0,offset:0,format:`float32x2`}]},{arrayStride:h,stepMode:`instance`,attributes:[{shaderLocation:1,offset:0,format:`float32x4`},{shaderLocation:2,offset:16,format:`float32x4`}]}]},fragment:{module:n,entryPoint:`dotFragmentMain`,targets:[{format:t,blend:{color:{srcFactor:`one`,dstFactor:`one`,operation:`add`},alpha:{srcFactor:`one`,dstFactor:`one-minus-src-alpha`,operation:`add`}}}]},primitive:{topology:`triangle-list`}}),!0}setDots(e,t,n){let r=Math.min(e.length,p);if(this.dotCount=r,r===0)return;let i=(performance.now()-this.startTimeMs)/1e3,{dpr:a,cssW:o,cssH:s}=this.getCanvasMetrics(),c=Math.max(10,n*1.3);for(let n=0;n<r;n+=1){let r=e[n],l=r.x/o,u=r.y/s,d=0;n<t?d=2:n===t&&(d=1),this.dotStates[n]!==d&&(this.dotStates[n]=d,this.dotStateTimes[n]=i,d===2&&this.emitDotCompleteBurst(i,{x:l,y:u},a));let f=c*(d===1?1.7:d===2?.95:.85)*a,p=n*m;this.dotScratch[p+0]=l,this.dotScratch[p+1]=u,this.dotScratch[p+2]=f,this.dotScratch[p+3]=d,this.dotScratch[p+4]=this.hashSeed(n),this.dotScratch[p+5]=this.dotStateTimes[n],this.dotScratch[p+6]=0,this.dotScratch[p+7]=0}this.device.queue.writeBuffer(this.dotBuffer,0,this.dotScratch.subarray(0,r*m))}updateMouseState(e,t,n,r){let i=(performance.now()-this.startTimeMs)/1e3,a=this.lastSimTimeSec===0?0:i-this.lastSimTimeSec,o=Math.max(0,Math.min(.05,a));if(this.lastSimTimeSec=i,o>0){let t=o/.45;this.mouseRamp=e?Math.min(1,this.mouseRamp+t):Math.max(0,this.mouseRamp-t)}let{dpr:s,cssW:c,cssH:l}=this.getCanvasMetrics(),u=Math.max(0,Math.min(1,t/c)),d=Math.max(0,Math.min(1,n/l)),f={...this.mousePos},p={x:u,y:d};if(this.mousePos=p,o>0){let e=(p.x-f.x)/o,t=(p.y-f.y)/o;this.mouseVelocity.x=this.mouseVelocity.x*.7+e*.3,this.mouseVelocity.y=this.mouseVelocity.y*.7+t*.3}let m=Math.hypot(this.mouseVelocity.x,this.mouseVelocity.y),h=w(.08,2.4,m);if(o>0&&this.mouseRamp>.02){this.haloPhase+=o*(1.2+h*5.5);let e=(18+h*140)*this.mouseRamp*b;this.haloAccumulator+=e*o;let t=Math.floor(this.haloAccumulator);this.haloAccumulator-=t,t=Math.min(t,10);let n=.014+.012*h;for(let e=0;e<t;e++){let e=this.haloPhase*Math.PI*2+this.rand01()*Math.PI*2,t=Math.cos(e)*n,r=Math.sin(e)*n,a=.18+h*.65,o=-Math.sin(e)*a+t*.8,c=Math.cos(e)*a+r*.8,l=.35+this.rand01()*.55,u=(.9+this.rand01()*2.2)*x,d=Math.min(u,S)*s;this.queueParticle(p.x+t,p.y+r,o,c,i,l,d,this.makeSeed(g))}}if(o>0&&(e&&m<.07?this.stillSeconds+=o:this.stillSeconds=0,e&&this.stillSeconds>.22&&i-this.lastStampTime>.35&&(this.stampHeroStar(i,p,s,10,18,1.1,1.8),this.emitDustRing(i,p,s,18),this.lastStampTime=i,this.stillSeconds=.12)),e&&!this.prevActive&&(this.emitRadialBurst(i,p,s,300),this.burstCooldown=.12),o>0&&(this.burstCooldown=Math.max(0,this.burstCooldown-o),e&&this.burstCooldown<=0&&h>.85&&this.mouseRamp>.25&&(this.emitDirectionalBurst(i,p,s,110),this.burstCooldown=.12)),o>0){let e=(120+h*1700)*this.mouseRamp*b;this.sparkAccumulator+=e*o;let t=Math.floor(this.sparkAccumulator);if(this.sparkAccumulator-=t,t=Math.min(t,120),t>0){let e=this.mouseVelocity.x,n=this.mouseVelocity.y,r=Math.hypot(e,n);if(r>1e-5)e/=r,n/=r;else{let t=this.rand01()*Math.PI*2;e=Math.cos(t),n=Math.sin(t)}let a=-n,o=e,c=1.25-h*.9;for(let r=0;r<t;r++){let t=this.rand01(),r=f.x+(p.x-f.x)*t,l=f.y+(p.y-f.y)*t,u=(this.rand01()-.5)*.01,d=a*u,m=o*u,g=(this.rand01()-.5)*c,v=Math.cos(g),b=Math.sin(g),w=e*v-n*b,ee=e*b+n*v,T=.1+this.rand01()*.2+h*.75,te=(this.rand01()-.5)*(.36-h*.18),ne=w*T+a*te*T,re=ee*T+o*te*T,E=h>.75&&this.rand01()<C,ie=E?y:_,ae=E?.45+this.rand01()*.55:.2+this.rand01()*.38+h*.18,D=(E?6+this.rand01()*10+h*10:1.4+this.rand01()*3.4+h*8)*x,O=Math.min(D,S)*s;this.queueParticle(r+d,l+m,ne,re,i,ae,O,this.makeSeed(ie))}}}if(o>0&&r.length>0){let e=(45+160*Math.min(1,r.length/160))*(.25+.75*this.mouseRamp)*b;this.dustAccumulator+=e*o;let t=Math.floor(this.dustAccumulator);this.dustAccumulator-=t,t=Math.min(t,45);for(let e=0;e<t;e++){let e=r[this.rand01()*r.length|0],t=Math.max(0,Math.min(1,e.x/c)),n=Math.max(0,Math.min(1,e.y/l)),a=this.rand01()*Math.PI*2,o=.014+this.rand01()*.03,u=Math.cos(a)*o,d=Math.sin(a)*o,f=.45+this.rand01()*1.2,p=(.9+this.rand01()*2.8)*x,m=Math.min(p,S)*s;this.queueParticle(t,n,u,d,i,f,m,this.makeSeed(g))}}e||(this.mouseVelocity.x*=.95,this.mouseVelocity.y*=.95),!e&&this.prevActive&&(this.stampHeroStar(i,p,s,14,26,1.4,2.4),this.emitDustRing(i,p,s,28)),this.prevActive=e}render(){let e=(performance.now()-this.startTimeMs)/1e3;this.flushPendingParticles(),this.uniformScratch[0]=e,this.uniformScratch[1]=0,this.uniformScratch[2]=this.canvas.width,this.uniformScratch[3]=this.canvas.height,this.uniformScratch[4]=this.mouseRamp,this.uniformScratch[5]=0,this.uniformScratch[6]=0,this.uniformScratch[7]=0,this.device.queue.writeBuffer(this.uniformBuffer,0,this.uniformScratch),this.dotUniformScratch[0]=e,this.dotUniformScratch[1]=0,this.dotUniformScratch[2]=this.canvas.width,this.dotUniformScratch[3]=this.canvas.height,this.dotUniformScratch[4]=1,this.dotUniformScratch[5]=this.mouseRamp,this.dotUniformScratch[6]=this.mousePos.x,this.dotUniformScratch[7]=this.mousePos.y,this.device.queue.writeBuffer(this.dotUniformBuffer,0,this.dotUniformScratch);let t=this.device.createCommandEncoder(),n=t.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:`clear`,storeOp:`store`}]});this.dotCount>0&&(n.setPipeline(this.dotPipeline),n.setBindGroup(0,this.dotBindGroup),n.setVertexBuffer(0,this.quadVertexBuffer),n.setVertexBuffer(1,this.dotBuffer),n.draw(6,this.dotCount,0,0)),n.setPipeline(this.pipeline),n.setBindGroup(0,this.bindGroup),n.setVertexBuffer(0,this.quadVertexBuffer),n.setVertexBuffer(1,this.particleBuffer),n.draw(6,l,0,0),n.end(),this.device.queue.submit([t.finish()])}resize(e,t){this.canvas.width=e,this.canvas.height=t;let n=navigator.gpu.getPreferredCanvasFormat();this.context.configure({device:this.device,format:n,alphaMode:`premultiplied`})}getCanvasMetrics(){let e=Math.max(1,this.canvas.clientWidth||this.canvas.width),t=Math.max(1,this.canvas.clientHeight||this.canvas.height),n=this.canvas.width/e;return{dpr:Math.max(.5,n),cssW:e,cssH:t}}rand01(){let e=this.rngState|0;return e^=e<<13,e^=e>>>17,e^=e<<5,this.rngState=e,(e>>>0)/4294967296}hashSeed(e){let t=(e+1)*2654435761;return t^=t<<13,t^=t>>>17,t^=t<<5,(t>>>0)/4294967296}makeSeed(e){return e+this.rand01()}queueParticle(e,t,n,r,i,a,o,s){if(this.pendingCount>=f)return;this.pendingCount===0&&(this.pendingStart=this.nextParticleIndex),this.nextParticleIndex=(this.nextParticleIndex+1)%l;let c=this.pendingCount*u,d=this.pendingData;d[c+0]=e,d[c+1]=t,d[c+2]=n,d[c+3]=r,d[c+4]=i,d[c+5]=a,d[c+6]=o,d[c+7]=s,this.pendingCount++}flushPendingParticles(){if(this.pendingCount===0)return;let e=this.pendingCount,t=this.pendingStart,n=Math.min(e,l-t),r=n*u;if(this.device.queue.writeBuffer(this.particleBuffer,t*d,this.pendingData.subarray(0,r)),e>n){let t=(e-n)*u;this.device.queue.writeBuffer(this.particleBuffer,0,this.pendingData.subarray(r,r+t))}this.pendingCount=0}emitRadialBurst(e,t,n,r){let i=Math.min(Math.floor(r*b),f);for(let r=0;r<i;r++){let r=this.rand01()*Math.PI*2,i=.14+this.rand01()*.8,a=Math.cos(r)*i,o=Math.sin(r)*i,s=.32+this.rand01()*.75,c=(2.2+this.rand01()*7)*x,l=Math.min(c,S)*n;this.queueParticle(t.x,t.y,a,o,e,s,l,this.makeSeed(_))}}emitDirectionalBurst(e,t,n,r){let i=this.mouseVelocity.x,a=this.mouseVelocity.y,o=Math.hypot(i,a);o>1e-5?(i/=o,a/=o):(i=1,a=0);let s=-a,c=i,l=Math.min(Math.floor(r*b),f);for(let r=0;r<l;r++){let r=(this.rand01()-.5)*.35,o=Math.cos(r),l=Math.sin(r),u=i*o-a*l,d=i*l+a*o,f=.17+this.rand01()*1,p=(this.rand01()-.5)*.25,m=u*f+s*p,h=d*f+c*p,g=.28+this.rand01()*.6,v=(2+this.rand01()*8.5)*x,y=Math.min(v,S)*n;this.queueParticle(t.x,t.y,m,h,e,g,y,this.makeSeed(_))}}stampHeroStar(e,t,n,r,i,a,o){let s=(r+this.rand01()*(i-r))*x,c=Math.min(s,S*1.5)*n,l=a+this.rand01()*(o-a),u=(this.rand01()-.5)*.05,d=-.05-this.rand01()*.07;this.queueParticle(t.x,t.y,u,d,e,l,c,this.makeSeed(v))}emitDustRing(e,t,n,r){let i=Math.min(Math.floor(r*b),30),a=.018;for(let r=0;r<i;r++){let r=this.rand01()*Math.PI*2,i=Math.cos(r)*a*(.6+this.rand01()*.6),o=Math.sin(r)*a*(.6+this.rand01()*.6),s=.03+this.rand01()*.08,c=i*s*30,l=o*s*30,u=.5+this.rand01()*.7,d=(.8+this.rand01()*2)*x,f=Math.min(d,S)*n;this.queueParticle(t.x+i,t.y+o,c,l,e,u,f,this.makeSeed(g))}}emitDotCompleteBurst(e,t,n){this.emitRadialBurst(e,t,n,320),this.emitDustRing(e,t,n,80),this.stampHeroStar(e,t,n,18,28,1.2,2)}},T=class{size;auroraScale=.5;sparkleScale=.75;constructor(e,t,n){this.auroraCanvas=e,this.gameCanvas=t,this.sparkleCanvas=n,this.size={width:0,height:0,dpr:1}}updateQuality(e){this.auroraScale=e.auroraScale,this.sparkleScale=e.sparkleScale}resize(){let e=window.innerWidth,t=window.innerHeight,n=Math.max(1,window.devicePixelRatio||1),r=Math.min(1,n),i=Math.max(.5,r*this.auroraScale),a=Math.max(.5,r*this.sparkleScale);return this.applySize(this.auroraCanvas,e,t,i),this.applySize(this.gameCanvas,e,t,n),this.sparkleCanvas&&this.applySize(this.sparkleCanvas,e,t,a),this.size={width:e,height:t,dpr:n},this.size}configureGameContext(e){let{dpr:t}=this.size;e.setTransform(t,0,0,t,0,0),e.imageSmoothingEnabled=!0}getPointFromEvent(e,t){let n=e.getBoundingClientRect();return{x:t.clientX-n.left,y:t.clientY-n.top}}getSize(){return this.size}applySize(e,t,n,r){e.width=Math.max(1,Math.floor(t*r)),e.height=Math.max(1,Math.floor(n*r)),e.style.width=`${t}px`,e.style.height=`${n}px`}},te=class{callbacks={};activePointerId=null;constructor(e,t){this.canvas=e,this.getPoint=t,this.onPointerDown=this.onPointerDown.bind(this),this.onPointerMove=this.onPointerMove.bind(this),this.onPointerUp=this.onPointerUp.bind(this),this.canvas.addEventListener(`pointerdown`,this.onPointerDown),this.canvas.addEventListener(`pointermove`,this.onPointerMove),this.canvas.addEventListener(`pointerup`,this.onPointerUp),this.canvas.addEventListener(`pointercancel`,this.onPointerUp),this.canvas.addEventListener(`pointerleave`,this.onPointerUp)}setCallbacks(e){this.callbacks=e}dispose(){this.canvas.removeEventListener(`pointerdown`,this.onPointerDown),this.canvas.removeEventListener(`pointermove`,this.onPointerMove),this.canvas.removeEventListener(`pointerup`,this.onPointerUp),this.canvas.removeEventListener(`pointercancel`,this.onPointerUp),this.canvas.removeEventListener(`pointerleave`,this.onPointerUp)}onPointerDown(e){if(this.activePointerId!==null)return;this.activePointerId=e.pointerId,this.canvas.setPointerCapture(e.pointerId),e.preventDefault();let t=this.buildPoint(e);this.callbacks.onStart?.(t)}onPointerMove(e){if(this.activePointerId!==e.pointerId)return;e.preventDefault();let t=this.buildPoint(e);this.callbacks.onMove?.(t)}onPointerUp(e){if(this.activePointerId!==e.pointerId)return;e.preventDefault();let t=this.buildPoint(e);this.callbacks.onEnd?.(t),this.canvas.releasePointerCapture(e.pointerId),this.activePointerId=null}buildPoint(e){let{x:t,y:n}=this.getPoint(e);return{x:t,y:n,id:e.pointerId,time:performance.now()/1e3}}},ne=t({default:()=>re}),re=[{order:1,x:771,y:353,display:`keystone`,note:`Start at Black Dot`},{order:2,x:660,y:255,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:580,y:215,display:`intermediate`,note:`Curve Up-Left`},{order:4,x:500,y:200,display:`keystone`,note:`Top Arch Center`},{order:5,x:420,y:220,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:350,y:265,display:`intermediate`,note:`Curve Down-Left`},{order:7,x:290,y:340,display:`keystone`,note:`Top Left Shoulder`},{order:8,x:250,y:435,display:`intermediate`,note:`Left Vertical`},{order:9,x:240,y:535,display:`intermediate`,note:`Left Vertical`},{order:10,x:260,y:635,display:`keystone`,note:`Bottom Left Turn`},{order:11,x:320,y:720,display:`intermediate`,note:`Bottom Curve`},{order:12,x:410,y:775,display:`intermediate`,note:`Bottom Curve`},{order:13,x:510,y:780,display:`keystone`,note:`Bottom Center`},{order:14,x:622,y:750,display:`intermediate`,note:`Curve Up-Right`},{order:15,x:714,y:694,display:`intermediate`,note:`Curve Up-Right`},{order:16,x:769,y:614,display:`keystone`,note:`Closing the Loop`},{order:17,x:782,y:532,display:`intermediate`,note:`Loop End`},{order:18,x:805,y:411,display:`intermediate`,note:`Loop End`},{order:19,x:810,y:256,display:`intermediate`,newStroke:!0,note:`Lift Pen -> Top of Stem`},{order:20,x:810,y:309,display:`keystone`,note:`Stem Down`},{order:21,x:810,y:350,display:`intermediate`,note:`Stem Down`},{order:22,x:810,y:435,display:`intermediate`,note:`Stem Down`},{order:23,x:810,y:520,display:`keystone`,note:`Stem Center`},{order:24,x:810,y:605,display:`intermediate`,note:`Stem Down`},{order:25,x:805,y:690,display:`keystone`,note:`Stem Bottom`},{order:26,x:800,y:778,display:`intermediate`,note:`Stem Finish`}],E=t({default:()=>ie}),ie=[{order:1,x:240,y:42,display:`keystone`,note:`Start Dot (Top of Stem)`},{order:2,x:239,y:166,display:`intermediate`,note:`Stem Down`},{order:3,x:239,y:263,display:`intermediate`,note:`Stem Down`},{order:4,x:240,y:390,display:`keystone`,note:`Stem Middle`},{order:5,x:240,y:470,display:`intermediate`,note:`Stem Down`},{order:6,x:240,y:550,display:`intermediate`,note:`Stem Down`},{order:7,x:240,y:630,display:`keystone`,note:`Stem Lower`},{order:8,x:240,y:710,display:`intermediate`,note:`Stem Down`},{order:9,x:240,y:790,display:`intermediate`,note:`Stem Down`},{order:10,x:240,y:864,display:`keystone`,note:`Stem Bottom Arrow`},{order:11,x:241,y:725,display:`keystone`,note:`Stem Bottom Arrow`},{order:12,x:236,y:522,display:`keystone`,note:`Stem Bottom Arrow`},{order:13,x:240,y:450,display:`intermediate`,newStroke:!0,note:`Lift Pen -> Start Loop (Mid-Stem)`},{order:14,x:290,y:390,display:`intermediate`,note:`Loop Curve Up-Right`},{order:15,x:380,y:350,display:`keystone`,note:`Loop Top Left`},{order:16,x:490,y:335,display:`intermediate`,note:`Loop Top Center`},{order:17,x:610,y:350,display:`intermediate`,note:`Loop Top Right`},{order:18,x:710,y:410,display:`keystone`,note:`Loop Top Curve`},{order:19,x:780,y:500,display:`intermediate`,note:`Loop Far Right`},{order:20,x:800,y:600,display:`intermediate`,note:`Loop Far Right Mid`},{order:21,x:780,y:700,display:`keystone`,note:`Loop Bottom Right Turn`},{order:22,x:710,y:790,display:`intermediate`,note:`Loop Bottom Curve`},{order:23,x:610,y:845,display:`intermediate`,note:`Loop Bottom Right`},{order:24,x:500,y:870,display:`keystone`,note:`Loop Bottom Center`},{order:25,x:391,y:871,display:`intermediate`,note:`Loop Bottom Left`},{order:26,x:315,y:870,display:`intermediate`,note:`Loop Closing In`},{order:27,x:241,y:863,display:`keystone`,note:`Loop Connect to Stem`}],ae=t({default:()=>D}),D=[{order:1,x:760,y:230,display:`keystone`,note:`Start Black Dot`},{order:2,x:700,y:190,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:635,y:165,display:`intermediate`,note:`Curve Up-Left`},{order:4,x:565,y:150,display:`keystone`,note:`Top Arch Right`},{order:5,x:495,y:150,display:`intermediate`,note:`Top Arch Center`},{order:6,x:425,y:165,display:`intermediate`,note:`Top Arch Left`},{order:7,x:360,y:195,display:`keystone`,note:`Curve Down-Left`},{order:8,x:300,y:240,display:`intermediate`,note:`Curve Down-Left`},{order:9,x:250,y:300,display:`intermediate`,note:`Upper Left`},{order:10,x:210,y:370,display:`keystone`,note:`Left Curve High`},{order:11,x:180,y:450,display:`intermediate`,note:`Left Curve Mid`},{order:12,x:170,y:530,display:`intermediate`,note:`Left Vertical Center`},{order:13,x:180,y:610,display:`keystone`,note:`Left Curve Low`},{order:14,x:210,y:690,display:`intermediate`,note:`Bottom Left Turn`},{order:15,x:260,y:760,display:`intermediate`,note:`Bottom Left Curve`},{order:16,x:320,y:810,display:`keystone`,note:`Bottom Arch Left`},{order:17,x:390,y:845,display:`intermediate`,note:`Bottom Arch Mid-Left`},{order:18,x:470,y:860,display:`intermediate`,note:`Bottom Arch Center`},{order:19,x:550,y:855,display:`keystone`,note:`Bottom Arch Right`},{order:20,x:620,y:830,display:`intermediate`,note:`Curve Up-Right`},{order:21,x:680,y:790,display:`intermediate`,note:`Curve Up-Right`},{order:22,x:730,y:740,display:`keystone`,note:`Lower Right Turn`},{order:26,x:780,y:660,display:`intermediate`,note:`Finish Arrow Tip`}],O=t({default:()=>k}),k=[{order:1,x:736,y:422,display:`keystone`,note:`Start Black Dot`},{order:2,x:631,y:311,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:512,y:266,display:`intermediate`,note:`Curve Up-Left`},{order:4,x:390,y:280,display:`keystone`,note:`Top of Loop`},{order:5,x:300,y:340,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:223,y:420,display:`intermediate`,note:`Curve Down-Left`},{order:7,x:183,y:550,display:`keystone`,note:`Leftmost Curve`},{order:8,x:207,y:680,display:`intermediate`,note:`Curve Down-Right`},{order:9,x:280,y:760,display:`intermediate`,note:`Curve Down-Right`},{order:10,x:400,y:800,display:`keystone`,note:`Bottom of Loop`},{order:11,x:520,y:780,display:`intermediate`,note:`Curve Up-Right`},{order:12,x:650,y:720,display:`intermediate`,note:`Approach Stem`},{order:13,x:780,y:620,display:`keystone`,note:`Connect to Stem (Go Up)`},{order:14,x:780,y:520,display:`intermediate`,note:`Stem Up`},{order:15,x:780,y:420,display:`intermediate`,note:`Stem Up`},{order:16,x:780,y:320,display:`keystone`,note:`Stem Up High`},{order:17,x:780,y:166,display:`intermediate`,note:`Stem Up`},{order:18,x:780,y:56,display:`intermediate`,note:`Reach Top`},{order:19,x:779,y:-79,display:`keystone`,note:`Top Turnaround`},{order:20,x:780,y:239,display:`intermediate`,note:`Stem Down (Retrace)`},{order:21,x:780,y:400,display:`intermediate`,note:`Stem Down`},{order:22,x:780,y:550,display:`keystone`,note:`Stem Down Mid`},{order:23,x:779,y:687,display:`keystone`,note:`Stem Bottom`}],A=t({default:()=>j}),j=[{order:1,x:220,y:440,display:`keystone`,note:`Start Black Dot`},{order:2,x:330,y:440,display:`intermediate`,note:`Horizontal Bar`},{order:3,x:440,y:440,display:`intermediate`,note:`Horizontal Bar`},{order:4,x:550,y:440,display:`keystone`,note:`Center of Bar`},{order:5,x:660,y:440,display:`intermediate`,note:`Horizontal Bar`},{order:6,x:770,y:440,display:`intermediate`,note:`End of Bar (Turn Up)`},{order:7,x:760,y:360,display:`keystone`,note:`Curve Up-Left`},{order:8,x:700,y:280,display:`intermediate`,note:`Curve Up-Left`},{order:9,x:620,y:220,display:`intermediate`,note:`Top Right Curve`},{order:10,x:500,y:190,display:`keystone`,note:`Top Center`},{order:11,x:380,y:200,display:`intermediate`,note:`Top Left Curve`},{order:12,x:280,y:250,display:`intermediate`,note:`Curve Down-Left`},{order:13,x:200,y:330,display:`keystone`,note:`Top Left Shoulder`},{order:14,x:150,y:430,display:`intermediate`,note:`Left Spine High`},{order:15,x:140,y:530,display:`intermediate`,note:`Left Spine Mid`},{order:16,x:160,y:630,display:`keystone`,note:`Bottom Left Turn`},{order:17,x:210,y:720,display:`intermediate`,note:`Curve Down-Right`},{order:18,x:290,y:790,display:`intermediate`,note:`Curve Down-Right`},{order:19,x:390,y:840,display:`keystone`,note:`Bottom Center`},{order:20,x:500,y:850,display:`intermediate`,note:`Bottom Right`},{order:21,x:600,y:820,display:`intermediate`,note:`Curve Up-Right`},{order:22,x:680,y:770,display:`keystone`,note:`Tail Curve`},{order:23,x:730,y:720,display:`intermediate`,note:`Approach Arrow`},{order:24,x:769,y:655,display:`keystone`,note:`Arrow Head`}],M=t({default:()=>oe}),oe=[{order:1,x:620,y:140,display:`keystone`,note:`Start Black Dot (Under Arch)`},{order:2,x:560,y:110,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:500,y:90,display:`intermediate`,note:`Top Arch Right`},{order:4,x:440,y:90,display:`keystone`,note:`Top Arch Center`},{order:5,x:390,y:130,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:380,y:190,display:`intermediate`,note:`Join Vertical Stem`},{order:7,x:380,y:270,display:`keystone`,note:`Stem High`},{order:8,x:380,y:360,display:`intermediate`,note:`Stem (Crossing Point)`},{order:9,x:380,y:450,display:`intermediate`,note:`Stem Middle`},{order:10,x:380,y:540,display:`keystone`,note:`Stem Mid-Low`},{order:11,x:380,y:630,display:`intermediate`,note:`Stem Lower`},{order:12,x:380,y:720,display:`intermediate`,note:`Stem Lower`},{order:13,x:380,y:810,display:`keystone`,note:`Stem Bottom Approach`},{order:14,x:380,y:860,display:`intermediate`,note:`Stem Arrow Start`},{order:15,x:380,y:900,display:`intermediate`,note:`Stem Arrow Body`},{order:16,x:380,y:930,display:`keystone`,note:`Stem Arrow Tip`},{order:17,x:380,y:940,display:`intermediate`,note:`Finish Stroke 1`},{order:18,x:380,y:940,display:`intermediate`,note:`Pen Lift Point`},{order:19,x:180,y:360,display:`keystone`,newStroke:!0,note:`Start Stroke 2 (The 'X')`},{order:20,x:250,y:360,display:`intermediate`,note:`Crossbar Left`},{order:21,x:320,y:360,display:`intermediate`,note:`Crossbar Approach`},{order:22,x:390,y:360,display:`keystone`,note:`Crossbar Center (After Stem)`},{order:23,x:460,y:360,display:`intermediate`,note:`Crossbar Right`},{order:24,x:530,y:360,display:`intermediate`,note:`Crossbar Right`},{order:25,x:600,y:360,display:`keystone`,note:`Crossbar Arrow Start`},{order:26,x:660,y:360,display:`intermediate`,note:`Finish Crossbar Arrow`}],N=t({default:()=>se}),se=[{order:1,x:742,y:260,display:`keystone`,note:`Start Black Dot`},{order:2,x:630,y:180,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:520,y:150,display:`intermediate`,note:`Top Curve`},{order:4,x:410,y:160,display:`keystone`,note:`Top Center`},{order:5,x:300,y:220,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:210,y:320,display:`intermediate`,note:`Left Curve High`},{order:7,x:180,y:440,display:`keystone`,note:`Left Side Center`},{order:8,x:210,y:560,display:`intermediate`,note:`Left Curve Low`},{order:9,x:300,y:640,display:`intermediate`,note:`Bottom Curve Left`},{order:10,x:430,y:660,display:`keystone`,note:`Bottom of Loop`},{order:11,x:580,y:620,display:`intermediate`,note:`Curve Up-Right`},{order:12,x:700,y:520,display:`intermediate`,note:`Approach Stem`},{order:13,x:780,y:400,display:`keystone`,note:`Connect to Stem`},{order:14,x:780,y:272,display:`intermediate`,note:`Stem Up`},{order:15,x:780,y:200,display:`keystone`,note:`Top of Stem`},{order:16,x:780,y:337,display:`intermediate`,note:`Stem Down (Retrace)`},{order:17,x:780,y:500,display:`intermediate`,note:`Stem Down Middle`},{order:18,x:780,y:700,display:`keystone`,note:`Descender Start`},{order:19,x:780,y:881,display:`intermediate`,note:`Descender Down`},{order:20,x:720,y:930,display:`intermediate`,note:`Tail Curve Start`},{order:21,x:599,y:979,display:`keystone`,note:`Tail Bottom Center`},{order:22,x:480,y:987,display:`intermediate`,note:`Tail Curve Left`},{order:23,x:359,y:945,display:`intermediate`,note:`Tail Curve Up`},{order:24,x:260,y:886,display:`keystone`,note:`Tail Arrow Head`},{order:25,x:200,y:800,display:`intermediate`,note:`Finish Tip`}],P=t({default:()=>F}),F=[{order:1,x:216,y:-91,display:`keystone`,note:`Start Black Dot`},{order:2,x:219,y:28,display:`intermediate`,note:`Stem Down`},{order:3,x:218,y:167,display:`intermediate`,note:`Stem Down`},{order:4,x:220,y:310,display:`intermediate`,note:`Stem Down`},{order:5,x:220,y:390,display:`keystone`,note:`Stem Upper Middle`},{order:6,x:220,y:470,display:`intermediate`,note:`Stem Middle`},{order:7,x:220,y:550,display:`intermediate`,note:`Stem Lower Middle`},{order:8,x:220,y:630,display:`keystone`,note:`Stem Low`},{order:9,x:220,y:710,display:`intermediate`,note:`Stem Lower`},{order:10,x:220,y:790,display:`intermediate`,note:`Stem Approach Bottom`},{order:11,x:220,y:870,display:`keystone`,note:`Stem Left Arrow`},{order:12,x:220,y:500,display:`intermediate`,newStroke:!0,note:`Branch Point (Mid-Stem)`},{order:13,x:280,y:420,display:`intermediate`,note:`Arch Up-Right`},{order:14,x:360,y:360,display:`keystone`,note:`Arch Left Shoulder`},{order:15,x:460,y:330,display:`intermediate`,note:`Arch Top Center`},{order:16,x:560,y:330,display:`intermediate`,note:`Arch Top Right`},{order:17,x:660,y:360,display:`keystone`,note:`Arch Right Shoulder`},{order:18,x:740,y:420,display:`intermediate`,note:`Arch Curve Down`},{order:19,x:780,y:500,display:`intermediate`,note:`Right Leg Start`},{order:20,x:780,y:580,display:`keystone`,note:`Right Leg Mid-High`},{order:21,x:780,y:650,display:`intermediate`,note:`Right Leg Middle`},{order:22,x:780,y:720,display:`intermediate`,note:`Right Leg Low`},{order:23,x:780,y:790,display:`keystone`,note:`Right Leg Lower`},{order:24,x:780,y:870,display:`keystone`,note:`Right Arrow Tip`}],ce=t({default:()=>I}),I=[{order:1,x:500,y:350,display:`keystone`,note:`Start Black Dot (Stem Top)`},{order:2,x:500,y:420,display:`intermediate`,note:`Stem Down`},{order:3,x:500,y:490,display:`intermediate`,note:`Stem Down`},{order:4,x:500,y:560,display:`keystone`,note:`Stem Middle`},{order:5,x:500,y:630,display:`intermediate`,note:`Stem Down`},{order:6,x:500,y:700,display:`intermediate`,note:`Stem Lower`},{order:7,x:500,y:770,display:`keystone`,note:`Begin Turn`},{order:8,x:510,y:830,display:`intermediate`,note:`Curve Right`},{order:9,x:535,y:875,display:`intermediate`,note:`Curve Bottom`},{order:10,x:570,y:900,display:`keystone`,note:`Hook Upwards`},{order:11,x:620,y:910,display:`intermediate`,note:`Finish Hook`},{order:12,x:500,y:150,display:`keystone`,newStroke:!0,note:`The Dot (Top X)`}],L=t({default:()=>le}),le=[{order:1,x:550,y:280,display:`keystone`,note:`Start Black Dot`},{order:2,x:550,y:380,display:`intermediate`,note:`Stem Down`},{order:3,x:550,y:480,display:`intermediate`,note:`Stem Down`},{order:4,x:550,y:580,display:`keystone`,note:`Stem Middle`},{order:5,x:550,y:680,display:`intermediate`,note:`Stem Down`},{order:6,x:550,y:760,display:`intermediate`,note:`Start Turn`},{order:7,x:530,y:840,display:`keystone`,note:`Curve Left`},{order:8,x:490,y:900,display:`intermediate`,note:`Curve Bottom Right`},{order:9,x:430,y:930,display:`intermediate`,note:`Curve Bottom Center`},{order:10,x:360,y:920,display:`keystone`,note:`Curve Bottom Left`},{order:11,x:310,y:890,display:`intermediate`,note:`Arrow Approach`},{order:12,x:280,y:870,display:`intermediate`,note:`Finish Arrow Tip`},{order:13,x:550,y:140,display:`keystone`,newStroke:!0,note:`The Dot (Top X)`}],R=t({default:()=>ue}),ue=[{order:1,x:220,y:180,display:`keystone`,note:`Start Black Dot (Top of Stem)`},{order:2,x:220,y:307,display:`intermediate`,note:`Stem Down`},{order:3,x:217,y:448,display:`intermediate`,note:`Stem Down`},{order:4,x:220,y:540,display:`intermediate`,note:`Stem Middle`},{order:5,x:220,y:720,display:`keystone`,note:`Stem Lower`},{order:6,x:220,y:807,display:`keystone`,note:`Stem Lower`},{order:7,x:220,y:900,display:`intermediate`,note:`Stem Finish Arrow`},{order:8,x:576,y:388,display:`keystone`,newStroke:!0,note:`Start 'X' (Top Right)`},{order:9,x:485,y:454,display:`intermediate`,note:`Diagonal In`},{order:10,x:411,y:506,display:`intermediate`,note:`Diagonal In`},{order:11,x:320,y:572,display:`intermediate`,note:`Diagonal In`},{order:12,x:229,y:637,display:`keystone`,note:`The Turn (Touch Stem)`},{order:13,x:337,y:699,display:`intermediate`,note:`Diagonal Out`},{order:14,x:441,y:759,display:`intermediate`,note:`Diagonal Out`},{order:15,x:520,y:818,display:`intermediate`,note:`Diagonal Out`},{order:16,x:613,y:892,display:`keystone`,note:`Leg Finish Arrow`}],de=t({default:()=>z}),z=[{order:1,x:500,y:150,display:`keystone`,note:`Start Black Dot`},{order:2,x:500,y:250,display:`intermediate`,note:`Stem Down`},{order:3,x:500,y:350,display:`intermediate`,note:`Stem Down`},{order:4,x:500,y:450,display:`keystone`,note:`Stem Middle`},{order:5,x:500,y:550,display:`intermediate`,note:`Stem Down`},{order:6,x:500,y:650,display:`intermediate`,note:`Stem Down`},{order:7,x:500,y:750,display:`keystone`,note:`Begin Turn`},{order:8,x:510,y:820,display:`intermediate`,note:`Curve Right`},{order:9,x:535,y:870,display:`intermediate`,note:`Curve Bottom`},{order:10,x:570,y:900,display:`keystone`,note:`Hook Upwards`},{order:11,x:620,y:910,display:`intermediate`,note:`Finish Hook`}],fe=t({default:()=>pe}),pe=[{order:1,x:179,y:251,display:`keystone`,note:`Start Black Dot (Top Left)`},{order:2,x:180,y:340,display:`intermediate`,note:`Left Stem Down`},{order:3,x:180,y:480,display:`intermediate`,note:`Left Stem Middle`},{order:4,x:180,y:620,display:`keystone`,note:`Left Stem Lower`},{order:5,x:180,y:760,display:`intermediate`,note:`Left Stem Down`},{order:6,x:180,y:900,display:`intermediate`,note:`Left Stem Finish Arrow`},{order:7,x:180,y:380,display:`keystone`,newStroke:!0,note:`Branch Point 1 (Start Middle Arch)`},{order:8,x:240,y:260,display:`intermediate`,note:`Curve Up-Right`},{order:9,x:340,y:210,display:`intermediate`,note:`Arch 1 Top`},{order:10,x:440,y:260,display:`keystone`,note:`Arch 1 Down-Right`},{order:11,x:500,y:380,display:`intermediate`,note:`Middle Stem Start`},{order:12,x:500,y:550,display:`intermediate`,note:`Middle Stem Down`},{order:13,x:500,y:720,display:`keystone`,note:`Middle Stem Lower`},{order:14,x:500,y:850,display:`intermediate`,note:`Middle Stem Approach`},{order:15,x:500,y:900,display:`intermediate`,note:`Middle Stem Finish Arrow`},{order:16,x:500,y:380,display:`keystone`,newStroke:!0,note:`Branch Point 2 (Start Right Arch)`},{order:17,x:560,y:260,display:`intermediate`,note:`Curve Up-Right`},{order:18,x:660,y:210,display:`intermediate`,note:`Arch 2 Top`},{order:19,x:760,y:260,display:`keystone`,note:`Arch 2 Down-Right`},{order:20,x:820,y:380,display:`intermediate`,note:`Right Stem Start`},{order:21,x:820,y:510,display:`intermediate`,note:`Right Stem Down`},{order:22,x:820,y:640,display:`keystone`,note:`Right Stem Middle`},{order:23,x:820,y:770,display:`intermediate`,note:`Right Stem Lower`},{order:24,x:820,y:850,display:`intermediate`,note:`Right Stem Approach`},{order:25,x:820,y:908,display:`keystone`,note:`Right Stem Arrow Body`}],B=t({default:()=>me}),me=[{order:1,x:251,y:279,display:`keystone`,note:`Start Black Dot (Top Left)`},{order:2,x:250,y:359,display:`intermediate`,note:`Left Stem Down`},{order:3,x:250,y:480,display:`intermediate`,note:`Left Stem Middle`},{order:4,x:250,y:620,display:`keystone`,note:`Left Stem Lower`},{order:5,x:250,y:760,display:`intermediate`,note:`Left Stem Down`},{order:6,x:250,y:900,display:`intermediate`,note:`Left Stem Finish Arrow`},{order:7,x:250,y:450,display:`keystone`,newStroke:!0,note:`Branch Point (Start Arch)`},{order:8,x:300,y:320,display:`intermediate`,note:`Curve Up-Right`},{order:9,x:400,y:240,display:`intermediate`,note:`Arch Top Left`},{order:10,x:500,y:210,display:`keystone`,note:`Arch Top Center`},{order:11,x:600,y:240,display:`intermediate`,note:`Arch Top Right`},{order:12,x:700,y:320,display:`intermediate`,note:`Curve Down-Right`},{order:13,x:750,y:450,display:`keystone`,note:`Right Stem Start`},{order:14,x:750,y:600,display:`intermediate`,note:`Right Stem Down`},{order:15,x:750,y:750,display:`intermediate`,note:`Right Stem Approach`},{order:16,x:750,y:900,display:`keystone`,note:`Right Stem Finish Arrow`}],V=t({default:()=>H}),H=[{order:1,x:740,y:260,display:`keystone`,note:`Start Black Dot`},{order:2,x:670,y:200,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:590,y:160,display:`intermediate`,note:`Curve Up-Left`},{order:4,x:500,y:150,display:`keystone`,note:`Top Center`},{order:5,x:410,y:160,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:330,y:200,display:`intermediate`,note:`Curve Down-Left`},{order:7,x:260,y:260,display:`keystone`,note:`Top Left Curve`},{order:8,x:200,y:330,display:`intermediate`,note:`Left Side High`},{order:9,x:160,y:410,display:`intermediate`,note:`Left Side Mid`},{order:10,x:150,y:500,display:`keystone`,note:`Left Center`},{order:11,x:160,y:590,display:`intermediate`,note:`Left Side Low`},{order:12,x:200,y:670,display:`intermediate`,note:`Bottom Left Turn`},{order:13,x:260,y:740,display:`keystone`,note:`Bottom Left Curve`},{order:14,x:330,y:800,display:`intermediate`,note:`Curve Down-Right`},{order:15,x:410,y:840,display:`intermediate`,note:`Bottom Curve`},{order:16,x:500,y:850,display:`keystone`,note:`Bottom Center`},{order:17,x:590,y:840,display:`intermediate`,note:`Bottom Curve Right`},{order:18,x:670,y:800,display:`intermediate`,note:`Curve Up-Right`},{order:19,x:740,y:740,display:`keystone`,note:`Bottom Right Curve`},{order:20,x:800,y:670,display:`intermediate`,note:`Right Side Low`},{order:21,x:840,y:590,display:`intermediate`,note:`Right Side Mid`},{order:22,x:850,y:500,display:`keystone`,note:`Right Center`},{order:23,x:840,y:410,display:`intermediate`,note:`Right Side High`},{order:24,x:800,y:330,display:`intermediate`,note:`Approach Top`},{order:25,x:760,y:290,display:`keystone`,note:`Close Loop`},{order:26,x:730,y:280,display:`intermediate`,note:`Finish Arrow Tip`}],U=t({default:()=>W}),W=[{order:1,x:231,y:203,display:`keystone`,note:`Start Black Dot (Top Left)`},{order:2,x:230,y:260,display:`intermediate`,note:`Stem Down`},{order:3,x:230,y:360,display:`intermediate`,note:`Stem Down`},{order:4,x:230,y:460,display:`keystone`,note:`Stem Middle`},{order:5,x:230,y:560,display:`intermediate`,note:`Stem Down`},{order:6,x:230,y:660,display:`intermediate`,note:`Stem Down`},{order:7,x:230,y:859,display:`keystone`,note:`Stem Lower`},{order:8,x:229,y:1015,display:`intermediate`,note:`Stem Finish Arrow`},{order:9,x:229,y:1153,display:`intermediate`,note:`Stem Finish Arrow`},{order:10,x:231,y:949,display:`intermediate`,note:`Stem Finish Arrow`},{order:11,x:231,y:745,display:`intermediate`,note:`Stem Finish Arrow`},{order:12,x:230,y:340,display:`keystone`,newStroke:!0,note:`Branch Point (Start Lobe)`},{order:13,x:280,y:240,display:`intermediate`,note:`Curve Up-Right`},{order:14,x:380,y:180,display:`intermediate`,note:`Top Curve Left`},{order:15,x:500,y:150,display:`keystone`,note:`Lobe Top Center`},{order:16,x:620,y:160,display:`intermediate`,note:`Top Curve Right`},{order:17,x:720,y:220,display:`intermediate`,note:`Curve Down-Right`},{order:18,x:800,y:320,display:`keystone`,note:`Right Side High`},{order:19,x:840,y:420,display:`intermediate`,note:`Right Side Middle`},{order:20,x:840,y:520,display:`intermediate`,note:`Right Side Low`},{order:21,x:800,y:620,display:`keystone`,note:`Bottom Right Curve`},{order:22,x:720,y:700,display:`intermediate`,note:`Curve Left-Down`},{order:23,x:620,y:740,display:`intermediate`,note:`Bottom Curve Right`},{order:24,x:500,y:750,display:`keystone`,note:`Lobe Bottom Center`},{order:25,x:380,y:740,display:`intermediate`,note:`Bottom Curve Left`},{order:26,x:320,y:710,display:`intermediate`,note:`Approach Stem`},{order:27,x:280,y:680,display:`keystone`,note:`Close Lobe Arrow`},{order:28,x:250,y:660,display:`intermediate`,note:`Connect`},{order:29,x:230,y:640,display:`intermediate`,note:`Join Stem`}],G=t({default:()=>K}),K=[{order:1,x:648,y:229,display:`keystone`,note:`Start Black Dot`},{order:2,x:537,y:161,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:380,y:160,display:`intermediate`,note:`Top Curve`},{order:4,x:280,y:200,display:`keystone`,note:`Top Left Curve`},{order:5,x:200,y:280,display:`intermediate`,note:`Left Side High`},{order:6,x:150,y:400,display:`intermediate`,note:`Left Side Mid`},{order:7,x:150,y:520,display:`keystone`,note:`Left Side Low`},{order:8,x:200,y:640,display:`intermediate`,note:`Bottom Left Turn`},{order:9,x:300,y:720,display:`intermediate`,note:`Bottom Curve`},{order:10,x:450,y:730,display:`keystone`,note:`Bottom Right Curve`},{order:11,x:580,y:680,display:`intermediate`,note:`Approach Stem`},{order:12,x:660,y:600,display:`intermediate`,note:`Join Stem`},{order:13,x:680,y:500,display:`keystone`,note:`Connect to Stem (Go Up)`},{order:14,x:680,y:380,display:`intermediate`,note:`Stem Up`},{order:15,x:680,y:260,display:`intermediate`,note:`Stem Up`},{order:16,x:679,y:168,display:`keystone`,note:`Top of Stem`},{order:17,x:680,y:300,display:`intermediate`,note:`Stem Down (Retrace)`},{order:18,x:680,y:500,display:`intermediate`,note:`Stem Down Mid`},{order:19,x:680,y:700,display:`keystone`,note:`Stem Down Low`},{order:20,x:680,y:820,display:`intermediate`,note:`Stem Bottom Approach`},{order:21,x:679,y:966,display:`intermediate`,note:`Stem Bottom Approach`},{order:22,x:680,y:1136,display:`intermediate`,note:`Stem Bottom Sharp Turn`},{order:23,x:745,y:997,display:`keystone`,note:`Kick Up-Right`},{order:24,x:802,y:871,display:`intermediate`,note:`Kick Diagonal`},{order:25,x:860,y:760,display:`intermediate`,note:`Kick Arrow Body`},{order:26,x:920,y:700,display:`keystone`,note:`Kick Arrow Head`},{order:27,x:950,y:670,display:`intermediate`,note:`Finish Tip`}],he=t({default:()=>q}),q=[{order:1,x:300,y:273,display:`keystone`,note:`Start Black Dot`},{order:2,x:300,y:341,display:`intermediate`,note:`Stem Down`},{order:3,x:300,y:480,display:`intermediate`,note:`Stem Middle`},{order:4,x:300,y:620,display:`keystone`,note:`Stem Lower`},{order:5,x:300,y:760,display:`intermediate`,note:`Stem Down`},{order:6,x:300,y:900,display:`keystone`,note:`Stem Finish Arrow`},{order:7,x:300,y:450,display:`keystone`,newStroke:!0,note:`Branch Point (Start Arch)`},{order:8,x:350,y:350,display:`intermediate`,note:`Curve Up-Right`},{order:9,x:420,y:290,display:`intermediate`,note:`Arch Curve`},{order:10,x:510,y:250,display:`keystone`,note:`Arch Top`},{order:11,x:600,y:240,display:`intermediate`,note:`Arch Extension`},{order:12,x:680,y:250,display:`intermediate`,note:`Approach Arrow`},{order:13,x:750,y:270,display:`keystone`,note:`Arch Finish Arrow`}],J=t({default:()=>Y}),Y=[{order:1,x:710,y:220,display:`keystone`,note:`Start Black Dot`},{order:2,x:640,y:170,display:`intermediate`,note:`Curve Up-Left`},{order:3,x:560,y:140,display:`intermediate`,note:`Curve Up-Left`},{order:4,x:470,y:140,display:`keystone`,note:`Top Center`},{order:5,x:390,y:170,display:`intermediate`,note:`Curve Down-Left`},{order:6,x:320,y:230,display:`intermediate`,note:`Top Left Curve`},{order:7,x:280,y:310,display:`keystone`,note:`Top Left Turn`},{order:8,x:290,y:400,display:`intermediate`,note:`Curve In`},{order:9,x:340,y:480,display:`intermediate`,note:`Start Diagonal`},{order:10,x:420,y:550,display:`keystone`,note:`Crossing Middle`},{order:11,x:529,y:597,display:`intermediate`,note:`Crossing Middle`},{order:12,x:617,y:628,display:`intermediate`,note:`Crossing Middle`},{order:13,x:710,y:699,display:`keystone`,note:`Right Side Turn`},{order:14,x:730,y:790,display:`intermediate`,note:`Right Side Curve`},{order:15,x:730,y:870,display:`intermediate`,note:`Bottom Right`},{order:16,x:680,y:930,display:`keystone`,note:`Bottom Curve Start`},{order:17,x:590,y:950,display:`intermediate`,note:`Bottom Center`},{order:18,x:500,y:940,display:`intermediate`,note:`Bottom Curve Left`},{order:19,x:404,y:930,display:`keystone`,note:`Bottom Left Turn`},{order:20,x:331,y:905,display:`intermediate`,note:`Curve Up-Left`},{order:21,x:247,y:858,display:`keystone`,note:`Arrow Head`}],ge=t({default:()=>X}),X=[{order:1,x:500,y:150,display:`keystone`,note:`Start Black Dot (Top)`},{order:2,x:500,y:270,display:`intermediate`,note:`Stem Down`},{order:3,x:500,y:390,display:`intermediate`,note:`Stem Down`},{order:4,x:500,y:510,display:`keystone`,note:`Stem Middle`},{order:5,x:500,y:630,display:`intermediate`,note:`Stem Down`},{order:6,x:500,y:750,display:`intermediate`,note:`Begin Turn`},{order:7,x:510,y:820,display:`keystone`,note:`Curve Right`},{order:8,x:540,y:880,display:`intermediate`,note:`Curve Bottom`},{order:9,x:580,y:910,display:`intermediate`,note:`Hook Upwards`},{order:10,x:630,y:920,display:`keystone`,note:`Finish Hook`},{order:11,x:300,y:350,display:`keystone`,newStroke:!0,note:`Start Crossbar (Left X)`},{order:12,x:400,y:350,display:`intermediate`,note:`Crossbar Left`},{order:13,x:500,y:350,display:`intermediate`,note:`Crossbar Center`},{order:14,x:600,y:350,display:`intermediate`,note:`Crossbar Right`},{order:15,x:700,y:350,display:`keystone`,note:`Finish Crossbar Arrow`}],Z=t({default:()=>Q}),Q=[{order:1,x:250,y:200,display:`keystone`,note:`Start Black Dot (Top Left)`},{order:2,x:250,y:300,display:`intermediate`,note:`Left Stem Down`},{order:3,x:250,y:400,display:`intermediate`,note:`Left Stem Down`},{order:4,x:250,y:500,display:`keystone`,note:`Left Stem Middle`},{order:5,x:250,y:600,display:`intermediate`,note:`Left Stem Low`},{order:6,x:260,y:700,display:`intermediate`,note:`Begin Left Curve`},{order:7,x:290,y:780,display:`keystone`,note:`Curve Down-Right`},{order:8,x:350,y:850,display:`intermediate`,note:`Bowl Left`},{order:9,x:420,y:890,display:`intermediate`,note:`Bowl Bottom Left`},{order:10,x:500,y:900,display:`keystone`,note:`Bowl Bottom Center`},{order:11,x:580,y:890,display:`intermediate`,note:`Bowl Bottom Right`},{order:12,x:650,y:850,display:`intermediate`,note:`Bowl Right`},{order:13,x:710,y:780,display:`keystone`,note:`Curve Up-Right`},{order:14,x:740,y:700,display:`intermediate`,note:`Begin Right Stem`},{order:15,x:750,y:600,display:`intermediate`,note:`Right Stem Up`},{order:16,x:750,y:500,display:`keystone`,note:`Right Stem Middle`},{order:17,x:750,y:400,display:`intermediate`,note:`Right Stem Up`},{order:18,x:750,y:300,display:`intermediate`,note:`Right Stem Up`},{order:19,x:750,y:240,display:`keystone`,note:`Arrow Approach`},{order:20,x:750,y:200,display:`intermediate`,note:`Finish Arrow Tip`},{order:21,x:751,y:442,display:`intermediate`,note:`Finish Arrow Tip`},{order:22,x:747,y:654,display:`intermediate`,note:`Finish Arrow Tip`},{order:23,x:750,y:814,display:`intermediate`,note:`Finish Arrow Tip`},{order:24,x:794,y:865,display:`intermediate`,note:`Finish Arrow Tip`},{order:25,x:842,y:891,display:`intermediate`,note:`Finish Arrow Tip`}],_e=t({default:()=>ve}),ve=[{order:1,x:220,y:200,display:`keystone`,note:`Start Black Dot (Top Left)`},{order:2,x:250,y:280,display:`intermediate`,note:`Diagonal Down`},{order:3,x:280,y:360,display:`intermediate`,note:`Diagonal Down`},{order:4,x:310,y:440,display:`keystone`,note:`Left Side Middle`},{order:5,x:340,y:520,display:`intermediate`,note:`Diagonal Down`},{order:6,x:370,y:600,display:`intermediate`,note:`Diagonal Down`},{order:7,x:400,y:680,display:`keystone`,note:`Left Side Low`},{order:8,x:430,y:760,display:`intermediate`,note:`Approach Bottom`},{order:9,x:460,y:840,display:`intermediate`,note:`Approach Vertex`},{order:10,x:500,y:900,display:`keystone`,note:`Bottom Vertex (Turn)`},{order:11,x:540,y:840,display:`intermediate`,note:`Diagonal Up`},{order:12,x:570,y:760,display:`intermediate`,note:`Diagonal Up`},{order:13,x:600,y:680,display:`keystone`,note:`Right Side Low`},{order:14,x:630,y:600,display:`intermediate`,note:`Diagonal Up`},{order:15,x:660,y:520,display:`intermediate`,note:`Diagonal Up`},{order:16,x:690,y:440,display:`keystone`,note:`Right Side Middle`},{order:17,x:720,y:360,display:`intermediate`,note:`Diagonal Up`},{order:18,x:750,y:280,display:`intermediate`,note:`Approach Top`},{order:19,x:780,y:220,display:`keystone`,note:`Arrow Head`},{order:20,x:790,y:200,display:`intermediate`,note:`Finish Tip`}],ye=t({default:()=>be}),be=[{order:1,x:150,y:200,display:`keystone`,note:`Start Black Dot (Top Left)`},{order:2,x:180,y:315,display:`intermediate`,note:`Diagonal Down 1`},{order:3,x:210,y:430,display:`intermediate`,note:`Diagonal Down 1`},{order:4,x:240,y:550,display:`keystone`,note:`Left Leg Middle`},{order:5,x:270,y:665,display:`intermediate`,note:`Diagonal Down 1`},{order:6,x:300,y:780,display:`intermediate`,note:`Approach Bottom 1`},{order:7,x:325,y:900,display:`keystone`,note:`First Bottom Turn`},{order:8,x:355,y:780,display:`intermediate`,note:`Diagonal Up 1`},{order:9,x:385,y:665,display:`intermediate`,note:`Diagonal Up 1`},{order:10,x:415,y:550,display:`keystone`,note:`Inner Left Middle`},{order:11,x:445,y:430,display:`intermediate`,note:`Diagonal Up 1`},{order:12,x:475,y:315,display:`intermediate`,note:`Approach Peak`},{order:13,x:500,y:200,display:`keystone`,note:`Middle Peak`},{order:14,x:525,y:315,display:`intermediate`,note:`Diagonal Down 2`},{order:15,x:555,y:430,display:`intermediate`,note:`Diagonal Down 2`},{order:16,x:585,y:550,display:`keystone`,note:`Inner Right Middle`},{order:17,x:615,y:665,display:`intermediate`,note:`Diagonal Down 2`},{order:18,x:645,y:780,display:`intermediate`,note:`Approach Bottom 2`},{order:19,x:675,y:900,display:`keystone`,note:`Second Bottom Turn`},{order:20,x:700,y:780,display:`intermediate`,note:`Diagonal Up 2`},{order:21,x:730,y:665,display:`intermediate`,note:`Diagonal Up 2`},{order:22,x:760,y:550,display:`keystone`,note:`Right Leg Middle`},{order:23,x:790,y:430,display:`intermediate`,note:`Diagonal Up 2`},{order:24,x:820,y:315,display:`intermediate`,note:`Approach Finish`},{order:25,x:850,y:200,display:`keystone`,note:`Arrow Head`},{order:26,x:860,y:180,display:`intermediate`,note:`Finish Tip`}],xe={"./assets/letters_json/a.json":ne,"./assets/letters_json/b.json":E,"./assets/letters_json/c.json":ae,"./assets/letters_json/d.json":O,"./assets/letters_json/e.json":A,"./assets/letters_json/f.json":M,"./assets/letters_json/g.json":N,"./assets/letters_json/h.json":P,"./assets/letters_json/i.json":ce,"./assets/letters_json/j.json":L,"./assets/letters_json/k.json":R,"./assets/letters_json/l.json":de,"./assets/letters_json/m.json":fe,"./assets/letters_json/n.json":B,"./assets/letters_json/o.json":V,"./assets/letters_json/p.json":U,"./assets/letters_json/q.json":G,"./assets/letters_json/r.json":he,"./assets/letters_json/s.json":J,"./assets/letters_json/t.json":ge,"./assets/letters_json/u.json":Z,"./assets/letters_json/v.json":_e,"./assets/letters_json/w.json":ye},$=Object.entries(xe).map(([e,t])=>{let n=t.default;return Array.isArray(n)?{name:(e.split(`/`).pop()??`unknown`).replace(/\.json$/i,``),points:n}:null}).filter(e=>e!==null);const Se=$.map(e=>e.name).sort();var Ce=e=>{if($.length===0)return{name:`unknown`,points:[]};let t=$;e&&e.size>0&&(t=$.filter(t=>e.has(t.name))),t.length===0&&(t=$);let n=t[Math.floor(Math.random()*t.length)];return console.info(`Selected letter:`,n.name),{name:n.name,points:n.points.map(e=>({order:e.order,x:e.x,y:e.y})).sort((e,t)=>e.order-t.order)}},we=e=>{if(e.length===0)return{minX:0,maxX:0,minY:0,maxY:0};let t=e[0].x,n=e[0].x,r=e[0].y,i=e[0].y;for(let a of e)a.x<t&&(t=a.x),a.x>n&&(n=a.x),a.y<r&&(r=a.y),a.y>i&&(i=a.y);return{minX:t,maxX:n,minY:r,maxY:i}},Te=3.5,Ee=e=>{let t=[...e].sort((e,t)=>e-t),n=Math.floor(t.length/2);return t.length%2==0?(t[n-1]+t[n])/2:t[n]},De=e=>{if(e.length<2)return 1/0;let t=[];for(let n=0;n<e.length-1;n+=1){let r=e[n+1].x-e[n].x,i=e[n+1].y-e[n].y;t.push(Math.hypot(r,i))}let n=t.filter(e=>e>0),r=n.length>0?n:t;return r.length===0?1/0:Ee(r)*Te},Oe=(e,t)=>{let n=[];if(e.length<2)return n;for(let r=0;r<e.length-1;r+=1){let i=e[r+1].x-e[r].x,a=e[r+1].y-e[r].y;Math.hypot(i,a)>t||n.push({from:r,to:r+1})}return n},ke=class{plotPoints=[];paths=[];currentPath=null;fadeSeconds=9;maxPoints=600;isDrawing=!1;currentMousePos={x:0,y:0};plotBounds={minX:0,maxX:0,minY:0,maxY:0};scaledPlotPoints=[];lastPlotSize={width:0,height:0};needsRescale=!0;currentTargetIndex=0;dotRadius=20;hitRadiusScale=1.5;penUpDistanceThreshold=1/0;lineSegmentIndex=0;lineSegmentT=0;linePauseRemaining=0;lineSegmentSeconds=.123;lineLoopPauseSeconds=.4;completionMessageUntil=0;pendingLetterReset=!1;completionMessageSeconds=2;callbacks;currentLetterName=`unknown`;enabledLetters=null;constructor(e,t={}){let n=Ce();this.plotPoints=n.points,this.plotBounds=we(this.plotPoints),this.callbacks=t,this.currentLetterName=n.name,this.callbacks.onLetterChange?.(this.currentLetterName),e.setCallbacks({onStart:e=>this.startPath(e),onMove:e=>this.extendPath(e),onEnd:e=>this.endPath(e)})}setEnabledLetters(e){this.enabledLetters=e}getDrawingState(){let e=this.paths.flatMap(e=>e);return{isDrawing:this.isDrawing,mouseX:this.currentMousePos.x,mouseY:this.currentMousePos.y,trailPoints:e}}getDotState(){return{dots:this.scaledPlotPoints,currentIndex:this.currentTargetIndex,radius:this.dotRadius}}clearUserPaths(){this.paths=[],this.currentPath=null,this.isDrawing=!1}resetCurrentLetter(){this.clearUserPaths(),this.currentTargetIndex=0,this.lineSegmentIndex=0,this.lineSegmentT=0,this.linePauseRemaining=0,this.pendingLetterReset=!1,this.completionMessageUntil=0}setViewportSize(e,t){if(this.plotPoints.length===0)return;let n=Math.max(1,e),r=Math.max(1,t);if(n===this.lastPlotSize.width&&r===this.lastPlotSize.height&&!this.needsRescale)return;this.lastPlotSize={width:n,height:r},this.needsRescale=!1;let i=this.plotBounds.maxX-this.plotBounds.minX,a=this.plotBounds.maxY-this.plotBounds.minY;if(i<=0||a<=0)return;let o=n*.5,s=r*.5,c=Math.min(o/i,s/a),l=(n-i*c)/2-this.plotBounds.minX*c,u=(r-a*c)/2-this.plotBounds.minY*c;this.scaledPlotPoints=this.plotPoints.map(e=>({order:e.order,x:e.x*c+l,y:e.y*c+u})),this.penUpDistanceThreshold=De(this.scaledPlotPoints)}update(e){this.advancePlotAnimation(e);let t=performance.now()/1e3;if(this.paths.length>0){let e=t-this.fadeSeconds;for(let t of this.paths){let n=0;for(;n<t.length&&t[n].time<e;)n+=1;n>0&&t.splice(0,n),t.length>this.maxPoints&&t.splice(0,t.length-this.maxPoints)}this.paths=this.paths.filter(e=>e.length>0)}this.pendingLetterReset&&t>=this.completionMessageUntil&&(this.resetForNewLetter(),this.pendingLetterReset=!1,this.completionMessageUntil=0)}render(e){if(e.save(),e.setTransform(1,0,0,1,0,0),e.clearRect(0,0,e.canvas.width,e.canvas.height),e.restore(),this.renderPlotLines(e),this.paths.length>0){e.strokeStyle=`rgba(200, 230, 255, 0.9)`,e.lineWidth=15,e.lineJoin=`round`,e.lineCap=`round`;for(let t of this.paths)t.length!==0&&(e.beginPath(),t.forEach((t,n)=>{n===0?e.moveTo(t.x,t.y):e.lineTo(t.x,t.y)}),e.stroke())}this.renderCompletionMessage(e)}startPath(e){this.pendingLetterReset||(this.isDrawing=!0,this.currentMousePos={x:e.x,y:e.y},this.currentPath=[{x:e.x,y:e.y,time:e.time}],this.paths.push(this.currentPath),this.tryAdvanceTarget(e))}extendPath(e){this.pendingLetterReset||(this.currentMousePos={x:e.x,y:e.y},this.currentPath&&(this.currentPath.push({x:e.x,y:e.y,time:e.time}),this.tryAdvanceTarget(e)))}endPath(e){this.pendingLetterReset||(this.currentMousePos={x:e.x,y:e.y},this.currentPath&&this.currentPath.push({x:e.x,y:e.y,time:e.time}),this.tryAdvanceTarget(e),this.isDrawing=!1,this.currentPath=null)}renderPlotLines(e){if(this.scaledPlotPoints.length<2)return;e.strokeStyle=`rgba(255, 220, 0, 0.5)`,e.lineWidth=9,e.lineJoin=`round`,e.lineCap=`round`,e.beginPath();let t=Math.min(Math.max(0,this.currentTargetIndex-1),this.scaledPlotPoints.length-1),n=this.scaledPlotPoints.slice(t),r=Oe(n,this.penUpDistanceThreshold);if(r.length!==0){if(this.linePauseRemaining>0)this.drawLineSegments(e,n,r,r.length-1);else{let t=Math.min(this.lineSegmentIndex,r.length-1);this.drawLineSegments(e,n,r,t-1),this.drawPartialLineSegment(e,n,r,t,this.lineSegmentT)}e.stroke()}}drawLineSegments(e,t,n,r){if(r<0)return;let i=Math.min(r,n.length-1),a=null;for(let r=0;r<=i;r+=1){let i=n[r],o=t[i.from],s=t[i.to];(!a||a.to!==i.from)&&e.moveTo(o.x,o.y),e.lineTo(s.x,s.y),a=i}}drawPartialLineSegment(e,t,n,r,i){if(r<0||r>=n.length)return;let a=n[r],o=t[a.from],s=t[a.to],c=r>0?n[r-1]:null;(!c||c.to!==a.from)&&e.moveTo(o.x,o.y);let l=o.x+(s.x-o.x)*i,u=o.y+(s.y-o.y)*i;e.lineTo(l,u)}tryAdvanceTarget(e){if(!this.isDrawing||this.scaledPlotPoints.length===0||this.currentTargetIndex>=this.scaledPlotPoints.length)return;let t=this.scaledPlotPoints[this.currentTargetIndex],n=e.x-t.x,r=e.y-t.y,i=this.getTargetDotRadius()*this.hitRadiusScale;n*n+r*r<=i*i&&(this.currentTargetIndex=Math.min(this.currentTargetIndex+1,this.scaledPlotPoints.length),this.callbacks.onPointHit?.(),this.lineSegmentIndex=0,this.lineSegmentT=0,this.linePauseRemaining=0,this.currentTargetIndex>=this.scaledPlotPoints.length&&this.handleLetterComplete())}getTargetDotRadius(){return Math.max(10,this.dotRadius*1.3)*1.45}advancePlotAnimation(e){if(e<=0||this.scaledPlotPoints.length<2)return;let t=Math.min(Math.max(0,this.currentTargetIndex-1),this.scaledPlotPoints.length-1),n=Oe(this.scaledPlotPoints.slice(t),this.penUpDistanceThreshold).length;if(n!==0){if(this.lineSegmentIndex=Math.min(this.lineSegmentIndex,n-1),this.linePauseRemaining>0){if(this.linePauseRemaining=Math.max(0,this.linePauseRemaining-e),this.linePauseRemaining>0)return;this.lineSegmentIndex=0,this.lineSegmentT=0}for(this.lineSegmentT+=e/this.lineSegmentSeconds;this.lineSegmentT>=1;)if(--this.lineSegmentT,this.lineSegmentIndex+=1,this.lineSegmentIndex>=n){this.linePauseRemaining=this.lineLoopPauseSeconds,this.lineSegmentIndex=0,this.lineSegmentT=0;break}}}handleLetterComplete(){this.pendingLetterReset||(this.completionMessageUntil=performance.now()/1e3+this.completionMessageSeconds,this.pendingLetterReset=!0,this.isDrawing=!1,this.currentPath=null)}resetForNewLetter(){let e=Ce(this.enabledLetters??void 0);this.plotPoints=e.points,this.plotBounds=we(this.plotPoints),this.currentLetterName=e.name,this.callbacks.onLetterChange?.(this.currentLetterName),this.scaledPlotPoints=[],this.currentTargetIndex=0,this.lineSegmentIndex=0,this.lineSegmentT=0,this.linePauseRemaining=0,this.paths=[],this.currentPath=null,this.isDrawing=!1,this.needsRescale=!0,this.lastPlotSize.width>0&&this.lastPlotSize.height>0&&this.setViewportSize(this.lastPlotSize.width,this.lastPlotSize.height)}renderCompletionMessage(e){if(performance.now()/1e3>=this.completionMessageUntil)return;let t=e.canvas.width/(window.devicePixelRatio||1),n=e.canvas.height/(window.devicePixelRatio||1);e.save(),e.textAlign=`center`,e.textBaseline=`middle`,e.fillStyle=`rgba(255, 244, 210, 0.95)`,e.strokeStyle=`rgba(0, 0, 0, 0.45)`,e.lineWidth=6,e.font=`700 48px system-ui, -apple-system, sans-serif`;let r=`Well done!`;e.strokeText(r,t/2,n*.45),e.fillText(r,t/2,n*.45),e.restore()}},Ae={low:{auroraScale:.33,sparkleScale:.5,level:`low`},medium:{auroraScale:.5,sparkleScale:.75,level:`medium`},high:{auroraScale:.75,sparkleScale:1,level:`high`},ultra:{auroraScale:1,sparkleScale:1,level:`ultra`}},je=class{frameTimes=[];lastFrameTime=performance.now();sampleSize=90;stabilizationFrames=60;frameCount=0;currentQuality;onQualityChange;lastAdjustmentFrame=0;adjustmentCooldown=120;constructor(e){this.currentQuality=e??this.detectInitialQuality()}detectInitialQuality(){let e=/iPhone|iPad|iPod|Android/i.test(navigator.userAgent),t=/iPhone [5-8]|iPad (Mini|Air(?! [3-9]))|Android [4-7]/i.test(navigator.userAgent),n=navigator.deviceMemory,r=navigator.hardwareConcurrency||2;return t?`low`:e?n&&n>=4?`medium`:`low`:n&&n>=8&&r>=8?`high`:`medium`}setOnQualityChange(e){this.onQualityChange=e}recordFrame(){let e=performance.now(),t=e-this.lastFrameTime;this.lastFrameTime=e,this.frameCount++,!(t>100)&&(this.frameTimes.push(t),this.frameTimes.length>this.sampleSize&&this.frameTimes.shift(),this.frameCount>this.stabilizationFrames&&this.frameCount-this.lastAdjustmentFrame>this.adjustmentCooldown&&this.frameTimes.length>=this.sampleSize/2&&this.maybeAdjustQuality())}getAverageFPS(){return this.frameTimes.length===0?60:1e3/(this.frameTimes.reduce((e,t)=>e+t,0)/this.frameTimes.length)}getPercentileFPS(e){if(this.frameTimes.length===0)return 60;let t=[...this.frameTimes].sort((e,t)=>t-e);return 1e3/t[Math.floor(t.length*(1-e/100))]}maybeAdjustQuality(){let e=this.getAverageFPS(),t=this.getPercentileFPS(5),n=[`low`,`medium`,`high`,`ultra`],r=n.indexOf(this.currentQuality),i=this.currentQuality;t<25||e<40?r>0&&(i=n[r-1]):e>58&&t>50&&r<n.length-1&&(i=n[r+1]),i!==this.currentQuality&&(console.info(`Quality adjusted: ${this.currentQuality} → ${i} (avg: ${e.toFixed(1)} fps, 5th%: ${t.toFixed(1)} fps)`),this.currentQuality=i,this.lastAdjustmentFrame=this.frameCount,this.frameTimes=[],this.onQualityChange?.(this.getQualitySettings()))}getQualitySettings(){return Ae[this.currentQuality]}getCurrentLevel(){return this.currentQuality}setQuality(e){e!==this.currentQuality&&(this.currentQuality=e,this.lastAdjustmentFrame=this.frameCount,this.frameTimes=[],this.onQualityChange?.(this.getQualitySettings()))}getDebugInfo(){let e=this.getAverageFPS(),t=this.getPercentileFPS(5);return`${this.currentQuality} | ${e.toFixed(0)} fps (5%: ${t.toFixed(0)})`}},Me=t({default:()=>Ne}),Ne=`/letter_constellations/assets/accordion-CyoQk0r2.webp`,Pe=t({default:()=>Fe}),Fe=`/letter_constellations/assets/acorn-B-P3IZra.webp`,Ie=t({default:()=>Le}),Le=`/letter_constellations/assets/airplane-DZQqSBvK.webp`,Re=t({default:()=>ze}),ze=`/letter_constellations/assets/alligator-Ce5bwrBv.webp`,Be=t({default:()=>Ve}),Ve=`/letter_constellations/assets/anchor-BV-M7FTg.webp`,He=t({default:()=>Ue}),Ue=`/letter_constellations/assets/ant-CvU0-Xki.webp`,We=t({default:()=>Ge}),Ge=`/letter_constellations/assets/apple-DE3ezPUv.webp`,Ke=t({default:()=>qe}),qe=`/letter_constellations/assets/arrow-C4cWjJLo.webp`,Je=t({default:()=>Ye}),Ye=`/letter_constellations/assets/astronaut-uRVMZixm.webp`,Xe=t({default:()=>Ze}),Ze=`/letter_constellations/assets/bananas-VvvO5iFW.webp`,Qe=t({default:()=>$e}),$e=`/letter_constellations/assets/beachball-BbGYNBZh.webp`,et=t({default:()=>tt}),tt=`/letter_constellations/assets/bear-CvctF6Z1.webp`,nt=t({default:()=>rt}),rt=`/letter_constellations/assets/bee-BCPxoZBd.webp`,it=t({default:()=>at}),at=`/letter_constellations/assets/bird-CPHJbX7S.webp`,ot=t({default:()=>st}),st=`/letter_constellations/assets/boat-5S1Aura4.webp`,ct=t({default:()=>lt}),lt=`/letter_constellations/assets/books-DzB7841Q.webp`,ut=t({default:()=>dt}),dt=`/letter_constellations/assets/bus-Cr08LBdp.webp`,ft=t({default:()=>pt}),pt=`/letter_constellations/assets/butterfly-DJ37Qz2P.webp`,mt=t({default:()=>ht}),ht=`/letter_constellations/assets/cake-C6PvIUA0.webp`,gt=t({default:()=>_t}),_t=`/letter_constellations/assets/car-BLjHiQLL.webp`,vt=t({default:()=>yt}),yt=`/letter_constellations/assets/castle-kQcpwewP.webp`,bt=t({default:()=>xt}),xt=`/letter_constellations/assets/cat-_X2VtWwN.webp`,St=t({default:()=>Ct}),Ct=`/letter_constellations/assets/clock-BqkzNyNk.webp`,wt=t({default:()=>Tt}),Tt=`/letter_constellations/assets/coat-OQ0hqgvS.webp`,Et=t({default:()=>Dt}),Dt=`/letter_constellations/assets/cow-BU0iB5gA.webp`,Ot=t({default:()=>kt}),kt=`/letter_constellations/assets/crayon-DT3SkfbL.webp`,At=t({default:()=>jt}),jt=`/letter_constellations/assets/cup-B6QWjkyx.webp`,Mt=t({default:()=>Nt}),Nt=`/letter_constellations/assets/dog-DC0ZhRtT.webp`,Pt=t({default:()=>Ft}),Ft=`/letter_constellations/assets/doll-DAHtNuVs.webp`,It=t({default:()=>Lt}),Lt=`/letter_constellations/assets/dolphin-C2ILYpRK.webp`,Rt=t({default:()=>zt}),zt=`/letter_constellations/assets/donkey-BMyke4TY.webp`,Bt=t({default:()=>Vt}),Vt=`/letter_constellations/assets/door-CVCIUMCt.webp`,Ht=t({default:()=>Ut}),Ut=`/letter_constellations/assets/dragon-BAjlAFRA.webp`,Wt=t({default:()=>Gt}),Gt=`/letter_constellations/assets/dress-DvrNR5gf.webp`,Kt=t({default:()=>qt}),qt=`/letter_constellations/assets/drum-CwxRV3Cf.webp`,Jt=t({default:()=>Yt}),Yt=`/letter_constellations/assets/duck-U27dc_u8.webp`,Xt=t({default:()=>Zt}),Zt=`/letter_constellations/assets/eagle-DiKLg3yz.webp`,Qt=t({default:()=>$t}),$t=`/letter_constellations/assets/ear-BgYxZnRS.webp`,en=t({default:()=>tn}),tn=`/letter_constellations/assets/earth-r6cd7hOH.webp`,nn=t({default:()=>rn}),rn=`/letter_constellations/assets/easel-EvkDAxpG.webp`,an=t({default:()=>on}),on=`/letter_constellations/assets/egg-Dm6XQc4n.webp`,sn=t({default:()=>cn}),cn=`/letter_constellations/assets/elephant-CIvbMIFa.webp`,ln=t({default:()=>un}),un=`/letter_constellations/assets/envelope-DU_Wnco4.webp`,dn=t({default:()=>fn}),fn=`/letter_constellations/assets/eraser-MC2wEqI8.webp`,pn=t({default:()=>mn}),mn=`/letter_constellations/assets/feather-DOx_OLgY.webp`,hn=t({default:()=>gn}),gn=`/letter_constellations/assets/fence-CwF42dkY.webp`,_n=t({default:()=>vn}),vn=`/letter_constellations/assets/fish-DyOqSklD.webp`,yn=t({default:()=>bn}),bn=`/letter_constellations/assets/flower-B__p0fXU.webp`,xn=t({default:()=>Sn}),Sn=`/letter_constellations/assets/foot-BXSJM_nF.webp`,Cn=t({default:()=>wn}),wn=`/letter_constellations/assets/fork-yucdu-VG.webp`,Tn=t({default:()=>En}),En=`/letter_constellations/assets/fox-BE5prvLj.webp`,Dn=t({default:()=>On}),On=`/letter_constellations/assets/frog-Dh1keBjF.webp`,kn=t({default:()=>An}),An=`/letter_constellations/assets/garden-BXEz5faC.webp`,jn=t({default:()=>Mn}),Mn=`/letter_constellations/assets/ghost-C3-5NTro.webp`,Nn=t({default:()=>Pn}),Pn=`/letter_constellations/assets/giraffe-BN8FpAKY.webp`,Fn=t({default:()=>In}),In=`/letter_constellations/assets/gloves-BJzKgdkI.webp`,Ln=t({default:()=>Rn}),Rn=`/letter_constellations/assets/goat-Cp1Bn2Pw.webp`,zn=t({default:()=>Bn}),Bn=`/letter_constellations/assets/goldfish-C02QaEXu.webp`,Vn=t({default:()=>Hn}),Hn=`/letter_constellations/assets/grapes-DhMMMjx_.webp`,Un=t({default:()=>Wn}),Wn=`/letter_constellations/assets/guitar-BhL2ZgvO.webp`,Gn=t({default:()=>Kn}),Kn=`/letter_constellations/assets/hammer-CfkG7qG8.webp`,qn=t({default:()=>Jn}),Jn=`/letter_constellations/assets/hand-C1darVYo.webp`,Yn=t({default:()=>Xn}),Xn=`/letter_constellations/assets/hat-5HvhUkFa.webp`,Zn=t({default:()=>Qn}),Qn=`/letter_constellations/assets/heart-qdzpJsno.webp`,$n=t({default:()=>er}),er=`/letter_constellations/assets/hedgehog-BaJaSoQw.webp`,tr=t({default:()=>nr}),nr=`/letter_constellations/assets/helicopter-DC0W3QSo.webp`,rr=t({default:()=>ir}),ir=`/letter_constellations/assets/horse-DfgajmWZ.webp`,ar=t({default:()=>or}),or=`/letter_constellations/assets/hotdog-DTQ4uHOu.webp`,sr=t({default:()=>cr}),cr=`/letter_constellations/assets/house-BZe7-xWI.webp`,lr=t({default:()=>ur}),ur=`/letter_constellations/assets/ice_cream-DT_mc0_P.webp`,dr=t({default:()=>fr}),fr=`/letter_constellations/assets/igloo-CC6ArNRt.webp`,pr=t({default:()=>mr}),mr=`/letter_constellations/assets/iguana-BXYH3WVB.webp`,hr=t({default:()=>gr}),gr=`/letter_constellations/assets/inkpot-DnxrzNHx.webp`,_r=t({default:()=>vr}),vr=`/letter_constellations/assets/instrument-1utLT-6T.webp`,yr=t({default:()=>br}),br=`/letter_constellations/assets/invitation-z-PWuchC.webp`,xr=t({default:()=>Sr}),Sr=`/letter_constellations/assets/iron-CtDYRgb4.webp`,Cr=t({default:()=>wr}),wr=`/letter_constellations/assets/island-B8Bp5Tsi.webp`,Tr=t({default:()=>Er}),Er=`/letter_constellations/assets/jaguar-DHXPvMuk.webp`,Dr=t({default:()=>Or}),Or=`/letter_constellations/assets/jar-Dwv58Vzr.webp`,kr=t({default:()=>Ar}),Ar=`/letter_constellations/assets/jean_jacket-BULMDhzs.webp`,jr=t({default:()=>Mr}),Mr=`/letter_constellations/assets/jelly-CVVY10k9.webp`,Nr=t({default:()=>Pr}),Pr=`/letter_constellations/assets/jellyfish-BC5AuiGF.webp`,Fr=t({default:()=>Ir}),Ir=`/letter_constellations/assets/jigsaw-BOvQFAyk.webp`,Lr=t({default:()=>Rr}),Rr=`/letter_constellations/assets/juggler-GzS6Gfiw.webp`,zr=t({default:()=>Br}),Br=`/letter_constellations/assets/juice-BqR4izBV.webp`,Vr=t({default:()=>Hr}),Hr=`/letter_constellations/assets/kangaroo-BxEtKT1L.webp`,Ur=t({default:()=>Wr}),Wr=`/letter_constellations/assets/key-C62Kti4m.webp`,Gr=t({default:()=>Kr}),Kr=`/letter_constellations/assets/kid-CjHrY2fb.webp`,qr=t({default:()=>Jr}),Jr=`/letter_constellations/assets/kite-BB0XKGyC.webp`,Yr=t({default:()=>Xr}),Xr=`/letter_constellations/assets/kiwi-B0KGm69A.webp`,Zr=t({default:()=>Qr}),Qr=`/letter_constellations/assets/koala-CjVCRMJZ.webp`,$r=t({default:()=>ei}),ei=`/letter_constellations/assets/ladybug-CpSwAR2D.webp`,ti=t({default:()=>ni}),ni=`/letter_constellations/assets/lamp-DUz1kO0o.webp`,ri=t({default:()=>ii}),ii=`/letter_constellations/assets/lemon-Dd__w2Pq.webp`,ai=t({default:()=>oi}),oi=`/letter_constellations/assets/lemur-DHKaat7q.webp`,si=t({default:()=>ci}),ci=`/letter_constellations/assets/lettuce-D8mc2q-U.webp`,li=t({default:()=>ui}),ui=`/letter_constellations/assets/library-DKV5tmFO.webp`,di=t({default:()=>fi}),fi=`/letter_constellations/assets/llama-Bwh9dRAG.webp`,pi=t({default:()=>mi}),mi=`/letter_constellations/assets/map-vQmueQII.webp`,hi=t({default:()=>gi}),gi=`/letter_constellations/assets/milk_carton-5cuoXE3T.webp`,_i=t({default:()=>vi}),vi=`/letter_constellations/assets/mitten-BeKduZUD.webp`,yi=t({default:()=>bi}),bi=`/letter_constellations/assets/monkey-Cx-TqYqO.webp`,xi=t({default:()=>Si}),Si=`/letter_constellations/assets/moon-jSXzUMnz.webp`,Ci=t({default:()=>wi}),wi=`/letter_constellations/assets/mountain-CZsGBOtO.webp`,Ti=t({default:()=>Ei}),Ei=`/letter_constellations/assets/mouse-CmzFJelu.webp`,Di=t({default:()=>Oi}),Oi=`/letter_constellations/assets/muffin-BhJF_GvY.webp`,ki=t({default:()=>Ai}),Ai=`/letter_constellations/assets/mushroom-DHqipsMi.webp`,ji=t({default:()=>Mi}),Mi=`/letter_constellations/assets/nail-2NdCu6nX.webp`,Ni=t({default:()=>Pi}),Pi=`/letter_constellations/assets/needle-WCtT707e.webp`,Fi=t({default:()=>Ii}),Ii=`/letter_constellations/assets/nest-Bs_bSd3C.webp`,Li=t({default:()=>Ri}),Ri=`/letter_constellations/assets/net-CcQz3K1r.webp`,zi=t({default:()=>Bi}),Bi=`/letter_constellations/assets/newt-DMKiaP-a.webp`,Vi=t({default:()=>Hi}),Hi=`/letter_constellations/assets/night-loJiPRO9.webp`,Ui=t({default:()=>Wi}),Wi=`/letter_constellations/assets/nose-BGs_8Ldz.webp`,Gi=t({default:()=>Ki}),Ki=`/letter_constellations/assets/nurse-DnB4a9sK.webp`,qi=t({default:()=>Ji}),Ji=`/letter_constellations/assets/nuts-dWD5IlYg.webp`,Yi=t({default:()=>Xi}),Xi=`/letter_constellations/assets/oar-BJP2E8X6.webp`,Zi=t({default:()=>Qi}),Qi=`/letter_constellations/assets/octopus-Bjrhg0Kz.webp`,$i=t({default:()=>ea}),ea=`/letter_constellations/assets/onion-D-sGtoXB.webp`,ta=t({default:()=>na}),na=`/letter_constellations/assets/orange-DUVPpQC1.webp`,ra=t({default:()=>ia}),ia=`/letter_constellations/assets/orangutan-GXcPQi2e.webp`,aa=t({default:()=>oa}),oa=`/letter_constellations/assets/ostrich-CbDM_d-O.webp`,sa=t({default:()=>ca}),ca=`/letter_constellations/assets/owl-Du9V2IZL.webp`,la=t({default:()=>ua}),ua=`/letter_constellations/assets/panda-Ce4wJnci.webp`,da=t({default:()=>fa}),fa=`/letter_constellations/assets/pen-XTsE5Tp9.webp`,pa=t({default:()=>ma}),ma=`/letter_constellations/assets/pencil-CiXeWLpi.webp`,ha=t({default:()=>ga}),ga=`/letter_constellations/assets/penguin-DQXXzk60.webp`,_a=t({default:()=>va}),va=`/letter_constellations/assets/pig-f6PyeKkO.webp`,ya=t({default:()=>ba}),ba=`/letter_constellations/assets/pineapple-DEyw_76-.webp`,xa=t({default:()=>Sa}),Sa=`/letter_constellations/assets/pirate-CR8dih_s.webp`,Ca=t({default:()=>wa}),wa=`/letter_constellations/assets/pizza-BHofEroO.webp`,Ta=t({default:()=>Ea}),Ea=`/letter_constellations/assets/pumpkin-BK91jTFE.webp`,Da=t({default:()=>Oa}),Oa=`/letter_constellations/assets/quail-BPC2-Z_g.webp`,ka=t({default:()=>Aa}),Aa=`/letter_constellations/assets/queen-fiZYgCAh.webp`,ja=t({default:()=>Ma}),Ma=`/letter_constellations/assets/question-WkEi71Wr.webp`,Na=t({default:()=>Pa}),Pa=`/letter_constellations/assets/quill-KEF9k-Kr.webp`,Fa=t({default:()=>Ia}),Ia=`/letter_constellations/assets/quilt-B-SSp3IP.webp`,La=t({default:()=>Ra}),Ra=`/letter_constellations/assets/quince-CGOlv5XI.webp`,za=t({default:()=>Ba}),Ba=`/letter_constellations/assets/quiver-ClNEjwQB.webp`,Va=t({default:()=>Ha}),Ha=`/letter_constellations/assets/rabbit-DZWfw0Tv.webp`,Ua=t({default:()=>Wa}),Wa=`/letter_constellations/assets/radishes-B0KgqPIP.webp`,Ga=t({default:()=>Ka}),Ka=`/letter_constellations/assets/rainbow-CsvdimZC.webp`,qa=t({default:()=>Ja}),Ja=`/letter_constellations/assets/ring-xmCTdpMO.webp`,Ya=t({default:()=>Xa}),Xa=`/letter_constellations/assets/robin-X5gA9bFr.webp`,Za=t({default:()=>Qa}),Qa=`/letter_constellations/assets/robot-ozHBRcKM.webp`,$a=t({default:()=>eo}),eo=`/letter_constellations/assets/rocket-CitKkJev.webp`,to=t({default:()=>no}),no=`/letter_constellations/assets/rose-DL-Oy0Ek.webp`,ro=t({default:()=>io}),io=`/letter_constellations/assets/ruler-C1DPZ1Nf.webp`,ao=t({default:()=>oo}),oo=`/letter_constellations/assets/sailboat-BlcB2Z3i.webp`,so=t({default:()=>co}),co=`/letter_constellations/assets/sheep-C-5w7NTE.webp`,lo=t({default:()=>uo}),uo=`/letter_constellations/assets/snake-CbOrYwEs.webp`,fo=t({default:()=>po}),po=`/letter_constellations/assets/snowman-DjpZB2fz.webp`,mo=t({default:()=>ho}),ho=`/letter_constellations/assets/socks-Dp64Nj0U.webp`,go=t({default:()=>_o}),_o=`/letter_constellations/assets/star-CjyC3Ct8.webp`,vo=t({default:()=>yo}),yo=`/letter_constellations/assets/strawberry-ftyABBXF.webp`,bo=t({default:()=>xo}),xo=`/letter_constellations/assets/sun-BSCyCAh2.webp`,So=t({default:()=>Co}),Co=`/letter_constellations/assets/table-ZTiC2LOW.webp`,wo=t({default:()=>To}),To=`/letter_constellations/assets/teapot-zWS6vboA.webp`,Eo=t({default:()=>Do}),Do=`/letter_constellations/assets/tiger-CZFaYbD4.webp`,Oo=t({default:()=>ko}),ko=`/letter_constellations/assets/tomato-C7BXJ1No.webp`,Ao=t({default:()=>jo}),jo=`/letter_constellations/assets/tractor-_lqPOHsY.webp`,Mo=t({default:()=>No}),No=`/letter_constellations/assets/train-CEU1TdYj.webp`,Po=t({default:()=>Fo}),Fo=`/letter_constellations/assets/tree-BWqxXn5Z.webp`,Io=t({default:()=>Lo}),Lo=`/letter_constellations/assets/turtle-DvU5iXKL.webp`,Ro=t({default:()=>zo}),zo=`/letter_constellations/assets/ukulele-DCuOtxfK.webp`,Bo=t({default:()=>Vo}),Vo=`/letter_constellations/assets/umbrella-Dz1MlrtA.webp`,Ho=t({default:()=>Uo}),Uo=`/letter_constellations/assets/unicorn-CriH96qk.webp`,Wo=t({default:()=>Go}),Go=`/letter_constellations/assets/unicycle-gQror61r.webp`,Ko=t({default:()=>qo}),qo=`/letter_constellations/assets/unidentified_object-CUDu-AOC.webp`,Jo=t({default:()=>Yo}),Yo=`/letter_constellations/assets/urchin-_JmHjkoD.webp`,Xo=t({default:()=>Zo}),Zo=`/letter_constellations/assets/utensils-DFAZ2efj.webp`,Qo=t({default:()=>$o}),$o=`/letter_constellations/assets/vacuum-BLJAk_jE.webp`,es=t({default:()=>ts}),ts=`/letter_constellations/assets/van-Dpdv5-Hp.webp`,ns=t({default:()=>rs}),rs=`/letter_constellations/assets/vase-CpSUtdT6.webp`,is=t({default:()=>as}),as=`/letter_constellations/assets/vegetables-ChjTiEM-.webp`,os=t({default:()=>ss}),ss=`/letter_constellations/assets/vest-D3gDumUU.webp`,cs=t({default:()=>ls}),ls=`/letter_constellations/assets/village-NwxI-P5i.webp`,us=t({default:()=>ds}),ds=`/letter_constellations/assets/violin-Bwy5HY6e.webp`,fs=t({default:()=>ps}),ps=`/letter_constellations/assets/volcano-DCRqkKRz.webp`,ms=t({default:()=>hs}),hs=`/letter_constellations/assets/vulture-C-ErfnZu.webp`,gs=t({default:()=>_s}),_s=`/letter_constellations/assets/wagon-DW5tu-i6.webp`,vs=t({default:()=>ys}),ys=`/letter_constellations/assets/watch-ahI5puCQ.webp`,bs=t({default:()=>xs}),xs=`/letter_constellations/assets/watermelon-DOsensRc.webp`,Ss=t({default:()=>Cs}),Cs=`/letter_constellations/assets/web-Ce6CPeZ_.webp`,ws=t({default:()=>Ts}),Ts=`/letter_constellations/assets/whale-DAEVlAKS.webp`,Es=t({default:()=>Ds}),Ds=`/letter_constellations/assets/wind-B6DREdK4.webp`,Os=t({default:()=>ks}),ks=`/letter_constellations/assets/wizard-C8Dnr-1B.webp`,As=t({default:()=>js}),js=`/letter_constellations/assets/wolf-aZD8PyPP.webp`,Ms=t({default:()=>Ns}),Ns=`/letter_constellations/assets/worm-Ds6LH9wX.webp`,Ps=t({default:()=>Fs}),Fs=`/letter_constellations/assets/accordion-DqQtriLm.webm?as=url`,Is=t({default:()=>Ls}),Ls=`/letter_constellations/assets/acorn-BIezBOPs.webm?as=url`,Rs=t({default:()=>zs}),zs=`/letter_constellations/assets/airplane-C_Wfm5g-.webm?as=url`,Bs=t({default:()=>Vs}),Vs=`/letter_constellations/assets/alligator-DYepK-g8.webm?as=url`,Hs=t({default:()=>Us}),Us=`/letter_constellations/assets/anchor-D_H3HrJm.webm?as=url`,Ws=t({default:()=>Gs}),Gs=`/letter_constellations/assets/ant-D6QhcluX.webm?as=url`,Ks=t({default:()=>qs}),qs=`/letter_constellations/assets/apple-CZDvf2t3.webm?as=url`,Js=t({default:()=>Ys}),Ys=`/letter_constellations/assets/arrow-Cs627Rik.webm?as=url`,Xs=t({default:()=>Zs}),Zs=`/letter_constellations/assets/astronaut-B9QACnUz.webm?as=url`,Qs=t({default:()=>$s}),$s=`/letter_constellations/assets/a-CMkzIMFv.webm?as=url`,ec=t({default:()=>tc}),tc=`/letter_constellations/assets/b-C2kyMVsg.webm?as=url`,nc=t({default:()=>rc}),rc=`/letter_constellations/assets/c-IjVD-Vre.webm?as=url`,ic=t({default:()=>ac}),ac=`/letter_constellations/assets/d-9vk5XyCw.webm?as=url`,oc=t({default:()=>sc}),sc=`/letter_constellations/assets/e-VRhdBt2g.webm?as=url`,cc=t({default:()=>lc}),lc=`/letter_constellations/assets/f-BVhXOumA.webm?as=url`,uc=t({default:()=>dc}),dc=`/letter_constellations/assets/g-DIYEe5HW.webm?as=url`,fc=t({default:()=>pc}),pc=`/letter_constellations/assets/h-DyPaSF0e.webm?as=url`,mc=t({default:()=>hc}),hc=`/letter_constellations/assets/i-C5tqm2EV.webm?as=url`,gc=t({default:()=>_c}),_c=`/letter_constellations/assets/j-Be2zckp_.webm?as=url`,vc=t({default:()=>yc}),yc=`/letter_constellations/assets/k-BakCspA1.webm?as=url`,bc=t({default:()=>xc}),xc=`/letter_constellations/assets/l-NUuunizq.webm?as=url`,Sc=t({default:()=>Cc}),Cc=`/letter_constellations/assets/m-BB98DVPZ.webm?as=url`,wc=t({default:()=>Tc}),Tc=`/letter_constellations/assets/n-CnIX4TYm.webm?as=url`,Ec=t({default:()=>Dc}),Dc=`/letter_constellations/assets/o-BiLMB4YT.webm?as=url`,Oc=t({default:()=>kc}),kc=`/letter_constellations/assets/p-C4ciUhaN.webm?as=url`,Ac=t({default:()=>jc}),jc=`/letter_constellations/assets/q-DF2D67v_.webm?as=url`,Mc=t({default:()=>Nc}),Nc=`/letter_constellations/assets/r-CS0DW1cp.webm?as=url`,Pc=t({default:()=>Fc}),Fc=`/letter_constellations/assets/s-An_rASvU.webm?as=url`,Ic=t({default:()=>Lc}),Lc=`/letter_constellations/assets/t-CVflKspQ.webm?as=url`,Rc=t({default:()=>zc}),zc=`/letter_constellations/assets/u-Dudg3DnT.webm?as=url`,Bc=t({default:()=>Vc}),Vc=`/letter_constellations/assets/v-WZ8s5vPh.webm?as=url`,Hc=t({default:()=>Uc}),Uc=`/letter_constellations/assets/w-BElagC9L.webm?as=url`,Wc=t({default:()=>Gc}),Gc=`/letter_constellations/assets/x-BCyjQ6tg.webm?as=url`,Kc=t({default:()=>qc}),qc=`/letter_constellations/assets/y-CxUp37AX.webm?as=url`,Jc=t({default:()=>Yc}),Yc=`/letter_constellations/assets/z-CmvHDWPa.webm?as=url`,Xc=t({default:()=>Zc}),Zc=`/letter_constellations/assets/bananas-h3WNqzRq.webm?as=url`,Qc=t({default:()=>$c}),$c=`/letter_constellations/assets/beachball-CHdPyc90.webm?as=url`,el=t({default:()=>tl}),tl=`/letter_constellations/assets/bear-5zJMp45m.webm?as=url`,nl=t({default:()=>rl}),rl=`/letter_constellations/assets/bee-CIa4b6b3.webm?as=url`,il=t({default:()=>al}),al=`/letter_constellations/assets/bird-C-mCHkrZ.webm?as=url`,ol=t({default:()=>sl}),sl=`/letter_constellations/assets/boat-Egh2j8Fn.webm?as=url`,cl=t({default:()=>ll}),ll=`/letter_constellations/assets/books-CDA8xLKG.webm?as=url`,ul=t({default:()=>dl}),dl=`/letter_constellations/assets/bus-BcHciuah.webm?as=url`,fl=t({default:()=>pl}),pl=`/letter_constellations/assets/butterfly-Cc832W2U.webm?as=url`,ml=t({default:()=>hl}),hl=`/letter_constellations/assets/cake-CE8GeVD6.webm?as=url`,gl=t({default:()=>_l}),_l=`/letter_constellations/assets/car-D5SmH24e.webm?as=url`,vl=t({default:()=>yl}),yl=`/letter_constellations/assets/castle-BoljANwC.webm?as=url`,bl=t({default:()=>xl}),xl=`/letter_constellations/assets/cat-BDJFtO1J.webm?as=url`,Sl=t({default:()=>Cl}),Cl=`/letter_constellations/assets/clock-BdR7qum2.webm?as=url`,wl=t({default:()=>Tl}),Tl=`/letter_constellations/assets/coat-16JLlN-R.webm?as=url`,El=t({default:()=>Dl}),Dl=`/letter_constellations/assets/cow-_gfKTNxD.webm?as=url`,Ol=t({default:()=>kl}),kl=`/letter_constellations/assets/crayon-DcBF0plc.webm?as=url`,Al=t({default:()=>jl}),jl=`/letter_constellations/assets/cup-U1NYHdiC.webm?as=url`,Ml=t({default:()=>Nl}),Nl=`/letter_constellations/assets/dog-L0-qQmmf.webm?as=url`,Pl=t({default:()=>Fl}),Fl=`/letter_constellations/assets/doll-DHzmlk22.webm?as=url`,Il=t({default:()=>Ll}),Ll=`/letter_constellations/assets/dolphin-DHN3M5dU.webm?as=url`,Rl=t({default:()=>zl}),zl=`/letter_constellations/assets/donkey-DZGy0jN1.webm?as=url`,Bl=t({default:()=>Vl}),Vl=`/letter_constellations/assets/door-D03rYjG6.webm?as=url`,Hl=t({default:()=>Ul}),Ul=`/letter_constellations/assets/dragon-Bv-oOD4o.webm?as=url`,Wl=t({default:()=>Gl}),Gl=`/letter_constellations/assets/dress-DioYKYZv.webm?as=url`,Kl=t({default:()=>ql}),ql=`/letter_constellations/assets/drum-Cf0je1UD.webm?as=url`,Jl=t({default:()=>Yl}),Yl=`/letter_constellations/assets/duck-Bgo1sGLJ.webm?as=url`,Xl=t({default:()=>Zl}),Zl=`/letter_constellations/assets/eagle-BgbhiRSg.webm?as=url`,Ql=t({default:()=>$l}),$l=`/letter_constellations/assets/ear-Dl7REk6c.webm?as=url`,eu=t({default:()=>tu}),tu=`/letter_constellations/assets/earth-B5sQyPAK.webm?as=url`,nu=t({default:()=>ru}),ru=`/letter_constellations/assets/easel-DrjyCnNQ.webm?as=url`,iu=t({default:()=>au}),au=`/letter_constellations/assets/egg-Cp_mjYOa.webm?as=url`,ou=t({default:()=>su}),su=`/letter_constellations/assets/elephant-DXDMMvRo.webm?as=url`,cu=t({default:()=>lu}),lu=`/letter_constellations/assets/engine-CGoW8nln.webm?as=url`,uu=t({default:()=>du}),du=`/letter_constellations/assets/envelope-CZkB1Qdz.webm?as=url`,fu=t({default:()=>pu}),pu=`/letter_constellations/assets/eraser-CL0SN2d1.webm?as=url`,mu=t({default:()=>hu}),hu=`/letter_constellations/assets/feather-BTNKnrGo.webm?as=url`,gu=t({default:()=>_u}),_u=`/letter_constellations/assets/fence-BgF_34t7.webm?as=url`,vu=t({default:()=>yu}),yu=`/letter_constellations/assets/firefly-B7E-EXaq.webm?as=url`,bu=t({default:()=>xu}),xu=`/letter_constellations/assets/fish-DsqXYDML.webm?as=url`,Su=t({default:()=>Cu}),Cu=`/letter_constellations/assets/flower-DzzwNf_U.webm?as=url`,wu=t({default:()=>Tu}),Tu=`/letter_constellations/assets/foot-CqvpFi2I.webm?as=url`,Eu=t({default:()=>Du}),Du=`/letter_constellations/assets/fork-BbXVs51Y.webm?as=url`,Ou=t({default:()=>ku}),ku=`/letter_constellations/assets/fox-xbmCA8tV.webm?as=url`,Au=t({default:()=>ju}),ju=`/letter_constellations/assets/frog-ylp_1Suh.webm?as=url`,Mu=t({default:()=>Nu}),Nu=`/letter_constellations/assets/garden-B_iCSqP1.webm?as=url`,Pu=t({default:()=>Fu}),Fu=`/letter_constellations/assets/ghost-DCvyj5gL.webm?as=url`,Iu=t({default:()=>Lu}),Lu=`/letter_constellations/assets/gift-Cy86EoFO.webm?as=url`,Ru=t({default:()=>zu}),zu=`/letter_constellations/assets/giraffe-CADFBhnh.webm?as=url`,Bu=t({default:()=>Vu}),Vu=`/letter_constellations/assets/gloves-BLFuRMLi.webm?as=url`,Hu=t({default:()=>Uu}),Uu=`/letter_constellations/assets/goat-BTaP2ceE.webm?as=url`,Wu=t({default:()=>Gu}),Gu=`/letter_constellations/assets/goldfish-CfKPbugz.webm?as=url`,Ku=t({default:()=>qu}),qu=`/letter_constellations/assets/grapes-CvsbRxCI.webm?as=url`,Ju=t({default:()=>Yu}),Yu=`/letter_constellations/assets/guitar-DdMBQt2m.webm?as=url`,Xu=t({default:()=>Zu}),Zu=`/letter_constellations/assets/hammer-Ceh7AMku.webm?as=url`,Qu=t({default:()=>$u}),$u=`/letter_constellations/assets/hand-KB-VfcaO.webm?as=url`,ed=t({default:()=>td}),td=`/letter_constellations/assets/hat-BiSruZqk.webm?as=url`,nd=t({default:()=>rd}),rd=`/letter_constellations/assets/heart-CgLg_Xw1.webm?as=url`,id=t({default:()=>ad}),ad=`/letter_constellations/assets/hedgehog-DM6G4aTi.webm?as=url`,od=t({default:()=>sd}),sd=`/letter_constellations/assets/helicopter-BDdKO2XC.webm?as=url`,cd=t({default:()=>ld}),ld=`/letter_constellations/assets/horse-BQ3oQppo.webm?as=url`,ud=t({default:()=>dd}),dd=`/letter_constellations/assets/hotdog-q3lfgc-H.webm?as=url`,fd=t({default:()=>pd}),pd=`/letter_constellations/assets/house-FiY5MHVH.webm?as=url`,md=t({default:()=>hd}),hd=`/letter_constellations/assets/ice_cream-uwet6mas.webm?as=url`,gd=t({default:()=>_d}),_d=`/letter_constellations/assets/igloo-Bm1n6RqU.webm?as=url`,vd=t({default:()=>yd}),yd=`/letter_constellations/assets/iguana-Cj9OFN_N.webm?as=url`,bd=t({default:()=>xd}),xd=`/letter_constellations/assets/inkpot-CedBlT6t.webm?as=url`,Sd=t({default:()=>Cd}),Cd=`/letter_constellations/assets/insect-BVXEJtav.webm?as=url`,wd=t({default:()=>Td}),Td=`/letter_constellations/assets/instrument-DEj5EZot.webm?as=url`,Ed=t({default:()=>Dd}),Dd=`/letter_constellations/assets/invitation-yDuZq8gj.webm?as=url`,Od=t({default:()=>kd}),kd=`/letter_constellations/assets/iron-Cx2pzUpR.webm?as=url`,Ad=t({default:()=>jd}),jd=`/letter_constellations/assets/island-fs6PoIj0.webm?as=url`,Md=t({default:()=>Nd}),Nd=`/letter_constellations/assets/jaguar-BBmfc63Y.webm?as=url`,Pd=t({default:()=>Fd}),Fd=`/letter_constellations/assets/jar-DHu15TBb.webm?as=url`,Id=t({default:()=>Ld}),Ld=`/letter_constellations/assets/jean_jacket-6ocVOvCD.webm?as=url`,Rd=t({default:()=>zd}),zd=`/letter_constellations/assets/jelly-EK6xdlfq.webm?as=url`,Bd=t({default:()=>Vd}),Vd=`/letter_constellations/assets/jellyfish-BRbFjms6.webm?as=url`,Hd=t({default:()=>Ud}),Ud=`/letter_constellations/assets/jet-ByhMGSXW.webm?as=url`,Wd=t({default:()=>Gd}),Gd=`/letter_constellations/assets/jigsaw-DZYjOBmp.webm?as=url`,Kd=t({default:()=>qd}),qd=`/letter_constellations/assets/juggler-DBOlauIy.webm?as=url`,Jd=t({default:()=>Yd}),Yd=`/letter_constellations/assets/juice-D-68Bu_u.webm?as=url`,Xd=t({default:()=>Zd}),Zd=`/letter_constellations/assets/kangaroo-CcIkE6dA.webm?as=url`,Qd=t({default:()=>$d}),$d=`/letter_constellations/assets/kettle_drum-D-Fw6JsA.webm?as=url`,ef=t({default:()=>tf}),tf=`/letter_constellations/assets/key-KD60Ug_d.webm?as=url`,nf=t({default:()=>rf}),rf=`/letter_constellations/assets/kid-tJFEsvg6.webm?as=url`,af=t({default:()=>of}),of=`/letter_constellations/assets/kite-BvkQpXQ4.webm?as=url`,sf=t({default:()=>cf}),cf=`/letter_constellations/assets/kiwi-DSByUtRu.webm?as=url`,lf=t({default:()=>uf}),uf=`/letter_constellations/assets/knight-BHnTxxNg.webm?as=url`,df=t({default:()=>ff}),ff=`/letter_constellations/assets/knot-BFtAc-vX.webm?as=url`,pf=t({default:()=>mf}),mf=`/letter_constellations/assets/koala-BOrbjEQC.webm?as=url`,hf=t({default:()=>gf}),gf=`/letter_constellations/assets/labrador-C2atZ3eI.webm?as=url`,_f=t({default:()=>vf}),vf=`/letter_constellations/assets/ladybug-D_CzUcBl.webm?as=url`,yf=t({default:()=>bf}),bf=`/letter_constellations/assets/lamp-CA6Xv_tn.webm?as=url`,xf=t({default:()=>Sf}),Sf=`/letter_constellations/assets/lemon-CPuugyKa.webm?as=url`,Cf=t({default:()=>wf}),wf=`/letter_constellations/assets/lemur-CPypRhBI.webm?as=url`,Tf=t({default:()=>Ef}),Ef=`/letter_constellations/assets/lettuce-9ltxxL4a.webm?as=url`,Df=t({default:()=>Of}),Of=`/letter_constellations/assets/library-Bt30cqOU.webm?as=url`,kf=t({default:()=>Af}),Af=`/letter_constellations/assets/llama-CnavT3OA.webm?as=url`,jf=t({default:()=>Mf}),Mf=`/letter_constellations/assets/lungs-CtTA6hzs.webm?as=url`,Nf=t({default:()=>Pf}),Pf=`/letter_constellations/assets/map-DpXTnY61.webm?as=url`,Ff=t({default:()=>If}),If=`/letter_constellations/assets/milk_carton-CYrSpFIF.webm?as=url`,Lf=t({default:()=>Rf}),Rf=`/letter_constellations/assets/mitten-CjMBbbHc.webm?as=url`,zf=t({default:()=>Bf}),Bf=`/letter_constellations/assets/monkey-Da7Co2b4.webm?as=url`,Vf=t({default:()=>Hf}),Hf=`/letter_constellations/assets/moon-BQlWDz1v.webm?as=url`,Uf=t({default:()=>Wf}),Wf=`/letter_constellations/assets/mountain-BXOdjgsP.webm?as=url`,Gf=t({default:()=>Kf}),Kf=`/letter_constellations/assets/mouse-CX-udEkX.webm?as=url`,qf=t({default:()=>Jf}),Jf=`/letter_constellations/assets/muffin-ZyoQalKv.webm?as=url`,Yf=t({default:()=>Xf}),Xf=`/letter_constellations/assets/mushroom-ma49-9Es.webm?as=url`,Zf=t({default:()=>Qf}),Qf=`/letter_constellations/assets/nail-BDgqn4Oi.webm?as=url`,$f=t({default:()=>ep}),ep=`/letter_constellations/assets/needle-DfSM5BBx.webm?as=url`,tp=t({default:()=>np}),np=`/letter_constellations/assets/nest-D8oW8sHs.webm?as=url`,rp=t({default:()=>ip}),ip=`/letter_constellations/assets/net-BOWyW7Pd.webm?as=url`,ap=t({default:()=>op}),op=`/letter_constellations/assets/newt-Di4rr-kM.webm?as=url`,sp=t({default:()=>cp}),cp=`/letter_constellations/assets/night-DYJXIJ6m.webm?as=url`,lp=t({default:()=>up}),up=`/letter_constellations/assets/nose-wU6bmTQo.webm?as=url`,dp=t({default:()=>fp}),fp=`/letter_constellations/assets/nurse-Dvz6wvdQ.webm?as=url`,pp=t({default:()=>mp}),mp=`/letter_constellations/assets/nuts-DOYhPI3p.webm?as=url`,hp=t({default:()=>gp}),gp=`/letter_constellations/assets/oar-NPG07m8x.webm?as=url`,_p=t({default:()=>vp}),vp=`/letter_constellations/assets/octopus-BlAhrR5x.webm?as=url`,yp=t({default:()=>bp}),bp=`/letter_constellations/assets/onion-C3UnQO2e.webm?as=url`,xp=t({default:()=>Sp}),Sp=`/letter_constellations/assets/orange-CAjV4sVP.webm?as=url`,Cp=t({default:()=>wp}),wp=`/letter_constellations/assets/orangutan-jdEjO_N5.webm?as=url`,Tp=t({default:()=>Ep}),Ep=`/letter_constellations/assets/orbit-Cd0ALAeU.webm?as=url`,Dp=t({default:()=>Op}),Op=`/letter_constellations/assets/ostrich-Dku7vhly.webm?as=url`,kp=t({default:()=>Ap}),Ap=`/letter_constellations/assets/overalls-CI6JhbtH.webm?as=url`,jp=t({default:()=>Mp}),Mp=`/letter_constellations/assets/owl-CQWsqBJC.webm?as=url`,Np=t({default:()=>Pp}),Pp=`/letter_constellations/assets/panda-BmfrrnU6.webm?as=url`,Fp=t({default:()=>Ip}),Ip=`/letter_constellations/assets/pen-ZTmVznwu.webm?as=url`,Lp=t({default:()=>Rp}),Rp=`/letter_constellations/assets/pencil-BVQ8rqGT.webm?as=url`,zp=t({default:()=>Bp}),Bp=`/letter_constellations/assets/penguin-DeD256Gp.webm?as=url`,Vp=t({default:()=>Hp}),Hp=`/letter_constellations/assets/pig-DzgF3-tq.webm?as=url`,Up=t({default:()=>Wp}),Wp=`/letter_constellations/assets/pineapple-DxiQhkpX.webm?as=url`,Gp=t({default:()=>Kp}),Kp=`/letter_constellations/assets/pirate-D-3TP_NB.webm?as=url`,qp=t({default:()=>Jp}),Jp=`/letter_constellations/assets/pizza-FIglZQ7d.webm?as=url`,Yp=t({default:()=>Xp}),Xp=`/letter_constellations/assets/pumpkin-BindYuWd.webm?as=url`,Zp=t({default:()=>Qp}),Qp=`/letter_constellations/assets/quaff-C0HVHSvx.webm?as=url`,$p=t({default:()=>em}),em=`/letter_constellations/assets/quail-B8-wr5Ky.webm?as=url`,tm=t({default:()=>nm}),nm=`/letter_constellations/assets/quarter-DYSIAC14.webm?as=url`,rm=t({default:()=>im}),im=`/letter_constellations/assets/queen-DJtNeLa0.webm?as=url`,am=t({default:()=>om}),om=`/letter_constellations/assets/question-V1fIR47w.webm?as=url`,sm=t({default:()=>cm}),cm=`/letter_constellations/assets/quill-Bh53ye5V.webm?as=url`,lm=t({default:()=>um}),um=`/letter_constellations/assets/quilt-Ju7WSa2i.webm?as=url`,dm=t({default:()=>fm}),fm=`/letter_constellations/assets/quince-DQk-aRBI.webm?as=url`,pm=t({default:()=>mm}),mm=`/letter_constellations/assets/quiver-KF2M2SC_.webm?as=url`,hm=t({default:()=>gm}),gm=`/letter_constellations/assets/rabbit-Bn1L-ab9.webm?as=url`,_m=t({default:()=>vm}),vm=`/letter_constellations/assets/radishes-Caj0bGTw.webm?as=url`,ym=t({default:()=>bm}),bm=`/letter_constellations/assets/rainbow-D_0DDL10.webm?as=url`,xm=t({default:()=>Sm}),Sm=`/letter_constellations/assets/ring-CvaEKfC1.webm?as=url`,Cm=t({default:()=>wm}),wm=`/letter_constellations/assets/robin-B4Wyw6Kt.webm?as=url`,Tm=t({default:()=>Em}),Em=`/letter_constellations/assets/robot-O0G0NIz8.webm?as=url`,Dm=t({default:()=>Om}),Om=`/letter_constellations/assets/rocket-DyCWYIqK.webm?as=url`,km=t({default:()=>Am}),Am=`/letter_constellations/assets/rose-DNpS-rzs.webm?as=url`,jm=t({default:()=>Mm}),Mm=`/letter_constellations/assets/ruler-3jiqdeeH.webm?as=url`,Nm=t({default:()=>Pm}),Pm=`/letter_constellations/assets/sailboat-C9PCPNTV.webm?as=url`,Fm=t({default:()=>Im}),Im=`/letter_constellations/assets/sheep-BD1Qex-a.webm?as=url`,Lm=t({default:()=>Rm}),Rm=`/letter_constellations/assets/snake-3VL3pI0b.webm?as=url`,zm=t({default:()=>Bm}),Bm=`/letter_constellations/assets/snowman-BVngaE17.webm?as=url`,Vm=t({default:()=>Hm}),Hm=`/letter_constellations/assets/socks-BlZGzMoX.webm?as=url`,Um=t({default:()=>Wm}),Wm=`/letter_constellations/assets/spoon-Bpezv9ur.webm?as=url`,Gm=t({default:()=>Km}),Km=`/letter_constellations/assets/star-ByhnbkyF.webm?as=url`,qm=t({default:()=>Jm}),Jm=`/letter_constellations/assets/strawberry-WDKIdltR.webm?as=url`,Ym=t({default:()=>Xm}),Xm=`/letter_constellations/assets/sun-OXh-YcDj.webm?as=url`,Zm=t({default:()=>Qm}),Qm=`/letter_constellations/assets/table-DdVbos9W.webm?as=url`,$m=t({default:()=>eh}),eh=`/letter_constellations/assets/teapot-C8QdlDfa.webm?as=url`,th=t({default:()=>nh}),nh=`/letter_constellations/assets/teddy_bear-DbuTfBst.webm?as=url`,rh=t({default:()=>ih}),ih=`/letter_constellations/assets/tiger-DDYpj3Yp.webm?as=url`,ah=t({default:()=>oh}),oh=`/letter_constellations/assets/tomato-WtQkNxZo.webm?as=url`,sh=t({default:()=>ch}),ch=`/letter_constellations/assets/tractor-CCgqU7qU.webm?as=url`,lh=t({default:()=>uh}),uh=`/letter_constellations/assets/train-C5Zj-m0t.webm?as=url`,dh=t({default:()=>fh}),fh=`/letter_constellations/assets/tree-nuB9A3vs.webm?as=url`,ph=t({default:()=>mh}),mh=`/letter_constellations/assets/turtle-BDkcySK2.webm?as=url`,hh=t({default:()=>gh}),gh=`/letter_constellations/assets/ukulele-CLqXYXhe.webm?as=url`,_h=t({default:()=>vh}),vh=`/letter_constellations/assets/umbrella-r_963ubY.webm?as=url`,yh=t({default:()=>bh}),bh=`/letter_constellations/assets/ungulate-CTO96oPD.webm?as=url`,xh=t({default:()=>Sh}),Sh=`/letter_constellations/assets/unicorn-CY2TK8f5.webm?as=url`,Ch=t({default:()=>wh}),wh=`/letter_constellations/assets/unicycle-DGKk_v6u.webm?as=url`,Th=t({default:()=>Eh}),Eh=`/letter_constellations/assets/unidentified_object--Ro5ICPR.webm?as=url`,Dh=t({default:()=>Oh}),Oh=`/letter_constellations/assets/uniform-DqfMcPDd.webm?as=url`,kh=t({default:()=>Ah}),Ah=`/letter_constellations/assets/urchin-iCRAGDls.webm?as=url`,jh=t({default:()=>Mh}),Mh=`/letter_constellations/assets/utensils-BzdOuCsZ.webm?as=url`,Nh=t({default:()=>Ph}),Ph=`/letter_constellations/assets/vacuum-DoCcb9z-.webm?as=url`,Fh=t({default:()=>Ih}),Ih=`/letter_constellations/assets/van-CQ1lJBiN.webm?as=url`,Lh=t({default:()=>Rh}),Rh=`/letter_constellations/assets/vase-CccRv94X.webm?as=url`,zh=t({default:()=>Bh}),Bh=`/letter_constellations/assets/vegetables-DLtroWc1.webm?as=url`,Vh=t({default:()=>Hh}),Hh=`/letter_constellations/assets/vest-DBXjbVnZ.webm?as=url`,Uh=t({default:()=>Wh}),Wh=`/letter_constellations/assets/village-CVG8tels.webm?as=url`,Gh=t({default:()=>Kh}),Kh=`/letter_constellations/assets/violin-BE21TgO1.webm?as=url`,qh=t({default:()=>Jh}),Jh=`/letter_constellations/assets/volcano-gCdpROrp.webm?as=url`,Yh=t({default:()=>Xh}),Xh=`/letter_constellations/assets/vulture-TiasgFRu.webm?as=url`,Zh=t({default:()=>Qh}),Qh=`/letter_constellations/assets/wagon-BpMibSu1.webm?as=url`,$h=t({default:()=>eg}),eg=`/letter_constellations/assets/watch-DR-zilf0.webm?as=url`,tg=t({default:()=>ng}),ng=`/letter_constellations/assets/watermelon-NcZQdRkY.webm?as=url`,rg=t({default:()=>ig}),ig=`/letter_constellations/assets/web-CjQIo_1Q.webm?as=url`,ag=t({default:()=>og}),og=`/letter_constellations/assets/whale-cDbKOtue.webm?as=url`,sg=t({default:()=>cg}),cg=`/letter_constellations/assets/wind-DK91ZQgX.webm?as=url`,lg=t({default:()=>ug}),ug=`/letter_constellations/assets/wizard-DhJfciPy.webm?as=url`,dg=t({default:()=>fg}),fg=`/letter_constellations/assets/wolf-DHbkNJpk.webm?as=url`,pg=t({default:()=>mg}),mg=`/letter_constellations/assets/worm-BWnc7YI5.webm?as=url`,hg={"./assets/items/a/accordion.png":Me,"./assets/items/a/acorn.png":Pe,"./assets/items/a/airplane.png":Ie,"./assets/items/a/alligator.png":Re,"./assets/items/a/anchor.png":Be,"./assets/items/a/ant.png":He,"./assets/items/a/apple.png":We,"./assets/items/a/arrow.png":Ke,"./assets/items/a/astronaut.png":Je,"./assets/items/b/bananas.png":Xe,"./assets/items/b/beachball.png":Qe,"./assets/items/b/bear.png":et,"./assets/items/b/bee.png":nt,"./assets/items/b/bird.png":it,"./assets/items/b/boat.png":ot,"./assets/items/b/books.png":ct,"./assets/items/b/bus.png":ut,"./assets/items/b/butterfly.png":ft,"./assets/items/c/cake.png":mt,"./assets/items/c/car.png":gt,"./assets/items/c/castle.png":vt,"./assets/items/c/cat.png":bt,"./assets/items/c/clock.png":St,"./assets/items/c/coat.png":wt,"./assets/items/c/cow.png":Et,"./assets/items/c/crayon.png":Ot,"./assets/items/c/cup.png":At,"./assets/items/d/dog.png":Mt,"./assets/items/d/doll.png":Pt,"./assets/items/d/dolphin.png":It,"./assets/items/d/donkey.png":Rt,"./assets/items/d/door.png":Bt,"./assets/items/d/dragon.png":Ht,"./assets/items/d/dress.png":Wt,"./assets/items/d/drum.png":Kt,"./assets/items/d/duck.png":Jt,"./assets/items/e/eagle.png":Xt,"./assets/items/e/ear.png":Qt,"./assets/items/e/earth.png":en,"./assets/items/e/easel.png":nn,"./assets/items/e/egg.png":an,"./assets/items/e/elephant.png":sn,"./assets/items/e/envelope.png":ln,"./assets/items/e/eraser.png":dn,"./assets/items/f/feather.png":pn,"./assets/items/f/fence.png":hn,"./assets/items/f/fish.png":_n,"./assets/items/f/flower.png":yn,"./assets/items/f/foot.png":xn,"./assets/items/f/fork.png":Cn,"./assets/items/f/fox.png":Tn,"./assets/items/f/frog.png":Dn,"./assets/items/g/garden.png":kn,"./assets/items/g/ghost.png":jn,"./assets/items/g/giraffe.png":Nn,"./assets/items/g/gloves.png":Fn,"./assets/items/g/goat.png":Ln,"./assets/items/g/goldfish.png":zn,"./assets/items/g/grapes.png":Vn,"./assets/items/g/guitar.png":Un,"./assets/items/h/hammer.png":Gn,"./assets/items/h/hand.png":qn,"./assets/items/h/hat.png":Yn,"./assets/items/h/heart.png":Zn,"./assets/items/h/hedgehog.png":$n,"./assets/items/h/helicopter.png":tr,"./assets/items/h/horse.png":rr,"./assets/items/h/hotdog.png":ar,"./assets/items/h/house.png":sr,"./assets/items/i/ice_cream.png":lr,"./assets/items/i/igloo.png":dr,"./assets/items/i/iguana.png":pr,"./assets/items/i/inkpot.png":hr,"./assets/items/i/instrument.png":_r,"./assets/items/i/invitation.png":yr,"./assets/items/i/iron.png":xr,"./assets/items/i/island.png":Cr,"./assets/items/j/jaguar.png":Tr,"./assets/items/j/jar.png":Dr,"./assets/items/j/jean_jacket.png":kr,"./assets/items/j/jelly.png":jr,"./assets/items/j/jellyfish.png":Nr,"./assets/items/j/jigsaw.png":Fr,"./assets/items/j/juggler.png":Lr,"./assets/items/j/juice.png":zr,"./assets/items/k/kangaroo.png":Vr,"./assets/items/k/key.png":Ur,"./assets/items/k/kid.png":Gr,"./assets/items/k/kite.png":qr,"./assets/items/k/kiwi.png":Yr,"./assets/items/k/koala.png":Zr,"./assets/items/l/ladybug.png":$r,"./assets/items/l/lamp.png":ti,"./assets/items/l/lemon.png":ri,"./assets/items/l/lemur.png":ai,"./assets/items/l/lettuce.png":si,"./assets/items/l/library.png":li,"./assets/items/l/llama.png":di,"./assets/items/m/map.png":pi,"./assets/items/m/milk_carton.png":hi,"./assets/items/m/mitten.png":_i,"./assets/items/m/monkey.png":yi,"./assets/items/m/moon.png":xi,"./assets/items/m/mountain.png":Ci,"./assets/items/m/mouse.png":Ti,"./assets/items/m/muffin.png":Di,"./assets/items/m/mushroom.png":ki,"./assets/items/n/nail.png":ji,"./assets/items/n/needle.png":Ni,"./assets/items/n/nest.png":Fi,"./assets/items/n/net.png":Li,"./assets/items/n/newt.png":zi,"./assets/items/n/night.png":Vi,"./assets/items/n/nose.png":Ui,"./assets/items/n/nurse.png":Gi,"./assets/items/n/nuts.png":qi,"./assets/items/o/oar.png":Yi,"./assets/items/o/octopus.png":Zi,"./assets/items/o/onion.png":$i,"./assets/items/o/orange.png":ta,"./assets/items/o/orangutan.png":ra,"./assets/items/o/ostrich.png":aa,"./assets/items/o/owl.png":sa,"./assets/items/p/panda.png":la,"./assets/items/p/pen.png":da,"./assets/items/p/pencil.png":pa,"./assets/items/p/penguin.png":ha,"./assets/items/p/pig.png":_a,"./assets/items/p/pineapple.png":ya,"./assets/items/p/pirate.png":xa,"./assets/items/p/pizza.png":Ca,"./assets/items/p/pumpkin.png":Ta,"./assets/items/q/quail.png":Da,"./assets/items/q/queen.png":ka,"./assets/items/q/question.png":ja,"./assets/items/q/quill.png":Na,"./assets/items/q/quilt.png":Fa,"./assets/items/q/quince.png":La,"./assets/items/q/quiver.png":za,"./assets/items/r/rabbit.png":Va,"./assets/items/r/radishes.png":Ua,"./assets/items/r/rainbow.png":Ga,"./assets/items/r/ring.png":qa,"./assets/items/r/robin.png":Ya,"./assets/items/r/robot.png":Za,"./assets/items/r/rocket.png":$a,"./assets/items/r/rose.png":to,"./assets/items/r/ruler.png":ro,"./assets/items/s/sailboat.png":ao,"./assets/items/s/sheep.png":so,"./assets/items/s/snake.png":lo,"./assets/items/s/snowman.png":fo,"./assets/items/s/socks.png":mo,"./assets/items/s/star.png":go,"./assets/items/s/strawberry.png":vo,"./assets/items/s/sun.png":bo,"./assets/items/t/table.png":So,"./assets/items/t/teapot.png":wo,"./assets/items/t/tiger.png":Eo,"./assets/items/t/tomato.png":Oo,"./assets/items/t/tractor.png":Ao,"./assets/items/t/train.png":Mo,"./assets/items/t/tree.png":Po,"./assets/items/t/turtle.png":Io,"./assets/items/u/ukulele.png":Ro,"./assets/items/u/umbrella.png":Bo,"./assets/items/u/unicorn.png":Ho,"./assets/items/u/unicycle.png":Wo,"./assets/items/u/unidentified_object.png":Ko,"./assets/items/u/urchin.png":Jo,"./assets/items/u/utensils.png":Xo,"./assets/items/v/vacuum.png":Qo,"./assets/items/v/van.png":es,"./assets/items/v/vase.png":ns,"./assets/items/v/vegetables.png":is,"./assets/items/v/vest.png":os,"./assets/items/v/village.png":cs,"./assets/items/v/violin.png":us,"./assets/items/v/volcano.png":fs,"./assets/items/v/vulture.png":ms,"./assets/items/w/wagon.png":gs,"./assets/items/w/watch.png":vs,"./assets/items/w/watermelon.png":bs,"./assets/items/w/web.png":Ss,"./assets/items/w/whale.png":ws,"./assets/items/w/wind.png":Es,"./assets/items/w/wizard.png":Os,"./assets/items/w/wolf.png":As,"./assets/items/w/worm.png":Ms},gg={"./assets/voice/a/accordion.webm":Ps,"./assets/voice/a/acorn.webm":Is,"./assets/voice/a/airplane.webm":Rs,"./assets/voice/a/alligator.webm":Bs,"./assets/voice/a/anchor.webm":Hs,"./assets/voice/a/ant.webm":Ws,"./assets/voice/a/apple.webm":Ks,"./assets/voice/a/arrow.webm":Js,"./assets/voice/a/astronaut.webm":Xs,"./assets/voice/alphabet/a.webm":Qs,"./assets/voice/alphabet/b.webm":ec,"./assets/voice/alphabet/c.webm":nc,"./assets/voice/alphabet/d.webm":ic,"./assets/voice/alphabet/e.webm":oc,"./assets/voice/alphabet/f.webm":cc,"./assets/voice/alphabet/g.webm":uc,"./assets/voice/alphabet/h.webm":fc,"./assets/voice/alphabet/i.webm":mc,"./assets/voice/alphabet/j.webm":gc,"./assets/voice/alphabet/k.webm":vc,"./assets/voice/alphabet/l.webm":bc,"./assets/voice/alphabet/m.webm":Sc,"./assets/voice/alphabet/n.webm":wc,"./assets/voice/alphabet/o.webm":Ec,"./assets/voice/alphabet/p.webm":Oc,"./assets/voice/alphabet/q.webm":Ac,"./assets/voice/alphabet/r.webm":Mc,"./assets/voice/alphabet/s.webm":Pc,"./assets/voice/alphabet/t.webm":Ic,"./assets/voice/alphabet/u.webm":Rc,"./assets/voice/alphabet/v.webm":Bc,"./assets/voice/alphabet/w.webm":Hc,"./assets/voice/alphabet/x.webm":Wc,"./assets/voice/alphabet/y.webm":Kc,"./assets/voice/alphabet/z.webm":Jc,"./assets/voice/b/bananas.webm":Xc,"./assets/voice/b/beachball.webm":Qc,"./assets/voice/b/bear.webm":el,"./assets/voice/b/bee.webm":nl,"./assets/voice/b/bird.webm":il,"./assets/voice/b/boat.webm":ol,"./assets/voice/b/books.webm":cl,"./assets/voice/b/bus.webm":ul,"./assets/voice/b/butterfly.webm":fl,"./assets/voice/c/cake.webm":ml,"./assets/voice/c/car.webm":gl,"./assets/voice/c/castle.webm":vl,"./assets/voice/c/cat.webm":bl,"./assets/voice/c/clock.webm":Sl,"./assets/voice/c/coat.webm":wl,"./assets/voice/c/cow.webm":El,"./assets/voice/c/crayon.webm":Ol,"./assets/voice/c/cup.webm":Al,"./assets/voice/d/dog.webm":Ml,"./assets/voice/d/doll.webm":Pl,"./assets/voice/d/dolphin.webm":Il,"./assets/voice/d/donkey.webm":Rl,"./assets/voice/d/door.webm":Bl,"./assets/voice/d/dragon.webm":Hl,"./assets/voice/d/dress.webm":Wl,"./assets/voice/d/drum.webm":Kl,"./assets/voice/d/duck.webm":Jl,"./assets/voice/e/eagle.webm":Xl,"./assets/voice/e/ear.webm":Ql,"./assets/voice/e/earth.webm":eu,"./assets/voice/e/easel.webm":nu,"./assets/voice/e/egg.webm":iu,"./assets/voice/e/elephant.webm":ou,"./assets/voice/e/engine.webm":cu,"./assets/voice/e/envelope.webm":uu,"./assets/voice/e/eraser.webm":fu,"./assets/voice/f/feather.webm":mu,"./assets/voice/f/fence.webm":gu,"./assets/voice/f/firefly.webm":vu,"./assets/voice/f/fish.webm":bu,"./assets/voice/f/flower.webm":Su,"./assets/voice/f/foot.webm":wu,"./assets/voice/f/fork.webm":Eu,"./assets/voice/f/fox.webm":Ou,"./assets/voice/f/frog.webm":Au,"./assets/voice/g/garden.webm":Mu,"./assets/voice/g/ghost.webm":Pu,"./assets/voice/g/gift.webm":Iu,"./assets/voice/g/giraffe.webm":Ru,"./assets/voice/g/gloves.webm":Bu,"./assets/voice/g/goat.webm":Hu,"./assets/voice/g/goldfish.webm":Wu,"./assets/voice/g/grapes.webm":Ku,"./assets/voice/g/guitar.webm":Ju,"./assets/voice/h/hammer.webm":Xu,"./assets/voice/h/hand.webm":Qu,"./assets/voice/h/hat.webm":ed,"./assets/voice/h/heart.webm":nd,"./assets/voice/h/hedgehog.webm":id,"./assets/voice/h/helicopter.webm":od,"./assets/voice/h/horse.webm":cd,"./assets/voice/h/hotdog.webm":ud,"./assets/voice/h/house.webm":fd,"./assets/voice/i/ice_cream.webm":md,"./assets/voice/i/igloo.webm":gd,"./assets/voice/i/iguana.webm":vd,"./assets/voice/i/inkpot.webm":bd,"./assets/voice/i/insect.webm":Sd,"./assets/voice/i/instrument.webm":wd,"./assets/voice/i/invitation.webm":Ed,"./assets/voice/i/iron.webm":Od,"./assets/voice/i/island.webm":Ad,"./assets/voice/j/jaguar.webm":Md,"./assets/voice/j/jar.webm":Pd,"./assets/voice/j/jean_jacket.webm":Id,"./assets/voice/j/jelly.webm":Rd,"./assets/voice/j/jellyfish.webm":Bd,"./assets/voice/j/jet.webm":Hd,"./assets/voice/j/jigsaw.webm":Wd,"./assets/voice/j/juggler.webm":Kd,"./assets/voice/j/juice.webm":Jd,"./assets/voice/k/kangaroo.webm":Xd,"./assets/voice/k/kettle_drum.webm":Qd,"./assets/voice/k/key.webm":ef,"./assets/voice/k/kid.webm":nf,"./assets/voice/k/kite.webm":af,"./assets/voice/k/kiwi.webm":sf,"./assets/voice/k/knight.webm":lf,"./assets/voice/k/knot.webm":df,"./assets/voice/k/koala.webm":pf,"./assets/voice/l/labrador.webm":hf,"./assets/voice/l/ladybug.webm":_f,"./assets/voice/l/lamp.webm":yf,"./assets/voice/l/lemon.webm":xf,"./assets/voice/l/lemur.webm":Cf,"./assets/voice/l/lettuce.webm":Tf,"./assets/voice/l/library.webm":Df,"./assets/voice/l/llama.webm":kf,"./assets/voice/l/lungs.webm":jf,"./assets/voice/m/map.webm":Nf,"./assets/voice/m/milk_carton.webm":Ff,"./assets/voice/m/mitten.webm":Lf,"./assets/voice/m/monkey.webm":zf,"./assets/voice/m/moon.webm":Vf,"./assets/voice/m/mountain.webm":Uf,"./assets/voice/m/mouse.webm":Gf,"./assets/voice/m/muffin.webm":qf,"./assets/voice/m/mushroom.webm":Yf,"./assets/voice/n/nail.webm":Zf,"./assets/voice/n/needle.webm":$f,"./assets/voice/n/nest.webm":tp,"./assets/voice/n/net.webm":rp,"./assets/voice/n/newt.webm":ap,"./assets/voice/n/night.webm":sp,"./assets/voice/n/nose.webm":lp,"./assets/voice/n/nurse.webm":dp,"./assets/voice/n/nuts.webm":pp,"./assets/voice/o/oar.webm":hp,"./assets/voice/o/octopus.webm":_p,"./assets/voice/o/onion.webm":yp,"./assets/voice/o/orange.webm":xp,"./assets/voice/o/orangutan.webm":Cp,"./assets/voice/o/orbit.webm":Tp,"./assets/voice/o/ostrich.webm":Dp,"./assets/voice/o/overalls.webm":kp,"./assets/voice/o/owl.webm":jp,"./assets/voice/p/panda.webm":Np,"./assets/voice/p/pen.webm":Fp,"./assets/voice/p/pencil.webm":Lp,"./assets/voice/p/penguin.webm":zp,"./assets/voice/p/pig.webm":Vp,"./assets/voice/p/pineapple.webm":Up,"./assets/voice/p/pirate.webm":Gp,"./assets/voice/p/pizza.webm":qp,"./assets/voice/p/pumpkin.webm":Yp,"./assets/voice/q/quaff.webm":Zp,"./assets/voice/q/quail.webm":$p,"./assets/voice/q/quarter.webm":tm,"./assets/voice/q/queen.webm":rm,"./assets/voice/q/question.webm":am,"./assets/voice/q/quill.webm":sm,"./assets/voice/q/quilt.webm":lm,"./assets/voice/q/quince.webm":dm,"./assets/voice/q/quiver.webm":pm,"./assets/voice/r/rabbit.webm":hm,"./assets/voice/r/radishes.webm":_m,"./assets/voice/r/rainbow.webm":ym,"./assets/voice/r/ring.webm":xm,"./assets/voice/r/robin.webm":Cm,"./assets/voice/r/robot.webm":Tm,"./assets/voice/r/rocket.webm":Dm,"./assets/voice/r/rose.webm":km,"./assets/voice/r/ruler.webm":jm,"./assets/voice/s/sailboat.webm":Nm,"./assets/voice/s/sheep.webm":Fm,"./assets/voice/s/snake.webm":Lm,"./assets/voice/s/snowman.webm":zm,"./assets/voice/s/socks.webm":Vm,"./assets/voice/s/spoon.webm":Um,"./assets/voice/s/star.webm":Gm,"./assets/voice/s/strawberry.webm":qm,"./assets/voice/s/sun.webm":Ym,"./assets/voice/t/table.webm":Zm,"./assets/voice/t/teapot.webm":$m,"./assets/voice/t/teddy_bear.webm":th,"./assets/voice/t/tiger.webm":rh,"./assets/voice/t/tomato.webm":ah,"./assets/voice/t/tractor.webm":sh,"./assets/voice/t/train.webm":lh,"./assets/voice/t/tree.webm":dh,"./assets/voice/t/turtle.webm":ph,"./assets/voice/u/ukulele.webm":hh,"./assets/voice/u/umbrella.webm":_h,"./assets/voice/u/ungulate.webm":yh,"./assets/voice/u/unicorn.webm":xh,"./assets/voice/u/unicycle.webm":Ch,"./assets/voice/u/unidentified_object.webm":Th,"./assets/voice/u/uniform.webm":Dh,"./assets/voice/u/urchin.webm":kh,"./assets/voice/u/utensils.webm":jh,"./assets/voice/v/vacuum.webm":Nh,"./assets/voice/v/van.webm":Fh,"./assets/voice/v/vase.webm":Lh,"./assets/voice/v/vegetables.webm":zh,"./assets/voice/v/vest.webm":Vh,"./assets/voice/v/village.webm":Uh,"./assets/voice/v/violin.webm":Gh,"./assets/voice/v/volcano.webm":qh,"./assets/voice/v/vulture.webm":Yh,"./assets/voice/w/wagon.webm":Zh,"./assets/voice/w/watch.webm":$h,"./assets/voice/w/watermelon.webm":tg,"./assets/voice/w/web.webm":rg,"./assets/voice/w/whale.webm":ag,"./assets/voice/w/wind.webm":sg,"./assets/voice/w/wizard.webm":lg,"./assets/voice/w/wolf.webm":dg,"./assets/voice/w/worm.webm":pg},_g={"./assets/voice/alphabet/a.webm":Qs,"./assets/voice/alphabet/b.webm":ec,"./assets/voice/alphabet/c.webm":nc,"./assets/voice/alphabet/d.webm":ic,"./assets/voice/alphabet/e.webm":oc,"./assets/voice/alphabet/f.webm":cc,"./assets/voice/alphabet/g.webm":uc,"./assets/voice/alphabet/h.webm":fc,"./assets/voice/alphabet/i.webm":mc,"./assets/voice/alphabet/j.webm":gc,"./assets/voice/alphabet/k.webm":vc,"./assets/voice/alphabet/l.webm":bc,"./assets/voice/alphabet/m.webm":Sc,"./assets/voice/alphabet/n.webm":wc,"./assets/voice/alphabet/o.webm":Ec,"./assets/voice/alphabet/p.webm":Oc,"./assets/voice/alphabet/q.webm":Ac,"./assets/voice/alphabet/r.webm":Mc,"./assets/voice/alphabet/s.webm":Pc,"./assets/voice/alphabet/t.webm":Ic,"./assets/voice/alphabet/u.webm":Rc,"./assets/voice/alphabet/v.webm":Bc,"./assets/voice/alphabet/w.webm":Hc,"./assets/voice/alphabet/x.webm":Wc,"./assets/voice/alphabet/y.webm":Kc,"./assets/voice/alphabet/z.webm":Jc},vg=e=>{if(typeof e==`string`)return e;if(e&&typeof e==`object`&&`default`in e){let t=e.default;return typeof t==`string`?t:``}return``},yg=e=>{let t=e.split(`/`),n=t[t.length-1]??``;return{folder:t[t.length-2]??``,name:n.replace(/\.[^.]+$/,``)}},bg=new Map;for(let[e,t]of Object.entries(gg)){let{folder:n,name:r}=yg(e);if(n===`alphabet`)continue;let i=vg(t);i&&(bg.has(n)||bg.set(n,new Map),bg.get(n).set(r,i))}var xg=new Map;for(let[e,t]of Object.entries(hg)){let{folder:n,name:r}=yg(e),i=r.replace(/_/g,` `),a=vg(t);if(!a)continue;let o=bg.get(n)?.get(r);xg.has(n)||xg.set(n,[]),xg.get(n).push({name:r,label:i,imageUrl:a,audioUrl:o})}var Sg=new Map;for(let[e,t]of Object.entries(_g)){let{name:n}=yg(e),r=vg(t);r&&Sg.set(n,r)}var Cg=(e,t)=>{let n=[...e];for(let e=n.length-1;e>0;--e){let t=Math.floor(Math.random()*(e+1));[n[e],n[t]]=[n[t],n[e]]}return n.slice(0,Math.min(t,n.length))};async function wg(){let e=document.querySelector(`#app`),t=new URL(`/letter_constellations/assets/aurora-BhVtVCtR.mp3`,``+import.meta.url).toString(),n=new Audio(t);n.loop=!0,n.preload=`auto`,n.volume=1;let r=new URL(`/letter_constellations/assets/chime-DLBpLcOd.mp3`,``+import.meta.url).toString(),i=new Audio(r);i.preload=`auto`;let a=null,s=null,c=.33,l=()=>{if(a)return;a=new AudioContext;let e=a.createMediaElementSource(n);s=a.createGain(),s.gain.value=.25*c,e.connect(s).connect(a.destination)},u=document.createElement(`canvas`);u.id=`aurora-canvas`,e.appendChild(u);let d=document.createElement(`canvas`);d.id=`sparkle-canvas`,e.appendChild(d);let f=document.createElement(`canvas`);f.id=`game-canvas`,e.appendChild(f);let p=document.createElement(`div`);p.className=`controls-bar`;let m=document.createElement(`div`);m.className=`top-controls`;let h=document.createElement(`button`);h.className=`control-button`,h.type=`button`,h.textContent=`Clear`,m.appendChild(h);let g=document.createElement(`button`);g.className=`control-button`,g.type=`button`,g.textContent=`Reset`,m.appendChild(g);let _=document.createElement(`button`);_.className=`mute-button`,_.type=`button`,_.textContent=`Mute`,_.setAttribute(`aria-pressed`,`false`),p.appendChild(m);let v=document.createElement(`div`);v.className=`volume-control`,v.innerHTML=`
    <label for="volume-slider">Volume</label>
    <input id="volume-slider" type="range" min="0" max="2" step="0.05" value="0.33" />
  `,p.appendChild(v),p.appendChild(_),e.appendChild(p);let y=document.createElement(`button`);y.className=`settings-button`,y.type=`button`,y.textContent=`Letters`,e.appendChild(y);let b=document.createElement(`div`);b.className=`settings-overlay`;let x=document.createElement(`div`);x.className=`settings-modal`;let S=document.createElement(`h2`);S.textContent=`Select Letters`,x.appendChild(S);let C=document.createElement(`div`);C.className=`settings-error`,C.textContent=`Please select at least one letter`,x.appendChild(C);let w=document.createElement(`div`);w.className=`letter-grid`;let ne=`abcdefghijklmnopqrstuvwxyz`.split(``),re=new Set(Se),E=`letterConstellations_enabledLetters`,ie=()=>{try{let e=localStorage.getItem(E);if(e){let t=JSON.parse(e).filter(e=>re.has(e));if(t.length>0)return new Set(t)}}catch{}return new Set(Se)},ae=e=>{try{localStorage.setItem(E,JSON.stringify([...e]))}catch{}},D=ie(),O=new Map;for(let e of ne){let t=re.has(e),n=document.createElement(`input`);n.type=`checkbox`,n.className=`letter-checkbox`,n.id=`letter-${e}`,n.checked=D.has(e),n.disabled=!t;let r=document.createElement(`label`);r.className=`letter-label`,r.htmlFor=`letter-${e}`,r.textContent=e,t||(r.title=`Coming soon`),w.appendChild(n),w.appendChild(r),t&&O.set(e,n)}x.appendChild(w);let k=document.createElement(`div`);k.className=`settings-buttons`;let A=document.createElement(`button`);A.className=`settings-btn`,A.type=`button`,A.textContent=`Select All`;let j=document.createElement(`button`);j.className=`settings-btn`,j.type=`button`,j.textContent=`Select None`;let M=document.createElement(`button`);M.className=`settings-btn primary`,M.type=`button`,M.textContent=`Done`,k.appendChild(A),k.appendChild(j),k.appendChild(M),x.appendChild(k),b.appendChild(x),e.appendChild(b);let oe=()=>{D=new Set;for(let[e,t]of O)t.checked&&D.add(e)},N=e=>{e?C.classList.add(`visible`):C.classList.remove(`visible`)},se=()=>{for(let[e,t]of O)t.checked=D.has(e);N(!1),b.classList.add(`visible`)},P=e=>{if(e){if(oe(),D.size===0){N(!0);return}ae(D),Y&&Y.setEnabledLetters(D)}N(!1),b.classList.remove(`visible`)};y.addEventListener(`click`,e=>{e.stopPropagation(),se()}),A.addEventListener(`click`,e=>{e.stopPropagation();for(let e of O.values())e.checked=!0;N(!1)}),j.addEventListener(`click`,e=>{e.stopPropagation();for(let e of O.values())e.checked=!1}),M.addEventListener(`click`,e=>{e.stopPropagation(),P(!0)}),b.addEventListener(`click`,e=>{e.target===b&&P(!1)}),document.addEventListener(`keydown`,e=>{e.key===`Escape`&&b.classList.contains(`visible`)&&P(!1)});let F=v.querySelector(`#volume-slider`);c=Number(F.value),i.volume=Math.min(1,c);let ce=new Map,I=new Set,L=null,le=0,R=null,ue=e=>{let t=ce.get(e);return t||(t=new Audio(e),t.preload=`auto`,ce.set(e,t),I.add(t)),t.volume=Math.min(1,Number(F.value)),t.muted=n.muted,t},de=()=>{L&&(L.pause(),L.currentTime=0)},z=async e=>{if(e.length===0)return;let t=++le;de();for(let n of e){if(t!==le)return;let e=ue(n);L=e,e.currentTime=0;try{await e.play()}catch{return}await new Promise(t=>{let n=()=>{e.removeEventListener(`ended`,n),e.removeEventListener(`error`,n),t()};e.addEventListener(`ended`,n),e.addEventListener(`error`,n)})}},fe=e=>{let t=Sg.get(e);if(t){if(!B){R=t;return}z([t])}},pe=()=>{let e=n.muted;_.textContent=e?`Unmute`:`Mute`,_.setAttribute(`aria-pressed`,String(e)),i.muted=e;for(let t of I)t.muted=e};pe();let B=!1,me=()=>{B&&(l(),a&&a.state===`suspended`&&a.resume().catch(()=>{}),n.play().catch(()=>{}))},V=()=>{if(B=!0,me(),R){let e=R;R=null,z([e])}},H={once:!0,capture:!0};document.addEventListener(`pointerdown`,V,H),document.addEventListener(`touchstart`,V,H),document.addEventListener(`click`,V,H),document.addEventListener(`keydown`,V,H),_.addEventListener(`click`,e=>{e.stopPropagation(),n.muted=!n.muted,n.muted||me(),pe()}),F.addEventListener(`input`,()=>{l();let e=Number(F.value);c=e,s&&(s.gain.value=.25*e),i.volume=Math.min(1,e);let t=Math.min(1,e);for(let e of I)e.volume=t});let U=document.createElement(`div`);U.className=`letter-panel`;let W=document.createElement(`button`);W.className=`letter-display`,W.type=`button`,W.setAttribute(`aria-label`,`Play letter sound`);let G=document.createElement(`div`);G.className=`item-tray letter-panel__item`,U.appendChild(W),U.appendChild(G),e.appendChild(U);let K=``,he=e=>{G.replaceChildren(),W.textContent=e,K=e;let t=xg.get(e)??[];if(t.length===0){let t=document.createElement(`div`);t.className=`item-empty`,t.textContent=`No ${e.toUpperCase()} items yet.`,G.appendChild(t);return}let n=Cg(t,1);for(let t of n){let n=document.createElement(`button`);n.type=`button`,n.className=`item-card`,n.setAttribute(`aria-label`,t.label),n.addEventListener(`click`,n=>{n.stopPropagation();let r=[],i=Sg.get(e);i&&r.push(i),t.audioUrl&&r.push(t.audioUrl),z(r)});let r=document.createElement(`img`);r.src=t.imageUrl,r.alt=t.label,r.loading=`lazy`,r.decoding=`async`,n.appendChild(r),G.appendChild(n)}},q=null,J=null,Y=null,ge=new je,X=new T(u,f,d);X.updateQuality(ge.getQualitySettings());let Z=f.getContext(`2d`);if(!Z)throw Error(`Failed to get 2D context for game canvas.`);let Q=()=>{let{width:e,height:t}=X.resize();X.configureGameContext(Z),Y&&Y.setViewportSize(e,t),q&&q.resize(u.width,u.height),J&&J.resize(d.width,d.height)};ge.setOnQualityChange(e=>{X.updateQuality(e),Q()}),Q(),window.addEventListener(`resize`,Q),q=new o(u);let _e=await q.initialize();J=new ee(d);let ve=await J.initialize(q.getDevice());if(Q(),!_e||!ve){let e=document.createElement(`div`);e.className=`error-message`,e.innerHTML=`
            <h1>WebGPU Not Supported</h1>
            <p>Your browser doesn't support WebGPU yet.</p>
            <p>Try using Chrome 113+ or Edge 113+ with WebGPU enabled.</p>
        `,document.body.appendChild(e);return}Y=new ke(new te(f,e=>X.getPointFromEvent(f,e)),{onPointHit:()=>{i.currentTime=0,i.play().catch(()=>{})},onLetterChange:e=>{he(e),fe(e)}}),Y.setEnabledLetters(D),W.addEventListener(`click`,e=>{e.stopPropagation(),K&&fe(K)}),h.addEventListener(`click`,e=>{e.stopPropagation(),Y?.clearUserPaths()}),g.addEventListener(`click`,e=>{e.stopPropagation(),Y?.resetCurrentLetter()});let ye=X.getSize();Y.setViewportSize(ye.width,ye.height);let be=performance.now(),xe=0;function $(e){ge.recordFrame();let t=(e-be)/1e3;if(be=e,!Y||!q||!J)return;Y.update(t);let n=Y.getDrawingState();J.updateMouseState(n.isDrawing,n.mouseX,n.mouseY,n.trailPoints);let r=Y.getDotState();J.setDots(r.dots,r.currentIndex,r.radius),xe++&1||q.render(),J.render(),Z&&Y.render(Z),requestAnimationFrame($)}$(performance.now())}wg();
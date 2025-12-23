// High-performance Sparkle Renderer (instanced GPU particles)
// - No fullscreen trail-distance scan
// - Particles spawned from cursor speed/direction + light trail twinkles
// - GPU computes motion from time - spawnTime (no compute pass)

import dotShaderCode from './shaders/dots.wgsl?raw';
import sparkShaderCode from './shaders/sparks.wgsl?raw';

export type TrailPoint = {
  x: number; // CSS pixels from Game
  y: number; // CSS pixels from Game
  time: number;
};

const MAX_PARTICLES = 24576;
const FLOATS_PER_PARTICLE = 8; // pos0x,pos0y, velx,vely, spawn,life, sizePx, seed
const BYTES_PER_PARTICLE = FLOATS_PER_PARTICLE * 4;
const MAX_SPAWNS_PER_FRAME = 1536;
const MAX_DOTS = 512;
const FLOATS_PER_DOT = 8; // posx,posy,sizePx,state, seed,...
const BYTES_PER_DOT = FLOATS_PER_DOT * 4;
const KIND_DUST = 0;
const KIND_SPARK = 1;
const KIND_STAR = 2;
const KIND_COMET = 3;

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

export class SparkleRenderer {
  private canvas: HTMLCanvasElement;
  private device!: GPUDevice;
  private context!: GPUCanvasContext;

  private pipeline!: GPURenderPipeline;
  private dotPipeline!: GPURenderPipeline;
  private uniformBuffer!: GPUBuffer;
  private dotUniformBuffer!: GPUBuffer;
  private bindGroup!: GPUBindGroup;
  private dotBindGroup!: GPUBindGroup;

  private quadVertexBuffer!: GPUBuffer;
  private particleBuffer!: GPUBuffer;
  private dotBuffer!: GPUBuffer;

  private startTimeMs: number;

  // Uniform scratch (32 bytes = 8 floats)
  private uniformScratch = new Float32Array(8);
  private dotUniformScratch = new Float32Array(8);

  // Mouse state (normalized 0..1)
  private mouseActive = false;
  private mouseRamp = 0;
  private prevActive = false;

  private mousePos = { x: 0.5, y: 0.5 };
  private mouseVelocity = { x: 0, y: 0 };

  private lastSimTimeSec = 0;

  // Trail points from Game (CSS px)
  private lastTrailPoints: TrailPoint[] = [];

  // Dot rendering state
  private dotCount = 0;
  private dotScratch = new Float32Array(MAX_DOTS * FLOATS_PER_DOT);

  // Particle ring buffer + batched writes (at most 2 writeBuffer calls/frame)
  private nextParticleIndex = 0;
  private pendingStart = 0;
  private pendingCount = 0;
  private pendingData = new Float32Array(MAX_SPAWNS_PER_FRAME * FLOATS_PER_PARTICLE);

  // Emission accumulators
  private sparkAccumulator = 0;
  private dustAccumulator = 0;
  private burstCooldown = 0;
  private haloPhase = 0;
  private haloAccumulator = 0;
  private stillSeconds = 0;
  private lastStampTime = -999;

  // Fast RNG (xorshift32)
  private rngState = 0x12345678;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.startTimeMs = performance.now();
  }

  async initialize(existingDevice?: GPUDevice): Promise<boolean> {
    if (existingDevice) {
      this.device = existingDevice;
    } else {
      if (!navigator.gpu) {
        console.error('WebGPU not supported');
        return false;
      }
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        console.error('Failed to get GPU adapter');
        return false;
      }
      this.device = await adapter.requestDevice();
    }

    this.context = this.canvas.getContext('webgpu') as GPUCanvasContext;
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    this.context.configure({
      device: this.device,
      format: presentationFormat,
      alphaMode: 'premultiplied',
    });

    // Uniform buffer (matches sparks.wgsl Uniforms struct; allocate 32 bytes)
    this.uniformBuffer = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.dotUniformBuffer = this.device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Quad vertices (two triangles, 6 verts, vec2 each)
    this.quadVertexBuffer = this.device.createBuffer({
      size: 6 * 2 * 4,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.quadVertexBuffer.getMappedRange()).set([
      -1, -1,
      +1, -1,
      -1, +1,
      -1, +1,
      +1, -1,
      +1, +1,
    ]);
    this.quadVertexBuffer.unmap();

    // Instance buffer (initialize to 0 so all particles start "dead")
    this.particleBuffer = this.device.createBuffer({
      size: MAX_PARTICLES * BYTES_PER_PARTICLE,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.particleBuffer.getMappedRange()).fill(0);
    this.particleBuffer.unmap();

    this.dotBuffer = this.device.createBuffer({
      size: MAX_DOTS * BYTES_PER_DOT,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.dotBuffer.getMappedRange()).fill(0);
    this.dotBuffer.unmap();

    const dotShaderModule = this.device.createShaderModule({ code: dotShaderCode });
    const shaderModule = this.device.createShaderModule({ code: sparkShaderCode });

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
      ],
    });

    this.bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
    });

    this.dotBindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.dotUniformBuffer } }],
    });

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });

    // Additive color blending (fireworks/glitter feel)
    this.pipeline = this.device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'sparkVertexMain',
        buffers: [
          {
            arrayStride: 2 * 4,
            stepMode: 'vertex',
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
          },
          {
            arrayStride: BYTES_PER_PARTICLE,
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 1, offset: 0, format: 'float32x4' },
              { shaderLocation: 2, offset: 16, format: 'float32x4' },
            ],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'sparkFragmentMain',
        targets: [
          {
            format: presentationFormat,
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
              // Keep alpha compositing sane while color is additive
              alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add',
              },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    });

    this.dotPipeline = this.device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: {
        module: dotShaderModule,
        entryPoint: 'dotVertexMain',
        buffers: [
          {
            arrayStride: 2 * 4,
            stepMode: 'vertex',
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }],
          },
          {
            arrayStride: BYTES_PER_DOT,
            stepMode: 'instance',
            attributes: [
              { shaderLocation: 1, offset: 0, format: 'float32x4' },
              { shaderLocation: 2, offset: 16, format: 'float32x4' },
            ],
          },
        ],
      },
      fragment: {
        module: dotShaderModule,
        entryPoint: 'dotFragmentMain',
        targets: [
          {
            format: presentationFormat,
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
              alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add',
              },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    });

    return true;
  }

  setDots(dots: Array<{ x: number; y: number }>, currentIndex: number, radius: number): void {
    const count = Math.min(dots.length, MAX_DOTS);
    this.dotCount = count;
    if (count === 0) {
      return;
    }

    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const cssW = Math.max(1, this.canvas.width / dpr);
    const cssH = Math.max(1, this.canvas.height / dpr);

    const baseRadius = Math.max(6, radius);

    for (let i = 0; i < count; i += 1) {
      const dot = dots[i];
      const nx = dot.x / cssW;
      const ny = dot.y / cssH;

      let state = 0;
      if (i < currentIndex) {
        state = 2;
      } else if (i === currentIndex) {
        state = 1;
      }

      const sizeCss =
        state === 1 ? baseRadius * 1.25 : state === 2 ? baseRadius * 1.05 : baseRadius * 0.95;
      const sizePx = sizeCss * dpr;

      const base = i * FLOATS_PER_DOT;
      this.dotScratch[base + 0] = nx;
      this.dotScratch[base + 1] = ny;
      this.dotScratch[base + 2] = sizePx;
      this.dotScratch[base + 3] = state;
      this.dotScratch[base + 4] = this.hashSeed(i);
      this.dotScratch[base + 5] = 0;
      this.dotScratch[base + 6] = 0;
      this.dotScratch[base + 7] = 0;
    }

    this.device.queue.writeBuffer(
      this.dotBuffer,
      0,
      this.dotScratch.subarray(0, count * FLOATS_PER_DOT)
    );
  }

  updateMouseState(active: boolean, x: number, y: number, trailPoints: TrailPoint[]): void {
    this.mouseActive = active;
    this.lastTrailPoints = trailPoints;

    const nowSec = (performance.now() - this.startTimeMs) / 1000;
    const dtRaw = this.lastSimTimeSec === 0 ? 0 : nowSec - this.lastSimTimeSec;
    const dt = Math.max(0, Math.min(0.05, dtRaw));
    this.lastSimTimeSec = nowSec;

    // Ramp in/out (prevents harsh pop)
    if (dt > 0) {
      const rampStep = dt / 0.45;
      this.mouseRamp = active
        ? Math.min(1, this.mouseRamp + rampStep)
        : Math.max(0, this.mouseRamp - rampStep);
    }

    // Convert CSS px -> normalized UV
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const cssW = Math.max(1, this.canvas.width / dpr);
    const cssH = Math.max(1, this.canvas.height / dpr);

    const nx = Math.max(0, Math.min(1, x / cssW));
    const ny = Math.max(0, Math.min(1, y / cssH));

    const prevPos = { ...this.mousePos };
    const curPos = { x: nx, y: ny };
    this.mousePos = curPos;

    // Velocity in UV/sec (EMA smoothed)
    if (dt > 0) {
      const rawVx = (curPos.x - prevPos.x) / dt;
      const rawVy = (curPos.y - prevPos.y) / dt;
      this.mouseVelocity.x = this.mouseVelocity.x * 0.7 + rawVx * 0.3;
      this.mouseVelocity.y = this.mouseVelocity.y * 0.7 + rawVy * 0.3;
    }

    const speed = Math.hypot(this.mouseVelocity.x, this.mouseVelocity.y);
    const speedBoost = smoothstep(0.08, 2.4, speed);

    // --- Magic wand halo: orbiting fairy dust around the cursor ---
    if (dt > 0 && this.mouseRamp > 0.02) {
      this.haloPhase += dt * (1.2 + speedBoost * 5.5);

      // small, steady emission; increases when moving fast
      const haloRate = (18 + speedBoost * 140) * this.mouseRamp; // dust/sec
      this.haloAccumulator += haloRate * dt;

      let haloCount = Math.floor(this.haloAccumulator);
      this.haloAccumulator -= haloCount;

      // hard cap keeps perf stable
      haloCount = Math.min(haloCount, 12);

      const radius = 0.014 + 0.012 * speedBoost;

      for (let i = 0; i < haloCount; i++) {
        const ang = this.haloPhase * Math.PI * 2 + this.rand01() * Math.PI * 2;

        const ox = Math.cos(ang) * radius;
        const oy = Math.sin(ang) * radius;

        // tangential motion + tiny outward drift
        const tang = 0.18 + speedBoost * 0.65;
        const vx = -Math.sin(ang) * tang + ox * 0.8;
        const vy = Math.cos(ang) * tang + oy * 0.8;

        const life = 0.35 + this.rand01() * 0.55;
        const sizeCss = 0.9 + this.rand01() * 2.2;
        const sizePx = sizeCss * dpr;

        this.queueParticle(
          curPos.x + ox,
          curPos.y + oy,
          vx,
          vy,
          nowSec,
          life,
          sizePx,
          this.makeSeed(KIND_DUST)
        );
      }
    }

    // --- “Star stamp” when the user pauses (super satisfying / kid-friendly) ---
    if (dt > 0) {
      if (active && speed < 0.07) {
        this.stillSeconds += dt;
      } else {
        this.stillSeconds = 0;
      }

      if (active && this.stillSeconds > 0.22 && nowSec - this.lastStampTime > 0.35) {
        this.stampHeroStar(nowSec, curPos, dpr, 10, 18, 1.1, 1.8);
        this.emitDustRing(nowSec, curPos, dpr, 18);
        this.lastStampTime = nowSec;
        this.stillSeconds = 0.12;
      }
    }

    // Bursts on press + occasional flick bursts
    if (active && !this.prevActive) {
      this.emitRadialBurst(nowSec, curPos, dpr, 300);
      this.burstCooldown = 0.12;
    }

    if (dt > 0) {
      this.burstCooldown = Math.max(0, this.burstCooldown - dt);
      if (active && this.burstCooldown <= 0 && speedBoost > 0.85 && this.mouseRamp > 0.25) {
        this.emitDirectionalBurst(nowSec, curPos, dpr, 110);
        this.burstCooldown = 0.12;
      }
    }

    // Continuous directional emission
    if (dt > 0) {
      const rate = (120 + speedBoost * 1700) * this.mouseRamp; // sparks/sec
      this.sparkAccumulator += rate * dt;

      let emitCount = Math.floor(this.sparkAccumulator);
      this.sparkAccumulator -= emitCount;
      emitCount = Math.min(emitCount, 180);

      if (emitCount > 0) {
        // Direction basis from cursor velocity
        let dirX = this.mouseVelocity.x;
        let dirY = this.mouseVelocity.y;
        const dirLen = Math.hypot(dirX, dirY);
        if (dirLen > 1e-5) {
          dirX /= dirLen;
          dirY /= dirLen;
        } else {
          // fallback direction if barely moving
          const a = this.rand01() * Math.PI * 2;
          dirX = Math.cos(a);
          dirY = Math.sin(a);
        }
        const perpX = -dirY;
        const perpY = dirX;

        // Narrower cone at high speed
        const cone = (1.25 - speedBoost * 0.9);

        for (let i = 0; i < emitCount; i++) {
          const u = this.rand01();
          const px = prevPos.x + (curPos.x - prevPos.x) * u;
          const py = prevPos.y + (curPos.y - prevPos.y) * u;

          const jitter = (this.rand01() - 0.5) * 0.01;
          const jx = perpX * jitter;
          const jy = perpY * jitter;

          const angleOff = (this.rand01() - 0.5) * cone;
          const ca = Math.cos(angleOff);
          const sa = Math.sin(angleOff);

          const rx = dirX * ca - dirY * sa;
          const ry = dirX * sa + dirY * ca;

          const vMag = 0.1 + this.rand01() * 0.2 + speedBoost * 0.75;
          const side = (this.rand01() - 0.5) * (0.36 - speedBoost * 0.18);
          const vx = rx * vMag + perpX * side * vMag;
          const vy = ry * vMag + perpY * side * vMag;

          const isComet = speedBoost > 0.75 && this.rand01() < 0.06;
          const kind = isComet ? KIND_COMET : KIND_SPARK;

          const life = isComet
            ? (0.45 + this.rand01() * 0.55)
            : (0.20 + this.rand01() * 0.38 + speedBoost * 0.18);

          const sizeCss = isComet
            ? (6.0 + this.rand01() * 10.0 + speedBoost * 10.0)
            : (1.4 + this.rand01() * 3.4 + speedBoost * 8.0);
          const sizePx = sizeCss * dpr;

          this.queueParticle(px + jx, py + jy, vx, vy, nowSec, life, sizePx, this.makeSeed(kind));
        }
      }
    }

    // Trail twinkles (very cheap, makes letters sparkle even when stopped)
    if (dt > 0 && trailPoints.length > 0) {
      const trailStrength = Math.min(1, trailPoints.length / 160);
      const base = 45 + 160 * trailStrength;
      const rate = base * (0.25 + 0.75 * this.mouseRamp); // dust/sec
      this.dustAccumulator += rate * dt;

      let dustCount = Math.floor(this.dustAccumulator);
      this.dustAccumulator -= dustCount;
      dustCount = Math.min(dustCount, 60);

      for (let i = 0; i < dustCount; i++) {
        const p = trailPoints[(this.rand01() * trailPoints.length) | 0];
        const tx = Math.max(0, Math.min(1, p.x / cssW));
        const ty = Math.max(0, Math.min(1, p.y / cssH));

        const a = this.rand01() * Math.PI * 2;
        const drift = 0.014 + this.rand01() * 0.03;
        const vx = Math.cos(a) * drift;
        const vy = Math.sin(a) * drift;

        const life = 0.45 + this.rand01() * 1.2;
        const sizeCss = 0.9 + this.rand01() * 2.8;
        const sizePx = sizeCss * dpr;

        this.queueParticle(tx, ty, vx, vy, nowSec, life, sizePx, this.makeSeed(KIND_DUST));
      }
    }

    // Decay velocity when not drawing
    if (!active) {
      this.mouseVelocity.x *= 0.95;
      this.mouseVelocity.y *= 0.95;
    }

    // --- Big “finish” star on release ---
    if (!active && this.prevActive) {
      this.stampHeroStar(nowSec, curPos, dpr, 14, 26, 1.4, 2.4);
      this.emitDustRing(nowSec, curPos, dpr, 28);
    }

    this.prevActive = active;
  }

  render(): void {
    const timeSec = (performance.now() - this.startTimeMs) / 1000;

    // Flush pending particle spawns (<= 2 writeBuffer calls)
    this.flushPendingParticles();

    // Update uniforms (must match sparks.wgsl layout)
    this.uniformScratch[0] = timeSec;
    this.uniformScratch[1] = 0;
    this.uniformScratch[2] = this.canvas.width;
    this.uniformScratch[3] = this.canvas.height;
    this.uniformScratch[4] = this.mouseRamp;
    this.uniformScratch[5] = 0;
    this.uniformScratch[6] = 0;
    this.uniformScratch[7] = 0;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformScratch);

    // Update dot uniforms (time + resolution + full intensity)
    this.dotUniformScratch[0] = timeSec;
    this.dotUniformScratch[1] = 0;
    this.dotUniformScratch[2] = this.canvas.width;
    this.dotUniformScratch[3] = this.canvas.height;
    this.dotUniformScratch[4] = 1;
    this.dotUniformScratch[5] = 0;
    this.dotUniformScratch[6] = 0;
    this.dotUniformScratch[7] = 0;
    this.device.queue.writeBuffer(this.dotUniformBuffer, 0, this.dotUniformScratch);

    const commandEncoder = this.device.createCommandEncoder();

    const pass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });

    if (this.dotCount > 0) {
      pass.setPipeline(this.dotPipeline);
      pass.setBindGroup(0, this.dotBindGroup);
      pass.setVertexBuffer(0, this.quadVertexBuffer);
      pass.setVertexBuffer(1, this.dotBuffer);
      pass.draw(6, this.dotCount, 0, 0);
    }

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.quadVertexBuffer);
    pass.setVertexBuffer(1, this.particleBuffer);
    pass.draw(6, MAX_PARTICLES, 0, 0);
    pass.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }

  resize(width: number, height: number): void {
    this.canvas.width = width;
    this.canvas.height = height;

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: presentationFormat,
      alphaMode: 'premultiplied',
    });
  }

  // -------- internals --------

  private rand01(): number {
    // xorshift32
    let x = this.rngState | 0;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.rngState = x;
    return ((x >>> 0) / 4294967296);
  }

  private hashSeed(index: number): number {
    let x = (index + 1) * 0x9e3779b1;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return ((x >>> 0) / 4294967296);
  }

  private makeSeed(kind: number): number {
    return kind + this.rand01();
  }

  private queueParticle(
    posX: number,
    posY: number,
    velX: number,
    velY: number,
    spawnTime: number,
    life: number,
    sizePx: number,
    seed: number
  ): void {
    if (this.pendingCount >= MAX_SPAWNS_PER_FRAME) return;

    if (this.pendingCount === 0) {
      this.pendingStart = this.nextParticleIndex;
    }

    this.nextParticleIndex = (this.nextParticleIndex + 1) % MAX_PARTICLES;

    const base = this.pendingCount * FLOATS_PER_PARTICLE;
    const a = this.pendingData;
    a[base + 0] = posX;
    a[base + 1] = posY;
    a[base + 2] = velX;
    a[base + 3] = velY;
    a[base + 4] = spawnTime;
    a[base + 5] = life;
    a[base + 6] = sizePx;
    a[base + 7] = seed;

    this.pendingCount++;
  }

  private flushPendingParticles(): void {
    if (this.pendingCount === 0) return;

    const total = this.pendingCount;
    const start = this.pendingStart;

    const firstLen = Math.min(total, MAX_PARTICLES - start);
    const firstFloats = firstLen * FLOATS_PER_PARTICLE;

    this.device.queue.writeBuffer(
      this.particleBuffer,
      start * BYTES_PER_PARTICLE,
      this.pendingData.subarray(0, firstFloats)
    );

    if (total > firstLen) {
      const secondLen = total - firstLen;
      const secondFloats = secondLen * FLOATS_PER_PARTICLE;

      this.device.queue.writeBuffer(
        this.particleBuffer,
        0,
        this.pendingData.subarray(firstFloats, firstFloats + secondFloats)
      );
    }

    this.pendingCount = 0;
  }

  private emitRadialBurst(nowSec: number, pos: { x: number; y: number }, dpr: number, count: number): void {
    const n = Math.min(count, MAX_SPAWNS_PER_FRAME);

    for (let i = 0; i < n; i++) {
      const a = this.rand01() * Math.PI * 2;
      const v = 0.14 + this.rand01() * 0.8;
      const vx = Math.cos(a) * v;
      const vy = Math.sin(a) * v;

      const life = 0.32 + this.rand01() * 0.75;
      const sizeCss = 2.2 + this.rand01() * 7.0;
      const sizePx = sizeCss * dpr;

      this.queueParticle(pos.x, pos.y, vx, vy, nowSec, life, sizePx, this.makeSeed(KIND_SPARK));
    }
  }

  private emitDirectionalBurst(nowSec: number, pos: { x: number; y: number }, dpr: number, count: number): void {
    let dx = this.mouseVelocity.x;
    let dy = this.mouseVelocity.y;
    const len = Math.hypot(dx, dy);
    if (len > 1e-5) {
      dx /= len;
      dy /= len;
    } else {
      dx = 1;
      dy = 0;
    }
    const px = -dy;
    const py = dx;

    const n = Math.min(count, MAX_SPAWNS_PER_FRAME);

    for (let i = 0; i < n; i++) {
      const cone = 0.35;
      const off = (this.rand01() - 0.5) * cone;
      const ca = Math.cos(off);
      const sa = Math.sin(off);
      const rx = dx * ca - dy * sa;
      const ry = dx * sa + dy * ca;

      const v = 0.17 + this.rand01() * 1.0;
      const side = (this.rand01() - 0.5) * 0.25;

      const vx = rx * v + px * side;
      const vy = ry * v + py * side;

      const life = 0.28 + this.rand01() * 0.6;
      const sizeCss = 2.0 + this.rand01() * 8.5;
      const sizePx = sizeCss * dpr;

      this.queueParticle(pos.x, pos.y, vx, vy, nowSec, life, sizePx, this.makeSeed(KIND_SPARK));
    }
  }

  private stampHeroStar(
    nowSec: number,
    pos: { x: number; y: number },
    dpr: number,
    sizeCssMin: number,
    sizeCssMax: number,
    lifeMin: number,
    lifeMax: number
  ): void {
    const sizeCss = sizeCssMin + this.rand01() * (sizeCssMax - sizeCssMin);
    const sizePx = sizeCss * dpr;

    const life = lifeMin + this.rand01() * (lifeMax - lifeMin);

    // slow float upward (UV y+ is down, so negative y goes up)
    const vx = (this.rand01() - 0.5) * 0.05;
    const vy = -0.05 - this.rand01() * 0.07;

    this.queueParticle(
      pos.x,
      pos.y,
      vx,
      vy,
      nowSec,
      life,
      sizePx,
      this.makeSeed(KIND_STAR)
    );
  }

  private emitDustRing(nowSec: number, pos: { x: number; y: number }, dpr: number, count: number): void {
    const n = Math.min(count, 40);
    const radius = 0.018;

    for (let i = 0; i < n; i++) {
      const a = this.rand01() * Math.PI * 2;
      const ox = Math.cos(a) * radius * (0.6 + this.rand01() * 0.6);
      const oy = Math.sin(a) * radius * (0.6 + this.rand01() * 0.6);

      const drift = 0.03 + this.rand01() * 0.08;
      const vx = ox * drift * 30;
      const vy = oy * drift * 30;

      const life = 0.5 + this.rand01() * 0.7;
      const sizePx = (0.8 + this.rand01() * 2.0) * dpr;

      this.queueParticle(
        pos.x + ox,
        pos.y + oy,
        vx,
        vy,
        nowSec,
        life,
        sizePx,
        this.makeSeed(KIND_DUST)
      );
    }
  }
}
